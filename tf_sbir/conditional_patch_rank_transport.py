"""Calibrate conditional neural activation ranks across the patches of a single image."""

import torch

from .semantic_neuron_gallery import nearest


FAMILY="conditional-patch-rank-transport"
LAYERS=(2,3)
GROUPS=("sensor","curvature","sensor_matched")
K=64
N=8
METHODS=("conditional_scene","conditional_shared","conditional_uniform","rotate_ranks","conditional_mean",
         "fixed_median","ordinary_histogram","shuffled_donors","identity")


def cdf(samples,values):
    # Linear between distinct observations, with the same midrank for tied measured activations.
    samples=samples.contiguous()
    values=values.contiguous().unsqueeze(-1)
    left=torch.searchsorted(samples,values,right=False)
    right=torch.searchsorted(samples,values,right=True)
    n=samples.shape[-1]
    low=samples.gather(-1,(left-1).clamp(0,n-1))
    high=samples.gather(-1,left.clamp(0,n-1))
    pos=left.float()-.5+(values-low)/(high-low).clamp_min(1e-20)
    rank=torch.where(left!=right,.5*(left+right).float(),pos)
    rank=torch.where((left==0)&(right==0),torch.zeros_like(rank),rank)
    rank=torch.where(left==n,torch.full_like(rank,float(n)),rank)
    return (rank/n).squeeze(-1).clamp(0,1)


def quantile(samples,probability):
    n=samples.size(-1)
    index=(n*probability-.5).clamp(0,n-1)
    lower=index.long()
    upper=(lower+1).clamp_max(n-1)
    low=torch.take_along_dim(samples,lower.unsqueeze(-1),dim=-1).squeeze(-1)
    high=torch.take_along_dim(samples,upper.unsqueeze(-1),dim=-1).squeeze(-1)
    return torch.lerp(low,high,index-lower.float())


def rank_among_patches(values):
    t=values.transpose(1,2).contiguous()
    ordered=t.sort(-1).values
    left=torch.searchsorted(ordered,t,right=False)
    right=torch.searchsorted(ordered,t,right=True)
    return ((left+right).float()/(2*t.size(-1))).transpose(1,2).contiguous()


def hold(values,near):
    return values.flatten(0,1)[near].float().permute(0,1,3,2).sort(-1).values


def prepare(table):
    photo=table["scene"]
    n=len(photo)
    layout={}
    global_values={}
    for l in LAYERS:
        for group in GROUPS:
            tag=l,group
            val=table["values"][tag][...,:N]
            layout[tag]=torch.empty_like(val)
            global_values[tag]=val.flatten(0,1).float().T.contiguous().sort(-1).values[None,None]
    # Compute each target scene template against patch activations of a disjoint set of photos.
    # Neither its own image nor another patch of that image can intervene in its template.
    for parity in (0,1):
        rows=torch.arange(parity,n,2,device=photo.device)
        pool=torch.arange(1-parity,n,2,device=photo.device)
        for batch in rows.split(24):
            idx=nearest(table["late"][batch],table["late"][pool],K)
            for tag in layout:
                pts=table["values"][tag][...,:N]
                sorted_donors=hold(pts[pool],idx)
                normalized=cdf(sorted_donors,pts[batch].float())
                layout[tag][batch]=normalized.sort(1).values.to(layout[tag].dtype)
    rng=torch.Generator().manual_seed(22469)
    return {"templates":layout,"unconditional":global_values,
            "shuffled_keys":torch.randperm(n*49,generator=rng).to(photo.device)}


def measure(source,table,model,want=METHODS):
    near=nearest(source["late"],table["late"],K)
    neighbors=nearest(source["scene"][:,None],table["scene"],16)[:,0]
    outputs,context={},{}
    for l in LAYERS:
        for group in GROUPS:
            key=l,group
            real=source["values"][key][...,:N].float()
            pool=table["values"][key][...,:N]
            observations=hold(pool,near)
            old_prob=cdf(observations,real)
            ordinal=rank_among_patches(old_prob)
            examples=model["templates"][key]
            conditional=examples[neighbors].float().mean(1).transpose(1,2)[:,None]
            shared=examples.float().mean(0).T[None,None]
            def apply(reference,ranks,observed):
                # Quantile reference tables are indexed by the within-query spatial order, with shared ties.
                probability=quantile(reference,ranks)
                return quantile(observed,probability)
            options={}
            if "conditional_scene" in want:options["conditional_scene"]=apply(conditional,ordinal,observations)
            if "conditional_shared" in want:options["conditional_shared"]=apply(shared,ordinal,observations)
            if "conditional_uniform" in want:options["conditional_uniform"]=quantile(observations,ordinal)
            if "rotate_ranks" in want:options["rotate_ranks"]=apply(conditional,ordinal.roll(13,1),observations)
            if "conditional_mean" in want:options["conditional_mean"]=observations.mean(-1)
            if "fixed_median" in want:options["fixed_median"]=quantile(observations,torch.full_like(ordinal,.5))
            if "ordinary_histogram" in want:options["ordinary_histogram"]=quantile(model["unconditional"][key],rank_among_patches(real))
            if "shuffled_donors" in want:
                shuffled=hold(pool,model["shuffled_keys"][near])
                shuffled_rank=rank_among_patches(cdf(shuffled,real))
                options["shuffled_donors"]=apply(conditional,shuffled_rank,shuffled)
            if "identity" in want:options["identity"]=real
            context[key]={"observations":observations,"PIT":old_prob,"rank":ordinal,"template":conditional,
                          "value":real,"near_saturation":((old_prob<=0)|(old_prob>=1)).float().mean((1,2))}
            outputs[key]=options
    return outputs,context
