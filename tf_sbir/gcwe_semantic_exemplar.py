"""Infer scene meanings from shared versus independent geometric exemplars of a public word."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .concept_benchmark import atomic_json
from .crossdomain_benchmark import revision
from .data import Images, configure_image_worker, normal_transform
from .depiction_path_diagnostic import class_interval
from .encoding import load_frozen_clip, save_tensor
from .gallery_panel_resume import semantic_inputs
from .geometry_word_flat import KD_FUNCTION_SHA, PrefixReplay, load_kd, make_word_bank, score_rows


FAMILY="gcwe-semantic-exemplar"


def comparison_design(device):
    coords=torch.cartesian_prod(torch.arange(7),torch.arange(7))
    dist=(coords[:,None]-coords[None,:]).square().sum(-1)
    anchor,close,far=((dist[:,:,None]>0)&(dist[:,:,None]<dist[:,None,:])).nonzero(as_tuple=True)
    radius_check=dist[anchor,close]*100+dist[anchor,far]
    _,group,counts=torch.unique(radius_check,sorted=True,return_inverse=True,return_counts=True)
    order=group.argsort(stable=True)
    first=49*anchor[order]+close[order]
    second=49*anchor[order]+far[order]
    return first.to(device),second.to(device),counts.to(device),counts.cumsum(0).to(device)-1


def observation(tokens,design):
    first,second,counts,ends=design
    affinity=(tokens@tokens.transpose(1,2)).flatten(1)
    left,right=affinity[:,first],affinity[:,second]
    half_wins=2*(left>right).to(torch.int32)+(left==right).to(torch.int32)
    cumulative=half_wins.cumsum(1,dtype=torch.int32)[:,ends]
    wins=cumulative.diff(dim=1,prepend=torch.zeros_like(cumulative[:,:1]))
    assert torch.all((wins>=0)&(wins<=2*counts)) and int(counts.max())<16000
    return wins.short()


def probabilities(wins,counts):
    return (wins.cuda().float()+1)/(2*(counts+1))


def read_grids(config,paths,vision,design,folder):
    stamp=hashlib.sha256(("native-ordinal-choice-v1\n"+"\n".join(map(str,paths))).encode()).hexdigest()
    file=folder/(stamp+".pt")
    if file.is_file():
        data=torch.load(file,weights_only=True,mmap=True)
        assert data.shape==(len(paths),len(design[2]))
        return data
    batches=DataLoader(Images(paths,normal_transform()),batch_size=128,num_workers=config.workers,
                       pin_memory=True,worker_init_fn=configure_image_worker,
                       generator=torch.Generator().manual_seed(42))
    rows=[]
    for x in batches:
        raw=vision.transformer(vision._embeds(x.cuda(non_blocking=True).to(vision.conv1.weight.dtype)))
        tokens=F.normalize((vision.ln_post(raw[:,1:])@vision.proj).float(),dim=-1)
        rows.append(observation(tokens,design).cpu())
    data=torch.cat(rows)
    save_tensor(data,file)
    return data


def likelihood(query,geometry,h,background):
    return 1+((query-background)*h/(background*(1-background)))@(geometry-background).T


def infer(k,conditional,permutation,prior,counts):
    mean=k@conditional
    results={}
    concentrations={}
    for m in counts:
        scaled=m*k.log()
        weights=(scaled-scaled.amax(1,keepdim=True)).exp()
        shared=weights@conditional
        independent=mean.log()*m
        results["shared_"+str(m)]=F.normalize(prior*shared,p=1,dim=1)
        results["independent_"+str(m)]=(prior.clamp_min(1e-30).log()+independent).softmax(1)
        results["shuffled_"+str(m)]=F.normalize(prior*(weights[:,permutation]@conditional),p=1,dim=1)
        concentrations[str(m)]=(weights.sum(1).square()/weights.square().sum(1)).cpu()
        if m==1:
            torch.testing.assert_close(results["shared_1"],results["independent_1"],atol=1e-6,rtol=1e-5)
    return results,concentrations


def validate(design):
    gen=torch.Generator().manual_seed(42)
    x=F.normalize(torch.randn(4,49,12,generator=gen).cuda(),dim=-1)
    raw=observation(x,design)
    perm=torch.arange(49).reshape(7,7).T.flatten().cuda()
    assert torch.equal(observation(x[:,perm],design),raw)
    tied=observation(torch.ones_like(x),design)
    torch.testing.assert_close(tied.long(),design[2].expand_as(tied))
    g=probabilities(raw,design[2])
    bg=g.mean(0)
    h=design[2].float()/design[2].sum()
    k=likelihood(g,g,h,bg)
    direct=((g*h/bg)@g.T)+(((1-g)*h/(1-bg))@(1-g).T)
    torch.testing.assert_close(k,direct,atol=3e-6,rtol=1e-5)
    torch.testing.assert_close(likelihood(bg[None],g,h,bg),torch.ones_like(k[:1]))
    torch.testing.assert_close(k.mean(1),torch.ones_like(k[:,0]))
    assert (k>0).all()


@torch.inference_mode()
def run(dataset,stage,m):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    w=Path("/content/sbir")
    out=w/"results"/FAMILY
    cfg,qp,qs,pp,gs,qf,pf,src,tgt,scale=semantic_inputs(w,dataset)
    design=comparison_design("cuda")
    validate(design)
    h=design[2].float()/design[2].sum()
    vision=load_frozen_clip(cfg.clip_checkpoint,torch.device("cuda")).visual
    cache=w/"cache"/FAMILY
    photo_obs=read_grids(cfg,pp,vision,design,cache)
    plans={}
    for split,(ids,labels) in qs.items():
        order=torch.randperm(len(ids),generator=torch.Generator().manual_seed(42))
        if stage=="transfer":
            assert dataset=="tuberlin"
            sample=torch.arange(len(ids))
        else:
            assert dataset=="sketchy"
            sample=order[:1536] if stage=="pilot" else order[1536:]
        plans[split]=ids[sample],labels[sample]
    all_ids=torch.cat([row[0] for row in plans.values()]).unique()
    query_obs=read_grids(cfg,[qp[i] for i in all_ids.tolist()],vision,design,cache)
    kd,_=load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    for split,(ids,labels) in plans.items():
        pi,gl=gs[split]
        bank=make_word_bank(F.normalize(pf[pi].cuda().float(),dim=-1),tgt,scale,192)
        groups=[v.cuda() for v in torch.randperm(len(pi),generator=torch.Generator().manual_seed(42)).tensor_split(2)]
        models=[]
        for ref,held in zip(reversed(groups),groups):
            evidence=bank[ref]
            p=probabilities(photo_obs[pi[ref.cpu()]],design[2])
            permutation=torch.randperm(len(ref),generator=torch.Generator().manual_seed(42)).cuda()
            models.append((held,bank[held],evidence/evidence.sum(0),p,p.mean(0),permutation))
        evaluator=PrefixReplay(kd,gl,split)
        counts=(1,4,16,64) if stage=="pilot" else (m,)
        rows={}
        neff={str(k):[] for k in counts}
        deviations=[]
        with ThreadPoolExecutor(max_workers=2) as pool:
            for batch in torch.arange(len(ids)).split(48):
                prior=(scale*(F.normalize(qf[ids[batch]].cuda().float(),dim=-1)@src.T)).softmax(1)
                qgeo=probabilities(query_obs[torch.searchsorted(all_ids,ids[batch])],design[2])
                scores={"word":prior@bank.T}
                for held,held_bank,conditional,p,bg,permutation in models:
                    k=likelihood(qgeo,p,h,bg)
                    torch.testing.assert_close(k.mean(1),torch.ones_like(k[:,0]),atol=2e-5,rtol=0)
                    assert k.min()>0
                    deviations.append(k.std(1).cpu())
                    posterior,n=infer(k,conditional,permutation,prior,counts)
                    for c,v in n.items():
                        neff[c].append(v/k.shape[1])
                    for name,val in posterior.items():
                        if name not in scores:
                            scores[name]=torch.empty_like(scores["word"])
                        scores[name][:,held]=val@held_bank.T
                for name,val in scores.items():
                    rows.setdefault(name,[]).append(score_rows(val,labels[batch],gl.cuda(),split,evaluator,pool)[:,:2])
        raw={k:torch.cat(v).double() for k,v in rows.items()}
        effects={}
        for c in counts:
            model="shared_"+str(c)
            for other in ("word","independent_"+str(c),"shuffled_"+str(c)):
                diff=100*(raw[model]-raw[other])
                effects[model+" minus "+other]=dict(points=diff.mean(0).tolist(),classes=class_interval(diff[:,0],labels,seed=42))
        metrics={k:(100*v.mean(0)).tolist() for k,v in raw.items()}
        stem=split+"-"+stage
        save_tensor(dict(qi=ids,pi=pi,labels=labels,rows=raw),out/(stem+".pt"))
        atomic_json(out/(stem+".json"),dict(commit=revision(),gpu=torch.cuda.get_device_name(),seed=42,
            queries=len(ids),gallery=len(pi),evaluator=KD_FUNCTION_SHA,metrics=metrics,contrasts=effects,
            mean_geometry_dispersion=float(torch.cat(deviations).mean()),
            effective_reference_fraction={k:float(torch.cat(v).mean()) for k,v in neff.items()}))
        print(stem,metrics,flush=True)
        del bank,models,evidence,held_bank,conditional


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",choices=("sketchy","tuberlin"),default="sketchy")
    parser.add_argument("--stage",choices=("pilot","confirmation","transfer"),default="pilot")
    parser.add_argument("--count",type=int,choices=(1,4,16,64),default=16)
    opts=parser.parse_args()
    run(opts.dataset,opts.stage,opts.count)
