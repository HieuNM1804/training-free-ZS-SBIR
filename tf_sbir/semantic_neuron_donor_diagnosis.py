"""After ranking, test whether the true patch-to-photo correspondence helps most when its selected gallery photos depict the target class."""

import argparse
from pathlib import Path

import torch

from .concept_benchmark import atomic_json, checksum, read
from .crossdomain_benchmark import revision
from .encoding import load_frozen_clip, save_tensor
from .gallery_panel_resume import semantic_inputs
from .retrieval_diagnosis import stratified_sample
from .semantic_neuron_gallery import FAMILY, batches, native, nearest, to_gpu


def adjusted_slope(purity,ap,change,labels):
    """Conditional association with class and within-class quintile of original query AP fixed."""
    x=purity.double()
    y=100*change.double()
    labs=labels.unique(return_inverse=True)[1]
    out=[]
    for c in labs.unique():
        ix=torch.where(labs==c)[0]
        order=ix[ap[ix].argsort(stable=True)]
        cross=0.;ss=0.;hi=0.;lo=0.;n=0
        for group in order.tensor_split(5):
            if len(group)<4:continue
            cx=x[group]-x[group].mean()
            cy=y[group]-y[group].mean()
            cross+=float((cx*cy).sum())
            ss+=float(cx.square().sum())
            ranked=group[x[group].argsort(stable=True)]
            half=len(ranked)//2
            lo+=float(y[ranked[:half]].sum());hi+=float(y[ranked[-half:]].sum());n+=half
        out.append((cross,ss,lo,hi,float(n)))
    table=torch.tensor(out,dtype=torch.float64)
    draw=torch.randint(len(table),(4000,len(table)),generator=torch.Generator().manual_seed(51709))
    boot=table[draw].sum(1)
    slope=.10*boot[:,0]/boot[:,1].clamp_min(1e-20)
    highlow=(boot[:,3]-boot[:,2])/boot[:,4].clamp_min(1)
    total=table.sum(0);points=torch.tensor([.025,.975],dtype=torch.float64)
    return {"change_pp_per_10pp_more_correct_class_donors":float(.10*total[0]/total[1].clamp_min(1e-20)),
            "class_bootstrap_95":slope.quantile(points).tolist(),"upper_half_minus_lower_half_gain_pp":float((total[3]-total[2])/total[4].clamp_min(1)),
            "upper_lower_CI95":highlow.quantile(points).tolist(),"groups":"class by original AP quintile",
            "upper_lower_paired_queries":int(total[4]*2)}


@torch.inference_mode()
def run(w,ds,phase):
    inputs=semantic_inputs(w,ds)
    cfg,qp,qs,_,gs,*_=inputs
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    root=w/"results"/FAMILY
    frozen=read(root/"freeze.json")
    main=frozen["configuration"]["nominee"]
    l,group,num,donor,channel,strength=main.split("/")
    l=int(l);num=int(num);k=int(donor.removeprefix("late"))
    matched=main.replace("/"+donor+"/","/shuffled/")
    allmean=frozen["configuration"]["selected"]["constant"]
    v=load_frozen_clip(cfg.clip_checkpoint,torch.device("cuda")).visual
    for s,(qi,y) in qs.items():
        named=s if phase=="full" else "pilot-official-"+s
        official=root/(named+".pt")
        path=root/("donor-diagnosis-"+named+".pt")
        if path.is_file() and path.with_suffix(".json").is_file():
            if read(path.with_suffix(".json"))["per_query_sha256"]!=checksum(path):raise ValueError("Changed diagnosis")
            continue
        obs=torch.load(official,mmap=True,weights_only=True)
        ind=torch.arange(len(qi)) if phase=="full" else stratified_sample(y,36,63241)
        if not torch.equal(obs["ids"],ind):raise ValueError("Query identities differ from official evaluation")
        labels=y[ind]
        memo=torch.load(w/"cache"/FAMILY/(s+"-memory-v2.pt"),mmap=True,weights_only=True)
        bank=to_gpu(memo["table"])
        units={ll:{g:ix.cuda() for g,ix in packs.items()} for ll,packs in memo["units"].items()}
        pi,gl=gs[s]
        positions=torch.full((int(pi.max())+1,),-1,dtype=torch.int64)
        positions[pi]=torch.arange(len(pi))
        chosen=positions[memo["donor"]]
        if (chosen<0).any() or not torch.equal(pi[chosen],memo["donor"]):raise ValueError("Photo IDs outside the current split")
        donor_labels=gl[chosen].cuda()
        pu,weighted,prior,photos=[],[],[],[]
        for ix,pix in enumerate(batches(qp,qi[ind],cfg.workers,48)):
            rows=labels[ix*48:ix*48+len(pix)].cuda()
            encoded=native(v,pix.cuda(non_blocking=True),units)
            near=nearest(encoded["late"],bank["late"],k)
            photo_id=near.div(49,rounding_mode="floor")
            counts=(donor_labels[photo_id]==rows[:,None,None]).float().mean(2)
            acts=bank["values"][l,group][...,:num].flatten(0,1)[near].float().mean(2)
            old=encoded["values"][l,group][...,:num].float()
            # Weight by the exact norm of the token intervention passed through the MLP output matrix.
            W=tuple(v.transformer.resblocks[l-1].mlp.children())[-1].weight[:,units[l][group][:num]].float()
            energy=((acts-old)@W.T).square().sum(-1)
            pu.append(counts.mean(1).cpu());weighted.append(((counts*energy).sum(1)/energy.sum(1).clamp_min(1e-20)).cpu())
            prior.append((donor_labels[None]==rows[:,None]).float().mean(1).cpu())
            photos.append(photo_id[:,:,0].short().cpu())
        purity=torch.cat(pu);weight_purity=torch.cat(weighted);chance=torch.cat(prior)
        true=obs["metrics"][main][:,0];base=obs["metrics"]["regional"][:,0]
        changes={"against_matched_shuffled":true-obs["metrics"][matched][:,0],"against_separately_tuned_mean":true-obs["metrics"][allmean][:,0],
                 "against_regional":true-base}
        rows={}
        for kname,val in changes.items():
            rows[kname]={"unweighted":adjusted_slope(purity,base,val,labels),"energy_weighted":adjusted_slope(weight_purity,base,val,labels)}
        save_tensor({"ids":ind,"labels":labels,"class_purity":purity,"weighted_purity":weight_purity,"class_prevalence":chance,
                     "nearest_donor_image_for_each_patch":torch.cat(photos),"donor_image_ids":memo["donor"],"changes":changes},path)
        atomic_json(path.with_suffix(".json"),{"revision":revision(),"lock":frozen["sha256"],"main":main,"control":matched,"split":s,
                    "queries":len(ind),"percent_correct_class_donors":float(100*purity.mean()),
                    "percent_energyweighted_correct_class":float(100*weight_purity.mean()),"percent_chance":float(100*chance.mean()),
                    "associations":rows,"per_query_sha256":checksum(path),"labels":"used only here, after the official retrieval has finished",
                    "interpretation":"A shared image class is a proxy for patch relevance. Conditional slope measures an association, not causal harm from out-of-class texture."})
        print(s,len(ind),float(purity.mean()),rows["against_matched_shuffled"]["unweighted"],flush=True)
        del bank


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--dataset",choices=("sketchy","tuberlin"),default="sketchy")
    p.add_argument("--stage",choices=("pilot","full"),default="pilot")
    a=p.parse_args()
    run(a.workspace,a.dataset,a.stage)
