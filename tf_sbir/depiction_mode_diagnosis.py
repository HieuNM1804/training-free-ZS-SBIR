"""Test if variation across photo views predicts how humans vary when drawing the same object."""

import argparse
from pathlib import Path

import torch

from .concept_benchmark import atomic_json
from .conditional_semantic_index import bare_words
from .foundation import load_words
from .regional_interventions import FAMILY
from .retrieval_diagnosis import global_inputs
from .semantic import read_words
from .shared_concept_likelihood import fit_prior


def centered(x):
    return x-x.mean(-1,keepdim=True)


def ranks(x):
    return x.argsort(dim=-1).argsort(dim=-1).double()


def corr(a,b):
    x,y=centered(a.double()),centered(b.double())
    return (x*y).sum(-1)/(x.norm(dim=-1)*y.norm(dim=-1)).clamp_min(1e-16)


def view_covariances(views,text,scale):
    between=[]
    within=torch.zeros(len(text),len(text),device=text.device,dtype=torch.float64)
    for part in views.split(192):
        z=centered(part.cuda().float()@text.T*scale).double()
        centered_view=z-z.mean(1,keepdim=True)
        flat=centered_view.flatten(0,1)
        within+=flat.T@flat/(len(views)*(z.shape[1]-1))
        between.append(z.mean(1))
    means=torch.cat(between)
    cov=torch.cov(means.T)
    return within,cov


def observation_modes(within,between):
    total=within+between
    eps=float(total.diag().mean())*1e-4
    ev,u=torch.linalg.eigh(total)
    use=ev>eps
    white=u[:,use]/ev[use].sqrt()
    noise,basis=torch.linalg.eigh(white.T@within@white)
    return noise.clamp(0,1),white@basis


def group_scatter(x,groups):
    count=groups.bincount().double()
    means=x.new_zeros(len(count),x.size(1)).index_add_(0,groups,x)/count[:,None]
    r=x-means[groups]
    var=x.new_zeros(len(count),x.size(1)).index_add_(0,groups,r.square())/count[:,None]
    same=(r.square()+var[groups])*count[groups,None]/(count[groups,None]-1).clamp_min(1)
    return r.square().sum(0)/(len(x)-len(count)),same,count[groups]>1


@torch.inference_mode()
def run(w):
    cfg,qp,qs,pp,gs,raw,photos=global_inputs(w,"sketchy")
    words=read_words(cfg.vocabulary)
    source,target=(x.cuda() for x in load_words(cfg,words));_,scale=bare_words(cfg,words)
    gbank=torch.load(w/"cache/nested-scale-word-evidence/sketchy/photo.pt",weights_only=True,mmap=True)
    bank=torch.load(w/"cache"/FAMILY/"sketchy/regions.pt",weights_only=True,mmap=True)
    result={}
    for s,(qi,y) in qs.items():
        pi,g=gs[s]
        p=photos[pi].cuda()
        em=p.new_empty((len(p),len(words)))
        for start in range(0,len(p),256):em[start:start+256]=((p[start:start+256]@target.T)*scale).softmax(-1)
        ids=fit_prior(em)[0]["mle"].topk(128).indices
        del em
        ppv=gbank[pi]
        wmat,bmat=view_covariances(ppv,target[ids],scale)
        n,axes=observation_modes(wmat,bmat)
        gallery_names={(Path(pp[i]).parent.name,Path(pp[i]).stem) for i in pi.tolist()}
        group_index={}; keys=[]
        for i in qi.tolist():
            path=Path(qp[i]);name=(path.parent.name,path.stem.rpartition("-")[0])
            if name not in gallery_names:raise ValueError("Unverified sketch-source identity")
            keys.append(group_index.setdefault(name,len(group_index)))
        group=torch.tensor(keys,device="cuda")
        z=centered(raw[qi].cuda()@source[ids].T*scale).double()@axes
        real,dispersion,valid=group_scatter(z,group)
        total=z.var(0)
        ratio=real/total.clamp_min(1e-20)
        buckets=[]
        for ix in torch.arange(len(n),device="cuda").tensor_split(5):
            buckets.append({"modes":len(ix),"gallery_within_photo_fraction_mean":float(n[ix].mean()),
                            "real_repeat_sketch_fraction_mean":float(ratio[ix].mean())})
        loc=torch.searchsorted(bank["ids"],qi)
        if not torch.equal(bank["ids"][loc],qi):raise ValueError("Missing independent window banks")
        pv=[]
        for part in loc.split(128):
            zoom=bank["features"][part,:,2].cuda().float()
            zs=centered(zoom@source[ids].T*scale).double()@axes
            pv.append(zs.var(1))
        pred=torch.cat(pv)
        random=torch.Generator().manual_seed(32111)
        shuffle=torch.randperm(len(z),generator=random).cuda()
        rn=ranks(n)
        # Remove per-mode scale before testing whether an individual sketch predicts
        # which modes specifically change more for another drawing of that photograph.
        scaled=pred/pred.mean(0).clamp_min(1e-20)
        truth=dispersion/real.clamp_min(1e-20)
        true_r=ranks(truth[valid])
        own=corr(ranks(scaled[valid]),true_r)
        ind=corr(ranks(scaled[shuffle][valid]),true_r)
        # Within-source control: reassign the per-view variances to another drawing of the same object.
        ordered=group.argsort(stable=True)
        cyclic=ordered.clone()
        for batch in ordered.split(group.bincount().cpu().tolist()):cyclic[batch]=batch.roll(1)
        sibling=corr(ranks(scaled[cyclic][valid]),true_r)
        result[s]={"queries":len(z),"gallery":len(p),"source_objects":len(group_index),"axes":len(n),
                   "gallery_to_real_within_object_noise_spearman":float(corr(rn,ranks(ratio))),
                   "noise_quintiles_in_gallery_order":buckets,"within_query_spearman_after_axis_rescaling":{
                       "own_views":float(own.mean()),"independent_query_views":float(ind.mean()),"other_drawing_same_object":float(sibling.mean())},
                   "description":"All representation and retrieval parameters were fixed before using the source IDs for post-retrieval diagnosis."}
        print(s,result[s],flush=True)
    atomic_json(w/"results"/FAMILY/"depiction-mode-diagnosis.json",result)


if __name__=="__main__":
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(a.parse_args().workspace)
