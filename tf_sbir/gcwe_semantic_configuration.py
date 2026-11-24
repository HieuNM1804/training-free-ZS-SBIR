"""Use a generative spatial observation to infer scene concepts before retrieving a full gallery."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import math
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


FAMILY="gcwe-semantic-configuration"


def distance_bins():
    grid=torch.cartesian_prod(torch.arange(7),torch.arange(7))
    squared=(grid[:,None]-grid[None,:]).square().sum(-1)
    support=squared.unique(sorted=True)[1:]
    bins=torch.searchsorted(support,squared).cuda()
    # Uniform ordered sampling without replacement, including the common anchor condition.
    allowed=~torch.eye(49,dtype=torch.bool)
    counts=F.one_hot(bins.cpu(),len(support))*allowed[:,:,None]
    counts=counts.sum(1).double()
    joint=(counts.T@counts)-torch.diag(counts.sum(0))
    prior=joint.flatten().cuda().float()/(49*48*47)
    torch.testing.assert_close(prior.sum(),prior.new_tensor(1.))
    return bins,prior


def configurations(patches,bins,n):
    sim=patches@patches.transpose(1,2)
    sim.diagonal(dim1=1,dim2=2).fill_(-torch.inf)
    order=sim.argsort(dim=2,descending=True,stable=True)[:,:,:48]
    distances=bins[torch.arange(49,device=patches.device)[None,:,None],order]
    event=distances[:,:,:24]*n+distances[:,:,24:].flip(-1)
    offset=torch.arange(len(event),device=patches.device)[:,None,None]*n*n
    hist=torch.bincount((event+offset).flatten(),minlength=len(event)*n*n).view(len(event),n*n)
    assert (hist.sum(1)==49*24).all()
    return hist.to(torch.int16)


def encode(config,paths,vision,bins,prior,folder):
    n=math.isqrt(len(prior))
    stamp=hashlib.sha256(("CLIP-B32-native-config-v1\n"+"\n".join(map(str,paths))).encode()).hexdigest()
    cache=folder/(stamp+".pt")
    if cache.is_file():
        data=torch.load(cache,weights_only=True,mmap=True)
        assert data.shape==(len(paths),n*n)
        return data
    loader=DataLoader(Images(paths,normal_transform()),batch_size=128,num_workers=config.workers,
                      pin_memory=True,worker_init_fn=configure_image_worker,
                      generator=torch.Generator().manual_seed(42))
    rows=[]
    for px in loader:
        tokens=vision.transformer(vision._embeds(px.cuda(non_blocking=True).to(vision.conv1.weight.dtype)))
        patches=F.normalize((vision.ln_post(tokens[:,1:])@vision.proj).float(),dim=-1)
        rows.append(configurations(patches,bins,n).cpu())
    result=torch.cat(rows)
    assert not result[:,prior.cpu()==0].any()
    save_tensor(result,cache)
    return result


def laws(bank,hist,reference,background):
    n=math.isqrt(len(background))
    mass=bank.new_zeros(bank.shape[1])
    joint=bank.new_zeros((bank.shape[1],len(background)))
    for batch in reference.split(512):
        b=bank[batch]
        mass+=b.sum(0)
        joint.addmm_(b.T,hist[batch.cpu()].cuda().float()/(49*24))
    joint=(joint+background)/(mass+1)[:,None]
    torch.testing.assert_close(joint.sum(1),torch.ones_like(mass),atol=2e-5,rtol=0)
    table=joint.view(-1,n,n)
    independent=(table.sum(2)[:,:,None]*table.sum(1)[:,None,:]).flatten(1)
    return joint,independent


def predictions(prior,observed,law,strengths):
    # Exact averaging over one observed event; unsupported universal grid events carry no query mass.
    predictive=prior@law
    assert not ((predictive<=0)&(observed>0)).any()
    update=prior*((observed/predictive.clamp_min(1e-30))@law.T)
    torch.testing.assert_close(update.sum(1),torch.ones_like(update[:,0]),atol=2e-5,rtol=0)
    results={"one":update}
    if strengths:
        loglik=observed@law.clamp_min(1e-30).log().T
        for m in strengths:
            results["count"+str(m)]=(prior.clamp_min(1e-30).log()+m*loglik).softmax(1)
    return results


def validate(bins,background):
    gen=torch.Generator().manual_seed(42)
    patches=F.normalize(torch.randn(5,49,24,generator=gen).cuda(),dim=-1)
    n=math.isqrt(len(background))
    h=configurations(patches,bins,n)
    reflect=torch.arange(49).reshape(7,7).flip(1).flatten().cuda()
    assert torch.equal(h,configurations(patches[:,reflect],bins,n))
    law=background.repeat(7,1)
    q=torch.randn(5,7,generator=gen).cuda().softmax(1)
    for x in predictions(q,h.float()/(49*24),law,(1,4)).values():
        torch.testing.assert_close(x,q,rtol=2e-5,atol=2e-6)


@torch.inference_mode()
def run(dataset,stage,selected):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    w=Path("/content/sbir")
    out=w/"results"/FAMILY
    cfg,qp,qs,pp,gs,qf,pf,src,tgt,scale=semantic_inputs(w,dataset)
    vision=load_frozen_clip(cfg.clip_checkpoint,torch.device("cuda")).visual
    bins,background=distance_bins()
    validate(bins,background)
    kd,_=load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    cache=w/"cache"/FAMILY
    photos_hist=encode(cfg,pp,vision,bins,background,cache)
    plans={}
    for split,(qi,labels) in qs.items():
        perm=torch.randperm(len(qi),generator=torch.Generator().manual_seed(42))
        if stage=="transfer":
            assert dataset=="tuberlin"
            use=torch.arange(len(qi))
        else:
            assert dataset=="sketchy"
            use=perm[:1536] if stage=="pilot" else perm[1536:]
        plans[split]=qi[use],labels[use]
    all_queries=torch.cat([r[0] for r in plans.values()]).unique()
    sketch_hist=encode(cfg,[qp[i] for i in all_queries.tolist()],vision,bins,background,cache)
    for split,(ids,labels) in plans.items():
        pi,gl=gs[split]
        bank=make_word_bank(F.normalize(pf[pi].cuda().float(),dim=1),tgt,scale,192)
        folds=[v.cuda() for v in torch.randperm(len(pi),generator=torch.Generator().manual_seed(42)).tensor_split(2)]
        hist=photos_hist[pi]
        distributions=[laws(bank,hist,ref,background) for ref in reversed(folds)]
        evaluator=PrefixReplay(kd,gl,split)
        raw={}
        all_strengths=(1,4,16) if stage=="pilot" else (() if selected=="one" else (int(selected[5:]),))
        with ThreadPoolExecutor(max_workers=2) as pool:
            for b in torch.arange(len(ids)).split(48):
                prior=(scale*(F.normalize(qf[ids[b]].cuda().float(),dim=-1)@src.T)).softmax(1)
                observed=sketch_hist[torch.searchsorted(all_queries,ids[b])].cuda().float()/(49*24)
                scores={"word":prior@bank.T}
                for held,models in zip(folds,distributions):
                    held_bank=bank[held]
                    for label,model in zip(("joint","independent"),models):
                        for rule,posterior in predictions(prior,observed,model,all_strengths).items():
                            name=label+"_"+rule
                            if name not in scores:
                                scores[name]=torch.empty_like(scores["word"])
                            scores[name][:,held]=posterior@held_bank.T
                for name,val in scores.items():
                    raw.setdefault(name,[]).append(score_rows(val,labels[b],gl.cuda(),split,evaluator,pool)[:,:2])
        raw={a:torch.cat(v).double() for a,v in raw.items()}
        contrasts={}
        for key in raw:
            if key.startswith("joint_"):
                for ctrl in ("word",key.replace("joint_","independent_")):
                    change=100*(raw[key]-raw[ctrl])
                    contrasts[key+" minus "+ctrl]=dict(points=change.mean(0).tolist(),classes=class_interval(change[:,0],labels,seed=42))
        stem=split+"-"+stage
        metrics={a:(100*v.mean(0)).tolist() for a,v in raw.items()}
        save_tensor(dict(qi=ids,pi=pi,labels=labels,rows=raw),out/(stem+".pt"))
        atomic_json(out/(stem+".json"),dict(commit=revision(),seed=42,gpu=torch.cuda.get_device_name(),
                    evaluator=KD_FUNCTION_SHA,queries=len(ids),gallery=len(pi),metrics=metrics,contrasts=contrasts))
        print(stem,metrics,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",choices=("sketchy","tuberlin"),default="sketchy")
    p.add_argument("--stage",choices=("pilot","confirmation","transfer"),default="pilot")
    p.add_argument("--rule",choices=("one","count1","count4","count16"),default="one")
    args=p.parse_args()
    run(args.dataset,args.stage,args.rule)
