"""Separate genuine two-sided correspondence, source-only contrast, and response even in the perturbation sign."""

import argparse
from pathlib import Path
import torch

from .concept_benchmark import atomic_json, checksum
from .crossdomain_benchmark import revision
from .encoding import save_tensor
from .gallery_panel_resume import semantic_inputs
from .recoverable_patch_information import FAMILY, MODES
from .recoverable_patch_inputs import restore
from .recoverable_patch_study import describe, locate, prepare
from .response_geometry_views import FAMILY as GEOMETRY, PAIR_NAMES


def wins(diff):
    return ((diff>0).double()+.5*(diff==0)).mean().item()


def stats(diff, labels):
    labels=labels.unique(return_inverse=True)[1]
    delta=diff.double()
    hits=(delta>0).double()+.5*(delta==0)
    count=labels.bincount().double()
    sums=torch.zeros_like(count).index_add_(0,labels,hits)
    idx=torch.randint(len(count),(2000,len(count)),generator=torch.Generator().manual_seed(8042))
    ci=(sums[idx].sum(1)/count[idx].sum(1)).quantile(torch.tensor([.025,.975],dtype=torch.double))
    return {"queries":len(diff),"correct_margin_increases":wins(delta),"mean_change":float(delta.mean()),
            "median_change":float(delta.median()),"tie_fraction":float((delta==0).double().mean()),"class_CI95":ci.tolist()}


def raw(q,p,temp):
    return temp*torch.einsum("bd,bkd->bk",q,p)


def lexical(q,p,source,target,temp):
    x=(temp*(q@source.T)).log_softmax(-1)
    y=(temp*(p@target.T)).log_softmax(-1)
    return torch.logsumexp(x[:,None]+y,-1)


def contrast(values):
    return values[:,0]-values[:,1]


def cosine_terms(q,dq,p,dp,temp):
    return {"source":contrast(raw(dq,p,temp)),"photo":contrast(raw(q,dp,temp)),
            "interaction":contrast(raw(dq,dp,temp)),"both":contrast(raw(dq,p,temp)+raw(q,dp,temp)+raw(dq,dp,temp))}


def word_terms(q,dq,p,dp,src,tgt,temp):
    old=contrast(lexical(q,p,src,tgt,temp))
    a=contrast(lexical(q+dq,p,src,tgt,temp))-old
    b=contrast(lexical(q,p+dp,src,tgt,temp))-old
    total=contrast(lexical(q+dq,p+dp,src,tgt,temp))-old
    return {"source":a,"photo":b,"interaction":total-a-b,"both":total}


def scale_diagnosis(bank,positions,modes):
    z=bank["change"][positions].float()
    out={}
    for m in MODES:
        a,b=[z[:,modes.index(f"{m}/{k}")] for k in (-2,2)]
        even=.5*(a+b)
        odd=.5*(b-a)
        total=even.square().sum(1)+odd.square().sum(1)
        out[m]={"mean_even_share_embedding_energy":float((even.square().sum(1)/total.clamp_min(1e-30)).mean()),
                "median_negative_embedding_change":float(a.norm(dim=1).median()),
                "median_positive_embedding_change":float(b.norm(dim=1).median()),
                "fraction_exact_unchanged_both":float((total==0).float().mean())}
    return out


@torch.inference_mode()
def run(w):
    inputs=semantic_inputs(w,"sketchy")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    _,_,_,_,_,_,_,src,tgt,temp=inputs
    sketch,sd=restore(w,"sketchy","pilot-query")
    photo,pd=restore(w,"sketchy","diagnostic-photo")
    modes=sd["readouts"]
    if modes!=pd["readouts"]:raise ValueError("Different image readouts")
    plan=torch.load(w/"cache"/GEOMETRY/"sketchy/plan.pt",mmap=True,weights_only=True)["splits"]
    root=w/"results"/FAMILY
    summary,record={},{}
    pp=torch.arange(len(photo["ids"]))
    photo_stats=describe(photo,pp)
    response={"photo":scale_diagnosis(photo,pp,modes),"sketch":scale_diagnosis(sketch,torch.arange(len(sketch["ids"])),modes)}
    for s in plan:
        p=prepare(w,s,inputs,sketch)
        if not torch.equal(plan[s]["ids"],p["ids"]):raise ValueError("Different error panel")
        heads=p["heads"]
        g=p["g"]
        ssum,files={},{}
        for panel,name in enumerate(PAIR_NAMES):
            ids=plan[s]["pairs"][:,panel]
            use=torch.where((ids>=0).all(1))[0]
            a=p["positions"][use]
            b=locate(photo["ids"],ids[use].reshape(-1)).reshape(-1,2)
            q=sketch["native"][a].cuda()
            photos=photo["native"][b].cuda()
            label=p["y"][use]
            current={}
            current["native/rgb"]=contrast(raw(q,photos,temp)).cpu()
            current["native/words"]=contrast(lexical(q,photos,src[heads],tgt[heads],temp)).cpu()
            tests={}
            for mode in MODES:
                for mag in (2,8):
                    sign_rows={}
                    for sign in (-1,1):
                        arm=modes.index(f"{mode}/{mag*sign}")
                        dq=sketch["change"][a,arm].cuda().float()
                        dp=photo["change"][b,arm].cuda().float()
                        rs=cosine_terms(q,dq,photos,dp,temp)
                        ws=word_terms(q,dq,photos,dp,src[heads],tgt[heads],temp)
                        vals={c+"/"+k:v for c,coll in (("rgb",rs),("words",ws)) for k,v in coll.items()}
                        sign_rows[sign]=vals
                    for key in sign_rows[-1]:
                        neg, pos=sign_rows[-1][key], sign_rows[1][key]
                        for term,value in (("negative",neg),("positive",pos),("odd",.5*(pos-neg)),("even",.5*(pos+neg))):
                            tests[f"{mode}/{mag}/{key}/{term}"]=value.cpu()
            current.update(tests)
            groups={"all":torch.ones(len(use),dtype=torch.bool)}
            if panel==0:
                order=p["base"][use].argmax(1)
                groups["baseline_top1_wrong"]=g[order]!=label
            ssum[name]={k:{r:stats(v[flag],label[flag]) for r,v in current.items()} for k,flag in groups.items()}
            files[name]={"sample_indices":use,"query_label":label,"images_relevant_and_irrelevant":ids[use],"contrasts":current}
            print("audited",s,name,len(use),flush=True)
        summary[s]=ssum
        record[s]=files
    output=root/"paired-image-interventions.pt"
    save_tensor(record,output)
    atomic_json(output.with_suffix(".json"),{"revision":revision(),"splits":summary,"photo":photo_stats,
                "response_by_domain":response,"per_query_sha256":checksum(output),"units":"CLIP scaled cosine or log compact lexical compatibility, no diffusion in paired lexical readout",
                "interpretation":"fraction of queries with increased margin for the fixed relevant photo, ties counted one-half; different from retrieval accuracy or mAP",
                "seed":8042,"GPU":torch.cuda.get_device_name()})
    print("saved",output,flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(cli.parse_args().workspace)
