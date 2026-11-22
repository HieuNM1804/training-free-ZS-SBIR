"""Measure fixed-pair rank reversals when only the other gallery images change."""

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from .concept_benchmark import atomic_json, checksum
from .conditional_semantic_index import bare_words
from .crossdomain_benchmark import digest, revision
from .crossdomain_inputs import CHECKPOINT
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .foundation import load_words
from .gallery_depiction_study import get_split, views
from .retrieval_diagnosis import global_inputs
from .semantic import read_words


FAMILY="gallery-stable-region-evidence"
FIELDS=("right_in_both","Ext1_right_only","Ext2_right_only","wrong_in_both",
        "top100_overlap","common_top1_same_image","common_top1_same_class","Ext1_common_top1_correct","Ext2_common_top1_correct")


def align(a,b):
    # File IDs are assigned once to the union before either split is evaluated.
    common=sorted(set(a.tolist()).intersection(b.tolist()))
    ids=torch.tensor(common,dtype=torch.long)
    lookup_a=torch.full((max(a.max(),b.max()).item()+1,),-1,dtype=torch.long)
    lookup_b=lookup_a.clone()
    lookup_a[a]=torch.arange(len(a))
    lookup_b[b]=torch.arange(len(b))
    return ids,lookup_a[ids],lookup_b[ids]


def crop_likelihood(readings,bank,reference=None):
    # The gallery mean over all candidates can be computed before looking at this sketch.
    evidence=(readings-readings.amax((1,2,3),keepdim=True)).exp()
    vals=(evidence.flatten(0,2)@bank.T).reshape(*evidence.shape[:3],len(bank)).clamp_min(1e-30)
    mass=vals.new_tensor([1/18]*9+[1/8]*4).view(1,1,13,1)
    use=vals if reference is None else vals/(evidence@reference[:,None]).clamp_min(1e-30)
    att=mass*use
    return ((vals*att).sum(2)/att.sum(2).clamp_min(1e-30)).mean(1).clamp_min(1e-30).log()


def diagnosis(a,b,y,g):
    r1=a.argsort(dim=1,descending=True,stable=True)
    r2=b.argsort(dim=1,descending=True,stable=True)
    pos=torch.arange(len(g),device=g.device).expand_as(r1)
    inv1=torch.empty_like(r1).scatter_(1,r1,pos)
    inv2=torch.empty_like(r2).scatter_(1,r2,pos)
    quantities=a.new_empty((len(y),len(FIELDS)))
    pairs=torch.zeros((4,),device=a.device,dtype=torch.long)
    for cls in y.unique():
        sel=y==cls
        pos1,neg1=inv1[sel][:,g==cls],inv1[sel][:,g!=cls]
        pos2,neg2=inv2[sel][:,g==cls],inv2[sel][:,g!=cls]
        is1=pos1[:,:,None]<neg1[:,None,:]
        is2=pos2[:,:,None]<neg2[:,None,:]
        both=(is1&is2).sum((1,2))
        win1=is1.sum((1,2));win2=is2.sum((1,2))
        count=pos1.size(1)*neg1.size(1)
        numbers=torch.stack((both,win1-both,win2-both,count-win1-win2+both),1)
        pairs+=numbers.sum(0)
        quantities[sel,:4]=numbers.float()/count
    common=((inv1<100)&(inv2<100)).sum(1).float()/100
    ia,ib=r1[:,0],r2[:,0]
    quantities[:,4:]=torch.stack((common,ia==ib,g[ia]==g[ib],g[ia]==y,g[ib]==y),1)
    return quantities.cpu(),pairs.cpu()


@torch.inference_mode()
def run(w):
    cfg,*inputs=global_inputs(w,"sketchy")
    qpaths,queries,ppaths,galleries,qraw,praw=inputs
    terms=read_words(cfg.vocabulary)
    if len(terms)!=20931:
        raise ValueError("Different public bank")
    src,tgt=(v.cuda() for v in load_words(cfg,terms)); _,scale=bare_words(cfg,terms)
    spl=("sketchy_1","sketchy_2")
    qids,q1,q2=align(queries[spl[0]][0],queries[spl[1]][0])
    pids,p1,p2=align(galleries[spl[0]][0],galleries[spl[1]][0])
    if not torch.equal(queries[spl[0]][0][q1],queries[spl[1]][0][q2]) or not torch.equal(galleries[spl[0]][0][p1],galleries[spl[1]][0][p2]):
        raise ValueError("Matched paths changed")
    def named(split,labels):
        return [UNSEEN_CLASSES[split][int(z)] for z in labels]
    qname=named(spl[0],queries[spl[0]][1][q1]);pname=named(spl[0],galleries[spl[0]][1][p1])
    if qname!=named(spl[1],queries[spl[1]][1][q2]) or pname!=named(spl[1],galleries[spl[1]][1][p2]):
        raise ValueError("Same image file had two different evaluation labels")
    classes=sorted(set(qname))
    if set(classes)!=set(pname):
        raise ValueError("Different overlapping categories in queries and photos")
    y=torch.tensor([classes.index(k) for k in qname],device="cuda")
    g=torch.tensor([classes.index(k) for k in pname],device="cuda")
    get_views=views(w,"sketchy",qpaths)
    banks=[]
    for s,subset in zip(spl,(p1,p2),strict=True):
        _,_,_,_,heads,small,large,hist,base=get_split(w,"sketchy",src,tgt,scale,inputs,s)
        if base is None:
            raise ValueError("Reconstruct both full official reference rankings first")
        pick=subset.cuda()
        banks.append({"words":heads,"small":small[pick],"sm":small.mean(0),"large":large[pick],"lg":large.mean(0),
                      "hist":hist,"base":base})
    photo=praw[pids].cuda().float()
    native_word=((photo@tgt.T)*scale).softmax(1)
    pooled=torch.stack((banks[0]["lg"],banks[1]["lg"])).mean(0)
    work=defaultdict(list); totals={}; check=[0.,0.]
    for off in range(0,len(qids),16):
        sl=slice(off,off+16)
        q=qraw[qids[sl]].cuda().float()
        z=get_views(qids[sl]).float()
        full=z@src.T*scale
        direct=crop_likelihood(full,native_word)
        # Pair-independent direct baselines have identical inputs and outputs by construction.
        global_direct=((q@src.T*scale).softmax(1)@native_word.T).clamp_min(1e-30).log()
        scores=[]
        for j,(bank,qi,pi) in enumerate(zip(banks,(q1,q2),(p1,p2),strict=True)):
            ix=qi[sl]
            zsmall=full[:,:,:,bank["words"]]
            hist=bank["hist"][ix][:,pi].cuda().clamp_min(1e-30).log()
            whole=bank["base"][ix][:,pi].cuda()
            exact=crop_likelihood(zsmall,bank["small"],bank["sm"])
            resid=whole-(.65*hist+.35*exact)
            check[j]=max(check[j],float((resid-resid.mean(1,keepdim=True)).abs().amax()))
            regional=(whole-.65*hist)/.35
            cond= native_word[:,bank["words"]].contiguous()
            cond=cond/cond.sum(1,keepdim=True).clamp_min(1e-30)
            large_local=crop_likelihood(full,bank["large"],bank["lg"])
            uncal=crop_likelihood(full,bank["large"])
            scores.append({"native_cosine":q@photo.T,"direct_global_words":global_direct,"direct_all_region":direct,
                           "gallery_global":hist,"locked_region":regional,"locked":whole,
                           "raw_norm_compact":crop_likelihood(zsmall,bank["small"]),
                           "support_only":crop_likelihood(zsmall,cond),"all_diffused_normalized":large_local,
                           "all_diffused_raw":uncal,"all_diffused_common_normalizer":crop_likelihood(full,bank["large"],pooled),
                           "all_regional_and_global":.65*hist+.35*large_local,
                           "all_regional_raw_and_global":.65*hist+.35*uncal})
        for k,a in scores[0].items():
            vals,count=diagnosis(a,scores[1][k],y[sl],g)
            work[k].append(vals)
            totals[k]=totals.get(k,0)+count
        if off%384==0:
            print("common queries",off,len(qids),"reference errors",check,flush=True)
    if max(check)>.02:
        raise ValueError(f"The original regional reference did not reconstruct {check}")
    details={k:torch.cat(v) for k,v in work.items()}
    for k in ("native_cosine","direct_global_words","direct_all_region"):
        if totals[k][1:3].sum()!=0:
            raise ValueError("Pair-independent negative control changed rank")
    folder=w/"results"/FAMILY
    f=folder/"shared-scene-diagnosis.pt"
    save_tensor({"measurements":details,"query_labels":y.cpu(),"common_query_ids":qids,"common_gallery_ids":pids,
                 "query_paths":[str(qpaths[i]) for i in qids],"gallery_paths":[str(ppaths[i]) for i in pids]},f)
    def summary(quant):
        avg=quant.double().mean(0)
        return dict(zip(FIELDS,avg.tolist()))|{"fixed_pos_neg_disagreement":float(avg[1:3].sum()),
              "conditional_pair_accuracy_Ext1":float(avg[0]+avg[1]),"conditional_pair_accuracy_Ext2":float(avg[0]+avg[2])}
    attrs={"revision":revision(),"checkpoint":CHECKPOINT,"vocabulary":checksum(cfg.vocabulary),
           "common_query_path_hash":digest([str(qpaths[i]) for i in qids]),"common_gallery_path_hash":digest([str(ppaths[i]) for i in pids]),
           "queries":len(qids),"gallery":len(pids),"classes":classes,"fields":FIELDS,
           "shared_anchor_words":len(set(banks[0]["words"].tolist())&set(banks[1]["words"].tolist())),
           "compact_size":len(banks[0]["words"]),"reference_centered_log_error":check,"per_query_sha256":checksum(f),
           "scores":{k:summary(v) for k,v in details.items()},
           "comparison_counts":{k:v.tolist() for k,v in totals.items()},
           "per_class":{cls:{k:summary(v[y.cpu()==i]) for k,v in details.items()} for i,cls in enumerate(classes)}}
    atomic_json(f.with_suffix(".json"),attrs)
    print("fixed pairs",attrs["queries"],attrs["gallery"],attrs["shared_anchor_words"],
          {k:round(v["fixed_pos_neg_disagreement"],5) for k,v in attrs["scores"].items()},flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(p.parse_args().workspace)
