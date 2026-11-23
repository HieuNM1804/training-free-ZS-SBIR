"""Integrate the full public vocabulary for both poses of every query window and independent target-image panels."""
import argparse
from collections import defaultdict
from pathlib import Path
import re

import torch

from .concept_benchmark import atomic_json,read
from .conditional_word_inputs import tree
from .crossdomain_benchmark import digest,revision
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .gallery_panel_resume import semantic_inputs,region_views
from .retrieval_diagnosis import stratified_sample
from .semantic import read_words


FAMILY="branch-conditioned-query-readout"
MIN_LEVEL=1
SEED=54857
VIEWS=14


def cap(w,ds):
    if ds=="tuberlin" and not (w/"results"/FAMILY/"freeze.json").is_file():raise ValueError("No TU inference before this method's freeze")


def prepare_gallery(w,s,g,target,scale,paths):
    file=w/"cache"/FAMILY/s/"gallery.pt";file.parent.mkdir(parents=True,exist_ok=True)
    sig=digest({"paths":paths,"seed":SEED,"level":MIN_LEVEL,"words":len(target),"recipe":"exact full vocabulary and smoothed half means v1"})
    stamp=file.with_suffix(".json")
    if file.is_file() and stamp.is_file():
        if read(stamp)["signature"]!=sig:raise ValueError("Different gallery")
        return torch.load(file,weights_only=True,mmap=True)
    torch.set_float32_matmul_precision("highest")
    ht=tree(w,s,g)
    torch.set_float32_matmul_precision("high")
    group=ht.ids[MIN_LEVEL];m=ht.sizes[MIN_LEVEL];v=len(target)
    count=torch.bincount(group,minlength=m).float()
    perm=torch.randperm(len(g),generator=torch.Generator().manual_seed(SEED)).to(g.device)
    mask=torch.zeros(len(g),dtype=torch.bool,device=g.device);mask[perm[:len(g)//2]]=True
    partial=[g.new_zeros((m,v)) for _ in range(2)]
    for i in range(0,len(g),256):
        sub=g[i:i+256];word=(scale*(sub@target.T)).softmax(1);ix=group[i:i+256];a=mask[i:i+256]
        for panel,valid in zip(partial,(a,~a),strict=True):panel.index_add_(0,ix[valid],word[valid])
    total=partial[0]+partial[1]
    banks=[total/count[:,None]]
    for panel,ids in zip(partial,(mask,~mask),strict=True):
        global_word=panel.sum(0)/int(ids.sum())
        sample_count=torch.bincount(group[ids],minlength=m)
        banks.append((panel+global_word)/(sample_count[:,None]+1))
    pack={"banks":torch.stack(banks).cpu(),"word_mean":(total.sum(0)/len(g)).cpu(),
          "ids":[x.cpu() for x in ht.ids[MIN_LEVEL:]],"parents":[x.cpu() for x in ht.parents[MIN_LEVEL:]],"sizes":ht.sizes[MIN_LEVEL:]}
    save_tensor(pack,file);atomic_json(stamp,{"signature":sig,"code":revision(),"gallery":len(g)})
    return pack


def canonical(name):return re.sub("[^a-z0-9]","",name.casefold())


def literal_word_ids(words,classes):
    idx=defaultdict(list)
    for i,w in enumerate(words):idx[canonical(w)].append(i)
    return [idx.get(canonical(x),[]) for x in classes]


@torch.inference_mode()
def run(w,ds,stage):
    cap(w,ds)
    cfg,qpaths,qs,ppaths,gs,q,p,src,tgt,scale=semantic_inputs(w,ds)
    words=read_words(cfg.vocabulary)
    for s,(qi,y) in qs.items():
        root=w/"cache"/FAMILY/s;out=root/stage;out.mkdir(parents=True,exist_ok=True)
        pi,gl=gs[s];pack=prepare_gallery(w,s,p[pi].cuda(),tgt,scale,[ppaths[i] for i in pi.tolist()])
        banks=pack["banks"].cuda().flatten(0,1).T.contiguous();n=pack["sizes"][0]
        prior=torch.bincount(pack["ids"][0],minlength=n).cuda().float()/len(pi)
        idx=stratified_sample(y,36,63241) if stage=="pilot" else torch.arange(len(qi))
        idword=literal_word_ids(words,UNSEEN_CLASSES[s])
        if (out/"done.json").is_file():
            if read(out/"done.json")["queries"]!=len(idx):raise ValueError("Query list changed")
            continue
        views,_=region_views(w,s,cfg,qpaths,qi,q)
        for j in range(0,len(idx),192):
            dest=out/f"{j:06}.pt"
            if dest.exists():continue
            ix=idx[j:j+192]
            x=q[qi[ix]].cuda();flip=views["flipped"][ix].float().cuda()
            full=torch.stack((x,flip),1)[:,None];other=views["regions"][ix].float().cuda()
            batch=torch.cat((full,other),1).reshape(-1,512)
            # Every observed viewing condition retains every gallery word; no word or hard shortlist is selected.
            buf=[];raw_ranks=[];root_ranks=[]
            for first in range(0,len(batch),VIEWS*2*16):
                end=min(first+VIEWS*2*16,len(batch))
                raw=scale*(batch[first:end]@src.T)
                pr=raw.softmax(1)
                evidence=(pr@banks).reshape(-1,3,n).clamp_min(1e-32)
                normal=(evidence*prior).sum(-1,keepdim=True)
                buf.append((evidence/normal).cpu())
                # Diagnostic labels are consulted only after all three gallery readings for this image have been produced.
                raw=raw.reshape(-1,VIEWS,2,len(src))
                rw=raw+pack["word_mean"].cuda().clamp_min(1e-32).log()
                for t in range(len(raw)):
                    terms=idword[int(y[ix[first//(VIEWS*2)+t]])]
                    for tensor,output in ((raw,raw_ranks),(rw,root_ranks)):
                        if not terms:output.append(torch.full((VIEWS,2),-1,dtype=torch.int32));continue
                        given=tensor[t,:,:,terms].amax(-1,keepdim=True)
                        output.append((1+(tensor[t]>given).sum(-1)).int().cpu())
            scores=torch.cat(buf).reshape(len(ix),VIEWS,2,3,n).contiguous()
            if not torch.isfinite(scores).all():raise ValueError("Invalid all-word observation")
            save_tensor({"ids":ix,"densities":scores,"literal_sketch_rank":torch.stack(raw_ranks),"literal_root_rank":torch.stack(root_ranks)},dest)
            if j%2304==0:print("observed",stage,s,j,len(idx),flush=True)
        atomic_json(out/"done.json",{"queries":len(idx),"word_count":len(src),"code":revision(),"vocabulary":"all Open Images names","literal_exact_matching":"post-scoring only"})
        print("ready",stage,s,len(idx),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    ap.add_argument("--dataset",choices=("sketchy","tuberlin"),default="sketchy")
    ap.add_argument("--stage",choices=("pilot","full"),default="pilot")
    arg=ap.parse_args();run(arg.workspace,arg.dataset,arg.stage)
