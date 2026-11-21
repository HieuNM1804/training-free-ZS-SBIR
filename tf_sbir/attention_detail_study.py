"""Test cross-scale detail, coarse attention conservation, and matched alternatives."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import inspect
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .attention_detail import features
from .concept_benchmark import atomic_json, checksum, read
from .conditional_agreement_study import Replay
from .conditional_semantic_index import SIZES,bare_words
from .crossdomain_benchmark import SPLIT_HASHES, digest, revision
from .crossdomain_inputs import CHECKPOINT
from .data import UNSEEN_CLASSES, Images, normal_transform, configure_image_worker
from .encoding import load_frozen_clip,save_tensor
from .foundation import load_words
from .map_diagnosis import map_values
from .quickdraw_locked import KD_FUNCTION_SHA
from .regional_intervention_study import index, mixscore
from .retrieval_diagnosis import global_inputs,stratified_sample
from .retrieval_evidence import load_kd,paired_delta
from .semantic import read_words

FAMILY="attention-preserving-detail"
SPLITS=("sketchy_1","sketchy_2")
PLAIN=("native","tile","large")


def key(c):
    return json.dumps(c,separators=(",",":"))


def grid():
    parts=[(name,0,0.) for name in PLAIN]
    parts += [("parent",j,v) for j in (4,8,12) for v in (1.,4.)]
    parts += [(name,j,1.) for name in ("mean","direct","free") for j in (8,12)]
    return [(*part,rho,mix) for part in parts for rho in ((1.,) if part[0]=="native" else (.5,1.)) for mix in (.4,.8)]


def pick(means,kinds,n):
    return [tuple(json.loads(k)) for k in sorted((k for k in means if json.loads(k)[0] in kinds),key=means.get,reverse=True)[:n]]


def multi_read(w,ds,cfg,paths,raw,ids,forms):
    cache=w/"cache"/FAMILY/ds
    cache.mkdir(parents=True,exist_ok=True)
    sig=digest({"checkpoint":CHECKPOINT,"paths":digest([str(paths[i]) for i in ids.tolist()]),"implementation":"v1"})
    def path(c):
        return cache/(sig[:12]+"-"+str(c[0])+"-"+str(c[1])+"-"+str(c[2])+".pt")
    need=sorted(c for c in forms if not path(c).is_file())
    if need:
        model=load_frozen_clip(cfg.clip_checkpoint,torch.device("cuda"))
        ps=[paths[i] for i in ids.tolist()]
        loader=DataLoader(Images(ps,normal_transform(),"rgb"),batch_size=40,num_workers=4,prefetch_factor=2,
                          pin_memory=True,worker_init_fn=configure_image_worker)
        blocks={c:[] for c in need}
        checks=[]
        offset=0
        for step,px in enumerate(loader):
            out,ch=features(model.visual,px,need,check=step==0)
            native=out["native",0,0.]
            err=float((1-(native*raw[ids[offset:offset+len(px)]].cuda()).sum(1)).abs().max())
            if err>.002:raise ValueError("Native tokens changed the B/32 embedding")
            if ch:checks.append(ch)
            for c in need:blocks[c].append(out[c].half().cpu())
            offset+=len(px)
            if step%64==0:print("detail features",ds,offset,"/",len(ids),flush=True)
        if offset!=len(ids):raise ValueError("Incomplete detailed features")
        for c,values in blocks.items():save_tensor(torch.cat(values),path(c))
        atomic_json(cache/(sig[:12]+"-"+digest(need)[:10]+"-checks.json"),{"identity":sig,"checks":checks})
    return {c:torch.load(path(c),weights_only=True,mmap=True) for c in forms}


@torch.inference_mode()
def run(w,phase,workers):
    root=w/"results"/FAMILY
    spec={"revision":revision(),"grid":grid(),"pilot_seed":63241,"pilot_per_class":36,"main":"parent",
          "controls":"same frozen weights and pixel values at both scales"}
    sig=digest(spec)
    if (root/"protocol.json").exists():
        if read(root/"protocol.json")["sha256"]!=sig:raise ValueError("Changed study")
    else:atomic_json(root/"protocol.json",spec|{"sha256":sig})
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:raise ValueError("KD digest changed")
    steps={"pilot":("pilot",),"lock":("shortlist",),"audit":("final",),"all":("pilot","shortlist","final")}[phase]
    for stage in steps:
        if stage=="pilot":cells=grid()
        elif stage=="shortlist":
            m=read(root/"pilot.json")["means"]
            cells=sorted(set(pick(m,{"parent"},3)+pick(m,{"mean","direct","free"},2)+pick(m,set(PLAIN),3)+pick(m,{"native"},1)))
        else:
            lock=read(root/"freeze.json")
            if lock["full_sketchy_sha256"]!=checksum(root/"shortlist.json") or lock["revision"]!=revision():raise ValueError("Broken Sketchy freeze")
            selected,plain,variant=(tuple(lock[k]) for k in ("primary","plain","alternative"))
            _,j,n,rho,mix=selected
            cells=sorted({selected,plain,variant,*[(m,j,n,rho,mix) for m in ("direct","mean","free","wrong_key","wrong_parent")],
                          *[(m,0,0.,rho,mix) for m in ("tile","large")],("native",0,0.,1.,mix),("native",0,0.,1.,0.)})
        table={}
        for ds in (("sketchy","tuberlin") if stage=="final" else ("sketchy",)):
            cfg,paths,qs,pp,gs,qraw,praw=global_inputs(w,ds)
            selection={s:(stratified_sample(y,36,63241) if stage=="pilot" else torch.arange(len(y))) for s,(_,y) in qs.items()}
            ids=torch.cat([qi[selection[s]] for s,(qi,y) in qs.items()]).unique(sorted=True)
            forms={c[:3] for c in cells}
            feature=multi_read(w,ds,cfg,paths,qraw,ids,forms)
            words=read_words(cfg.vocabulary)
            if len(words)!=20931:raise ValueError("Unexpected vocabulary")
            _,scale=bare_words(cfg,words)
            src,tgt=(a.cuda() for a in load_words(cfg,words))
            for s,(qi,y) in qs.items():
                pi,g=gs[s]
                if (len(qi),len(pi),len(g.unique()))!=SIZES[s] or digest(UNSEEN_CLASSES[s])!=SPLIT_HASHES[s]:
                    raise ValueError("Changed split")
                dest=root/(s+".json" if stage=="final" else stage+"-"+s+".json")
                if dest.is_file():
                    rec=read(dest)
                    if rec["signature"]!=sig or (stage=="final" and rec["per_query_sha256"]!=checksum(root/(s+".pt"))):
                        raise ValueError("Checkpoint mismatch")
                    table[s]=rec["values"]
                    continue
                pickids=selection[s]; loc=torch.searchsorted(ids,qi[pickids])
                if not torch.equal(ids[loc],qi[pickids]):raise ValueError("Query order mismatch")
                info=digest({"signature":sig,"gallery":digest([str(pp[i]) for i in pi.tolist()]),"dictionary":checksum(cfg.vocabulary)})[:16]
                word,b=index(w,s,praw[pi].cuda(),tgt,scale,info)
                hist=torch.load(w/"cache/pragmatic-gallery-competition"/(s+"-named-rgb16.pt"),weights_only=True,mmap=True)
                if hist.shape!=(len(qi),len(pi)):raise ValueError("Wrong comparison gallery")
                rows={key(c):[] for c in cells}
                errors={}
                tops=[]
                replay=Replay(kd,g,s) if stage=="final" else None
                ctx=ThreadPoolExecutor(max_workers=workers) if replay else nullcontext()
                with ctx as pool:
                    for offset in range(0,len(pickids),80):
                        group=pickids[offset:offset+80]
                        ind=loc[offset:offset+len(group)]
                        q=qraw[qi[group]].cuda()
                        labels=y[group]
                        base=hist[group].cuda().clamp_min(1e-30).log()
                        for shape in sorted(forms):
                            detail=feature[shape][ind].cuda().float()
                            for rho in sorted({c[3] for c in cells if c[:3]==shape}):
                                code=F.normalize((1-rho)*q+rho*detail)
                                if shape[0]=="native":code=q
                                z=(code@src[word].T)*scale
                                for cell in (c for c in cells if c[:4]==(*shape,rho)):
                                    name=key(cell)
                                    val,order,_,_=map_values(mixscore(z,b,base,cell[4]),labels.cuda(),g.cuda(),s)
                                    if replay:
                                        exact=replay.score(order,labels,pool)
                                        e=float((exact-val.cpu()).abs().max())
                                        if e>2e-6:raise ValueError("Original KD metric mismatch")
                                        errors[name]=max(errors.get(name,0.),e)
                                        if cell==selected:tops.append(order[:,:25].int().cpu())
                                    else:exact=val.cpu()
                                    rows[name].append(exact)
                rows={k:torch.cat(v) for k,v in rows.items()}
                table[s]={k:v.mean(0).tolist() for k,v in rows.items()}
                rec={"queries":len(pickids),"gallery":len(pi),"values":table[s],"signature":sig}
                if stage=="final":
                    path=root/(s+".pt");save_tensor({"metrics":rows,"query_labels":y,"top25":torch.cat(tops)},path)
                    rec.update({"protocol":{"split":s,"queries":len(qi),"gallery":len(pi),"classes":len(g.unique()),
                                            "kd_sha256":KD_FUNCTION_SHA,"checkpoint":CHECKPOINT,"vocabulary_sha256":checksum(cfg.vocabulary),
                                            "revision":revision(),"gpu":torch.cuda.get_device_name(),"map_k":200 if s=="sketchy_2" else 0,
                                            "p_k":200 if s=="sketchy_2" else 100},
                                "primary":key(selected),"best_contrast":key(variant),"best_tta":key(plain),
                                "baseline":key(("native",0,0.,1.,0.)),"scores":{k:dict(zip(("mAP","precision"),v)) for k,v in table[s].items()},
                                "vs_controls":{k:paired_delta(rows[key(selected)],v,y) for k,v in rows.items()},
                                "per_query_sha256":checksum(path),"kd_max_abs_error":errors,"original_kd_queries_per_condition":len(qi)})
                atomic_json(dest,rec)
                win=max(table[s],key=table[s].get);print(stage,s,win,table[s][win],flush=True)
        if stage!="final":
            m={k:[sum(table[s][k][j] for s in SPLITS)/2 for j in (0,1)] for k in table[SPLITS[0]]}
            atomic_json(root/(stage+".json"),{"means":m,"splits":table})
            if stage=="shortlist":
                lock={"primary":pick(m,{"parent"},1)[0],"alternative":pick(m,{"mean","direct","free"},1)[0],
                      "plain":pick(m,set(PLAIN),1)[0],"revision":revision(),"full_sketchy_sha256":checksum(root/"shortlist.json")}
                atomic_json(root/"freeze.json",lock|{"sha256":digest(lock)})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    parser.add_argument("--phase",choices=("pilot","lock","audit","all"),default="all")
    parser.add_argument("--workers",type=int,default=8)
    args=parser.parse_args()
    run(args.workspace,args.phase,args.workers)
