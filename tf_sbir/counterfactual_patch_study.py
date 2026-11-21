"""Evaluate physical sketch–photo patch evidence as a reliability weight on gallery seeds."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from .attention_witness_cache import FAMILY,LAYERS
from .concept_benchmark import atomic_json,checksum,read
from .conditional_agreement_study import Replay
from .conditional_semantic_index import SIZES,bare_words
from .counterfactual_patch_witness import PatchExchange,MODES
from .cross_diffusion_map import Index,Config
from .crossdomain_benchmark import revision,digest,SPLIT_HASHES
from .crossdomain_inputs import CHECKPOINT
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .foundation import load_words
from .map_diagnosis import map_values
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_diagnosis import global_inputs,stratified_sample
from .retrieval_evidence import load_kd,paired_delta
from .semantic import read_words
from .shared_concept_likelihood import emission


def key(c):
    return json.dumps(list(c),separators=(",",":"))


class Base:
    def __init__(self,p,words,scale):
        self.photo=((p@words.T)*scale).softmax(1)
        config=Config(neighbors=16,bandwidth=.36,iterations=2,mode="rgb")
        index=Index(SimpleNamespace(p=p),rgb_only=True)
        index.prepare(config)
        ker=index.kernels[0]
        matrix=ker.status
        for _ in range(config.iterations):
            left=torch.sparse.mm(ker.local,matrix)
            matrix=torch.sparse.mm(ker.local,left.T.contiguous()).T.contiguous()
            del left
        self.trans=matrix/matrix.sum(1,keepdim=True)
        self.right=self.trans.T.contiguous()
        probe=torch.linspace(.01,1,len(p),device=p.device)[None,:]
        direct=index.score(config,probe)
        error=float(((probe@self.right)-direct).abs().max())
        if error>8e-4:
            raise ValueError(f"Galerie diffusion disagrees with frozen operator: {error}")
        self.check=error
        # Sparse updates need only the right-multiplication layout.
        del self.trans

    def scores(self,queries,words,scale):
        l=emission(queries,words,scale)
        seed=l@self.photo.T
        return seed,seed@self.right

    def modify(self,base,seed,ids,verification,c):
        _,mode,k,beta,where=c
        if mode=="historical":
            return base
        ix=ids[:,:k].contiguous()
        stat=verification[:,:k].float()
        z=(stat-stat.mean(1,keepdim=True))/stat.std(1,keepdim=True,correction=0).clamp_min(1e-8)
        lr=z*beta
        change=(lr-lr.amax(1,keepdim=True)).exp()
        previous=(seed if where=="witness" else base).gather(1,ix)
        new=previous*change
        new=new*(previous.sum(1,keepdim=True)/new.sum(1,keepdim=True).clamp_min(1e-30))
        diff=new-previous
        if where=="terminal":
            return base.scatter(1,ix,new)
        col,perm=ix.sort(1)
        ptr=torch.arange(0,(len(ix)+1)*k,k,device=ix.device,dtype=torch.int32)
        sparse=torch.sparse_csr_tensor(ptr,col.flatten().int(),diff.gather(1,perm).flatten(),size=seed.shape)
        return base+torch.sparse.mm(sparse,self.right)


def grid():
    return [(layer,mode,k,beta,pos) for layer in LAYERS for mode in ("exchange","conditional")
            for k in (128,512,1024) for beta in (.25,.75,1.5) for pos in ("terminal","witness")]


def pick(scores,mode,k):
    return [tuple(json.loads(c)) for c in sorted((s for s in scores if json.loads(s)[1]==mode),
                                                 key=scores.get,reverse=True)[:k]]


def collect_scores(w,cache,dataset,split,base,source,scale,q,qi,pi,indices,layers,take,all_controls):
    digest_ids=hashlib.sha256(indices.numpy().tobytes()).hexdigest()[:14]
    memo=cache/f"{split}-{digest_ids}-ids{take}.pt"
    if memo.exists():
        ids=torch.load(memo,weights_only=True)
    else:
        ids=[]
        for qq in q[indices].split(144):
            _,s=base.scores(qq,source,scale)
            ids.append(s.topk(take,dim=1,sorted=True).indices.cpu().int())
        ids=torch.cat(ids)
        save_tensor(ids,memo)
    measures={}
    for layer in layers:
        file=cache/f"{split}-{digest_ids}-layer{layer}-{take}-c{int(all_controls)}.pt"
        if file.is_file():
            for m,v in torch.load(file,weights_only=True,mmap=True).items(): measures[layer,m]=v
            continue
        # Index query token rows only for the query set actually needed by this stage.
        model=PatchExchange(w,dataset,layer,qi[indices],pi)
        values={}
        step=64
        for i in range(0,len(ids),step):
            chunk=ids[i:i+step].cuda().long()
            scores=model.scores(torch.arange(i,i+len(chunk),device="cuda"),chunk,all_controls)
            for mode,v in scores.items():
                if not torch.isfinite(v).all():
                    raise ValueError("Non-finite attention evidence")
                values.setdefault(mode,[]).append(v.half().cpu())
            if i%2048==0:
                print(split,"physical evidence",layer,i+len(chunk),"/",len(ids),flush=True)
        values={m:torch.cat(v) for m,v in values.items()}
        save_tensor(values,file)
        measures.update({(layer,m):v for m,v in values.items()})
        del model,values
    return ids,measures


def evaluate(base,source,scale,q,y,g,split,idx,ids,measures,configs,replay=None,workers=10):
    result,errors,top={}, {},{}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for first in range(0,len(idx),64):
            local=idx[first:first+64]
            seed,original=base.scores(q[local],source,scale)
            labels=y[local]
            candidates=ids[first:first+len(local)].cuda().long()
            for cfg in configs:
                layer,mode,*_=cfg
                evidence=None if mode=="historical" else measures[layer,mode][first:first+len(local)].cuda()
                scores=base.modify(original,seed,candidates,evidence,cfg)
                if not torch.isfinite(scores).all():
                    raise ValueError("Nonfinite verification ranking")
                fast,order,_,_=map_values(scores,labels.cuda(),g.cuda(),split)
                nm=key(cfg)
                if replay:
                    out=replay.score(order,labels,pool)
                    err=float((out-fast.cpu()).abs().max())
                    if err>2e-6: raise ValueError("KD replay mismatch")
                    errors[nm]=max(errors.get(nm,0.),err)
                    top.setdefault(nm,[]).append(order[:,:100].cpu().int())
                else: out=fast.cpu()
                result.setdefault(nm,[]).append(out)
    return ({k:torch.cat(v) for k,v in result.items()},errors,{k:torch.cat(v) for k,v in top.items()})


@torch.inference_mode()
def run(w,workers):
    folder=w/"results"/FAMILY
    cache=w/"cache"/FAMILY/"rank-input-v1"
    protocol={"revision":revision(),"primary":"exchange","competitor":"conditional","pilot_per_class":24,
              "pilot_seed":47933,"grid":grid(),"references":64,"thresholds":"none for pairing"}
    atomic_json(folder/"protocol.json",protocol|{"sha256":digest(protocol)})
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:
        raise ValueError("Changed KD evaluator")
    for stage in ("pilot","shortlist","final"):
        if stage=="pilot":
            configs=grid()
        elif stage=="shortlist":
            prev=read(folder/"pilot.json")["means"]
            configs=pick(prev,"exchange",2)+pick(prev,"conditional",2)
        else:
            means=read(folder/"shortlist.json")["means"]
            primary=pick(means,"exchange",1)[0]
            competitor=pick(means,"conditional",1)[0]
            lock={"primary":primary,"competitor":competitor,"revision":revision(),
                  "sketchy_full_sha256":checksum(folder/"shortlist.json")}
            atomic_json(folder/"freeze.json",lock|{"sha256":digest(lock)})
            configs=sorted({primary,competitor,(primary[0],"historical",0,0,"terminal"),
                            *((primary[0],m,*primary[2:]) for m in MODES),
                            (*primary[:4],"witness" if primary[4]=="terminal" else "terminal")})
        scores_stage={}
        for dataset in (("sketchy","tuberlin") if stage=="final" else ("sketchy",)):
            cfg,_,qs,_,gs,qraw,praw=global_inputs(w,dataset)
            terms=read_words(cfg.vocabulary)
            _,scale=bare_words(cfg,terms)
            source,target=(v.cuda() for v in load_words(cfg,terms))
            for split,(qi,y) in qs.items():
                pi,g=gs[split]
                if (len(qi),len(pi),len(g.unique()))!=SIZES[split] or digest(UNSEEN_CLASSES[split])!=SPLIT_HASHES[split]:
                    raise ValueError("Incomplete split")
                indices=stratified_sample(y,24,47933) if stage=="pilot" else torch.arange(len(qi))
                q,p=qraw[qi].cuda(),praw[pi].cuda()
                base=Base(p,target,scale)
                layers=sorted(set(c[0] for c in configs))
                # A common full-stage bank can be reused by shortlisted cells and the subsequent ablations.
                ids,measure=collect_scores(w,cache,dataset,split,base,source,scale,q,qi,pi,indices,layers,
                                           max(c[2] for c in configs),stage!="pilot")
                values,err,top=evaluate(base,source,scale,q,y,g,split,indices,ids,measure,configs,
                                        Replay(kd,g,split) if stage=="final" else None,workers)
                scores_stage[split]={k:v.mean(0).tolist() for k,v in values.items()}
                print(stage,split,"best",max(scores_stage[split],key=scores_stage[split].get),
                      max(scores_stage[split].values()),flush=True)
                if stage=="final":
                    details=folder/(split+".pt")
                    save_tensor({"metrics":values,"query_labels":y,"top100":top[key(primary)]},details)
                    baseline=next(k for k in values if json.loads(k)[1]=="historical")
                    rel={k:paired_delta(values[key(primary)],v,y) for k,v in values.items()}
                    ap=(values[key(primary)]-values[baseline])[:,0]
                    report={"protocol":{"split":split,"queries":len(y),"gallery":len(g),"classes":len(g.unique()),
                            "map_k":200 if split=="sketchy_2" else 0,"p_k":200 if split=="sketchy_2" else 100,
                            "checkpoint":CHECKPOINT,"kd_sha256":KD_FUNCTION_SHA,"vocabulary_sha256":checksum(cfg.vocabulary),
                            "revision":revision(),"gpu":torch.cuda.get_device_name()},
                            "scores":{k:dict(zip(("mAP","precision"),v.mean(0).tolist())) for k,v in values.items()},
                            "primary":key(primary),"competitor":key(competitor),"baseline":baseline,"vs_controls":rel,
                            "original_kd_per_condition":len(y),"kd_max_abs_error":err,"original_graph_max_abs":base.check,
                            "primary_minus_baseline_ap":{"query_up":int((ap>0).sum()),"query_down":int((ap<0).sum()),
                                                          "quantiles":ap.quantile(torch.tensor([.1,.5,.9])).tolist()},
                            "per_query_sha256":checksum(details)}
                    atomic_json(folder/(split+".json"),report)
                del base,measure
        if stage!="final":
            means={k:[sum(scores_stage[s][k][i] for s in ("sketchy_1","sketchy_2"))*.5 for i in range(2)]
                   for k in scores_stage["sketchy_1"]}
            atomic_json(folder/(stage+".json"),{"scores":scores_stage,"means":means})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    parser.add_argument("--workers",type=int,default=min(10,len(os.sched_getaffinity(0))))
    args=parser.parse_args()
    run(args.workspace,args.workers)
