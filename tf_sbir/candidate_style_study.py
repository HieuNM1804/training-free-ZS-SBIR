"""Separate a matched gallery style response from the direct effect of the style."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from .candidate_style_cache import FAMILY,FRACTIONS
from .concept_benchmark import atomic_json,checksum,read
from .conditional_agreement_study import Replay
from .conditional_semantic_index import SIZES,bare_words
from .cross_diffusion_map import Config,Index
from .crossdomain_benchmark import digest,revision,SPLIT_HASHES
from .crossdomain_inputs import CHECKPOINT
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .foundation import load_words
from .map_diagnosis import map_values
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_diagnosis import global_inputs,stratified_sample
from .retrieval_evidence import load_kd,paired_delta
from .semantic import read_words
from .shared_concept_likelihood import fit_prior


def key(c):return json.dumps(list(c),separators=(",",":"))


def grid():
    return [(prior,alpha,text,factor,mode) for prior in ("uniform","mle") for alpha in FRACTIONS
            for text in ("source","target") for factor in (.4,1.,1.8) for mode in ("interaction","global")]


def choose(table,method,k):
    return [tuple(json.loads(s)) for s in sorted((s for s in table if json.loads(s)[4]==method),key=table.get,reverse=True)[:k]]


class StyleEvidence:
    def __init__(self,w,split,q,p,prob,src,tgt,scale,alpha):
        file=w/"cache"/FAMILY/split/f"views-{alpha:g}.pt"
        rec=torch.load(file,weights_only=True,mmap=True)
        if (len(q),len(p),CHECKPOINT)!=(rec["queries"],rec["gallery"],rec["checkpoint"]):raise ValueError("Mismatched style bank")
        self.qview=rec["q"].cuda()
        self.styles=rec["styles"].cuda()
        self.K=rec["style_count"]
        self.ref=(rec["reference"].float().cuda()-p[rec["reference_ids"]][:,None]).mean(0)
        self.q,self.scale,self.src,self.tgt=q,scale,src,tgt
        self.prob=prob
        self.groups=[(self.styles==s).nonzero().flatten() for s in range(self.K)]
        alternate=self.styles[torch.randperm(len(p),generator=torch.Generator().manual_seed(5731))]
        self.wrong=[(alternate==s).nonzero().flatten() for s in range(self.K)]
        self.mixture=torch.bincount(self.styles,minlength=self.K).float()/len(p)
        self.encoder_error=rec["native_input_max_error"]

    def seed(self,ids,text,factor,mode):
        q=self.q[ids]
        fixed=(q@self.src.T)*self.scale
        original=(fixed-fixed.amax(1,keepdim=True)).exp()@self.prob.T
        if mode=="original": return original,0.
        view=self.qview[ids].float()
        delta=view-q[:,None]
        bank=self.src if text=="source" else self.tgt
        result=torch.empty_like(original)
        max_error=0.
        for s,photos in enumerate(self.wrong if mode=="shuffled" else self.groups):
            if mode=="global":diff=delta[:,-1]-self.ref[-1]
            elif mode=="averaged":diff=((delta[:,:self.K]-self.ref[:self.K])*self.mixture[None,:,None]).sum(1)
            elif mode=="uncorrected":diff=delta[:,s]
            elif mode=="reference_only":diff=-self.ref[s].expand(len(ids),-1)
            else:diff=delta[:,s]-self.ref[s]
            logit=fixed+factor*self.scale*(diff@bank.T)
            if mode=="unconserved":
                new=(logit-fixed.amax(1,keepdim=True)).exp()@self.prob[photos].T
            else:
                new=(logit-logit.amax(1,keepdim=True)).exp()@self.prob[photos].T
                ref=original[:,photos].sum(1,keepdim=True)
                new*=ref/new.sum(1,keepdim=True).clamp_min(1e-30)
                max_error=max(max_error,float(((new.sum(1,keepdim=True)-ref).abs()/ref.clamp_min(1e-30)).max()))
            result[:,photos]=new
        return result,max_error


@torch.inference_mode()
def run(w,workers):
    root=w/"results"/FAMILY
    protocol={"revision":revision(),"grid":grid(),"primary":"candidate-specific double difference",
              "control":"same style for everyone, independently tuned","reference_photos":128,"pilot_seed":57913,"per_class":24}
    atomic_json(root/"protocol.json",protocol|{"sha256":digest(protocol)})
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:raise ValueError("Changed KD evaluator")
    for stage in ("pilot","shortlist","final"):
        if stage=="pilot": cells=grid()
        elif stage=="shortlist":
            means=read(root/"pilot.json")["means"]
            cells=choose(means,"interaction",3)+choose(means,"global",2)
        else:
            means=read(root/"shortlist.json")["means"]
            selected=choose(means,"interaction",1)[0]
            control=choose(means,"global",1)[0]
            lock={"revision":revision(),"primary":selected,"fixed_global":control,"full_sketchy_sha256":checksum(root/"shortlist.json")}
            atomic_json(root/"freeze.json",lock|{"sha256":digest(lock)})
            cells={selected,control,("uniform",selected[1],selected[2],0.,"original")}
            cells.update((*selected[:4],m) for m in ("global","original","averaged","uncorrected","shuffled","reference_only","unconserved"))
            cells=sorted(cells)
        table={}
        for dataset in (("sketchy","tuberlin") if stage=="final" else ("sketchy",)):
            cfg,_,qs,_,gs,qraw,praw=global_inputs(w,dataset)
            terms=read_words(cfg.vocabulary)
            if len(terms)!=20931:raise ValueError("Wrong dictionary")
            _,scale=bare_words(cfg,terms)
            src,tgt=(v.cuda() for v in load_words(cfg,terms))
            for split,(qi,y) in qs.items():
                pi,g=gs[split]
                if (len(qi),len(pi),len(g.unique()))!=SIZES[split] or digest(UNSEEN_CLASSES[split])!=SPLIT_HASHES[split]:
                    raise ValueError("Incorrect split")
                p,q=praw[pi].cuda(),qraw[qi].cuda()
                graph=Index(SimpleNamespace(p=p),rgb_only=True)
                config=Config(neighbors=16,bandwidth=.36,iterations=2,mode="rgb")
                graph.prepare(config)
                logits=(p@tgt.T)*scale
                base=logits.softmax(1)
                prior=fit_prior(base)[0]["mle"]
                del logits
                ix=stratified_sample(y,24,57913) if stage=="pilot" else torch.arange(len(qi))
                replay=Replay(kd,g,split) if stage=="final" else None
                per,errors,inspect_data,tops={},{},{},{}
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    for prior_kind,alpha in sorted(set(c[:2] for c in cells)):
                        if prior_kind=="uniform":probs=base
                        else:
                            probs=base*prior
                            probs/=probs.sum(1,keepdim=True).clamp_min(1e-30)
                        style=StyleEvidence(w,split,q,p,probs,src,tgt,scale,alpha)
                        for c in [c for c in cells if c[:2]==(prior_kind,alpha)]:
                            nm=key(c)
                            max_error=0.
                            for group in ix.split(144):
                                seed,noise=style.seed(group,*c[2:])
                                s=graph.score(config,seed)
                                max_error=max(max_error,noise)
                                if not torch.isfinite(s).all():raise ValueError("Nonfinite score")
                                fast,order,_,_=map_values(s,y[group].cuda(),g.cuda(),split)
                                if replay:
                                    val=replay.score(order,y[group],pool)
                                    error=float((val-fast.cpu()).abs().max())
                                    errors[nm]=max(errors.get(nm,0.),error)
                                    if error>2e-6:raise ValueError("KD mismatch")
                                    if c==selected:tops.setdefault(nm,[]).append(order[:,:25].int().cpu())
                                else:val=fast.cpu()
                                per.setdefault(nm,[]).append(val)
                            if max_error>2e-5:raise ValueError("Style strata changed mass unexpectedly")
                            inspect_data[nm]={"relative_style_mass_error":max_error,
                                "style_fractions":style.mixture.tolist(),"native_encoding_error":style.encoder_error}
                        del style,probs
                    del graph
                per={k:torch.cat(v) for k,v in per.items()}
                table[split]={k:v.mean(0).tolist() for k,v in per.items()}
                best=max(table[split],key=table[split].get)
                print(stage,split,best,table[split][best],flush=True)
                if stage=="final":
                    file=root/(split+".pt")
                    save_tensor({"query_labels":y,"metrics":per,"top25":torch.cat(tops[key(selected)])},file)
                    basekey=key(("uniform",selected[1],selected[2],0.,"original"))
                    obj={"protocol":{"split":split,"queries":len(q),"gallery":len(p),"classes":len(g.unique()),
                                     "checkpoint":CHECKPOINT,"vocabulary_sha256":checksum(cfg.vocabulary),"kd_sha256":KD_FUNCTION_SHA,
                                     "gpu":torch.cuda.get_device_name(),"revision":revision(),"map_k":200 if split=="sketchy_2" else 0,
                                     "p_k":200 if split=="sketchy_2" else 100},
                         "scores":{k:dict(zip(("mAP","precision"),v)) for k,v in table[split].items()},
                         "primary":key(selected),"fixed_global":key(control),"baseline":basekey,"checks":inspect_data,
                         "vs_controls":{k:paired_delta(per[key(selected)],v,y) for k,v in per.items()},
                         "original_kd_queries_per_condition":len(qi),"kd_max_abs_error":errors,"per_query_sha256":checksum(file)}
                    atomic_json(root/(split+".json"),obj)
        if stage!="final":
            means={k:[sum(table[s][k][j] for s in ("sketchy_1","sketchy_2"))/2 for j in range(2)] for k in table["sketchy_1"]}
            atomic_json(root/(stage+".json"),{"splits":table,"means":means})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    parser.add_argument("--workers",type=int,default=min(10,len(os.sched_getaffinity(0))))
    args=parser.parse_args()
    run(args.workspace,args.workers)
