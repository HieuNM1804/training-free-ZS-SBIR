"""Decompose a frozen word correction into changing word identities and changing the logit spectrum."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
from pathlib import Path

import torch

from .concept_benchmark import atomic_json, checksum, read
from .conditional_agreement_study import Replay
from .conditional_semantic_index import bare_words
from .crossdomain_benchmark import digest, revision
from .crossdomain_inputs import CHECKPOINT
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .foundation import load_words
from .map_diagnosis import map_values
from .paired_prompt_full import cached_full, edited_batch
from .paired_prompt_intervention import text_reference
from .paired_prompt_study import FAMILY, label_diagnostic, literal_rank, log_readout
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_diagnosis import global_inputs
from .retrieval_evidence import load_kd, paired_delta
from .semantic import read_words


def rearrange(values, ranks):
    return torch.empty_like(values).scatter_(1,ranks,values)


def entropy(logits):
    logp=logits.log_softmax(1)
    return -(logp.exp()*logp).sum(1)


def cases(metrics,top,paths,query_ids,cls,y,before,after,source_words,edited_words):
    change=metrics["full"][:,0]-metrics["base"][:,0]
    fixed=(y!=top["base"])&(y==top["full"])
    broken=(y==top["base"])&(y!=top["full"])
    entries={}
    for name,mask in (("fixed",fixed),("broken",broken)):
        loc=mask.nonzero().flatten()
        if not len(loc):
            entries[name]=None
            continue
        center=change[loc].median()
        j=int(loc[(change[loc]-center).abs().argmin()])
        path=Path(paths[query_ids[j]])
        entries[name]={"sketch":"/".join(path.parts[-3:]),"class":cls[y[j]],"base_top_class":cls[top["base"][j]],
                       "edited_top_class":cls[top["full"][j]],"base_AP":float(metrics["base"][j,0]),
                       "edited_AP":float(metrics["full"][j,0]),"native_word":source_words[j],"edited_word":edited_words[j],
                       "rank_exact_class_before":int(before[j]),"rank_exact_class_after":int(after[j])}
    return entries


@torch.inference_mode()
def run(w,workers):
    folder=w/"results"/FAMILY
    freeze=read(folder/"freeze.json")
    method=freeze["primary"]
    if freeze["revision"]!="ea8c50ff121ddaadd0a423d18664454e0fe5ba86":
        raise ValueError("The intended original lock is missing")
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:
        raise ValueError("Original evaluator changed")
    out=w/"results"/"prompt-spectrum-audit"
    spec={"revision":revision(),"primary_choice":method,"lock_hash":checksum(folder/"freeze.json"),
          "arms":["base","rank_only","spectrum_only","full"],"selection":"none; all arms fixed",
          "checkpoint":CHECKPOINT,"evaluator":KD_FUNCTION_SHA}
    sig=digest(spec)
    atomic_json(out/"protocol.json",spec|{"sha256":sig})
    for ds in ("sketchy","tuberlin"):
        cfg,*inputs=global_inputs(w,ds)
        names=read_words(cfg.vocabulary)
        src,tgt=(x.cuda() for x in load_words(cfg,names))
        _,scale=bare_words(cfg,names)
        for s,data,y,g in cached_full(w,ds,cfg,inputs,src,tgt,scale):
            dest=out/(s+".json")
            if dest.exists():
                old=read(dest)
                if old["signature"]!=sig or old["per_query_sha256"]!=checksum(dest.with_suffix(".pt")):
                    raise ValueError("Changed outcome")
                continue
            torch.backends.cuda.matmul.allow_tf32=True
            bank=torch.load(w/"cache/pose-conditional-layout"/(s+"-gallery-uniform.pt"),weights_only=True,mmap=True).cuda()
            axes=text_reference(src,tgt,48)
            rng=torch.Generator().manual_seed(8129)
            perm=torch.randperm(src.size(1),generator=rng).cuda()
            sign=(2*torch.randint(2,(src.size(1),),generator=rng)-1).float().cuda()
            replay=Replay(kd,g,s)
            classes=UNSEEN_CLASSES[s]
            class_ids=label_diagnostic(names,classes)
            metrics,top=defaultdict(list),defaultdict(list)
            obs=[]
            errors={}
            good_id=g.cuda()
            word_check=0.
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for start in range(0,len(y),64):
                    loc=slice(start,start+64)
                    q=data["query_native"][loc].cuda().float()
                    heads=data["selected_heads"][loc].cuda().float()
                    base=data["regional_score"][loc].cuda()
                    label=y[loc]
                    _,edited=next(edited_batch(q,heads,src,tgt,axes,perm,sign,[method]))
                    u=q@src.T*scale
                    # The full study evaluated edited embeddings in fixed groups of six.
                    v=(edited.repeat(6,1)@src.T*scale)[:len(q)]
                    us,ui=u.sort(dim=1,descending=True,stable=True)
                    vs,vi=v.sort(dim=1,descending=True,stable=True)
                    order_only=rearrange(us,vi)
                    value_only=rearrange(vs,ui)
                    orig=log_readout(u,bank)
                    word_check=max(word_check,float((rearrange(vs,vi)-v).abs().max()))
                    raw_truth=class_ids[label.cuda()]
                    obs.append(torch.stack((literal_rank(u,raw_truth),literal_rank(v,raw_truth),entropy(u),entropy(v),
                                            ui[:,0],vi[:,0]),1).cpu())
                    # Use a fixed batch shape for the identical full-reference multiplication.
                    for k,logits in (("base",None),("rank_only",order_only),("spectrum_only",value_only),("full",v)):
                        if logits is None:
                            score=base
                        else:
                            new=log_readout(logits.repeat(6,1),bank)[:len(q)]
                            score=base+.65*(new-orig)
                        fast,rank,_,_=map_values(score,label.cuda(),good_id,s)
                        exact=replay.score(rank,label,pool)
                        er=float((exact-fast.cpu()).abs().max())
                        if er>2e-6:
                            raise ValueError("KD-SBIR disagreed")
                        errors[k]=max(errors.get(k,0.),er)
                        metrics[k].append(exact)
                        top[k].append(g[rank[:,0].cpu()])
                    if start%3072==0:
                        print("spectrum",s,start,len(y),flush=True)
            vals={k:torch.cat(v) for k,v in metrics.items()}
            labels={k:torch.cat(v) for k,v in top.items()}
            reference=torch.load(folder/(s+".pt"),weights_only=True,mmap=True)["metrics"]
            difference={k:float((vals[k]-reference[a]).abs().max()) for k,a in (("base","regional"),("full",method))}
            # GPU matrix tiling can reorder near ties between nearly equal photos.
            mean_deviation={k:float((vals[k].mean(0)-reference[a].mean(0)).abs().max()) for k,a in (("base","regional"),("full",method))}
            if max(mean_deviation.values())>3e-5 or word_check!=0:
                raise ValueError(f"Frozen terminal or the two-factor reconstruction changed {mean_deviation}, {word_check}")
            signals=torch.cat(obs)
            before,after=signals[:,:2].long().T
            valid=before>0
            misses=valid&(y!=labels["base"])
            picks=cases(vals,labels,inputs[0],inputs[1][s][0],classes,y,before,after,
                        [names[int(i)] for i in signals[:,4]],[names[int(i)] for i in signals[:,5]])
            per_class={}
            for j,c in enumerate(classes):
                ix=y==j
                per_class[c]={k:(100*(v[ix]-vals["base"][ix]).mean(0)).tolist() for k,v in vals.items()}
            path=dest.with_suffix(".pt")
            save_tensor({"metrics":vals,"top1_class":labels,"literal_and_entropy":signals,"query_labels":y},path)
            row={"signature":sig,"queries":len(y),"gallery":len(g),"classes":len(classes),
                 "values":{k:v.mean(0).tolist() for k,v in vals.items()},"primary":method,
                 "vs_base":{k:paired_delta(v,vals["base"],y) for k,v in vals.items()},
                 "per_class_delta_pp":per_class,"word_movement":{"base_misses_with_public_literal":int(misses.sum()),
                 "literal_rose":int((misses&(after<before)).sum()),"literal_fell":int((misses&(after>before)).sum()),
                 "native_and_edited_mean_entropy":signals[:,2:4].mean(0).tolist()},
                 "example_nearest_median_AP_change_in_group":picks,"per_query_sha256":checksum(path),
                 "original_kd_queries_per_arm":len(y),"kd_max_abs_error":errors,"frozen_max_error":difference,
                 "frozen_mean_error":mean_deviation,"word_exact_rearrangement_error":word_check}
            atomic_json(dest,row)
            print("saved",s,row["values"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--workers",type=int,default=5)
    a=p.parse_args()
    run(a.workspace,a.workers)
