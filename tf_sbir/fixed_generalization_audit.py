"""Audit fixed-formula retrieval and distinguish transfer of the procedure from transfer of its prior."""

import argparse
import csv
from pathlib import Path

import torch
from torch.nn import functional as F

from .concept_benchmark import atomic_json,checksum,read
from .conditional_semantic_index import bare_words
from .crossdomain_benchmark import digest
from .data import UNSEEN_CLASSES
from .foundation import load_words
from .retrieval_diagnosis import global_inputs
from .retrieval_evidence import check_record,paired_delta,tensor
from .semantic import read_words
from .shared_concept_likelihood import FAMILY,OTHER_PRIMARY,PRIMARY,REFERENCE,emission,fit_prior,posterior,target_inputs
from .verify_locked_ranks import native_top_ids


def nameset(name):
    return " ".join(name.casefold().replace("_"," ").replace("-"," ").split())


def metric(v):
    means=v.mean(0).tolist()
    return {"mAP":means[0],"precision":means[1]}


def wrong_ranking(top,ql,pl):
    error=pl[top]!=ql[:,None]
    count=top[error].bincount(minlength=len(pl))
    return {"wrong_at1":int(error[:,0].sum()),"wrong_at10":int(error.sum()),
        "wrong_top10_share_top1pct":float(count.topk(max(1,round(len(pl)*.01))).values.sum()/count.sum().clamp_min(1))}


def compare_pairs(q,p,ql,pl,native,selected,pst,uniform,src,scale,terms,qp,pp):
    y,z=ql.cuda(),pl.cuda(); wrong=pl[native[:,0]]!=ql
    recovered=[]; uniform_fixed=[]; examples=[]
    for start in range(0,len(q),160):
        data=q[start:start+160]; good=z[None]==y[start:start+len(data),None]
        scores=data@p.T
        true=scores.masked_fill(~good,-torch.inf).argmax(1)
        bad=native[start:start+len(data),0].cuda()
        mass=emission(data,src,scale)
        recovered.append(((mass*pst[true]).sum(1)>(mass*pst[bad]).sum(1)).cpu())
        uniform_fixed.append(((mass*uniform[true]).sum(1)>(mass*uniform[bad]).sum(1)).cpu())
    new=pl[selected[:,0]]==ql
    masks={"rescued":wrong&new,"regressed":~wrong&~new,"still_wrong":wrong&~new}
    for reason,mask in masks.items():
        ids=mask.nonzero().flatten()
        rng=torch.Generator().manual_seed(5345)
        for j in ids[torch.randperm(len(ids),generator=rng)[:3]].tolist():
            i=int(selected[j,0])
            observation=emission(q[j:j+1],src,scale)[0]*pst[i]
            rates,words=(observation/observation.sum().clamp_min(1e-38)).topk(3)
            examples.append({"case":reason,"sketch":"/".join(Path(qp[j]).parts[-3:]),
                "native_top1":"/".join(Path(pp[int(native[j,0])]).parts[-3:]),
                "method_top1":"/".join(Path(pp[i]).parts[-3:]),
                "top_shared_word_contributions":[(terms[h],v) for h,v in zip(words.tolist(),rates.tolist())]})
    return {"native_wrong_top1":int(wrong.sum()),"primary_ranks_native_first_positive_above_native_wrong":int(torch.cat(recovered)[wrong].sum()),
            "uniform_ranks_native_first_positive_above_native_wrong":int(torch.cat(uniform_fixed)[wrong].sum()),
            "rescued":int(masks["rescued"].sum()),"regressed":int(masks["regressed"].sum()),"examples":examples}


@torch.inference_mode()
def audit(w):
    root=w/"results"; folder=root/FAMILY
    joint=read(folder/"joint-freeze.json"); sha=joint.pop("sha256")
    if digest(joint)!=sha or any(r["tu_rows_prelock"]!=0 for r in joint["studies"].values()):
        raise ValueError("Invalid joint lock")
    def criterion(name):
        return tuple(sum(row[m] for row in joint["studies"][name]["scores"])/2 for m in ("precision","mAP"))
    if joint["selected"]!=max(joint["studies"],key=criterion):
        raise ValueError("Winner contradicts the selection rule")
    prior_lock=read(folder/"strict/source-lock.json"); plan=read(folder/"transfer-tests-plan.json")
    if (prior_lock["prior_sha256"]!=checksum(folder/"strict/sketchy-prior.pt") or prior_lock["new_tu_scores_at_lock"]
            or prior_lock["plan_sha256"]!=digest(plan) or plan["utc"]>=joint["utc"]):
        raise ValueError("Strict transfer was not frozen")
    for name,study in joint["studies"].items():
        signed=read(root/name/"protocol.json")
        b=signed.pop("sha256")
        if b!=study["freeze_sha256"] or digest(signed)!=b:
            raise ValueError("Altered hypothesis")
        for row in study["scores"]:
            p=root/name/"sketchy"/f"{row['split']}-{study['primary']}.json"
            if row["sha256"]!=checksum(p):
                raise ValueError("Changed frozen result")
    report={"joint":sha,"chosen":joint["selected"],"strict_source_prior":prior_lock,"splits":{}}
    records=[]
    sk_names={nameset(n) for s in ("sketchy_1","sketchy_2") for n in UNSEEN_CLASSES[s]}
    for dataset in ("sketchy","tuberlin"):
        a,qpaths,qs,ppaths,gs,qraw,praw=global_inputs(w,dataset)
        terms=read_words(a.vocabulary); _,scale=bare_words(a,terms)
        src,tgt=(x.cuda() for x in load_words(a,terms))
        oldname="reverse-prompt-diffusion"
        hist=read(root/"inquiry-joint-freeze.json")["families"][oldname]["scores"][0]["method"]
        extra=read(folder/"strict"/f"{dataset}-transfer.json")
        for s,(qi,ql) in qs.items():
            pi,pl=gs[s]; names=UNSEEN_CLASSES[s]
            sub="heldout" if dataset=="tuberlin" else "sketchy"
            conditions={}; per={}
            for fam in (FAMILY,REFERENCE,"patch-style-invariance"):
                path=root/fam/sub
                for f in sorted(path.glob(s+"-*.json")):
                    row=read(f); key=fam+"/"+row["method"]
                    values=check_record(row,path,ql,pl,s)
                    expected=read(root/fam/"protocol.json")["sha256"]
                    if row["freeze_sha256"]!=expected or row["exploratory_unseen"]:
                        raise ValueError("Wrong protocol in result")
                    per[key]=values;conditions[key]=metric(values)
                    records.append([s,fam,row["method"],row["precision"],row["mAP"],row["revision"],expected])
            for v,check in extra["splits"][s].items():
                vals=tensor(folder/"strict/per-query"/f"{s}-{v}.pt")
                if vals.shape!=(len(ql),2) or max(abs(metric(vals)[m]-check[m]) for m in ("mAP","precision"))>1e-6:
                    raise ValueError("Strict transfer query result mismatch")
                per["prior/"+v]=vals;conditions["prior/"+v]=metric(vals)
            historical=root/oldname/sub
            stored=tensor(historical/f"{s}-labels.pt")
            if not torch.equal(stored["query"],ql) or not torch.equal(stored["gallery"],pl):
                raise ValueError("Dataset ordering mismatch")
            per["previous"]=check_record(read(historical/f"{s}-{hist}.json"),historical,ql,pl,s)
            conditions["previous"]=metric(per["previous"])
            primary=per[FAMILY+"/"+PRIMARY]
            controls={k:paired_delta(primary,v,ql) for k,v in per.items()
                      if k.startswith((FAMILY+"/","prior/")) or k in ("previous",REFERENCE+"/"+OTHER_PRIMARY,REFERENCE+"/native")}
            named_result=read(folder/f"{dataset}-complete.json")["splits"][s]
            kd=named_result["scores"][PRIMARY]["original_KD"]
            if (kd["queries"]!=len(ql) or max(kd["kd_minus_pipeline_max_abs_same_order_AP_P"])!=0 or
                    max(abs(kd[m]-metric(primary)[m]) for m in ("mAP","precision"))>1e-6):
                raise ValueError("Original KD evaluator fails for the scored rankings")
            q,p=qraw[qi].cuda(),praw[pi].cuda()
            native=native_top_ids(q,p)
            top=tensor(folder/sub/f"{s}-top10.pt")[PRIMARY].long()
            em=target_inputs(p,tgt,scale); priors,stat=fit_prior(em)
            out=compare_pairs(q,p,ql,pl,native,top,posterior(em,priors["mle"]),posterior(em,priors["uniform"]),
                              src,scale,terms,[qpaths[i] for i in qi.tolist()],[ppaths[i] for i in pi.tolist()])
            sets={}
            if dataset=="tuberlin":
                for key,keep in (("names_shared_with_Sketchy",True),("names_absent_from_Sketchy",False)):
                    ids=[i for i,n in enumerate(names) if (nameset(n) in sk_names)==keep]
                    mask=torch.isin(ql,torch.tensor(ids))
                    labels=torch.unique(ql[mask],return_inverse=True)[1]
                    methods={m:metric(per[m][mask]) for m in (FAMILY+"/"+PRIMARY,FAMILY+"/uniform",REFERENCE+"/"+OTHER_PRIMARY,
                        REFERENCE+"/native","prior/source-prior","prior/reference-half-a","prior/reference-half-b")}
                    sets[key]={"classes":[names[i] for i in ids],"queries":int(mask.sum()),"methods":methods,
                        "primary_vs_native":paired_delta(primary[mask],per[REFERENCE+"/native"][mask],labels)}
            report["splits"][s]={"queries":len(ql),"gallery":len(pl),"classes":len(names),"conditions":conditions,
                "primary_contrasts":controls,"prior":stat,"KD_check":kd,"error_pairs":out,"name_disjoint_checks":sets,
                "native_errors":wrong_ranking(native,ql,pl),"primary_errors":wrong_ranking(top,ql,pl)}
            print(s,metric(primary),out["rescued"],flush=True)
            del em,priors
    atomic_json(folder/"generalization-audit.json",report)
    with (folder/"fixed-configurations.csv").open("w",newline="") as f:
        writer=csv.writer(f,lineterminator="\n")
        writer.writerow(["split","family","method","precision","mAP","revision","protocol"])
        writer.writerows(records)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    audit(p.parse_args().workspace)
