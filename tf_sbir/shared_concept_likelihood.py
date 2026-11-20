"""Likelihood ratio for shared versus independent latent concepts, using frozen CLIP emissions."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path

import torch
from torch.nn import functional as F

from .concept_benchmark import atomic_json,checksum,read
from .concept_likelihood import ranked_scores
from .conditional_semantic_index import SIZES,bare_words
from .crossdomain_benchmark import SPLIT_HASHES,digest,revision
from .crossdomain_inputs import CHECKPOINT
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .foundation import load_words
from .retrieval_diagnosis import global_inputs,stratified_sample
from .retrieval_evidence import load_kd,paired_delta,tensor
from .semantic import read_words
from .transfer_benchmark import write_result
from .verify_locked_ranks import replay


FAMILY="shared-concept-likelihood"
PRIMARY="named-likelihood"
REFERENCE="conditional-semantic-index"
OTHER_PRIMARY="bare-witness"
NAMES=(PRIMARY,"uniform","one-update","posterior-mean","shuffled-identity","bare-likelihood")


def lock(root):
    path=root/"protocol.json"
    fields={"revision":revision(),"primary":PRIMARY,"controls":list(NAMES),"checkpoint":CHECKPOINT,
            "external_words":20931,"source_protocol_sha256":checksum(Path(__file__).resolve().parents[1]/"docs/SHARED_CONCEPT_PROTOCOL_VI.md"),
            "parameter_selection":"none","prior":"MLE of fixed vMF mixture on unlabeled gallery"}
    if path.exists():
        r=read(path); sig=r.pop("sha256")
        if digest(r)!=sig or any(r[k]!=v for k,v in fields.items()):
            raise ValueError("Invalid probability-model protocol")
        return r|{"sha256":sig}
    if any(root.glob("*/*.json")):
        raise ValueError("Preexisting trial without a protocol")
    fields["utc"]=datetime.now(timezone.utc).isoformat()
    atomic_json(path,fields|{"sha256":digest(fields)})
    return read(path)


def join(w):
    root=w/"results"; path=root/FAMILY/"joint-freeze.json"
    if path.exists():
        obj=read(path); sig=obj.pop("sha256")
        if digest(obj)!=sig:
            raise ValueError("Damaged family selection")
        return obj["selected"]
    studies={}
    for name,method in ((REFERENCE,OTHER_PRIMARY),(FAMILY,PRIMARY)):
        folder=root/name
        pre=read(folder/"protocol.json")
        sig=pre.pop("sha256")
        if digest(pre)!=sig or pre["primary"]!=method or any((folder/"heldout").glob("tuberlin-*.json")):
            raise ValueError("Protocol mismatch or TU already scored")
        if read(folder/"sketchy-complete.json")["freeze_sha256"]!=sig:
            raise ValueError("Sketchy program incomplete")
        scores=[]
        for s in ("sketchy_1","sketchy_2"):
            f=folder/"sketchy"/f"{s}-{method}.json"; row=read(f)
            if row["freeze_sha256"]!=sig or row["split_sha256"]!=SPLIT_HASHES[s]:
                raise ValueError("Invalid Sketchy score")
            scores.append({k:row[k] for k in ("split","mAP","precision")}|{"sha256":checksum(f)})
        studies[name]={"primary":method,"freeze_sha256":sig,"tu_rows_prelock":0,"scores":scores,"revision":pre["revision"]}
    def criterion(name):
        return tuple(sum(v[key] for v in studies[name]["scores"])/2 for key in ("precision","mAP"))
    selected=max(studies,key=criterion)
    obj={"utc":datetime.now(timezone.utc).isoformat(),"studies":studies,"selected":selected,
         "rule":"mean Sketchy precision then mean Sketchy mAP, two fixed hypotheses"}
    atomic_json(path,obj|{"sha256":digest(obj)})
    return selected


def emission(images,directions,scale):
    values=(images@directions.T)*scale
    return (values-values.amax(1,keepdim=True)).exp_()


def fit_prior(a):
    n,w=a.shape
    uniform=a.new_full((w,),1/w)
    prior=uniform
    last=None; first=None
    eps=torch.finfo(a.dtype).eps
    for step in range(1,2001):
        sums=(a@prior).clamp_min(torch.finfo(a.dtype).tiny)
        objective=float(sums.log().double().mean())
        if last is not None:
            if objective < last-8*eps:
                raise ValueError("Non-monotone mixture log likelihood")
            if objective-last <= eps**.5:
                break
        new=prior*(a.T@sums.reciprocal())/n
        prior=new/new.sum()
        if first is None:
            first=prior
        last=objective
    else:
        raise RuntimeError("Gallery prior did not converge")
    entropy=-(prior*prior.clamp_min(1e-38).log()).sum()
    return {"uniform":uniform,"one-update":first,"mle":prior},{"iterations":step-1,
        "relative_log_likelihood":objective,"last_increase":objective-last,"tolerance_nats":eps**.5,
        "effective_number_of_concepts":float(entropy.exp()),"maximum_mass":float(prior.max())}


def posterior(emissions,prior):
    mass=emissions*prior
    return mass/mass.sum(1,keepdim=True).clamp_min(torch.finfo(mass.dtype).tiny)


def target_inputs(p,bank,scale):
    # A single dense tensor preserves the contribution of every word. Rows are
    # normalized only up to their maximum, which cancels in the likelihood ratio.
    em=p.new_empty(len(p),len(bank))
    for i in range(0,len(p),384):
        em[i:i+384]=emission(p[i:i+384],bank,scale)
    return em


def score_blocks(q,pst,source,scale,mode):
    if mode=="posterior-mean":
        code=F.normalize(pst@source)
        for a in q.split(192):
            yield a@code.T
        return
    if mode=="shuffled-identity":
        perm=torch.randperm(len(source),generator=torch.Generator().manual_seed(6727)).to(source.device)
        source=source[perm]
    for a in q.split(192):
        like=emission(a,source,scale)
        yield like@pst.T


@torch.inference_mode()
def run(w,stage):
    folder=w/"results"/FAMILY
    signed=lock(folder)
    if stage=="joint":
        print(join(w))
        return
    if stage=="tuberlin":
        selection=read(folder/"joint-freeze.json"); sig=selection.pop("sha256")
        if digest(selection)!=sig or selection["studies"][FAMILY]["freeze_sha256"]!=signed["sha256"]:
            raise ValueError("Incorrect TU decision")
    a,qp,qs,pp,gs,raw,photos=global_inputs(w,"tuberlin" if stage=="tuberlin" else "sketchy")
    terms=read_words(a.vocabulary); bare,scale=bare_words(a,terms)
    source,target=(b.cuda() for b in load_words(a,terms))
    if len(terms)!=20931:
        raise ValueError("Unexpected dictionary")
    kd,kd_sha=load_kd(w)
    report={"freeze_sha256":signed["sha256"],"scale":scale,"kd_source_sha256":kd_sha,"splits":{}}
    for s,(qi,ql) in qs.items():
        pi,pl=gs[s]
        if (len(ql),len(pl),len(ql.unique()))!=SIZES[s] or len(pl.unique())!=SIZES[s][2] or digest(UNSEEN_CLASSES[s])!=SPLIT_HASHES[s]:
            raise ValueError("Unexpected split")
        pilot=stage=="pilot"
        if pilot:
            subset=stratified_sample(ql,32,9001);qi,ql=qi[subset],ql[subset]
        q,p=raw[qi].cuda(),photos[pi].cuda()
        k=200 if s=="sketchy_2" else 100
        place=folder/("heldout" if stage=="tuberlin" else "sketchy")
        info={}; rows={}; tops={}; details={}
        for variant,modes in (("named",NAMES[:-1]),("bare",[NAMES[-1]])):
            src=source if variant=="named" else bare
            tgt=target if variant=="named" else bare
            em=target_inputs(p,tgt,scale)
            priors,stat=fit_prior(em)
            vals,ids=priors["mle"].topk(10)
            info[variant]=stat|{"common_terms":[[terms[i],v] for i,v in zip(ids.tolist(),vals.tolist())]}
            for mode in modes:
                prior=priors.get(mode,priors["mle"])
                pst=posterior(em,prior)
                if pilot:
                    scores=torch.cat(list(score_blocks(q,pst,src,scale,mode)))
                    top=scores.topk(k).indices
                    correct=pl.cuda()[top]==ql.cuda()[:,None]
                    rows[mode]={"precision":float(correct.float().mean()),"top1":float(correct[:,0].float().mean())}
                    del scores
                else:
                    # Cache primary scores only for independent per-query KD replay.
                    retained=[]; headlines=[]
                    def generate():
                        for b in score_blocks(q,pst,src,scale,mode):
                            if not torch.isfinite(b).all():
                                raise ValueError("Nonfinite likelihood ratio")
                            if mode==PRIMARY:
                                retained.append(b.cpu())
                            headlines.append(b.argsort(dim=1,descending=True,stable=True)[:,:10].int().cpu())
                            yield from ranked_scores(b).cpu()
                    a.output=place;a.metric_workers=5 if k==100 else 3
                    meta={"revision":signed["revision"],"gpu":torch.cuda.get_device_name(),"external_vocabulary":len(terms),
                          "backbone":"OpenAI CLIP ViT-B/32","checkpoint_sha256":CHECKPOINT,"classes":len(UNSEEN_CLASSES[s]),
                          "freeze_sha256":signed["sha256"],"split_sha256":SPLIT_HASHES[s],"exploratory_unseen":False}
                    write_result(a,s,mode,generate(),ql,pl,meta,{"mode":mode,"scale":scale,"prior_estimation":"fixed emissions"})
                    if not headlines:
                        for b in score_blocks(q,pst,src,scale,mode):
                            headlines.append(b.argsort(dim=1,descending=True,stable=True)[:,:10].int().cpu())
                            if mode==PRIMARY:
                                retained.append(b.cpu())
                    tops[mode]=torch.cat(headlines)
                    r=read(place/f"{s}-{mode}.json"); details[mode]=tensor(place/"per-query"/f"{s}-{mode}.pt")
                    if details[mode].shape!=(len(q),2) or max(abs(float(details[mode][:,i].mean())-r[z]) for i,z in enumerate(("mAP","precision")))>1e-6:
                        raise ValueError("Incorrect summary")
                    rows[mode]={"mAP":r["mAP"],"precision":r["precision"]}
                    if mode==PRIMARY:
                        with ThreadPoolExecutor(max_workers=6) as pool:
                            check,_=replay(kd,pool,retained,ql,pl,details[mode],s)
                        rows[mode]["original_KD"]=check
                        if max(check["kd_minus_pipeline_max_abs_same_order_AP_P"])>1e-7:
                            raise ValueError("Metric mismatch")
                    del retained,headlines
                del pst
            del em
        summary={"queries":len(ql),"gallery":len(pl),"scores":rows,"prior":info}
        if not pilot:
            save_tensor(tops,place/f"{s}-top10.pt")
            summary["paired"]={key:paired_delta(details[PRIMARY],val,ql) for key,val in details.items() if key!=PRIMARY}
            summary["per_class_AP_P"]={n:{key:values[ql==i].mean(0).tolist() for key,values in details.items()}
                                       for i,n in enumerate(UNSEEN_CLASSES[s])}
        report["splits"][s]=summary
        print(s,{m:round(100*r["precision"],3) for m,r in rows.items()},flush=True)
    atomic_json(folder/f"{stage}-complete.json",report)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    parser.add_argument("--stage",choices=("pilot","sketchy","joint","tuberlin"),required=True)
    args=parser.parse_args()
    run(args.workspace,args.stage)
