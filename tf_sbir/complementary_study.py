"""Evaluate a fixed minibatch assignment of lexical evidence on independent queries."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import os
from pathlib import Path
import time

import torch

from .complementary_concepts import ComplementaryIndex,joint_distribution
from .concept_benchmark import atomic_json,checksum
from .conditional_agreement_study import Replay
from .conditional_semantic_index import SIZES,bare_words
from .crossdomain_benchmark import revision,digest,SPLIT_HASHES
from .crossdomain_inputs import CHECKPOINT
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .map_diagnosis import map_values
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_diagnosis import global_inputs
from .retrieval_evidence import load_kd,paired_delta
from .semantic import read_words


FAMILY="complementary-concept-witness"


def enumerate_matchings(b):
    ids=torch.arange(len(b),device=b.device)
    u,x,v,y=torch.cartesian_prod(ids,ids,ids,ids).T
    aligned=(u==v).int()+(x==y).int()
    crossed=(u==y).int()+(x==v).int()
    choose=(aligned<crossed).double()+.5*(aligned==crossed).double()
    mass=b[u]*b[x]*b[v]*b[y]*.5
    result=b.new_zeros(len(b)*len(b))
    for i,j,prob in ((u,v,choose),(x,y,choose),(u,y,1-choose),(x,v,1-choose)):
        result.scatter_add_(0,i*len(b)+j,mass*prob)
    return result.reshape(len(b),len(b))


def verify(index,q,p,words,scale):
    errors=[]
    for row in index.model.profile[:4]:
        b=row.topk(9).values.double()
        b=b/b.sum()
        exact=enumerate_matchings(b)
        form=joint_distribution(b)
        errors.append([float((exact-form).abs().max()),float((form.sum(0)-b).abs().max()),
                       float((form.sum(1)-b).abs().max())])
    if max(max(v) for v in errors)>1e-10:
        raise ValueError("Matching expectation or marginals disagree with exhaustive assignment")
    # Check both independent evidence paths through every query-word intervention.
    select=torch.linspace(0,len(p)-1,72,device=p.device).long()
    small=ComplementaryIndex(p[select],words,scale)
    l=small.likelihood(q[0])
    score=small.score_likelihood(l)["paired"][0]
    first,second=score.argsort(descending=True,stable=True)[:2].tolist()
    answer=small.explain_pair(q[0],first,second)
    gap=[]
    for ids in torch.arange(len(words),device=p.device).split(192):
        altered=l.expand(len(ids),-1).clone()
        altered.scatter_(1,ids[:,None],0.)
        scores=small.score_likelihood(altered)["paired"]
        gap.append(scores[:,first]-scores[:,second])
    actual=torch.cat(gap)
    reference=answer["margin_after_each_deletion"]
    tol=2e-5*score[first]
    error=float((actual-reference).abs().max()/score[first])
    flips=int(((actual*reference<0)&(actual.abs()>tol)&(reference.abs()>tol)).sum())
    if error>2e-5 or flips:
        raise ValueError("Counterfactual margin verification failed")
    return {"exact_matching_fixtures":len(errors),"enumerated_assignments_per_fixture":9**4,
            "joint_coefficient_max_error":max(v[0] for v in errors),
            "marginal_max_error":max(max(v[1:]) for v in errors),
            "deletion_query_gallery":len(select),"individual_word_deletions":len(words),
            "deletion_margin_max_relative_error":error,"discordant_signs_beyond_tolerance":flips}


@torch.inference_mode()
def run(w,workers):
    root=w/"results"/FAMILY
    protocol={"method":"minimum identity collisions between two pairs, uniform tie-breaking",
              "selection":"fixed before all scores of this study","conditioning":"distinct witness image",
              "dictionary":"literal Open Images terms, identical in both roles","graph":"one non-self gallery Gibbs step",
              "scale":"native CLIP temperature","revision":revision()}
    atomic_json(root/"protocol.json",protocol|{"sha256":digest(protocol)})
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:
        raise ValueError("Wrong KD evaluator")
    for dataset in ("sketchy","tuberlin"):
        if dataset=="tuberlin":
            atomic_json(root/"freeze.json",{"revision":revision(),"primary":"paired","selection":"none",
                        "prior_scores":{s:checksum(root/(s+".json")) for s in ("sketchy_1","sketchy_2")}})
        cfg,_,queries,_,galleries,qraw,praw=global_inputs(w,dataset)
        terms=read_words(cfg.vocabulary)
        if len(terms)!=20931:
            raise ValueError("Public vocabulary changed")
        words,scale=bare_words(cfg,terms)
        for split,(qi,ql) in queries.items():
            pi,pl=galleries[split]
            if (len(qi),len(pi),len(pl.unique()))!=SIZES[split] or digest(UNSEEN_CLASSES[split])!=SPLIT_HASHES[split]:
                raise ValueError("Incomplete data")
            q,p=qraw[qi].cuda(),praw[pi].cuda()
            start=time.perf_counter()
            index=ComplementaryIndex(p,words,scale)
            verify_result=verify(index,q,p,words,scale)
            ref=Replay(kd,pl,split)
            per,errors,best={}, {},{}
            y_g=pl.cuda()
            numeric=0.
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for k in range(0,len(q),192):
                    y=ql[k:k+192]
                    scores=index.score(q[k:k+192])
                    # The law of total expectation decomposes the unrestricted independent model.
                    decomposed=index.collision*scores["collision_only"]+index.complement*scores["reject_repeats"]
                    numeric=max(numeric,float(((decomposed-scores["independent"]).abs()/
                                               scores["independent"].amax(1,keepdim=True).clamp_min(1e-30)).max()))
                    for mode,s in scores.items():
                        if not torch.isfinite(s).all():
                            raise ValueError("Nonfinite "+mode)
                        quick,order,_,_=map_values(s,y.cuda(),y_g,split)
                        val=ref.score(order,y,pool)
                        err=float((val-quick.cpu()).abs().max())
                        if err>2e-6:
                            raise ValueError("KD replay mismatch")
                        errors[mode]=max(errors.get(mode,0.),err)
                        per.setdefault(mode,[]).append(val)
                        best.setdefault(mode,[]).append(order[:,0].cpu().int())
                    if k%3840==0:
                        print(split,k+len(y),"/",len(ql),flush=True)
            per={k:torch.cat(v) for k,v in per.items()}
            best={k:torch.cat(v) for k,v in best.items()}
            if any(v.shape!=(len(q),2) for v in per.values()):
                raise ValueError("Incomplete metric replay")
            details=root/(split+".pt")
            save_tensor({"query_labels":ql,"metrics":per,"top1":best},details)
            rec={"protocol":{"split":split,"queries":len(ql),"gallery":len(pl),"classes":len(pl.unique()),
                             "revision":revision(),"checkpoint":CHECKPOINT,"kd_function_sha256":KD_FUNCTION_SHA,
                             "vocabulary_sha256":checksum(cfg.vocabulary),"gpu":torch.cuda.get_device_name()},
                 "map_k":ref.cutoffs[0],"p_k":ref.cutoffs[1],"kd_max_abs_error":errors,"kd_queries_per_method":len(ql),
                 "scores":{k:dict(zip(("mAP","precision"),v.mean(0).tolist())) for k,v in per.items()},
                 "paired_vs":{k:paired_delta(per["paired"],v,ql) for k,v in per.items()},
                 "reject_vs_marginals":paired_delta(per["reject_repeats"],per["rejection_marginals"],ql),
                 "mean_probability_same_word":{"independent":float(index.collision.mean()),
                                               "paired":float((index.collision-index.removed).mean())},
                 "rank1_identity_changes_from_independent":int((best["paired"]!=best["independent"]).sum()),
                 "control_decomposition_max_relative_error":numeric,"verification":verify_result,
                 "per_query_sha256":checksum(details),"elapsed_seconds":time.perf_counter()-start}
            atomic_json(root/(split+".json"),rec)
            print(split,rec["scores"],flush=True)
            del index


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    parser.add_argument("--workers",type=int,default=min(10,len(os.sched_getaffinity(0))))
    args=parser.parse_args()
    run(args.workspace,args.workers)
