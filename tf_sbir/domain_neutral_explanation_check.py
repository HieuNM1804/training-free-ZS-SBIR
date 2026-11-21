"""Exhaustively check latent-word pair explanations on domain-neutral retrieval fixtures."""

import argparse
from pathlib import Path

import torch

from .concept_benchmark import atomic_json
from .conditional_semantic_index import bare_words
from .crossdomain_benchmark import revision
from .domain_neutral_witness import DomainNeutralIndex
from .retrieval_diagnosis import global_inputs
from .semantic import read_words


@torch.inference_mode()
def run(w):
    records={}
    for dataset in ("sketchy","tuberlin"):
        cfg,_,queries,_,gallery,qraw,praw=global_inputs(w,dataset)
        terms=read_words(cfg.vocabulary)
        words,scale=bare_words(cfg,terms)
        for split,(qi,_) in queries.items():
            pi,_=gallery[split]
            sample=torch.linspace(0,len(pi)-1,72).long()
            index=DomainNeutralIndex(praw[pi[sample]].cuda(),words,scale)
            stats=[]
            for qid in (len(qi)//3,2*len(qi)//3):
                query=qraw[qi[qid]].cuda()
                l=index.likelihood(query)
                score=index.score(query)["witness"][0]
                first,second=score.argsort(descending=True,stable=True)[:2].tolist()
                truth=index.explain_pair(query,first,second)
                changes=[]
                for ids in torch.arange(len(terms),device=words.device).split(192):
                    edited=l.expand(len(ids),-1).clone()
                    edited.scatter_(1,ids[:,None],0.)
                    actual=index.model.scores(edited)["witness"]
                    changes.append(actual[:,first]-actual[:,second])
                tested=torch.cat(changes)
                predicted=truth["margin_after_each_deletion"]
                tol=float(score[first])*2e-5
                discrepancies=int(((tested*predicted<0) & (tested.abs()>tol) & (predicted.abs()>tol)).sum())
                error=float((tested-predicted).abs().max()/score[first])
                if error>2e-5 or discrepancies:
                    raise ValueError("Public API certificate disagrees with exhaustive intervention")
                stats.append({"query_index":qid,"gallery_count":len(sample),"tested_deletions":len(terms),
                              "maximum_scale_relative_error":error,"discordant_signs_outside_tolerance":discrepancies,
                              "explanation_word":terms[truth["most_damaging_word"]],"observed_reversals_beyond_tolerance":int((tested < -tol).sum())})
            records[split]=stats
    atomic_json(w/"results/domain-agnostic-word-feedback/explanation-check.json",{"revision":revision(),"splits":records})
    print(records,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(parser.parse_args().workspace)
