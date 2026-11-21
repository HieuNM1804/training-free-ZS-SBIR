"""Recover actual first photos before and after a single latent word intervention."""

import argparse
from pathlib import Path

import torch

from .concept_benchmark import atomic_json, checksum, read
from .conditional_semantic_index import bare_words
from .crossdomain_benchmark import revision
from .foundation import load_words
from .gallery_lexical_feedback import LexicalFeedback
from .lexical_deletion_certificate import DeletionAudit, FAMILY, TOL
from .retrieval_diagnosis import global_inputs
from .semantic import read_words
from .shared_concept_likelihood import emission


@torch.inference_mode()
def run(w):
    folder=w/"results"/FAMILY
    report={}
    for dataset in ("sketchy","tuberlin"):
        cfg,qp,queries,pp,galleries,qraw,praw=global_inputs(w,dataset)
        words=read_words(cfg.vocabulary)
        _,scale=bare_words(cfg,words)
        source,target=(bank.cuda() for bank in load_words(cfg,words))
        for split,(qi,ql) in queries.items():
            pi,pl=galleries[split]
            result=read(folder/(split+".json"))
            cases=result["sample_records"]
            selected={}
            for row in cases:
                if row["deleted_words_changing_predicted_class"]:
                    selected.setdefault("first_class_change",row["query"])
                elif row["deleted_words_changing_top1"]:
                    selected.setdefault("first_same_class_change",row["query"])
            index=LexicalFeedback(praw[pi].cuda(),target,scale)
            audit=DeletionAudit(index)
            del index
            split_results={}
            for kind,pos in selected.items():
                l=emission(qraw[qi[pos:pos+1]].cuda(),source,scale)
                score,h=audit.scores(l)
                best=score[0].argmax()
                original=score[0,best]
                runner=score[0].topk(2).values[1]
                drops,_=audit.drops(l,h,best.view(1))
                risky=(drops[0]>=original-runner-TOL*original).nonzero().flatten()
                for group in risky.split(128):
                    if not len(group):
                        continue
                    altered=audit.delete(l.expand(len(group),-1),group)
                    new=altered.argmax(1).cpu()
                    match=(pl[new]!=pl[int(best)]) if kind=="first_class_change" else (new!=int(best))
                    if match.any():
                        row=int(match.nonzero()[0,0])
                        term=int(group[row])
                        challenger=int(new[row])
                        split_results[kind]={"query_index":pos,"query":str(qp[int(qi[pos])]),
                            "original":str(pp[int(pi[int(best)])]),"after_deletion":str(pp[int(pi[challenger])]),
                            "term":words[term],"term_rank_in_sketch":int((l[0]>l[0,term]).sum())+1,
                            "original_correct":bool(pl[int(best)]==ql[pos]),"after_correct":bool(pl[challenger]==ql[pos]),
                            "old_normalized_margin":float((original-score[0,challenger])/original),
                            "new_margin_on_same_scale":float((altered[row,best]-altered[row,challenger])/original)}
                        break
                if kind not in split_results:
                    raise ValueError("Failed to reproduce a stored intervention: "+kind)
            report[split]={"audit_sha256":checksum(folder/(split+".json")),"examples":split_results}
            del audit
    atomic_json(folder/"case-details.json",{"revision":revision(),"splits":report})
    print(report,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(parser.parse_args().workspace)
