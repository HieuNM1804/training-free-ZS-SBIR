"""Exact single-word interventions in the quadratic supporting-photo score."""

import argparse
import fcntl
from pathlib import Path

import torch

from .concept_benchmark import atomic_json, checksum, read
from .conditional_semantic_index import SIZES, bare_words
from .crossdomain_benchmark import digest, revision, SPLIT_HASHES
from .crossdomain_inputs import CHECKPOINT
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .foundation import load_words
from .gallery_lexical_feedback import LexicalFeedback
from .retrieval_diagnosis import global_inputs, stratified_sample
from .semantic import read_words
from .shared_concept_likelihood import emission


FAMILY = "word-deletion-certificates"
TOL = 2e-5


def accurate_product(left, right):
    # Long dot products with a different batch shape can accumulate enough FP32
    # rounding error to exceed a near-tie margin. Use short GPU products and
    # accumulate their partial sums in FP64 without changing the fixed index.
    result = left.new_zeros((len(left),right.size(1)),dtype=torch.double)
    for start in range(0,left.size(1),768):
        sl=slice(start,start+768)
        result.add_(left[:,sl] @ right[sl])
    return result.float()


class DeletionAudit:
    def __init__(self, index):
        self.b = index.index
        self.d = index.excluded
        self.count = index.count.clamp_min(1e-30)
        diag = torch.zeros_like(self.count, dtype=torch.double)
        for block in self.b.split(256):
            diag += block.double().square().sum(0)
        self.diagonal = (diag / self.count).float()

    def scores(self, l):
        seed = accurate_product(l,self.b.T)
        response = accurate_product(seed,self.b) / self.count
        joint = l * response
        score = (accurate_product(joint,self.b.T) + accurate_product(joint,self.d.T)
                 - seed * accurate_product(l,self.d.T))
        return score, response

    def drops(self, l, response, candidate):
        b, d = self.b[candidate], self.d[candidate]
        total = b + d
        backwards = accurate_product(accurate_product((total*l)/self.count,self.b.T),self.b)
        forward = total * response - b * (d*l).sum(1,keepdim=True)
        backwards -= d * (b*l).sum(1,keepdim=True)
        diagonal = total * self.diagonal - b*d
        return l * (forward + backwards - l*diagonal), l * forward

    def delete(self, l, words):
        altered = l.clone()
        altered.scatter_(1,words[:,None],0.)
        return self.scores(altered)[0]

    def complete_check(self, l, scores, drops, labels, brute):
        best = scores.argmax()
        original = scores[best]
        runner = scores.topk(2).values[1]
        tol = TOL * original.abs()
        risk = drops >= original - runner - tol
        ids = torch.arange(len(l),device=l.device) if brute else risk.nonzero().flatten()
        changed = class_changed = pruned_violations = near_ties = 0
        observed_words = []
        max_original_discrepancy = 0.
        for subset in ids.split(128):
            if not len(subset):
                continue
            recomputed = self.delete(l.expand(len(subset),-1),subset)
            pair = recomputed.topk(2,dim=1)
            rank1 = recomputed.argmax(1)
            flipped = rank1 != best
            near_ties += int((pair.values[:,0] - pair.values[:,1] <= tol).sum())
            changed += int(flipped.sum())
            class_changed += int((labels[rank1] != labels[best]).sum())
            pruned_violations += int((~risk[subset] & flipped).sum())
            err = (original - drops[subset] - recomputed[:,best]).abs() / original.abs().clamp_min(1e-30)
            max_original_discrepancy = max(max_original_discrepancy,float(err.max()))
            if len(observed_words) < 3:
                observed_words.extend(subset[flipped][:3-len(observed_words)].tolist())
        if pruned_violations or max_original_discrepancy > TOL:
            raise ValueError(f"Deletion check failed: missed={pruned_violations}, error={max_original_discrepancy:.7g}")
        return {"words_in_dictionary":len(l),"survive_bound":int(risk.sum()),"actually_evaluated":len(ids),
                "brute_force":brute,"deleted_words_changing_top1":changed,
                "deleted_words_changing_predicted_class":class_changed,
                "unnecessary_words_seen_to_flip":pruned_violations,"near_ties_evaluated":near_ties,
                "winner_max_scale_relative_error":max_original_discrepancy,"example_word_ids":observed_words}


def aggregate_flags(records):
    number = len(records)
    return {"queries":number,"fully_enumerated_reference_queries":sum(x["brute_force"] for x in records),
            "robust_top1":sum(x["deleted_words_changing_top1"]==0 for x in records),
            "robust_predicted_class":sum(x["deleted_words_changing_predicted_class"]==0 for x in records),
            "screened_words":sum(x["words_in_dictionary"]-x["survive_bound"] for x in records),
            "query_word_pairs":sum(x["words_in_dictionary"] for x in records),
            "words_recomputed":sum(x["actually_evaluated"] for x in records),
            "maximum_normalized_error":max(x["winner_max_scale_relative_error"] for x in records)}


@torch.inference_mode()
def run(w):
    root=w/"results"/FAMILY
    for dataset in ("sketchy","tuberlin"):
        if dataset=="tuberlin":
            atomic_json(root/"freeze.json",{"revision":revision(),"sample_per_class":4,"seed":79307,
                        "tolerance_of_winner_score":TOL,
                        "sketchy":{s:checksum(root/(s+".json")) for s in ("sketchy_1","sketchy_2")}})
        cfg,qp,queries,pp,galleries,qraw,praw = global_inputs(w,dataset)
        terms=read_words(cfg.vocabulary)
        _,scale=bare_words(cfg,terms)
        source,target=(v.cuda() for v in load_words(cfg,terms))
        for split,(qi,ql) in queries.items():
            pi,pl=galleries[split]
            if len(terms)!=20931 or (len(qi),len(pi),len(pl.unique()))!=SIZES[split] or digest(UNSEEN_CLASSES[split])!=SPLIT_HASHES[split]:
                raise ValueError("Protocol changed")
            path, details = root/(split+".json"), root/(split+".pt")
            if path.is_file() and details.is_file():
                old=read(path)
                if old["protocol"]["revision"]==revision() and old["per_query_sha256"]==checksum(details):
                    continue
            q,p=qraw[qi].cuda(),praw[pi].cuda()
            idx=LexicalFeedback(p,target,scale)
            audit=DeletionAudit(idx)
            del idx
            labels=pl.cuda()
            selected=set(stratified_sample(ql,4,79307).tolist())
            brute_index=min(selected)
            global_checks=[]
            measurements={}
            examples={}
            for i in range(0,len(q),192):
                l=emission(q[i:i+192],source,scale)
                s,h=audit.scores(l)
                top=s.argsort(dim=1,descending=True,stable=True)[:,:2]
                row=torch.arange(len(l),device=q.device)
                best,other=top.unbind(1)
                scale_ref=s[row,best].abs().clamp_min(1e-30)
                gap=s[row,best]-s[row,other]
                best_drop,bd=audit.drops(l,h,best)
                next_drop,nd=audit.drops(l,h,other)
                influence=best_drop-next_drop
                max_drop,term=influence.max(1)
                fields={"top1":best,"runner":other,"gap":gap/scale_ref,
                        "worst_margin":(gap-max_drop)/scale_ref,"word":term,
                        "top1_correct":labels[best]==ql[i:i+len(l)].cuda()}
                for name, ids in (("exact",term),("highest_sketch",l.argmax(1)),("forward_only",(bd-nd).argmax(1))):
                    altered=audit.delete(l,ids)
                    margin=altered[row,best]-altered[row,other]
                    err=(gap-influence[row,ids]-margin).abs()/scale_ref
                    if err.max()>TOL:
                        raise ValueError("Analytic and full deletion scores disagree "+str(float(err.max())))
                    fields[name+"_error"]=err
                    fields[name+"_pair_flip"]=(margin < -TOL*scale_ref)
                    fields[name+"_global_flip"]=altered.argmax(1)!=best
                for j in range(len(l)):
                    pos=i+j
                    if pos not in selected:
                        continue
                    record=audit.complete_check(l[j],s[j],best_drop[j],labels,pos==brute_index)
                    record["query"]=pos
                    record["query_label"]=int(ql[pos])
                    record["original_correct"]=bool(fields["top1_correct"][j])
                    global_checks.append(record)
                    key="global_changes_class" if record["deleted_words_changing_predicted_class"] else "class_stable"
                    if key not in examples:
                        examples[key]={"query":str(qp[int(qi[pos])]),"initial_photo":str(pp[int(pi[int(best[j])])]),
                                       "runner":str(pp[int(pi[int(other[j])])]),"max_pair_influence":terms[int(term[j])],
                                       "word_examples_changing_photo":[terms[k] for k in record["example_word_ids"]],
                                       "changes_of_first_photo":record["deleted_words_changing_top1"],
                                       "changes_of_predicted_class":record["deleted_words_changing_predicted_class"]}
                for name,value in fields.items():
                    measurements.setdefault(name,[]).append(value.cpu())
                if i%3840==0:
                    print(split,i+len(l),"/",len(q),"global audits",len(global_checks),flush=True)
            if len(global_checks)!=len(selected):
                raise ValueError("Stratified sample not exhausted")
            values={k:torch.cat(v) for k,v in measurements.items()}
            if any(len(v)!=len(q) for v in values.values()):
                raise ValueError("Incomplete pair audit")
            positive=values["worst_margin"]>TOL
            negative=values["worst_margin"]<-TOL
            save_tensor({"query_labels":ql,"fields":values},details)
            result={"protocol":{"revision":revision(),"split":split,"queries":len(q),"gallery":len(p),"terms":len(terms),
                                "vocabulary_sha256":checksum(cfg.vocabulary),"classes":len(pl.unique()),
                                "checkpoint":CHECKPOINT,"gpu":torch.cuda.get_device_name()},
                    "pair":{"certainly_stable":int(positive.sum()),"certainly_flippable":int(negative.sum()),
                            "near_numerical_tie":int((~positive & ~negative).sum()),
                            "top1_correct_among_stable":int(values["top1_correct"][positive].sum()),
                            "top1_correct_among_flippable":int(values["top1_correct"][negative].sum()),
                            "selection":{name:{"pair_flips":int(values[name+"_pair_flip"].sum()),
                                               "global_top1_changes":int(values[name+"_global_flip"].sum()),
                                               "max_error_relative_to_original_winner":float(values[name+"_error"].max())}
                                         for name in ("exact","highest_sketch","forward_only")}},
                    "full_gallery":aggregate_flags(global_checks),"sample_records":global_checks,"examples":examples,
                    "per_query_sha256":checksum(details)}
            atomic_json(path,result)
            print(split,result["pair"],result["full_gallery"],flush=True)
            del audit


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    workspace=parser.parse_args().workspace
    output=workspace/"results"/FAMILY
    output.mkdir(parents=True,exist_ok=True)
    with (output/".audit.lock").open("a") as guard:
        try:
            fcntl.flock(guard.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print("An audit is already running in this Colab workspace.",flush=True)
        else:
            run(workspace)
