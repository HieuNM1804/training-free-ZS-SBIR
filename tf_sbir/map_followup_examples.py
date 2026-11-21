"""Read stored ranks and AP to identify representative changes after the mAP selection."""

import argparse
from pathlib import Path

import torch

from .concept_benchmark import atomic_json,read
from .data import SKETCHY_SPLITS,UNSEEN_CLASSES,collect_paths
from .map_followup import METHODS,OLD,signed
from .retrieval_evidence import tensor


def short(path):
    return "/".join(Path(path).parts[-3:])


def changes(w):
    results=w/"results"
    joint,signature=signed(results/"map-study/joint-freeze.json")
    primary=joint["primary"]
    compare=primary if primary in METHODS else max(METHODS,key=lambda n:sum(s["mAP"] for s in joint["families"][n]["scores"]))
    sig=joint["families"][compare]["name"]
    report={"primary":primary,"examined_new_method":compare,"joint_lock":signature,"splits":{}}
    for split in (*SKETCHY_SPLITS,"tuberlin"):
        stage="heldout" if split=="tuberlin" else "sketchy"
        loc=Path(read(w/("tuberlin-data.json" if split=="tuberlin" else "ready.json"))["root"])
        classes=UNSEEN_CLASSES[split]
        qp,ql=collect_paths(loc,classes,"sketch")
        pp,pl=collect_paths(loc,classes,"photo")
        def scores(family,id_):
            return tensor(results/family/stage/"per-query"/f"{split}-{id_}.pt")[:,0]
        base=scores(OLD,joint["families"][OLD]["name"])
        result=scores(compare,sig)
        uniform=scores("shared-concept-likelihood","uniform")
        cosine=scores("conditional-semantic-index","native")
        top=tensor(results/compare/stage/"top10"/f"{split}-{sig}.pt").long()
        if len(base)!=len(qp) or top.shape!=(len(qp),10):
            raise ValueError("Wrong query or rank count")
        delta=(result-base)*100
        class_gain=[{"class":c,"queries":int((ql==i).sum()),"mAP_gain_pp":float(delta[ql==i].mean())}
                    for i,c in enumerate(classes)]
        cases=[]
        first_correct=pl[top[:,0]]==ql
        for reason,mask in (("improved_AP",delta>10),("lost_AP",delta< -10)):
            indices=mask.nonzero().flatten()
            if len(indices):
                order=indices[delta[indices].argsort()]
                for i in order[torch.linspace(.25,.75,min(3,len(order))).mul(len(order)-1).long()].unique().tolist():
                    cases.append({"event":reason,"query":short(qp[i]),"query_category":classes[int(ql[i])],
                                  "top1_photo":short(pp[int(top[i,0])]),"top1_relevant":bool(first_correct[i]),
                                  "native_AP":float(cosine[i]),"uniform_AP":float(uniform[i]),
                                  "incumbent_AP":float(base[i]),"new_AP":float(result[i]),
                                  "relevant_in_top10":int((pl[top[i]]==ql[i]).sum())})
        cohorts=[]
        for lo,hi in ((0,.25),(.25,.5),(.5,.75),(.75,1.000001)):
            mask=(base>=lo)&(base<hi)
            cohorts.append({"original_AP_interval":[lo,min(hi,1.)],"count":int(mask.sum()),
                            "new_mAP_gain_pp":float(delta[mask].mean()) if mask.any() else None})
        report["splits"][split]={"examples":cases,"per_class_gain":class_gain,"by_original_AP":cohorts}
    atomic_json(results/"map-followup/example-queries.json",report)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    changes(p.parse_args().workspace)
