"""Factor a pair decision into within-category photo transfer and a matched depiction change."""
import json
from pathlib import Path

import torch

from .concept_benchmark import atomic_json
from .crossdomain_benchmark import revision
from .depiction_path_diagnostic import class_interval
from .gallery_panel_resume import semantic_inputs
from .semantic_invariant_question import Question
from .semantic_invariant_question_study import ARMS, FAMILY, accuracy, word_rows


READINGS = ("candidate_source","source_source","sketch_source","source_target","sketch_target")


def match_sources(rows, paths, photos, gallery_ids, gallery_labels):
    lookup = {(Path(photos[i]).parent.name,Path(photos[i]).stem):i for i in gallery_ids.tolist()}
    labels = dict(zip(gallery_ids.tolist(),gallery_labels.tolist()))
    by_query = {}
    for r in rows:
        path = Path(paths[r["q"]])
        original = lookup[path.parent.name,path.stem.rsplit("-",1)[0]]
        assert original not in (r["pos"],r["neg"])
        assert labels[original]==labels[r["pos"]]!=labels[r["neg"]]
        by_query[r["q"]] = original
    return by_query


def decisions(rows, sources, query_words, photo_words):
    details=[]
    for r in rows:
        q, pos, neg = r["q"],r["pos"],r["neg"]
        a,b = sorted((pos,neg))
        orient = 1 if a==pos else -1
        readings = (photo_words[pos][0],photo_words[sources[q]][0],query_words[q][0],
                    photo_words[sources[q]][1],query_words[q][1])
        rec = dict(q=q,pos=pos,neg=neg,original=sources[q],cls=r["mode"])
        for arm in (*ARMS,"direct"):
            if arm=="direct":
                raw = [float(z @ (photo_words[pos][1]-photo_words[neg][1])) for z in readings]
                answers = {k:[accuracy(x),x] for k,x in zip(READINGS,raw)}
                assert answers["sketch_source"][0]==r["direct"]
            else:
                test=Question(photo_words[a],photo_words[b],invariant=arm=="invariant")
                results=[test.read(z) for z in readings]
                answers={k:[accuracy(orient*x["signed_certificate"]),orient*x["response"]]
                         for k,x in zip(READINGS,results)}
                assert answers["sketch_source"][0]==r[arm]["acc"]
                drift=abs(answers["sketch_source"][1]-r[arm]["answer"])
                if drift>=1e-6:
                    raise AssertionError(f"Sketch vote replay changed by {drift} for {(q,a,b,arm)}")
            rec[arm]=answers
        details.append(rec)
    return details


def summary(rows,arm):
    labels = torch.tensor([r["cls"] for r in rows])
    measurements = {k:torch.tensor([r[arm][k] for r in rows],dtype=torch.double) for k in READINGS}
    transitions = {}
    for first, second in (("candidate_source","source_source"),("source_source","sketch_source"),
                          ("source_target","sketch_target"),("source_target","source_source"),
                          ("sketch_target","sketch_source")):
        a,b=measurements[first][:,0],measurements[second][:,0]
        diff=measurements[second]-measurements[first]
        transitions[first+" to "+second]=dict(delta=diff.mean(0).tolist(),
            class_accuracy=class_interval(diff[:,0],labels,seed=42),
            class_answer=class_interval(diff[:,1],labels,seed=42),
            transitions=[[int(((a==x)&(b==y)).sum()) for y in (0.,.5,1.)] for x in (0.,.5,1.)])
    return dict(means={k:v.mean(0).tolist() for k,v in measurements.items()},transitions=transitions)


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    work=Path("/content/sbir")
    root=work/"results"/FAMILY
    _,paths,_,photos,galleries,qf,pf,src,tgt,scale=semantic_inputs(work,"sketchy")
    records={}
    original_queries=set()
    original_candidates=set()
    source_ids={}
    for split,(ids,labels) in galleries.items():
        previous=json.loads((root/(split+"-pairs.json")).read_bytes())
        records[split]=previous["boundary"]
        # The pilot GEMMs used the sorted union across both cohorts and both splits.
        # Keeping the last arithmetic batch identical matters for non-unique optimal votes.
        for cohort in previous.values():
            for r in cohort:
                original_queries.add(r["q"])
                original_candidates.update((r["pos"],r["neg"]))
        source_ids[split]=match_sources(records[split],paths,photos,ids,labels)
    all_photos=set(i for lookup in source_ids.values() for i in lookup.values())
    qw=word_rows(qf,original_queries,(src,tgt),scale)
    pw=word_rows(pf,original_candidates,(src,tgt),scale)
    pw.update(word_rows(pf,all_photos-original_candidates,(src,tgt),scale))
    for split,rows in records.items():
        details=decisions(rows,source_ids[split],qw,pw)
        stats={arm:summary(details,arm) for arm in (*ARMS,"direct")}
        atomic_json(root/(split+"-transfer-pairs.json"),details)
        atomic_json(root/(split+"-transfer.json"),dict(commit=revision(),seed=42,gpu=torch.cuda.get_device_name(),
                    decisions=len(rows),sketches=len(source_ids[split]),source_photos=len(set(source_ids[split].values())),
                    measurements=["accuracy","signed_answer"],transition_order=["wrong","tie","correct"],results=stats))
        print(split,stats["invariant"]["means"],flush=True)


if __name__=="__main__":
    run()
