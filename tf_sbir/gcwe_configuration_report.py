"""Verify full-query ranks and test whether geometric evidence predicts words of relevant photographs."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .concept_benchmark import atomic_json
from .crossdomain_benchmark import revision
from .depiction_path_diagnostic import class_interval
from .encoding import save_tensor
from .gallery_panel_resume import semantic_inputs
from .gcwe_semantic_configuration import FAMILY, distance_bins, laws, predictions
from .geometry_word_flat import KD_FUNCTION_SHA, make_word_bank


def cached(paths,folder):
    stamp=hashlib.sha256(("CLIP-B32-native-config-v1\n"+"\n".join(map(str,paths))).encode()).hexdigest()
    return torch.load(folder/(stamp+".pt"),weights_only=True,mmap=True)


def verify_rank_rows(folder,split,ids,labels,pi,stages):
    records=[]
    for stage in stages:
        stem=split+"-"+stage
        rec=torch.load(folder/(stem+".pt"),weights_only=True)
        summary=json.loads((folder/(stem+".json")).read_bytes())
        assert summary["evaluator"]==KD_FUNCTION_SHA and summary["seed"]==42 and torch.equal(rec["pi"],pi)
        for k,v in rec["rows"].items():
            torch.testing.assert_close(100*v.double().mean(0),torch.tensor(summary["metrics"][k],dtype=torch.double),atol=1e-8,rtol=0)
        records.append(rec)
    observed=torch.cat([r["qi"] for r in records])
    assert len(observed)==len(observed.unique())==len(ids)
    positions={v:i for i,v in enumerate(observed.tolist())}
    order=torch.tensor([positions[v] for v in ids.tolist()])
    assert torch.equal(torch.cat([r["labels"] for r in records])[order],labels)
    common=set.intersection(*(set(r["rows"]) for r in records))
    rows={k:torch.cat([r["rows"][k] for r in records])[order].double() for k in sorted(common)}
    result={}
    for k,v in rows.items():
        baseline="rgb_word" if k.startswith("rgb_") else "word"
        diff=100*(v-rows[baseline])
        result[k]=dict(metrics=(100*v.mean(0)).tolist(),delta=diff.mean(0).tolist(),class_ap=class_interval(diff[:,0],labels,seed=42))
    return result


def semantic_target(bank,held,labels):
    gl=labels[held.cpu()]
    classes=labels.unique(sorted=True)
    assert torch.equal(gl.unique(sorted=True),classes)
    means=torch.stack([bank[held[(gl==c).cuda()]].mean(0) for c in classes])
    other=(means.sum(0,keepdim=True)-means)/(len(means)-1)
    torch.testing.assert_close((means-other).mean(0),torch.zeros_like(means[0]),atol=1e-7,rtol=0)
    return means,other


@torch.inference_mode()
def run(dataset):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    w=Path("/content/sbir")
    folder=w/"results"/FAMILY
    cache=w/"cache"/FAMILY
    _,qp,qs,pp,gs,qf,pf,src,tgt,scale=semantic_inputs(w,dataset)
    _,background=distance_bins()
    ph=cached(pp,cache)
    stages=("pilot","confirmation") if dataset=="sketchy" else ("transfer",)
    stage_hists=[]
    for s in stages:
        partial=[torch.load(folder/(split+"-"+s+".pt"),weights_only=True,mmap=True)["qi"] for split in qs]
        qids=torch.cat(partial).unique()
        stage_hists.append((qids,cached([qp[i] for i in qids.tolist()],cache)))
    # Split overlap can place an ID in both stage caches. Check that its observed events agree.
    by_id={}
    for qids,hist in stage_hists:
        for q,obs in zip(qids.tolist(),hist):
            if q in by_id:
                assert torch.equal(by_id[q],obs)
            by_id[q]=obs
    for split,(ids,labels) in qs.items():
        pi,gl=gs[split]
        ranked=verify_rank_rows(folder,split,ids,labels,pi,stages)
        qhist=torch.stack([by_id[q] for q in ids.tolist()])
        bank=make_word_bank(F.normalize(pf[pi].cuda().float(),dim=1),tgt,scale,192)
        folds=[v.cuda() for v in torch.randperm(len(pi),generator=torch.Generator().manual_seed(42)).tensor_split(2)]
        hist=ph[pi]
        values={}
        for ref,held in zip(reversed(folds),folds):
            j,ind=laws(bank,hist,ref,background)
            fake=laws(bank,hist,ref,background,shuffle=True)[0]
            means,other=semantic_target(bank,held,gl)
            chunks={}
            for b in torch.arange(len(ids)).split(96):
                prior=(scale*(F.normalize(qf[ids[b]].cuda().float(),dim=-1)@src.T)).softmax(1)
                obs=qhist[b].cuda().float()/(49*24)
                original=prior.clamp_min(1e-30).log()
                for name,law in (("joint",j),("independent",ind),("shuffled",fake)):
                    for rule,post in predictions(prior,obs,law,(16,)).items():
                        diff=post.clamp_min(1e-30).log()-original
                        y=labels[b].cuda()
                        correct=(means[y]*diff).sum(1)
                        unrelated=(other[y]*diff).sum(1)
                        chunks.setdefault(name+"_"+rule,[]).append(torch.stack((correct,unrelated,correct-unrelated),1).cpu())
            for name,parts in chunks.items():
                values.setdefault(name,[]).append(torch.cat(parts))
        all_values={k:torch.stack(v).mean(0).double() for k,v in values.items()}
        stats={}
        for k,v in all_values.items():
            stats[k]=dict(nats=v.mean(0).tolist(),class_correct=class_interval(v[:,0],labels,seed=42),
                          class_discrimination=class_interval(v[:,2],labels,seed=42))
        effects={}
        for rule in ("one","count16"):
            for ctrl in ("independent","shuffled"):
                delta=all_values["joint_"+rule]-all_values[ctrl+"_"+rule]
                effects[rule+" minus "+ctrl]=dict(nats=delta.mean(0).tolist(),class_discrimination=class_interval(delta[:,2],labels,seed=42))
        save_tensor(dict(qi=ids,pi=pi,labels=labels,gain=all_values),folder/(split+"-word-target.pt"))
        atomic_json(folder/(split+"-full.json"),dict(commit=revision(),seed=42,gpu=torch.cuda.get_device_name(),
                    queries=len(ids),gallery=len(pi),classes=int(labels.unique().numel()),metrics=ranked,
                    word_gain=stats,contrasts=effects,units=["relevant","unrelated","relevant_minus_unrelated"]))
        print(split,{k:v["nats"] for k,v in stats.items()},flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("dataset",choices=("sketchy","tuberlin"))
    run(p.parse_args().dataset)
