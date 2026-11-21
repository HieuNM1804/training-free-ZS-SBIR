"""Choose a single method and settings by mean official Sketchy mAP before scoring TU."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path

import torch

from .capture_query_metrics import evaluate_with_details
from .concept_benchmark import atomic_json,checksum,read
from .concept_likelihood import ranked_scores
from .conditional_semantic_index import SIZES,bare_words
from .crossdomain_benchmark import SPLIT_HASHES,digest,revision
from .crossdomain_inputs import CHECKPOINT
from .data import SKETCHY_SPLITS,UNSEEN_CLASSES
from .encoding import save_tensor
from .foundation import load_words
from .map_diagnosis import by_rank,map_values
from .retrieval_diagnosis import global_inputs,stratified_sample
from .retrieval_evidence import load_kd,paired_delta,tensor
from .semantic import read_words
from .shared_concept_likelihood import FAMILY,PRIMARY,emission,target_inputs
from .verify_locked_ranks import replay


def config_id(config):
    return digest(asdict(config))[:16]


def score_key(rows):
    return tuple(sum(r[x] for r in rows)/len(rows) for x in ("mAP","precision"))


def settings_from_pilot(module,folder):
    records=[read(folder/f"pilot-{s}.json")["settings"] for s in SKETCHY_SPLITS]
    candidates={config_id(c):c for c in module.candidates()}
    if any(t.keys()!=candidates.keys() for t in records):
        raise ValueError("Incomplete mAP pilot")
    keys=sorted(candidates,key=lambda cid:(-score_key([t[cid] for t in records])[0],
        -score_key([t[cid] for t in records])[1],cid))
    return list(dict.fromkeys([*(candidates[k] for k in keys[:5]),module.canonical()]))


def selected(module,folder):
    choices=settings_from_pilot(module,folder)
    return max(choices,key=lambda c:score_key([read(folder/"sketchy"/f"{s}-{config_id(c)}.json") for s in SKETCHY_SPLITS]))


def make_lock(module,folder):
    file=folder/"freeze.json"
    if file.exists():
        x=read(file); signature=x.pop("sha256")
        if digest(x)!=signature or x["revision"]!=revision():
            raise ValueError("Invalid mAP lock")
        return x|{"sha256":signature}
    if any((folder/"heldout").glob("tuberlin-*.json")):
        raise ValueError("TU scored before selection")
    c=selected(module,folder); cs=module.controls(c)
    sources={}
    for s in SKETCHY_SPLITS:
        for v in set(settings_from_pilot(module,folder))|set(cs.values()):
            p=folder/"sketchy"/f"{s}-{config_id(v)}.json"; row=read(p)
            if row["revision"]!=revision() or row["parameters"]!=asdict(v) or row["dataset_sha256"]!=SPLIT_HASHES[s]:
                raise ValueError("Wrong source configuration")
            sources[p.name]=checksum(p)
    record={"utc":datetime.now(timezone.utc).isoformat(),"revision":revision(),"criterion":"equal mean Sketchy mAP, then precision",
            "canonical":asdict(module.canonical()),"winner":asdict(c),"winner_id":config_id(c),
            "controls":{name:asdict(v) for name,v in cs.items()},"sources":sources,"tu_rows_before_freeze":0}
    atomic_json(file,record|{"sha256":digest(record)})
    return read(file)


def joint(w,studies):
    root=w/"results"; file=root/"map-study/joint-freeze.json"
    if file.exists():
        old=read(file);h=old.pop("sha256")
        if digest(old)!=h or set(old["families"])!=set(studies)|{FAMILY}:
            raise ValueError("Conflicting joint decision")
        return old["primary"]
    entries={}
    for name in [FAMILY,*studies]:
        folder=root/name;historical=name==FAMILY
        if historical:
            historical_protocol=read(folder/"protocol.json")
            sig=historical_protocol["sha256"]; chosen=PRIMARY
        else:
            locked=read(folder/"freeze.json");sig=locked.pop("sha256")
            if digest(locked)!=sig or any((folder/"heldout").glob("tuberlin-*.json")):
                raise ValueError("Invalid prospective lock")
            chosen=locked["winner_id"]
        rows=[]
        for s in SKETCHY_SPLITS:
            p=folder/"sketchy"/f"{s}-{chosen}.json";r=read(p)
            if not historical and checksum(p)!=locked["sources"][p.name]:
                raise ValueError("Changed Sketchy score")
            rows.append({k:r[k] for k in ("mAP","precision")}|{"split":s,"sha256":checksum(p)})
        entries[name]={"scores":rows,"name":chosen,"freeze":sig,"historical":historical,
                       "tu_rows_prelock":None if historical else 0}
    primary=max(entries,key=lambda n:score_key(entries[n]["scores"]))
    record={"utc":datetime.now(timezone.utc).isoformat(),"families":entries,"primary":primary,
            "selection":"equal mean mAP@all Ext-1 and mAP@200 Ext-2, then mean P","selection_data":list(SKETCHY_SPLITS)}
    atomic_json(file,record|{"sha256":digest(record)})
    return primary


class Data:
    def __init__(self,q,p,source,target,scale):
        self.q,self.p,self.source,self.target,self.scale=q,p,source,target,scale
        self.a=target_inputs(p,target,scale)
        self.a/=self.a.sum(1,keepdim=True)
        self.query_emission=emission(q,source,scale)
        self.base=self.query_emission@self.a.T


def evaluate(a,path,split,scores,ql,pl,c,meta,perform_replay,kd):
    cid=config_id(c);file=path/f"{split}-{cid}.json"
    pt=path/"per-query"/f"{split}-{cid}.pt"
    if file.exists() and pt.exists():
        r=read(file)
        if r["parameters"]!=asdict(c) or any(r[k]!=meta[k] for k in meta):
            raise ValueError("Stale evaluation")
        if not perform_replay or "kd_check" in r:
            return
    if scores.shape!=(len(ql),len(pl)) or not torch.isfinite(scores).all():
        raise ValueError("Invalid score matrix")
    def rows():
        for b in scores.split(96):
            yield from ranked_scores(b).cpu()
    nworkers=5 if split!="sketchy_2" else 3
    (ap,prec,mk,pk),details=evaluate_with_details(rows(),ql,pl,split,f"{split} {cid}",nworkers)
    row=meta|{"name":cid,"parameters":asdict(c),"mAP":float(ap),"precision":float(prec),"map_k":mk,"p_k":pk}
    if perform_replay:
        with ThreadPoolExecutor(max_workers=6) as pool:
            kdcheck,top=replay(kd,pool,scores.split(128),ql,pl,details,split)
        if max(kdcheck["kd_minus_pipeline_max_abs_same_order_AP_P"])!=0:
            raise ValueError("Independent evaluator disagrees")
        row["kd_check"]=kdcheck
    else:
        top=torch.cat([b.argsort(dim=1,descending=True,stable=True)[:,:10].cpu() for b in scores.split(100)])
    save_tensor(details,pt); save_tensor(top.int(),path/"top10"/f"{split}-{cid}.pt")
    atomic_json(file,row)


@torch.inference_mode()
def run(workspace,name,stage):
    module=import_module("tf_sbir."+name.replace("-","_"))
    root=workspace/"results"; folder=root/name
    if stage=="freeze":
        make_lock(module,folder)
        return
    held=stage=="heldout";pilot=stage=="pilot";locked=None
    if held:
        locked=make_lock(module,folder);dec=read(root/"map-study/joint-freeze.json")
        sig=dec.pop("sha256")
        if digest(dec)!=sig or dec["families"][name]["freeze"]!=locked["sha256"]:
            raise ValueError("Missing pre-TU decision")
        configs=[module.Config(**p) for p in locked["controls"].values()]
    elif pilot:
        configs=module.candidates()
    elif stage=="sketchy":
        configs=settings_from_pilot(module,folder)
    else:
        configs=list(module.controls(selected(module,folder)).values())
    a,_,queries,_,galleries,raw,rgb=global_inputs(workspace,"tuberlin" if held else "sketchy")
    words=read_words(a.vocabulary);_,scale=bare_words(a,words)
    src,tgt=(x.cuda() for x in load_words(a,words)); kd,kd_sha=load_kd(workspace)
    for s,(qi,ql) in queries.items():
        pi,pl=galleries[s]
        if (len(qi),len(pi),len(ql.unique()))!=SIZES[s] or len(pl.unique())!=SIZES[s][2] or digest(UNSEEN_CLASSES[s])!=SPLIT_HASHES[s]:
            raise ValueError("Invalid data protocol")
        if pilot:
            ids=stratified_sample(ql,48,4783);qi,ql=qi[ids],ql[ids]
        data=Data(raw[qi].cuda(),rgb[pi].cuda(),src,tgt,scale)
        module.prepare_data(data,a,words)
        index=module.Index(data)
        out=folder/("heldout" if held else "sketchy")
        meta={"revision":revision(),"family":name,"queries":len(ql),"gallery":len(pl),"classes":len(pl.unique()),
              "checkpoint_sha256":CHECKPOINT,"dataset_sha256":SPLIT_HASHES[s],"kd_sha256":kd_sha,
              "gpu":torch.cuda.get_device_name(),"external_vocabulary":len(words),"exploratory_unseen":not held,
              "freeze_sha256":locked["sha256"] if held else None}
        report={}; cs=list(dict.fromkeys(configs))
        winner_id=locked["winner_id"] if held else (config_id(selected(module,folder)) if stage=="controls" else None)
        if not pilot:
            save_tensor({"query":ql,"gallery":pl},out/f"{s}-labels.pt")
        for cfg in sorted(cs,key=module.cache_key):
            cid=config_id(cfg);replay_needed=cid==winner_id and not pilot
            file=out/f"{s}-{cid}.json"
            if not pilot and file.exists() and (out/"per-query"/f"{s}-{cid}.pt").is_file():
                r=read(file)
                if r["parameters"]!=asdict(cfg) or any(r[k]!=meta[k] for k in meta):
                    raise ValueError("Stale record")
                if not replay_needed or "kd_check" in r:
                    continue
            scores=index.score(cfg)
            if pilot:
                if not torch.isfinite(scores).all():
                    raise ValueError("Nonfinite pilot ranking")
                vals=torch.cat([map_values(block,ql[i:i+len(block)].cuda(),pl.cuda(),s)[0]
                                for i,block in zip(range(0,len(ql),96),scores.split(96),strict=True)])
                report[cid]={"parameters":asdict(cfg),"mAP":float(vals[:,0].mean()),"precision":float(vals[:,1].mean())}
            else:
                evaluate(a,out,s,scores,ql,pl,cfg,meta,replay_needed,kd)
            del scores
        if pilot:
            atomic_json(folder/f"pilot-{s}.json",meta|{"settings":report,"seed":4783,"diagnostic":index.info})
            print(s,sorted([(r["mAP"],r["parameters"]) for r in report.values()],key=lambda x:-x[0])[:2],flush=True)
        else:
            atomic_json(out/f"index-{stage}-{s}.json",meta|{"diagnostic":index.info})
        del index,data


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--family",nargs="+",required=True)
    p.add_argument("--stage",choices=("pilot","sketchy","controls","freeze","joint","heldout"),required=True)
    o=p.parse_args()
    if o.stage=="joint":
        print(joint(o.workspace,o.family))
    else:
        if len(o.family)!=1: p.error("Exactly one family is required")
        run(o.workspace,o.family[0],o.stage)
