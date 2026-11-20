"""A gallery-independent Hellinger score on frozen CLIP's full posterior response."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path

import torch
from torch.nn import functional as F

from .capture_query_metrics import evaluate_with_details
from .concept_benchmark import atomic_json,checksum,read
from .concept_likelihood import ranked_scores
from .conditional_semantic_index import bare_words,SIZES
from .crossdomain_benchmark import digest,revision
from .crossdomain_inputs import CHECKPOINT
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .foundation import load_words
from .retrieval_diagnosis import global_inputs
from .retrieval_evidence import load_kd,paired_delta,tensor
from .semantic import read_words
from .shared_concept_likelihood import FAMILY
from .verify_locked_ranks import replay


def encode(x,bank,scale):
    return torch.cat([((b@bank.T)*scale).softmax(1) for b in x.split(192)])


@torch.inference_mode()
def run(w,stage):
    root=w/"results/semantic-posterior-geometry"
    proto=root/"protocol.json"
    spec={"revision":revision(),"primary":"hellinger","scale":"native checkpoint","test_data":"Sketchy only for selection",
          "criterion":"P > uniform on both Sketchy splits and higher mean mAP","checkpoint":CHECKPOINT,
          "source_sha256":checksum(Path(__file__).resolve().parents[1]/"docs/POSTERIOR_GEOMETRY_PROTOCOL_VI.md")}
    if proto.exists():
        prior=read(proto); h=prior.pop("sha256")
        if digest(prior)!=h or any(prior[k]!=v for k,v in spec.items()):
            raise ValueError("Different preregistration")
    else:
        spec["utc"]=datetime.now(timezone.utc).isoformat()
        atomic_json(proto,spec|{"sha256":digest(spec)})
    result={}
    if stage=="tuberlin" and not read(root/"sketchy.json")["qualified"]:
        print("Sketchy criterion failed. TU evaluation omitted as prespecified.")
        return
    a,_,qs,_,gs,raw,p=global_inputs(w,stage)
    words=read_words(a.vocabulary);_,scale=bare_words(a,words)
    source,target=(x.cuda() for x in load_words(a,words));kd,kdsha=load_kd(w)
    for s,(qi,ql) in qs.items():
        pi,pl=gs[s]
        if (len(qi),len(pi),len(UNSEEN_CLASSES[s]))!=SIZES[s]:
            raise ValueError("Invalid split")
        photo=encode(p[pi].cuda(),target,scale)
        sketch=encode(raw[qi].cuda(),source,scale)
        sub="heldout" if stage=="tuberlin" else "sketchy"
        baseline=tensor(w/"results"/FAMILY/sub/"per-query"/f"{s}-uniform.pt")
        rows={}
        for name in ("hellinger","cosine_posterior"):
            if name=="hellinger":
                q,bank=sketch.sqrt(),photo.sqrt()
            else:
                q,bank=F.normalize(sketch),F.normalize(photo)
            stored=[]
            def ordered_rows():
                for query in q.split(160):
                    scores=query@bank.T
                    if name=="hellinger":
                        stored.append(scores.cpu())
                    yield from ranked_scores(scores).cpu()
            (ap,prec,*_),detail=evaluate_with_details(ordered_rows(),ql,pl,s,name,5 if s!="sketchy_2" else 3)
            save_tensor(detail,root/"per-query"/f"{s}-{name}.pt")
            rows[name]={"precision":float(prec),"mAP":float(ap),"paired_vs_uniform":paired_delta(detail,baseline,ql)}
            if name=="hellinger":
                with ThreadPoolExecutor(max_workers=5) as pool:
                    check,_=replay(kd,pool,stored,ql,pl,detail,s)
                rows[name]["kd"]=check
                if max(check["kd_minus_pipeline_max_abs_same_order_AP_P"])!=0:
                    raise ValueError("Metric disagreement")
            del q,bank,stored
        rows["uniform"]={"mAP":float(baseline[:,0].mean()),"precision":float(baseline[:,1].mean())}
        result[s]={"queries":len(ql),"gallery":len(pl),"scores":rows}
        print(s,[(k,v["precision"],v["mAP"]) for k,v in rows.items()],flush=True)
        del photo,sketch
    output={"kd_sha256":kdsha,"splits":result}
    if stage=="sketchy":
        pairs=[x["scores"] for x in result.values()]
        output["qualified"]=all(v["hellinger"]["precision"]>v["uniform"]["precision"] for v in pairs) and sum(
            v["hellinger"]["mAP"]-v["uniform"]["mAP"] for v in pairs)>0
        output["utc"]=datetime.now(timezone.utc).isoformat()
        output["tu_rows_prelock"]=0 if not (root/"tuberlin.json").exists() else 1
    atomic_json(root/(stage+".json"),output)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--stage",choices=("sketchy","tuberlin"),required=True)
    a=p.parse_args()
    run(a.workspace,a.stage)
