"""Read pre-annotated attribute choices in isolation and in shared context, with frozen CLIP."""

import argparse
import csv
import json

import numpy as np
import open_clip
import torch
from torch.nn.functional import normalize

from .concept_benchmark import atomic_json, checksum
from .crossdomain_inputs import CHECKPOINT
from .encoding import load_frozen_clip, save_tensor
from .kaggle_unseen_features import DEFAULT_CACHE, load_split
from .native_block_probe import SEED, deterministic
from .visible_attribute_panels import HERE, OUT


NEUTRAL = (
    "This is an image.",
    "An object appears in an image.",
    "The object is visible in this particular image.",
    "Here is an image that shows an object in view.",
    "This is a picture in which an object can be seen in the frame.",
)
CONDITIONS = ("alone","context_before","context_after","neutral_before","neutral_after")
ENDPOINTS = ("auc","balanced_sentence_accuracy","both_correct")


def read_tsv(name):
    with (HERE/"docs"/name).open(encoding="utf-8",newline="") as f:
        return {r["class"]:r for r in csv.DictReader(f,delimiter="\t")}


def prompts(row,tokenizer):
    statement=row["context"]
    def length(s):
        return int(tokenizer([s]).ne(0).sum())-2
    neutral=min(NEUTRAL,key=lambda s:abs(length(s)-length(statement)))
    text={}
    for k in CONDITIONS:
        if k=="alone":
            text[k]=(row["A"],row["B"])
        else:
            filler=neutral if k.startswith("neutral") else statement
            text[k]=tuple(f"{filler} {p}" if k.endswith("before") else f"{p} {filler}" for p in (row["A"],row["B"]))
    return text,{"context":statement,"neutral":neutral,"context_tokens":length(statement),"neutral_tokens":length(neutral)}


def encode_text(model,tokenizer,prompts_for_class):
    sents=[s for k in CONDITIONS for s in prompts_for_class[k]]
    tokens=tokenizer(sents).cuda()
    if (tokens.ne(0).sum(1)>=tokens.shape[1]).any():
        raise ValueError("A caption would be truncated")
    return normalize(model.encode_text(tokens).float(),dim=1).double().reshape(len(CONDITIONS),2,-1)


def half_credit(x):
    return (x>0).double()+.5*(x==0).double()


def discriminate(scores,labels):
    a,b=scores[labels==1],scores[labels==-1]
    if not len(a) or not len(b):
        return None
    auc=half_credit(a[:,None]-b[None,:]).mean().item()
    yes,no=half_credit(a).mean().item(),half_credit(-b).mean().item()
    return dict(zip(ENDPOINTS,(auc,(yes+no)/2,yes*no),strict=True))


def retrieve(choice,target,labels):
    # Same-class, opposite-state photos. Require another positive source on both sides.
    if min((labels==s).sum().item() for s in (1,-1))<2:
        return None
    actual,oracle=[],[]
    for state in (1,-1):
        observed_for_state,oracle_for_state=[],[]
        for i in torch.where(labels==state)[0].tolist():
            pos=labels==state
            pos[i]=False
            neg=labels==-state
            correct=half_credit(target[pos,None,:]-target[None,neg,:]).mean((0,1))
            q=half_credit(choice[i])
            observed_for_state.append((q*correct[0]+(1-q)*correct[1]).item())
            oracle_for_state.append(correct[0 if state==1 else 1].item())
        actual.append(np.mean(observed_for_state))
        oracle.append(np.mean(oracle_for_state))
    return {"nominated_text":float(np.mean(actual)),"known_state_text":float(np.mean(oracle))}


def class_estimate(values):
    x=np.asarray(values,dtype=np.float64)
    if not len(x): return None
    draws=np.random.default_rng(SEED).integers(len(x),size=(10000,len(x)))
    return {"classes":len(x),"mean_percent":float(100*x.mean()),
            "class_bootstrap95":(100*np.quantile(x[draws].mean(1),(.025,.975))).tolist()}


def summarize(rows):
    report={}
    for population,views,metrics in (
        ("shared",("sketch","sketch_two_views","photo"),ENDPOINTS),
        ("own",("sketch","photo"),ENDPOINTS),
        ("retrieval",("sketch","sketch_two_views"),("nominated_text","known_state_text")),
    ):
        results={}
        for view in views:
            usable=[r for r in rows if r[population][view]["alone"] is not None]
            by_metric={}
            for metric in metrics:
                vals={c:np.array([r[population][view][c][metric] for r in usable]) for c in CONDITIONS}
                effects={k:class_estimate(v) for k,v in vals.items()}
                true=(vals["context_before"]+vals["context_after"])/2
                neutral=(vals["neutral_before"]+vals["neutral_after"])/2
                for k,v in (("context_minus_alone",true-vals["alone"]),
                            ("neutral_minus_alone",neutral-vals["alone"]),("context_minus_neutral",true-neutral)):
                    effects[k]=class_estimate(v)
                by_metric[metric]=effects
            results[view]={"eligible_classes":[r["class"] for r in usable],"metrics":by_metric}
        report[population]=results
    return report


def class_rows(row,mapping,labels_by_mode,split,features,text,notes):
    cls=row["class"]
    subject=row["confirm"]
    labels={}
    ids={}
    for mode in ("sketch","photo"):
        seq=[r[mode] for r in subject]
        y=labels_by_mode[mode][cls]["states_in_display_order"].replace(" ","")
        if len(y)!=len(seq) or not set(y)<=set("ABU"): raise ValueError("Bad annotation")
        byid=dict(zip(mapping[mode],y,strict=True))
        labels[mode]=torch.tensor([{"A":1,"B":-1,"U":0}[byid[i]] for i in seq],dtype=torch.int8)
        ids[mode]=seq
    match=(labels["sketch"]==labels["photo"]) & (labels["sketch"]!=0)
    signed=torch.where(match,labels["sketch"],0)
    scores={k:(x.double() @ text.flatten(0,1).T).reshape(len(subject),len(CONDITIONS),2).cpu() for k,x in features.items()}
    output={"class":cls,"split":split,"subjects":len(subject),"prompt_control":notes,
            "sketch_definite":int((labels["sketch"]!=0).sum()),"photo_definite":int((labels["photo"]!=0).sum()),
            "joint_states":[int((signed==v).sum()) for v in (1,-1)],
            "conflicting":int((labels["sketch"]*labels["photo"]==-1).sum()),
            "shared":{},"own":{},"retrieval":{}}
    for view,s in scores.items():
        margin=s[:,:,0]-s[:,:,1]
        output["shared"][view]={c:discriminate(margin[:,i],signed) for i,c in enumerate(CONDITIONS)}
        if view in labels:
            output["own"][view]={c:discriminate(margin[:,i],labels[view]) for i,c in enumerate(CONDITIONS)}
        if view!="photo":
            output["retrieval"][view]={c:retrieve(margin[:,i],scores["photo"][:,i,:],signed) for i,c in enumerate(CONDITIONS)}
    dirs=normalize(text[:,0]-text[:,1],dim=-1)
    output["cosine_with_isolated_text_contrast"]={c:float((dirs[i]*dirs[0]).sum()) for i,c in enumerate(CONDITIONS)}
    raw={"ids":ids,"labels":labels,"scores":scores,"texts":text.cpu()}
    return output,raw


@torch.inference_mode()
def score(revision,annotation_revision):
    manifest=json.loads((DEFAULT_CACHE/"manifest.json").read_bytes())
    if manifest["sha256"]!=CHECKPOINT: raise ValueError("Unexpected checkpoint")
    board=OUT/"boards/confirm"
    m=json.loads((board/"manifest.json").read_bytes())
    if checksum(board/"mapping.json")!=m["mapping_sha256"]: raise ValueError("Display order not verified")
    lookup={r["class"]:r for r in json.loads((board/"mapping.json").read_bytes())}
    panel=json.loads((OUT/"subject-ids.json").read_bytes())
    prop=read_tsv("visible-attribute-predicates.tsv")
    labels={k:read_tsv(f"visible-attribute-{k}-labels.tsv") for k in ("photo","sketch")}
    if any(set(d)!=set(lookup) for d in (prop,*labels.values())):raise ValueError("Incomplete confirmation judgments")
    model=load_frozen_clip(manifest["checkpoint"],torch.device("cuda"))
    tok=open_clip.get_tokenizer("ViT-B-32")
    rows,raw=[],{}
    for split in ("sketchy_1","sketchy_2"):
        ds,_,_=load_split(DEFAULT_CACHE,split)
        qi={i:j for j,i in enumerate(ds.query_ids.tolist())}
        pi={i:j for j,i in enumerate(ds.gallery_ids.tolist())}
        for row in (r for r in panel if r["split"]==split):
            cls=row["class"]
            captions,notes=prompts(prop[cls],tok)
            text=encode_text(model,tok,captions)
            qs=[qi[r["sketch"]] for r in row["confirm"]]
            ps=[pi[r["photo"]] for r in row["confirm"]]
            x=normalize(ds.sketch[qs].float(),dim=1)
            features={"sketch":x,"sketch_two_views":normalize(x+normalize(ds.mirror[qs].float(),dim=1),dim=1),
                      "photo":normalize(ds.photo[ps].float(),dim=1)}
            out,stored=class_rows(row,lookup[cls],labels,split,features,text,notes)
            rows.append(out)
            raw[cls]={**stored,"captions":captions}
    dest=OUT/"scores.pt"
    save_tensor(raw,dest)
    atomic_json(OUT/"summary.json",{"revision":revision,"annotation_revision":annotation_revision,"seed":SEED,
                "checkpoint":CHECKPOINT,"classes":len(rows),"subjects":sum(v["subjects"] for v in rows),
                "device":torch.cuda.get_device_name(),"summary":summarize(rows),"class_rows":rows,
                "semantics":{"A":1,"B":-1,"unclear":0},"conditions":CONDITIONS,"raw_sha256":checksum(dest),
                "map_sha256":m["mapping_sha256"],"subjects_sha256":checksum(OUT/"subject-ids.json"),
                "label_sha256":{s:checksum(HERE/"docs"/f"visible-attribute-{s}-labels.tsv") for s in labels}})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision",required=True)
    parser.add_argument("--annotation-revision",required=True)
    args=parser.parse_args()
    deterministic()
    score(args.revision,args.annotation_revision)
