"""Test the same visible distinctions against actual positive/negative category errors."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json

import torch
from torch.nn.functional import normalize

from .branch_view_study import PrefixReplay
from .concept_benchmark import atomic_json,checksum
from .conditional_completion_study import evaluate
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .kaggle_unseen_features import DEFAULT_CACHE,load_split
from .named_word_axis_audit import OUT as AXES,logits,text_readouts
from .native_block_probe import deterministic
from .replicated_simplex_study import evaluator
from .semantic_boxable import DEFAULT_PROMPTS
from .visible_attribute_panels import OUT
from .visible_attribute_readout import class_estimate,half_credit


def report_group(classes,mask,outcome,metrics,original,observed,selected):
    group=classes[mask].unique()
    index=torch.arange(len(observed))
    by_arm={}
    for i,reader in enumerate(("single","contrast")):
        pair=outcome[:,i]
        scores=metrics[:,i]
        real=pair[index,observed]
        actual_cl=query_balanced(classes,group,mask,real)
        shuffled_q=torch.zeros_like(real)
        shuffled_m=torch.zeros_like(scores[:,0])
        for c in group.tolist():
            ix=torch.where(mask & (classes==c))[0]
            n_a=(observed[ix]==0).double().mean()
            shuffled_q[ix]=n_a*pair[ix,0]+(1-n_a)*pair[ix,1]
            # Exact expectation under a uniform permutation. AP/P use the same class relevance mask.
            shuffled_m[ix]=n_a*scores[ix,0]+(1-n_a)*scores[ix,1]
            if (scores[ix,0]-scores[ix[0],0]).abs().max()>2e-6 or (scores[ix,1]-scores[ix[0],1]).abs().max()>2e-6:
                raise ValueError("Identical text ranking and class relevance gave unequal official metrics")
            # Explicit nontrivial permutation verifies the invariance for complete rankings.
            assigned=observed[ix].roll(1)
            check=scores[ix,assigned]
            if (check.mean(0)-scores[ix,observed[ix]].mean(0)).abs().max()>2e-6:
                raise ValueError("Within-class permutation should preserve aggregated retrieval metrics")
        actual_m=scores[index,observed]
        if (shuffled_m[mask].mean(0)-actual_m[mask].mean(0)).abs().max()>2e-6:
            raise ValueError("Expected within-class permutation failed to preserve AP/P")
        estimated=pair[index,selected]
        pairs={"observed":real,"selected":estimated,"within_class_shuffle":shuffled_q}
        counts={k:{"accuracy_percent":(100*v[mask].mean()).item(),
                   "repairs_relative_to_semantic":v[mask & ~original].sum().item(),
                   "damage_relative_to_semantic":(1-v[mask & original]).sum().item()} for k,v in pairs.items()}
        delta=real-shuffled_q
        by_arm[reader]={"observed_all_gallery_mAP_P_percent":(100*actual_m[mask].mean(0)).tolist(),
                        "selected_all_gallery_mAP_P_percent":(100*scores[index,selected][mask].mean(0)).tolist(),
                        "pair":counts,"observed_state_macro":class_estimate(actual_cl),
                        "observed_minus_shuffle_macro":class_estimate(query_balanced(classes,group,mask,delta)),
                        "observed_minus_shuffle_micro_points":(100*delta[mask].mean()).item()}
    return {"sketches":int(mask.sum()),"classes":len(group),"semantic_correct":int(original[mask].sum()),
            "semantic_wrong":int((~original[mask]).sum()),"arms":by_arm}


def query_balanced(classes,group,mask,vec):
    return [vec[mask & (classes==c)].mean().item() for c in group.tolist()]


@torch.inference_mode()
def run(revision):
    data=torch.load(OUT/"scores.pt",mmap=True,weights_only=True)
    core=json.loads((OUT/"summary.json").read_bytes())
    if checksum(OUT/"scores.pt")!=core["raw_sha256"]:raise ValueError("Invalid diagnosis data")
    eligible=set(core["summary"]["shared"]["sketch"]["eligible_classes"])
    aggr={k:[] for k in ("class","original","observed","selected","known","both","primary","outcome","metrics","semantic")}
    ids_out=[]
    for split_name in ("sketchy_1","sketchy_2"):
        ds,txt,_=load_split(DEFAULT_CACHE,split_name)
        photos=normalize(ds.photo.float(),dim=1)
        qids={i:j for j,i in enumerate(ds.query_ids.tolist())}
        baseline_meta=json.loads((AXES/f"{split_name}-core.json").read_bytes())
        baseline_path=AXES/f"{split_name}-core.pt"
        if checksum(baseline_path)!=baseline_meta["tensor_sha256"]:raise ValueError("Unverified absolute baseline")
        baseline=torch.load(baseline_path,mmap=True,weights_only=True)
        qorder,truth,obs,chosen,joint,prim,clsids=[],[],[],[],[],[],[]
        sentence=[]
        for cls,row in data.items():
            if not any(a["class"]==cls and a["split"]==split_name for a in core["class_rows"]):continue
            which=[qids[x] for x in row["ids"]["sketch"]]
            clsid=sorted(data).index(cls)
            qorder.extend(which)
            labels=row["labels"]["sketch"]
            truth.extend(labels.tolist())
            obs.extend((labels<0).long().tolist())
            chosen.extend((row["scores"]["sketch"][:,0,0]<row["scores"]["sketch"][:,0,1]).long().tolist())
            same=(labels!=0) & (labels==row["labels"]["photo"])
            joint.extend(same.tolist());prim.extend((same & (cls in eligible)).tolist())
            clsids.extend([clsid]*len(which))
            sentence.extend([row["texts"][0].float()]*len(which))
            if not all(UNSEEN_CLASSES[split_name][v]==cls for v in ds.sketch_labels[which].tolist()):raise ValueError("Query/class mismatch")
        qi=torch.tensor(qorder)
        sample=replace(ds,sketch=ds.sketch[qi],mirror=ds.mirror[qi],query_ids=ds.query_ids[qi],sketch_labels=ds.sketch_labels[qi])
        zq,zp=logits(sample,photos,*(txt[k] for k in DEFAULT_PROMPTS))
        _,_,q,p=text_readouts(zq,zp)
        s=q @ p.T.contiguous();relevant=sample.sketch_labels.cuda()[:,None]==ds.photo_labels.cuda()[None,:]
        pos=s.masked_fill(~relevant,-torch.inf).argmax(1);neg=s.masked_fill(relevant,-torch.inf).argmax(1)
        right=relevant[torch.arange(len(q),device=q.device),s.argmax(1)].cpu()
        ids_out.extend(zip([split_name]*len(q),sample.query_ids.tolist(),ds.gallery_ids[pos.cpu()].tolist(),
                           ds.gallery_ids[neg.cpu()].tolist(),strict=True))
        vec=torch.stack(sentence).cuda()
        reader=torch.stack((vec,torch.stack((vec[:,0]-vec[:,1],vec[:,1]-vec[:,0]),1)),1)
        pref=half_credit(torch.einsum("nrsd,nd->nrs",reader.double(),(photos[pos]-photos[neg]).double())).cpu()
        kd=evaluator(DEFAULT_CACHE.parent/"real-sketchy-smoke/evaluator.py")
        replay=PrefixReplay(kd,ds.photo_labels,split_name)
        with ThreadPoolExecutor(max_workers=4) as pool:
            metrics=torch.stack([torch.stack([evaluate(sample,reader[:,r,j].contiguous(),photos,replay,pool).double()
                                             for j in (0,1)],1) for r in (0,1)],1)
        values=(torch.tensor(clsids),right,torch.tensor(obs),torch.tensor(chosen),torch.tensor(truth)!=0,torch.tensor(joint),
                torch.tensor(prim),pref,metrics,baseline["rows"]["NN"][qi])
        for v,k in zip(values,aggr,strict=True):aggr[k].append(v)
    x={k:torch.cat(v) for k,v in aggr.items()}
    reports={}
    for k in ("primary","both","known"):
        reports[k]=report_group(x["class"],x[k],x["outcome"],x["metrics"],x["original"],x["observed"],x["selected"])
        reports[k]["semantic_subset_AP_P_mixed_cutoffs"]=(100*x["semantic"][x[k]].mean(0)).tolist()
    split_results={}
    for name in ("sketchy_1","sketchy_2"):
        mask=torch.tensor([r[0]==name for r in ids_out]) & x["primary"]
        split_results[name]=report_group(x["class"],mask,x["outcome"],x["metrics"],x["original"],x["observed"],x["selected"])
        split_results[name]["semantic_subset_AP_P_percent"]=(100*x["semantic"][mask].mean(0)).tolist()
    path=OUT/"category-errors.pt"
    save_tensor({**x,"class_order":sorted(data),"split_query_positive_negative_ids":ids_out},path)
    atomic_json(OUT/"category-errors.json",{"revision":revision,"evaluator":"original KD-SBIR",
                "source_sha256":core["raw_sha256"],"sha256":checksum(path),"groups":reports,"primary_by_split":split_results})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision",required=True)
    args=parser.parse_args()
    deterministic()
    run(args.revision)
