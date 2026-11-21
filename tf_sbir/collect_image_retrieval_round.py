"""Audit the six retrieval transfers, their controls, and the official per-query metrics on Colab."""

import argparse
import csv
import importlib.util
from pathlib import Path
import sys
import zipfile

import torch
from torch.nn import functional as F

from .concept_benchmark import atomic_json,checksum,read
from .conditional_semantic_index import SIZES,bare_words
from .crossdomain_benchmark import digest,revision
from .crossdomain_inputs import CHECKPOINT
from .data import UNSEEN_CLASSES
from .foundation import load_words
from .image_retrieval_round import DECISION,FAMILIES,HISTORY,verified
from .map_study import Data
from .retrieval_diagnosis import global_inputs,stratified_sample
from .retrieval_evidence import paired_delta,tensor
from .semantic import read_words

EXTRA="retrieval-soft-assignment"
NAMES=[*FAMILIES,EXTRA]
NATIVE="conditional-semantic-index"
UNIFORM="shared-concept-likelihood"
REF="public-semantic-translation"


def source(w,name,metadata):
    sha=metadata["revision"][:12]
    mod=name.replace("-","_")
    file=w/"worktrees"/("map-"+sha)/"tf_sbir"/(mod+".py")
    spec=importlib.util.spec_from_file_location("tf_sbir."+mod,file)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def group_scores(values,labels,mask):
    return {n:{"mAP":float(t[mask,0].mean()),"precision":float(t[mask,1].mean())} for n,t in values.items()}


def example_paths(w,split,ql,pl,qi,pi,qp,pp,metrics,leaves,primary):
    place="heldout" if split=="tuberlin" else "sketchy"
    def top(n):
        return tensor(w/"results"/n/place/"top10"/f"{split}-{leaves[n]}.pt")[:,0].long()
    before,after=top(REF),top(primary)
    mask_before,mask_after=pl[before]==ql,pl[after]==ql
    delta=metrics[primary][:,0]-metrics[REF][:,0]
    selections={"rescued_top1":~mask_before&mask_after,"regressed_top1":mask_before&~mask_after,
                "top1_unchanged_AP_up_025":(before==after)&(delta>.25)}
    paths=lambda p:"/".join(Path(p).parts[-3:])
    result={}
    for tag,mask in selections.items():
        ids=torch.where(mask)[0]
        rng=torch.Generator().manual_seed(97235)
        pick=ids[torch.randperm(len(ids),generator=rng)[:4]]
        result[tag]={"count":int(mask.sum()),"examples":[{"sketch":paths(qp[int(qi[j])]),
            "old_top1":paths(pp[int(pi[before[j]])]),"new_top1":paths(pp[int(pi[after[j]])]),
            "native_AP":float(metrics[NATIVE][j,0]),"old_AP":float(metrics[REF][j,0]),
            "new_AP":float(metrics[primary][j,0])} for j in pick]}
    return result


def load_split(w,s,ql,pl,decision,best_new):
    leaves={n:row["name"] for n,row in decision["families"].items()}|{NATIVE:"native",UNIFORM:"uniform"}
    place="heldout" if s=="tuberlin" else "sketchy"
    values,rows,csvrows={},{},[]
    for n,cid in leaves.items():
        folder=w/"results"/n/place
        path=folder/f"{s}-{cid}.json"
        record=read(path)
        v=tensor(folder/"per-query"/(path.stem+".pt"))
        if (len(ql),len(pl),len(pl.unique()))!=SIZES[s] or v.shape!=(len(ql),2):
            raise ValueError(f"Invalid archive sizes {path}")
        if record["checkpoint_sha256"]!=CHECKPOINT or max(abs(float(v[:,i].mean())-record[k])
                   for i,k in enumerate(("mAP","precision")))>1e-6:
            raise ValueError(f"Invalid checkpoint or aggregate {path}")
        item={k:record[k] for k in ("mAP","precision","map_k","p_k","queries","gallery","classes","revision")}
        item["source_sha256"]=checksum(path)
        if n in decision["families"] and s!="tuberlin":
            expected=next(x["sha256"] for x in decision["families"][n]["scores"] if x["split"]==s)
            if expected!=item["source_sha256"]:
                raise ValueError(f"Sketchy score differs from freeze: {path}")
        if n in NAMES:
            lock=verified(w/"results"/n/"freeze.json")
            if digest(lock)!=decision["families"][n]["freeze"]:
                raise ValueError(f"Configuration identity mismatch {n}")
            if s=="tuberlin" and digest(lock)!=record["freeze_sha256"]:
                raise ValueError(f"Wrong heldout configuration {n}")
            kd=record["kd_check"]
            if kd["queries"]!=len(ql) or max(kd["kd_minus_pipeline_max_abs_same_order_AP_P"])!=0:
                raise ValueError(f"Different official metrics {path}")
            item["KD"]=kd
            controls=dict(lock["controls"])
            controls["registered_canonical"]=lock["canonical"]
            item["controls"]={}
            for label,params in controls.items():
                file=folder/f"{s}-{digest(params)[:16]}.json"
                # Canonical is always selected for full Sketchy. If a family did not designate it
                # as a held-out control, do not fabricate a target result.
                if not file.exists() and label=="registered_canonical" and s=="tuberlin":
                    continue
                c=read(file)
                control=tensor(folder/"per-query"/(file.stem+".pt"))
                if c["parameters"]!=params or not torch.isfinite(control).all():
                    raise ValueError(f"Invalid control {file}")
                comparison=paired_delta(v,control,ql)["mAP"]
                item["controls"][label]={"mAP":c["mAP"],"precision":c["precision"],
                      "primary_minus_control_mAP":comparison,"sha256":checksum(file)}
                csvrows.append([s,n,label,c["mAP"],c["precision"],c["map_k"],c["p_k"],record["revision"],checksum(file)])
        else:
            csvrows.append([s,n,"reference",record["mAP"],record["precision"],record["map_k"],record["p_k"],
                            record["revision"],checksum(path)])
        rows[n]=item
        values[n]=v
    for n in rows:
        rows[n]["versus_exact_word"]=paired_delta(values[n],values[UNIFORM],ql)
        if n in NAMES:
            past=REF if decision["families"][n]["scope"]=="pairwise" else "neighborhood-calibrated-likelihood"
            rows[n]["versus_reference_of_same_scope"]=paired_delta(values[n],values[past],ql)
    class_rows=[]
    for i,c in enumerate(UNSEEN_CLASSES[s]):
        means=group_scores(values,ql,ql==i)
        class_rows.extend([s,c,n,int((ql==i).sum()),x["mAP"],x["precision"]] for n,x in means.items())
    return rows,values,csvrows,class_rows,leaves


def calculated_index(module,params,q,p,banks,scale,args,qpaths,ppaths,qi,pi):
    src,tgt,literal=banks
    data=Data(q,p,src,tgt,scale)
    if getattr(module,"USES_LITERAL",False):
        data.literal=literal
    if hasattr(module,"prepare_ir_data"):
        module.prepare_ir_data(data,args,qpaths,ppaths,qi,pi)
    model=module.Index(data)
    scores=model.score(module.Config(**params))
    return scores,model


def real_pair_checks(w,decision,modules,a,qpaths,ppaths,qi,pi,raw,rgb,ql,pl,banks,scale):
    qids=stratified_sample(ql,2,19265)
    gallery_ids=stratified_sample(pl,24,8609)
    retained=torch.arange(len(gallery_ids)-1,-1,-3)
    sketch_keep=torch.arange(len(qids)-1,-1,-2)
    out={}
    for name,module in modules.items():
        args=decision["families"][name]["settings"]
        def build(qix,pix):
            return calculated_index(module,args,raw[qi[qix]].cuda(),rgb[pi[pix]].cuda(),banks,scale,
                                   a,qpaths,ppaths,qi[qix],pi[pix])[0]
        full=build(qids,gallery_ids)
        small=build(qids[sketch_keep],gallery_ids[retained])
        maximum=float((full[sketch_keep][:,retained]-small).abs().max())
        single=build(qids[:1],gallery_ids[:1])
        unit=float((full[0,0]-single[0,0]).abs())
        if max(maximum,unit)>1e-4:
            raise ValueError(f"Pairwise scores changed materially in {name}: {maximum}, {unit}")
        out[name]={"queries":len(qids),"reference_photos":len(gallery_ids),"retained_queries":len(sketch_keep),
                   "retained_photos":len(retained),"max_abs_reversed_subset":maximum,"singleton_abs":unit}
    return out


def entropy(model,images,bank,scale,config,side):
    measures=[]
    factor=config.query_scale if side=="sketch" else config.photo_scale
    for x in images.split(192):
        logit=scale*(x.cuda()@bank.T)
        results=[]
        for mul,k in ((1.,0),(factor,config.terms)):
            vals=logit.topk(k,dim=1).values if k else logit
            p=(mul*vals).softmax(1)
            results.append(torch.stack(((-(p*p.clamp_min(1e-30).log()).sum(1)).exp(),p.amax(1)),1).cpu())
        measures.append(torch.stack(results,1))
    measures=torch.cat(measures)
    return {name:{"median_effective_words":float(measures[:,i,0].median()),
                 "median_peak_mass":float(measures[:,i,1].median())} for i,name in enumerate(("checkpoint","selected"))}


def repeat_region(module,dataset,args,paths,indices,global_vectors):
    x=module.region_inputs(args,paths,dataset)["5"][indices[:64]].cuda().float()
    x=F.normalize(x,dim=2)
    g=global_vectors[indices[:64]].cuda()[:,None]
    blocks=torch.cat((g,x),1)
    enlarged=torch.cat((blocks,blocks[:,1:2].expand(-1,3,-1)),1)
    result={}
    for mode in ("mean","democratic"):
        a=module.aggregate(blocks,mode)
        b=module.aggregate(enlarged,mode)
        change=(1.-(a*b).sum(1)).clamp_min(0.)
        result[mode]={"mean_cosine_distance_after_3_extra_copies":float(change.mean()),
                      "median":float(change.median())}
    return result


def write_csv(path,headers,rows):
    with path.open("w",encoding="utf-8",newline="") as f:
        out=csv.writer(f,lineterminator="\n");out.writerow(headers);out.writerows(rows)


@torch.inference_mode()
def run(w):
    root=w/"results"
    five=verified(root/DECISION)
    final=verified(root/EXTRA/"extra-joint.json")
    if final["prior_five_lock"]!=digest(five):
        raise ValueError("Different prior decision")
    eps=lambda n:tuple(sum(r[k] for r in final["families"][n]["scores"])/2 for k in ("mAP","precision"))
    best_new=max((n for n in NAMES if final["families"][n]["scope"]=="pairwise"),key=eps)
    modules={n:source(w,n,final["families"][n]) for n in NAMES if final["families"][n]["scope"]=="pairwise"}
    dest=root/"image-retrieval-round"
    result={"revision":revision(),"gpu":torch.cuda.get_device_name(),"five_freeze":digest(five),
            "final_freeze":digest(final),"best_new_pairwise":best_new,"pairwise_primary":final["pairwise"],
            "all_primary":final["primary"],"families":final["families"],"splits":{}}
    metrics=[];classes=[]
    for ds in ("sketchy","tuberlin"):
        a,qp,qs,pp,gs,raw,photos=global_inputs(w,ds)
        words=read_words(a.vocabulary)
        literal,scale=bare_words(a,words)
        src,tgt=(x.cuda() for x in load_words(a,words))
        banks=(src,tgt,literal.cuda())
        for s,(qi,ql) in qs.items():
            pi,pl=gs[s]
            rows,values,table,categories,leaves=load_split(w,s,ql,pl,final,best_new)
            metrics.extend(table);classes.extend(categories)
            details={"scores":rows,"example_errors":example_paths(w,s,ql,pl,qi,pi,qp,pp,values,leaves,best_new)}
            details["pairwise_invariance"]=real_pair_checks(w,final,modules,a,qp,pp,qi,pi,raw,photos,ql,pl,banks,scale)
            soft=modules[EXTRA];params=soft.Config(**final["families"][EXTRA]["settings"])
            details["assignment_entropy"]={"sketch":entropy(soft,raw[qi],src,scale,params,"sketch"),
                                           "photo":entropy(soft,photos[pi],tgt,scale,params,"photo")}
            regions=modules["retrieval-democratic-regions"]
            details["repeat_region"]={mode:repeat_region(regions,mode,a,paths,ids[index],vectors)
                for mode,paths,ids,index,vectors in (("sketch",qp,qi,stratified_sample(ql,3,816),raw),
                                                     ("photo",pp,pi,stratified_sample(pl,3,876),photos))}
            if s=="tuberlin":
                norm=lambda name:" ".join(name.casefold().replace("-"," ").replace("_"," ").split())
                prior={norm(x) for k in ("sketchy_1","sketchy_2") for x in UNSEEN_CLASSES[k]}
                distinct=[i for i,n in enumerate(UNSEEN_CLASSES[s]) if norm(n) not in prior]
                mask=torch.isin(ql,torch.tensor(distinct))
                details["new_literal_classes"]={"classes":len(distinct),"queries":int(mask.sum()),
                                                "gallery":len(pl),"scores":group_scores(values,ql,mask)}
            result["splits"][s]=details
            atomic_json(dest/(s+"-audit.json"),details)
            print(s,best_new,rows[best_new]["mAP"],flush=True)
    atomic_json(dest/"report.json",result)
    write_csv(dest/"metrics.csv",["split","family","control","mAP","precision","map_k","p_k","revision","sha256"],metrics)
    write_csv(dest/"classes.csv",["split","class","family","queries","mAP","precision"],classes)
    files=[p for name in [*NAMES,"image-retrieval-round"] for p in sorted((root/name).rglob("*"))
           if p.is_file() and p.suffix in {".pt",".json",".csv"}]
    path=w/"image-retrieval-round-evidence.zip"
    with zipfile.ZipFile(path,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=4) as archive:
        for f in files: archive.write(f,f.relative_to(w).as_posix())
    atomic_json(w/"image-retrieval-round-archive-info.json",{"sha":checksum(path),"size":path.stat().st_size,"count":len(files)})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(p.parse_args().workspace)
