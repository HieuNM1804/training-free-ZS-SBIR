"""Benchmark native-unit replacements and independently selected controls on identical galleries with original KD-SBIR rank replay."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import inspect
from pathlib import Path

import torch
from torch.nn import functional as F

from .concept_benchmark import atomic_json, checksum, read
from .conditional_agreement_study import Replay
from .crossdomain_benchmark import digest, revision, SPLIT_HASHES
from .crossdomain_inputs import CHECKPOINT
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .gallery_panel_resume import semantic_inputs
from .map_diagnosis import map_values
from .quickdraw_locked import KD_FUNCTION_SHA
from .recoverable_patch_study import word_log
from .regional_intervention_study import index
from .response_geometry_views import study_base
from .retrieval_diagnosis import stratified_sample
from .retrieval_evidence import load_kd, paired_delta
from .semantic_neuron_counterfactuals import ARMS, code, restore
from .semantic_neuron_gallery import FAMILY


GAINS=(.12,.45,1.)
CHANNELS=("words","rgb","both")
MAIN_GROUPS=("sensor","curvature")


def grid():
    return ["regional",*(f"{code(a)}/{c}/{g:g}" for a in ARMS for c in CHANNELS for g in GAINS)]


def choose(means,groups,methods):
    return max((k for k in means if k!="regional" and k.split("/")[1] in groups and k.split("/")[3] in methods),
               key=means.get)


def selection(means):
    main=choose(means,MAIN_GROUPS,("late4","late32"))
    l,g,n,d,c,w=main.split("/")
    matched="sensor_matched" if g=="sensor" else "matched"
    best={"nominee":main,"constant":choose(means,MAIN_GROUPS,("mean",)),
          "zero":choose(means,MAIN_GROUPS,("zero",)),"shuffled":choose(means,MAIN_GROUPS,("shuffled",)),
          "early":choose(means,MAIN_GROUPS,("early",)),"scene":choose(means,MAIN_GROUPS,("scene",)),
          "native_activity":choose(means,("native",),("late4","late32")),
          "matched_units":choose(means,("matched","sensor_matched"),("late4","late32")),
          "random_units":choose(means,("random",),("late4","late32"))}
    fixed=[f"{l}/{g}/{n}/{variant}/{c}/{w}" for variant in ("shuffled","mean","early","zero")]
    fixed.extend(f"{l}/{variant}/{n}/{d}/{c}/{w}" for variant in (matched,"random","native"))
    return best,sorted(set(fixed))


def parse_tables(names,readouts):
    records=defaultdict(list)
    direct=defaultdict(list)
    for k in names:
        if k in ("regional","native_rgb"):continue
        if k.startswith("standalone:"):
            name=k.split(":",1)[1]
            direct[readouts.index(name)].append(k)
            continue
        r,ch,gain=k.rsplit("/",2)
        records[readouts.index(r)].append((k,ch,float(gain)))
    return records,direct


def setup(w,s,inputs,data,stage):
    _,_,qs,pp,gs,_,photo,src,tgt,scale=inputs
    qi,labels=qs[s]
    pi,gl=gs[s]
    if digest(UNSEEN_CLASSES[s])!=SPLIT_HASHES[s]:raise ValueError("Wrong split")
    ids=stratified_sample(labels,36,63241) if stage=="pilot-query" else torch.arange(len(labels))
    if not torch.equal(data["ids"],qi[ids]):raise ValueError("Query order differs from the benchmark")
    rgb=photo[pi].cuda()
    sig=digest({"gallery":[str(pp[int(i)]) for i in pi],"model":CHECKPOINT,"source":"regional-64/RGB16"})
    axes,bank=index(w,s,rgb,tgt,scale,sig[:16])
    return {"ids":ids,"y":labels[ids],"g":gl,"rgb":rgb,"src":src[axes],"bank":bank,"scale":scale,
            "base":study_base(w,s,inputs,ids)}


def evaluate(s,data,info,study,names,kd,workers):
    forms,direct=parse_tables(names,info["arms"])
    replay=Replay(kd,study["g"],s) if kd else None
    gt=study["g"].cuda()
    metrics,top,errors=defaultdict(list),defaultdict(list),{}
    with (ThreadPoolExecutor(max_workers=workers) if kd else nullcontext()) as pool:
        for begin in range(0,len(data["ids"]),48):
            sl=slice(begin,begin+48)
            native=data["native"][sl].cuda()
            delta=data["change"][sl].cuda().float()
            baseline=study["base"][sl].cuda()
            labels=study["y"][sl]
            def score(name,matrix):
                val,order,hits,_=map_values(matrix,labels.cuda(),gt,s)
                val=val.cpu()
                if replay:
                    exact=replay.score(order,labels,pool)
                    err=float((val-exact).abs().max())
                    if err>2e-6:raise ValueError("Changed original KD-SBIR evaluation")
                    errors[name]=max(errors.get(name,0.),err)
                    val=exact
                metrics[name].append(val);top[name].append(hits[:,0].cpu())
            score("regional",baseline)
            if "native_rgb" in names:score("native_rgb",native@study["rgb"].T)
            old=word_log(native,study["src"],study["bank"],study["scale"])
            for arm in sorted(forms.keys()|direct.keys()):
                step=delta[:,arm]
                cur=native+step
                rgb=study["scale"]*(step@study["rgb"].T)
                words=word_log(cur,study["src"],study["bank"],study["scale"])-old
                diff={"rgb":rgb,"words":words,"both":.5*(rgb+words)}
                for k,c,g in forms[arm]:
                    score(k,baseline+g*diff[c])
                for k in direct[arm]:score(k,F.normalize(cur)@study["rgb"].T)
            if begin%576==0:print(s,"KD" if kd else "scan",begin,len(data["ids"]),flush=True)
    return {k:torch.cat(r) for k,r in metrics.items()},{k:torch.cat(r) for k,r in top.items()},errors


def reconstruction(bank):
    loss=bank["reconstruction_error"].float()
    summary={}
    for i,arm in enumerate(ARMS):
        l,g,n,method=arm
        ref=loss[:,ARMS.index((l,g,n,"mean"))].clamp_min(1e-8)
        relative=loss[:,i]/ref
        summary[code(arm)]={"mean_scaled_mse":float(loss[:,i].mean()),"median_relative_to_unconditional":float(relative.median()),
                            "fraction_improved_vs_unconditional":float((relative<1).float().mean())}
    return summary


def official(s,folder,stage,data,info,context,names,main,kd,workers,locked):
    f=folder/((s if stage=="all-query" else "pilot-official-"+s)+".json")
    expected={"lock":locked,"feature_manifest":info["signature"],"rows":names}
    if f.is_file():
        old=read(f)
        if old["protocol"]!=expected or checksum(f.with_suffix(".pt"))!=old["per_query_sha256"]:raise ValueError("Different earlier result")
        return old
    met,top,er=evaluate(s,data,info,context,names,kd,workers)
    label=context["y"].unique(return_inverse=True)[1]
    base=met["regional"]
    check=0.
    if stage=="pilot-query":
        initial=torch.load(folder/("pilot-"+s+".pt"),weights_only=True,mmap=True)
        check=max(float((met[k]-initial["metrics"][k]).abs().max()) for k in names if k in initial["metrics"])
        if check>2e-6:raise ValueError("Finalist rankings changed")
    save_tensor({"ids":context["ids"],"labels":label,"metrics":met,"top1":top},f.with_suffix(".pt"))
    summary={"protocol":expected,"queries":len(label),"gallery":len(context["g"]),"classes":int(label.max()+1),
             "values":{k:v.double().mean(0).tolist() for k,v in met.items()},
             "paired_main_minus_each":{k:paired_delta(met[main],v,label) for k,v in met.items()},
             "top1":{"fixed":int((top[main]&~top["regional"]).sum()),"broken":int((~top[main]&top["regional"]).sum())},
             "KD_max_abs_error":max(er.values()),"shortlist_repeat":check,"per_query_sha256":checksum(f.with_suffix(".pt"))}
    atomic_json(f,summary)
    return summary


@torch.inference_mode()
def run(w,stage,workers,ds):
    inputs=semantic_inputs(w,ds)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:raise ValueError("Wrong evaluator")
    folder=w/"results"/FAMILY
    if stage=="full":
        frozen=read(folder/"freeze.json")
        if digest(frozen["configuration"])!=frozen["sha256"]:raise ValueError("The freeze changed")
        key=frozen["configuration"]["nominee"]
        for s in inputs[2]:
            data,info=restore(w,s,"all-query")
            context=setup(w,s,inputs,data,"all-query")
            rec=official(s,folder,"all-query",data,info,context,frozen["full_rows"],key,kd,workers,frozen["sha256"])
            print("final",s,rec["values"][key],rec["values"]["regional"],flush=True)
            del context,data
        return
    if ds!="sketchy":raise ValueError("Development is confined to the two Sketchy splits")
    rows=grid()
    meta={"revision":revision(),"device":torch.cuda.get_device_name(),"model":CHECKPOINT,"KD":KD_FUNCTION_SHA,
          "words_sha256":checksum(inputs[0].vocabulary),"rows":rows,"pilot_query_seed":63241,
          "primary_arms":"either target-discovered sensor or translation neurons, conditional on late native patch semantics",
          "controls":"constant, removal, shuffled donors, shallow donors, global donors, matched/native/random units optimized independently",
          "selection":"mean unseen mAP of the two balanced Sketchy pilots; equal precision breaks ties; development may overfit"}
    sig=digest(meta)
    if (folder/"protocol.json").is_file() and read(folder/"protocol.json")["signature"]!=sig:raise ValueError("Different pilot protocol")
    atomic_json(folder/"protocol.json",meta|{"signature":sig})
    tables={}
    for s in inputs[2]:
        data,info=restore(w,s,"pilot-query")
        context=setup(w,s,inputs,data,"pilot-query")
        f=folder/("pilot-"+s+".json")
        if f.is_file():
            rec=read(f)
            if rec["signature"]!=sig or checksum(f.with_suffix(".pt"))!=rec["per_query_sha256"]:raise ValueError("Inconsistent pilot")
            tables[s]=rec["values"]
        else:
            met,top,_=evaluate(s,data,info,context,rows,None,workers)
            tables[s]={k:v.double().mean(0).tolist() for k,v in met.items()}
            save_tensor({"ids":context["ids"],"labels":context["y"],"metrics":met,"top1":top},f.with_suffix(".pt"))
            target,_=restore(w,s,"photo-holdout")
            memory=read(w/"cache"/FAMILY/(s+"-memory-v2.json"))
            atomic_json(f,{"signature":sig,"queries":len(data["ids"]),"gallery":len(context["g"]),"values":tables[s],
                           "native_prediction":{"heldout_gallery":reconstruction(target),"sketch":reconstruction(data)},
                           "heldout_photos":len(target["ids"]),"discovery":memory["recurrence"],"per_query_sha256":checksum(f.with_suffix(".pt"))})
        del data,context
    means={k:[sum(t[k][i] for t in tables.values())/len(tables) for i in (0,1)] for k in rows}
    chosen,matched=selection(means)
    main=chosen["nominee"]
    checks=sorted({"regional","native_rgb",*chosen.values(),*matched,"standalone:"+main.rsplit("/",2)[0]})
    # Full-query resources go to the previously declared primary and the strongest simpler control of each relevant kind.
    full=sorted({"regional","native_rgb",main,chosen["constant"],chosen["zero"],chosen["shuffled"],chosen["early"],
                 chosen["native_activity"],chosen["matched_units"],chosen["random_units"],"standalone:"+main.rsplit("/",2)[0],
                 main.replace("/late32/","/shuffled/").replace("/late4/","/shuffled/")})
    config={"nominee":main,"selected":chosen,"matched":matched,"full_rows":full,"pilot_signature":sig,
            "source_splits":list(tables),"attribution":"two translated or opposite tiny-noise observations within each calibration photo"}
    lock=digest(config)
    inputs_used=sorted({k.split(":",1)[1] if k.startswith("standalone:") else k.rsplit("/",2)[0]
                        for k in full if k not in ("regional","native_rgb")})
    atomic_json(folder/"freeze.json",{"configuration":config,"sha256":lock,"full_rows":full,"input_embeddings":inputs_used})
    audited={}
    for s in inputs[2]:
        data,info=restore(w,s,"pilot-query")
        context=setup(w,s,inputs,data,"pilot-query")
        audited[s]=official(s,folder,"pilot-query",data,info,context,checks,main,kd,workers,lock)
        del data,context
    atomic_json(folder/"pilot.json",{"chosen":chosen,"matched":matched,"audited":audited,"frozen":lock,"mean":means})
    print("locked",main,means[main],means["regional"],lock,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--dataset",choices=("sketchy","tuberlin"),default="sketchy")
    p.add_argument("--stage",choices=("pilot","full"),required=True)
    p.add_argument("--workers",type=int,default=2)
    opts=p.parse_args()
    run(opts.workspace,opts.stage,opts.workers,opts.dataset)
