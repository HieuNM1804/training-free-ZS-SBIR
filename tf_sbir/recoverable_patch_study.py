"""Identify which part of a frozen patch projection changes full-gallery retrieval on real queries."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import inspect
from pathlib import Path

import torch

from .concept_benchmark import atomic_json, checksum, read
from .conditional_agreement_study import Replay
from .crossdomain_benchmark import digest, revision, SPLIT_HASHES
from .crossdomain_inputs import CHECKPOINT
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .gallery_panel_resume import semantic_inputs
from .map_diagnosis import map_values
from .quickdraw_locked import KD_FUNCTION_SHA
from .recoverable_patch_information import FAMILY, MODES
from .recoverable_patch_inputs import restore
from .regional_intervention_study import index as load_index
from .response_geometry_views import study_base
from .retrieval_diagnosis import stratified_sample
from .retrieval_evidence import load_kd, paired_delta


CHANNELS=("rgb","words","both")
GAINS=(.15,.6,1.5)


def locate(sorted_ids, query):
    index=torch.searchsorted(sorted_ids,query.contiguous())
    if (index>=len(sorted_ids)).any() or not torch.equal(sorted_ids[index],query):
        raise ValueError("An image is missing from the signed embedding bank")
    return index


def word_log(feature, axes, bank, temperature):
    logits=(feature@axes.T)*temperature
    mass=(logits-logits.amax(1,keepdim=True)).exp()
    return (mass@bank.T).clamp_min(1e-30).log()


def grid(readouts):
    return ["regional",*(f"{r}/{c}/{g:g}" for r in readouts for c in CHANNELS for g in GAINS)]


def select(means):
    best={m:max((k for k in means if k.startswith(m+"/")),key=means.get) for m in MODES}
    _,amount,channel,gain=best["hidden"].split("/")
    matched=[f"{m}/{amount}/{channel}/{gain}" for m in MODES]
    return best,matched


def describe(data,positions):
    fields=("achromatic_invisible_energy_fraction","native_leak","recovered_token_norm_fraction",
            "reflected_out_of_range_fraction","reflected_RGB_RMS","relative_placebo_norm_error")
    values=data["diagnosis"][positions].float()
    checks=("float_token_reflection","centered_reflection_cos","openclip_native_cos","phase_adjoint",
            "physical_fp16_reflection_cos","cache_native_cos")
    return {"image_count":len(positions),"mean":dict(zip(fields,values.mean(0).tolist(),strict=True)),
            "median":dict(zip(fields,values.median(0).values.tolist(),strict=True)),
            "max_over_bank_checks":dict(zip(checks,data["checks"].amax(0).tolist(),strict=True))}


def prepare(w,s,inputs,embeds):
    _,_,qs,pp,gs,_,photo,src,tgt,scale=inputs
    qi,y=qs[s]
    pi,g=gs[s]
    if digest(UNSEEN_CLASSES[s])!=SPLIT_HASHES[s]:
        raise ValueError("Changed benchmark categories")
    ids=stratified_sample(y,36,63241)
    rgb=photo[pi].cuda()
    sig=digest({"gallery":[str(pp[int(j)]) for j in pi],"model":CHECKPOINT,"source":"regional-64/RGB16"})
    heads,weights=load_index(w,s,rgb,tgt,scale,sig[:16])
    return {"ids":ids,"positions":locate(embeds["ids"],qi[ids]),"y":y[ids],"g":g,"rgb":rgb,
            "axes":src[heads],"bank":weights,"scale":scale,"base":study_base(w,s,inputs,ids)}


def evaluate(s,prepared,data,readouts,rows,kd,workers):
    p=prepared
    splits=defaultdict(list)
    for name in rows:
        if name!="regional":
            m,a,c,w=name.split("/")
            splits[readouts.index(m+"/"+a)].append((name,c,float(w)))
    metrics,tops,errors=defaultdict(list),defaultdict(list),{}
    replay=Replay(kd,p["g"],s) if kd else None
    gl=p["g"].cuda()
    with (ThreadPoolExecutor(max_workers=workers) if kd else nullcontext()) as pool:
        for first in range(0,len(p["y"]),64):
            sl=slice(first,first+64)
            indexes=p["positions"][sl]
            ref=data["native"][indexes].cuda()
            change=data["change"][indexes].cuda().float()
            base=p["base"][sl].cuda()
            label=p["y"][sl]
            truth=label.cuda()
            def score(name,table):
                vals,order,hits,_=map_values(table,truth,gl,s)
                vals=vals.cpu()
                if replay:
                    official=replay.score(order,label,pool)
                    err=float((vals-official).abs().max())
                    errors[name]=max(errors.get(name,0.),err)
                    if err>2e-6:
                        raise ValueError("Rank replay differs from the KD-SBIR evaluator")
                    vals=official
                metrics[name].append(vals)
                tops[name].append(hits[:,0].cpu())
            score("regional",base)
            original=word_log(ref,p["axes"],p["bank"],p["scale"])
            for index,configs in splits.items():
                step=change[:,index]
                rgb=p["scale"]*(step@p["rgb"].T)
                lex=word_log(ref+step,p["axes"],p["bank"],p["scale"])-original
                diff={"rgb":rgb,"words":lex,"both":.5*(rgb+lex)}
                for name,channel,gain in configs:
                    score(name,base+gain*diff[channel])
            if first%320==0:
                print(s,"audit" if replay else "grid",first,len(p["y"]),flush=True)
    return ({k:torch.cat(v) for k,v in metrics.items()},{k:torch.cat(v) for k,v in tops.items()},errors)


@torch.inference_mode()
def run(w,workers):
    folder=w/"results"/FAMILY
    inputs=semantic_inputs(w,"sketchy")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:
        raise ValueError("Wrong evaluator")
    emb,desc=restore(w,"sketchy","pilot-query")
    readouts=desc["readouts"]
    rows=grid(readouts)
    spec={"revision":revision(),"checkpoint":CHECKPOINT,"preprocessing":desc["signature"],
          "tokenizer":desc["tokenizer"],"KD":KD_FUNCTION_SHA,"public_vocabulary_sha256":checksum(inputs[0].vocabulary),
          "rows":rows,"queries_per_class":36,"sample_seed":63241,"device":torch.cuda.get_device_name(),
          "selection":"equal mean of unseen Sketchy mAP, equal precision for ties; independently optimize each competing family",
          "unit":"query-only causal change on fixed regional candidate scores, complete gallery"}
    sig=digest(spec)
    if (folder/"protocol.json").is_file() and read(folder/"protocol.json")["signature"]!=sig:
        raise ValueError("A different protocol already has this result directory")
    atomic_json(folder/"protocol.json",spec|{"signature":sig})
    tables={}
    for s in inputs[2]:
        f=folder/("pilot-"+s+".json")
        if f.is_file():
            old=read(f)
            if old["signature"]!=sig or checksum(f.with_suffix(".pt"))!=old["per_query_sha256"]:
                raise ValueError("Inconsistent pilot checkpoint")
            tables[s]=old["values"]
            continue
        p=prepare(w,s,inputs,emb)
        metrics,top,_=evaluate(s,p,emb,readouts,rows,None,workers)
        tables[s]={k:v.double().mean(0).tolist() for k,v in metrics.items()}
        save_tensor({"ids":p["ids"],"query_labels":p["y"],"metrics":metrics,"top1":top},f.with_suffix(".pt"))
        atomic_json(f,{"signature":sig,"values":tables[s],"queries":len(p["ids"]),"gallery":len(p["g"]),
                       "input":describe(emb,p["positions"]),"per_query_sha256":checksum(f.with_suffix(".pt"))})
        del p
    means={k:[sum(t[k][i] for t in tables.values())/len(tables) for i in (0,1)] for k in rows}
    best,matched=select(means)
    keep=sorted({"regional",*best.values(),*matched})
    audited,paired,changed,errors={},{},{},{}
    main=best["hidden"]
    for s in inputs[2]:
        p=prepare(w,s,inputs,emb)
        met,top,err=evaluate(s,p,emb,readouts,keep,kd,workers)
        raw=torch.load(folder/("pilot-"+s+".pt"),weights_only=True,mmap=True)
        mismatch=max(float((raw["metrics"][k]-v).abs().max()) for k,v in met.items())
        if mismatch>2e-6:raise ValueError("Shortlist scores changed")
        audited[s]={k:v.double().mean(0).tolist() for k,v in met.items()}
        labels=p["y"].unique(return_inverse=True)[1]
        paired[s]={k:paired_delta(met[main],v,labels) for k,v in met.items()}
        changed[s]={"fixed":int((~top["regional"]&top[main]).sum()),"broken":int((top["regional"]&~top[main]).sum())}
        errors[s]={"queries":len(labels),"max_original_evaluator_error":max(err.values()),"repeat_error":mismatch}
        del p
    atomic_json(folder/"pilot.json",{"main":main,"best_by_family":best,"matched":matched,"audited":audited,
                "paired":paired,"top1":changed,"KD":errors,"mean":means,"protocol":sig})
    print("Selected",main,means[main],means["regional"],flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    cli.add_argument("--workers",type=int,default=2)
    args=cli.parse_args()
    run(args.workspace,args.workers)
