"""Does intervening on the origin of a symmetric response improve actual full-gallery ranks beyond terminal averaging?"""

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
from .crossdomain_benchmark import digest, revision
from .crossdomain_inputs import CHECKPOINT
from .encoding import save_tensor
from .gallery_panel_resume import semantic_inputs
from .map_diagnosis import map_values
from .patch_mediation_inputs import ARMS, CUTS, TYPES, load
from .patch_response_mediation import FAMILY
from .quickdraw_locked import KD_FUNCTION_SHA
from .recoverable_patch_study import prepare, word_log
from .retrieval_evidence import load_kd, paired_delta


# Terminal averaging has the same two observations as each proposed midpoint comparison.
EFFECTS={"terminal":("even","base"),"one_sided":("negative","base"),"odd":("positive","negative"),
         "block1":("m1","input"),"block2":("m2","m1"),"block3":("m3","m2"),"block4":("m4","m3"),
         "attention2":("a2","m1"),"mlp2":("m2","a2"),"attention3":("a3","m2"),"mlp3":("m3","a3"),
         "through4":("m4","base"),"late":("even","m4")}
GAINS=(-.6,.15,.6,1.5)
PRIMARY=("block2","block3","attention2","mlp2","attention3","mlp3")
DIRECTIONS=(*TYPES,"uniform_mean")


def rows():
    return ["regional",*(f"{m}/{term}/{c}/{a:g}" for m in DIRECTIONS for term in EFFECTS
                         for c in ("words","rgb") for a in (GAINS if c=="words" else (.6,)))]


def best(means,dirs,terms):
    subset=[k for k in means if k!="regional" and k.split("/")[0] in dirs and k.split("/")[1] in terms]
    return max(subset,key=means.get)


def shortlist(means):
    # The deployable primary has no hidden-pixel calculation: it averages two independent isotropic directions.
    main=best(means,("uniform_mean",),PRIMARY)
    _,stage,ch,g=main.split("/")
    options={"nominee":main,"terminal":best(means,("uniform_mean",),("terminal",)),
             "one_sided":best(means,("uniform_mean",),("one_sided",)),"odd":best(means,("uniform_mean",),("odd",)),
             "hidden":best(means,("hidden",),PRIMARY),"shuffled":best(means,("shuffled",),PRIMARY),
             "later":best(means,("uniform_mean",),("late",)),"full_early":best(means,("uniform_mean",),("through4",))}
    matches={f"{m}/{stage}/{ch}/{g}" for m in DIRECTIONS}|{f"uniform_mean/{k}/{ch}/{g}" for k in ("terminal","one_sided","odd","through4","late")}
    return options,sorted(matches)


def endpoints(base,shift,mode,terms,p,channels):
    needed=set()
    for name in terms:needed.update(EFFECTS[name])
    need_even="even" in needed
    if need_even:needed|={"positive","negative"}
    obs={"rgb":{},"words":{}}
    for term in sorted(needed-{"even"}):
        step=shift[:,ARMS.index((mode,term))] if term!="base" else torch.zeros_like(base)
        if "rgb" in channels:
            obs["rgb"][term]=p["scale"]*(step@p["rgb"].T)
        if "words" in channels:
            obs["words"][term]=word_log(base+step,p["axes"],p["bank"],p["scale"])
    if need_even:
        for ch in channels:obs[ch]["even"]=.5*(obs[ch]["positive"]+obs[ch]["negative"])
    return obs


def evaluate(s,p,bank,names,kd,workers):
    configs=defaultdict(lambda:defaultdict(list))
    for name in names:
        if name=="regional":continue
        m,term,ch,g=name.split("/")
        configs[m][term,ch].append((name,float(g)))
    scores,top,errors=defaultdict(list),defaultdict(list),{}
    detail=Replay(kd,p["g"],s) if kd is not None else None
    truth=p["g"].cuda()
    with (ThreadPoolExecutor(max_workers=workers) if kd else nullcontext()) as pool:
        for start in range(0,len(p["y"]),48):
            sl=slice(start,start+48)
            loc=p["positions"][sl]
            z=bank["native"][loc].cuda()
            change=bank["shift"][loc].cuda().float()
            base=p["base"][sl].cuda()
            yl=p["y"][sl]
            def record(k,score):
                v,order,r,_=map_values(score,yl.cuda(),truth,s)
                v=v.cpu()
                if detail:
                    exact=detail.score(order,yl,pool)
                    error=float((v-exact).abs().max())
                    if error>2e-6:raise ValueError("Original evaluator disagrees")
                    errors[k]=max(errors.get(k,0.),error)
                    v=exact
                scores[k].append(v)
                top[k].append(r[:,0].cpu())
            record("regional",base)
            uniform={}
            for mode in TYPES:
                needed=set(configs[mode])
                if mode.startswith("uniform_"):needed.update(configs["uniform_mean"])
                if not needed:continue
                terms={k for k,_ in needed}
                channels={k for _,k in needed}
                obs=endpoints(z,change,mode,terms,p,channels)
                for term,ch in needed:
                    end,begin=EFFECTS[term]
                    delta=obs[ch][end]-obs[ch][begin]
                    if term=="odd":delta*=.5
                    for name,gain in configs[mode].get((term,ch),[]):record(name,base+gain*delta)
                    if (term,ch) in configs["uniform_mean"] and mode.startswith("uniform_"):
                        if mode=="uniform_a":
                            uniform[term,ch]=delta
                        else:
                            avg=.5*(delta+uniform.pop((term,ch)))
                            for name,gain in configs["uniform_mean"][term,ch]:record(name,base+gain*avg)
                del obs
            if start%288==0:print(s,"KD" if kd else "grid",start,len(p["y"]),flush=True)
    return {k:torch.cat(v) for k,v in scores.items()},{k:torch.cat(v) for k,v in top.items()},errors


def profiles(bank):
    out={}
    for mode in TYPES:
        pos,neg=[bank["shift"][:,ARMS.index((mode,k))].float() for k in ("positive","negative")]
        even=.5*(pos+neg)
        den=even.square().sum(1).clamp_min(1e-20)
        vals={t:((bank["shift"][:,ARMS.index((mode,t))].float()*even).sum(1)/den) for t in CUTS}
        increments={f"{b}_minus_{a}":vals[b]-vals[a] for a,b in zip(CUTS,CUTS[1:])}
        out[mode]={t:{"median":float(v.median()),"mean":float(v.mean()),"quartiles":v.quantile(torch.tensor([.25,.75])).tolist()}
                   for t,v in (vals|increments).items()}
    return out


@torch.inference_mode()
def run(w,workers):
    inputs=semantic_inputs(w,"sketchy")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:raise ValueError("Wrong evaluator")
    data,config=load(w,"sketchy","pilot-query")
    grid=rows()
    version={"revision":revision(),"data":config["signature"],"vocab":checksum(inputs[0].vocabulary),"weights":CHECKPOINT,
             "KD":KD_FUNCTION_SHA,"grid":grid,"cuts":EFFECTS,"seed":63241,"queries_per_class":36,
             "selection":"equal-weight mAP of two unseen Sketchy splits, precision as tie breaker. Uniform two-direction early-block effect is primary. Terminal and one-sided controls tune separately."}
    sig=digest(version)
    root=w/"results"/FAMILY
    if (root/"protocol.json").exists() and read(root/"protocol.json")["signature"]!=sig:raise ValueError("Changed study")
    atomic_json(root/"protocol.json",version|{"signature":sig,"device":torch.cuda.get_device_name()})
    tables={}
    for s in inputs[2]:
        f=root/("pilot-"+s+".json")
        if f.is_file():
            older=read(f)
            if older["signature"]!=sig or checksum(f.with_suffix(".pt"))!=older["per_query_sha256"]:raise ValueError("Changed pilot checkpoint")
            tables[s]=older["values"]
            continue
        p=prepare(w,s,inputs,data)
        m,top,_=evaluate(s,p,data,grid,None,workers)
        tables[s]={k:v.double().mean(0).tolist() for k,v in m.items()}
        save_tensor({"ids":p["ids"],"labels":p["y"],"metrics":m,"top1":top},f.with_suffix(".pt"))
        atomic_json(f,{"signature":sig,"queries":len(p["ids"]),"gallery":len(p["g"]),"values":tables[s],
                       "per_query_sha256":checksum(f.with_suffix(".pt"))})
        del p
    means={k:[sum(t[k][i] for t in tables.values())/len(tables) for i in (0,1)] for k in grid}
    options,matches=shortlist(means)
    keep=sorted({"regional",*options.values(),*matches})
    main=options["nominee"]
    audit,paired,repairs,errs={},{},{},{}
    for s in inputs[2]:
        p=prepare(w,s,inputs,data)
        m,top,er=evaluate(s,p,data,keep,kd,workers)
        old=torch.load(root/("pilot-"+s+".pt"),mmap=True,weights_only=True)
        diff=max(float((old["metrics"][k]-v).abs().max()) for k,v in m.items())
        if diff>2e-6:raise ValueError("Changed shortlist values")
        labels=p["y"].unique(return_inverse=True)[1]
        paired[s]={k:paired_delta(m[main],v,labels) for k,v in m.items()}
        repairs[s]={"corrected":int((top[main]&~top["regional"]).sum()),"broken":int((~top[main]&top["regional"]).sum())}
        audit[s]={k:v.double().mean(0).tolist() for k,v in m.items()}
        errs[s]={"queries":len(labels),"original_evaluator_max_error":max(er.values()),"repeat_max":diff}
        del p
    atomic_json(root/"pilot.json",{"signature":sig,"selected":options,"matched":matches,"mean":means,"audited":audit,
                "paired":paired,"top1":repairs,"KD":errs,"numerical_L2_max":data["numerical_controls"].amax(0).tolist(),
                "sketch_mediation":profiles(data)})
    print(main,means[main],means["regional"],flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    cli.add_argument("--workers",type=int,default=2)
    args=cli.parse_args()
    run(args.workspace,args.workers)
