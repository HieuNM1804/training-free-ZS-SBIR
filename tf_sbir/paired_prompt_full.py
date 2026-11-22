"""Full test of enhancing depiction-dependent evidence after the invariance pilot."""

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
from .conditional_semantic_index import SIZES, bare_words
from .crossdomain_benchmark import digest, revision
from .crossdomain_inputs import CHECKPOINT
from .encoding import save_tensor
from .foundation import load_words
from .map_diagnosis import map_values
from .paired_prompt_inputs import prepare
from .paired_prompt_intervention import fit_panels, intervention, preserve_spread, reference_bases, text_reference
from .paired_prompt_study import FAMILY, log_readout
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_diagnosis import global_inputs
from .retrieval_evidence import load_kd, paired_delta
from .semantic import read_words


KINDS = ("heads","rotate","exchange","text","unshrunk")
BLENDS = (-1.,-.6,-.3,-.1,.1,.3,.6,1.)


def suffix(key):
    return key.split("/",1)[1]


def selections(pilot):
    mean = pilot["means"]
    rows = set()
    for kind, number in (("opposite",3),("heads",2),("text",1),("unshrunk",1),("rotate",1),("exchange",1)):
        choices = sorted((k for k in mean if k.startswith(kind+"/")),key=mean.get,reverse=True)
        rows.update(suffix(k) for k in choices[:number])
    return sorted(rows)


def all_cells(params):
    return ["regional",*(f"mix/{v:g}" for v in BLENDS),
            *(f"{kind}/{sign}/{p}" for kind in KINDS for sign in ("reduce","enhance") for p in params)]


def edited_batch(q, heads, source, target, axes, perm, sign, cells):
    mid = (source+target)*.5
    noun = (q@mid.T).topk(192).indices
    bases = reference_bases(heads,axes,perm,sign)
    plan = defaultdict(list)
    for k in cells:
        if k=="regional" or k.startswith("mix/"):
            continue
        kind,direction,width,ridge,strength,fraction=k.split("/")
        family = "heads" if kind=="unshrunk" else kind
        plan[int(width),family].append((float(ridge),float(strength),float(fraction),kind,direction,k))
    for (width, family), configs in plan.items():
        names = noun[:,:width]
        basis = bases[family]
        panels = fit_panels(q,basis,source[names],target[names],family=="exchange")
        for ridge,strength in sorted({c[:2] for c in configs}):
            step,raw,_=intervention(basis,panels,ridge,strength)
            for r,st,frac,kind,direction,key in configs:
                if (r,st)!=(ridge,strength):
                    continue
                change = raw if kind=="unshrunk" else step
                if direction=="enhance":
                    change = -change
                yield key, preserve_spread(q,frac*change,mid[names])


def rankscores(q,heads,base,bank,source,target,scale,axes,perm,sign,cells):
    old = q@source.T*scale
    ref = log_readout(old,bank)
    yield "regional",base
    text_mid = q@((source+target)*.5).T*scale
    for k in cells:
        if k.startswith("mix/"):
            weight=float(suffix(k))
            correction=log_readout(torch.lerp(old,text_mid,weight),bank)-ref
            yield k, base+.65*correction
    block,keys=[],[]
    def flush():
        padded=block+[block[-1]]*(6-len(block))
        likelihood=log_readout(torch.cat(padded)@source.T*scale,bank).split(len(q))
        for k,adjusted in zip(keys,likelihood[:len(keys)],strict=True):
            yield k,base+.65*(adjusted-ref)
        block.clear()
        keys.clear()
    for k,vec in edited_batch(q,heads,source,target,axes,perm,sign,cells):
        block.append(vec)
        keys.append(k)
        if len(block)==6:
            yield from flush()
    if block:
        yield from flush()


def cached_full(w,ds,cfg,inputs,source,target,scale):
    expect = {s:len(x[0]) for s,x in inputs[1].items()}
    home = w/"cache"/FAMILY/"full"
    if not all((home/(s+".json")).is_file() and (home/(s+".pt")).is_file() for s in expect):
        samples={s:torch.arange(length) for s,length in expect.items()}
        data,positions,info=prepare(w,FAMILY,ds,cfg,inputs,source,target,scale,samples)
        for s,(pack,y,g) in data.items():
            if not torch.equal(positions[s],samples[s]) or (len(y),len(g),len(g.unique()))!=SIZES[s]:
                raise ValueError("Incorrect split rows in preparation")
            path=home/(s+".pt")
            save_tensor(pack,path)
            atomic_json(path.with_suffix(".json"),{"source":info,"split":s,"queries":len(y),"gallery":len(g),
                                                 "vocabulary":checksum(cfg.vocabulary)})
    for s,length in expect.items():
        p=home/(s+".pt")
        meta=read(p.with_suffix(".json"))
        pack=torch.load(p,weights_only=True,mmap=True)
        if (meta["split"]!=s or meta["queries"]!=length or meta["vocabulary"]!=checksum(cfg.vocabulary)
                or tuple(pack["regional_score"].shape)!=SIZES[s][:2]
                or tuple(pack["selected_heads"].shape)!=(length,48,512)):
            raise ValueError("Incompatible cached full representations")
        yield s,pack,inputs[1][s][1],inputs[3][s][1]


def evaluate(s,data,y,g,src,tgt,scale,bank,cells,kd,workers):
    replayer=Replay(kd,g,s) if kd else None
    axes=text_reference(src,tgt,48)
    rng=torch.Generator().manual_seed(8129)
    perm=torch.randperm(src.size(1),generator=rng).cuda()
    sign=(2*torch.randint(2,(src.size(1),),generator=rng)-1).float().cuda()
    metrics,correct,errors=defaultdict(list),defaultdict(list),{}
    gr=g.cuda()
    with (ThreadPoolExecutor(max_workers=workers) if kd else nullcontext()) as pool:
        for start in range(0,len(y),64):
            loc=slice(start,start+64)
            labels=y[loc]
            scored=rankscores(data["query_native"][loc].cuda().float(),data["selected_heads"][loc].cuda().float(),
                              data["regional_score"][loc].cuda(),bank,src,tgt,scale,axes,perm,sign,cells)
            for k,row in scored:
                approximate,order,rel,_=map_values(row,labels.cuda(),gr,s)
                result=approximate.cpu()
                if kd:
                    actual=replayer.score(order,labels,pool)
                    error=float((actual-result).abs().max())
                    errors[k]=max(error,errors.get(k,0.))
                    if error>2e-6:
                        raise ValueError("Original KD-SBIR evaluator differs")
                    result=actual
                    correct[k].append(rel[:,0].cpu())
                metrics[k].append(result)
            if start%1536==0:
                print("audit" if kd else "lock",s,start,len(y),len(cells),flush=True)
    if set(metrics)!=set(cells):
        raise ValueError("Missing scores")
    return {k:torch.cat(v) for k,v in metrics.items()},{k:torch.cat(v) for k,v in correct.items()},errors


@torch.inference_mode()
def run(w,phase,dataset,workers):
    root=w/"results"/FAMILY
    src_doc=Path(__file__).resolve().parents[1]/"docs/results"/FAMILY/"pilot.json"
    pilot=read(src_doc)
    params=selections(pilot)
    cells=all_cells(params)
    sig_data={"revision":revision(),"primary":"heads/enhance","settings":params,"cells":cells,
              "pilot":checksum(src_doc),"checkpoint":CHECKPOINT,"evaluator":KD_FUNCTION_SHA,
              "selection":"equal full Sketchy mAP followed by equal precision"}
    signature=digest(sig_data)
    if phase!="cache":
        path=root/"full-protocol.json"
        if path.exists() and read(path)["sha256"]!=signature:
            raise ValueError("Full protocol/revision changed")
        atomic_json(path,sig_data|{"sha256":signature})
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:
        raise ValueError("Changed official scoring function")
    if phase=="audit":
        frozen=read(root/"freeze.json")
        if frozen["revision"]!=revision() or frozen["full_sketchy_sha256"]!=checksum(root/"shortlist.json"):
            raise ValueError("Frozen choices do not match full Sketchy")
        primary=frozen["primary"]
        suffix_params=primary.split("/",2)[2]
        cells=sorted(set(["regional",primary,frozen["simple"],frozen["best_control"],frozen["best_reduce"],
                          frozen["negative_text"],*(f"{kind}/enhance/{suffix_params}" for kind in KINDS),
                          f"heads/reduce/{suffix_params}"]))
    for ds in ((dataset,) if phase=="cache" else (("sketchy","tuberlin") if phase=="audit" else ("sketchy",))):
        cfg,*inputs=global_inputs(w,ds)
        names=read_words(cfg.vocabulary)
        if len(names)!=20931:
            raise ValueError("Changed public dictionary")
        src,tgt=[v.cuda() for v in load_words(cfg,names)]
        _,scale=bare_words(cfg,names)
        for s,packed,y,g in cached_full(w,ds,cfg,inputs,src,tgt,scale):
            if phase=="cache":
                print("cache",s,len(y),len(g),flush=True)
                continue
            dst=root/((s if phase=="audit" else "shortlist-"+s)+".json")
            if dst.is_file():
                saved=read(dst)
                if saved["signature"]!=signature or (phase=="audit" and saved["per_query_sha256"]!=checksum(dst.with_suffix(".pt"))):
                    raise ValueError("Stale result")
                continue
            bank=torch.load(w/"cache/pose-conditional-layout"/(s+"-gallery-uniform.pt"),weights_only=True,mmap=True).cuda()
            torch.backends.cuda.matmul.allow_tf32=True
            metrics,top,errors=evaluate(s,packed,y,g,src,tgt,scale,bank,cells,kd if phase=="audit" else None,workers)
            vals={k:v.mean(0).tolist() for k,v in metrics.items()}
            report={"signature":signature,"queries":len(y),"gallery":len(g),"values":vals}
            if phase=="audit":
                archive=dst.with_suffix(".pt")
                save_tensor({"metrics":metrics,"query_labels":y,"top1_correct":top},archive)
                report.update({"primary":primary,"baseline":"regional","matched_base":frozen["simple"],"best_contrast":frozen["best_control"],
                               "protocol":{"split":s,"gpu":torch.cuda.get_device_name(),"revision":revision(),"checkpoint":CHECKPOINT,
                                           "classes":len(g.unique()),"vocabulary_sha256":checksum(cfg.vocabulary),"evaluator":KD_FUNCTION_SHA},
                               "scores":{k:dict(zip(("mAP","precision"),v)) for k,v in vals.items()},
                               "vs_controls":{k:paired_delta(metrics[primary],v,y) for k,v in metrics.items()},
                               "per_query_sha256":checksum(archive),"original_kd_queries_per_condition":len(y),"kd_max_abs_error":errors,
                               "top1_changes":{"errors_fixed":int((~top["regional"]&top[primary]).sum()),
                                               "hits_broken":int((top["regional"]&~top[primary]).sum())}})
            atomic_json(dst,report)
            print("stored",s,"regional",vals["regional"],"best",max(vals,key=vals.get),flush=True)
            del bank
    if phase=="lock":
        tables={s:read(root/("shortlist-"+s+".json"))["values"] for s in ("sketchy_1","sketchy_2")}
        mean={k:[sum(t[k][i] for t in tables.values())/2 for i in (0,1)] for k in cells}
        def best(predicate):
            return max((k for k in mean if predicate(k)),key=mean.get)
        primary=best(lambda k:k.startswith("heads/enhance/"))
        atomic_json(root/"shortlist.json",{"splits":tables,"means":mean})
        frozen={"revision":revision(),"primary":primary,"simple":best(lambda k:k=="regional" or k.startswith("mix/")),
                "best_control":best(lambda k:not k.startswith("heads/enhance/")),
                "best_reduce":best(lambda k:k.startswith("heads/reduce/")),
                "negative_text":best(lambda k:k.startswith("mix/-")),"full_sketchy_sha256":checksum(root/"shortlist.json")}
        atomic_json(root/"freeze.json",frozen|{"sha256":digest(frozen)})
        print("LOCK",primary,mean[primary],frozen["best_control"],mean[frozen["best_control"]],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--phase",choices=("cache","lock","audit"),required=True)
    p.add_argument("--dataset",choices=("sketchy","tuberlin"),default="sketchy")
    p.add_argument("--workers",type=int,default=8)
    a=p.parse_args()
    run(a.workspace,a.phase,a.dataset,a.workers)
