"""Evaluate the cost of losing gallery coverage before selecting a rank change."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
from pathlib import Path
import time

import torch

from .concept_benchmark import atomic_json, checksum, read
from .conditional_agreement_study import Replay
from .conditional_semantic_index import bare_words
from .coverage_transport import Coverage, inspection_cost, intent_weights, populations, utility
from .crossdomain_benchmark import digest, revision
from .crossdomain_inputs import CHECKPOINT
from .encoding import save_tensor
from .foundation import load_words
from .map_diagnosis import map_values
from .paired_prompt_inputs import prepare
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_diagnosis import global_inputs, stratified_sample
from .retrieval_evidence import load_kd, paired_delta
from .semantic import read_words
from .word_conditional_verifier import zscore

FAMILY = "semantic-coverage-transport"
SOURCE_CACHE = "paired-prompt-head-intervention"
ALPHAS = (.035,.12,.35,.7,1.)
CAPS = (.12,.35,1.)
POWERS = (0., .5, 1.)
PENALTIES = (0., .25, 1., 4.)
PRIORS = ("both","average","lexical","visual")
METRICS = ("regional","native","mean",*(f"word{n}" for n in range(1,9)))
ACTIONS = [("regional",0.)] + [(name, a) for name in METRICS[1:] for a in ALPHAS]


def name_action(i):
    name, alpha = ACTIONS[i]
    return "regional" if i == 0 else f"fixed/{name}/{alpha:g}"


def rule_key(kind, prior, cost, risk, cap):
    return f"{kind}/{prior}/{cost:g}/{risk:g}/{cap:g}"


def policies(raw):
    # Dimensions: query, action, inspection cost, posterior endpoint, measured quantity.
    cap = torch.tensor([a for _, a in ACTIONS])
    b = len(raw)
    for j in range(len(ACTIONS)):
        yield name_action(j), torch.full((b,), j, dtype=torch.long)
    for k, power in enumerate(POWERS):
        for kind, signal, loss in (("semantic",0,1),("identity",0,2),("shuffled",3,4)):
            for prior in PRIORS:
                for risk in PENALTIES:
                    by_action = utility(raw[:,:,k,:,signal],raw[:,:,k,:,loss],risk,prior)
                    for maximum in CAPS:
                        valid = cap <= maximum+1e-6
                        choice = by_action.masked_fill(~valid[None],-torch.inf).argmax(1)
                        yield rule_key(kind,prior,power,risk,maximum), choice


def generate_orders(base, native, score_mean, word_photo):
    yield base
    sources = [native, score_mean, *word_photo[:, :8].unbind(1)]
    for source in sources:
        standardized = zscore(source)
        for a in ALPHAS:
            yield (1-a)*base + a*standardized


def schedule_reference(pool, replay, ranks, labels):
    cpu = ranks.to(device="cpu",dtype=torch.int32)
    return [pool.submit(replay.chunk,(p,y)) for p,y in zip(cpu.split(16),labels.split(16),strict=True)]


def measure(split, data, y, g, ids, positions, photos, source, target, scale, kd, pool):
    device = photos.device
    group = populations(photos)
    rng = torch.Generator(device=device).manual_seed(84731)
    shuffled = group[torch.randperm(len(group),device=device,generator=rng)]
    gamma = inspection_cost(len(g),device,POWERS)
    replay = Replay(kd,g,split)
    kept = defaultdict(list)
    checks = torch.zeros(2,device=device)
    kd_error = 0.
    for at in range(0,len(ids),32):
        block = ids[at:at+32]
        slot = positions[block]
        base = zscore(data["regional_score"][slot].cuda().float())
        q = data["query_native"][block].cuda().float()
        native = q @ photos.T
        logits, words = (q@source.T*scale).topk(16)
        word_photo = target[words] @ photos.T
        score_mean = ((.3*logits).softmax(1)[:,:,None]*word_photo).sum(1)
        order0 = base.argsort(dim=1,descending=True,stable=True)
        book = Coverage(order0,group,intent_weights(logits,word_photo,native,group,scale),gamma)
        null = Coverage(order0,shuffled,intent_weights(logits,word_photo,native,shuffled,scale),gamma)
        measures, quick, top, scheduled = [],[],[],[]
        query_labels = y[block]
        gallery_labels = g.cuda()
        for proposed in generate_orders(base,native,score_mean,word_photo):
            actual,order,relevant,_ = map_values(proposed,query_labels.cuda(),gallery_labels,split)
            val,delay,identity,sg,lv = book.read(order)
            shuffle_val,shuffle_delay,_,sg2,lv2 = null.read(order)
            checks = torch.maximum(checks,torch.stack((torch.maximum(sg,sg2),torch.maximum(lv,lv2))))
            measures.append(torch.stack((val,delay,identity,shuffle_val,shuffle_delay),-1).transpose(1,2).cpu())
            quick.append(actual.cpu())
            top.append(relevant[:,0].cpu())
            scheduled.append(schedule_reference(pool,replay,order,query_labels))
        if len(quick)!=len(ACTIONS):
            raise ValueError("Not every candidate was materialized")
        original = torch.stack([torch.cat([job.result() for job in pieces]) for pieces in scheduled],1)
        discrepancy = float((torch.stack(quick,1)-original).abs().max())
        kd_error = max(kd_error,discrepancy)
        if discrepancy>2e-6 or float(checks.max())>3e-5:
            raise ValueError(f"Order/evaluator or transport identity failed: {discrepancy}, {checks.tolist()}")
        raw = torch.stack(measures,1)
        if raw[:,0].abs().max().item()>2e-6:
            raise ValueError("An unchanged ranking acquired regret")
        kept["measures"].append(raw)
        kept["action_metrics"].append(original)
        kept["top_correct"].append(torch.stack(top,1))
        if at%192==0:
            print(split,at,len(ids),"actions",len(ACTIONS),"KD",kd_error,"transport",checks.tolist(),flush=True)
    outcome = {k:torch.cat(v) for k,v in kept.items()}
    outcome.update({"query_ids":ids,"query_labels":y[ids]})
    audit={"original_KD_queries_per_action":len(ids),"actions":len(ACTIONS),"KD_max_abs_error":kd_error,
           "max_signed_coupling_difference":checks[0].item(),"largest_lower_bound_violation":checks[1].item()}
    return outcome,audit


def follow_choice(stored,index):
    return stored["action_metrics"][torch.arange(len(index)),index]


def decision_details(stored,indices):
    row=torch.arange(len(indices))
    base=stored["action_metrics"][:,0]
    chosen=follow_choice(stored,indices)
    old=stored["top_correct"][:,0]
    new=stored["top_correct"][row,indices]
    native=torch.tensor([name == "native" for name,_ in ACTIONS])[indices]
    reduced=chosen[:,1] < base[:,1]-1e-7
    return {"adopted":int((indices!=0).sum()),"native_actions":int(native.sum()),
            "first_errors_fixed":int((~old&new).sum()),"first_hits_broken":int((old&~new).sum()),
            "lost_all_at_precision_cutoff":int(((base[:,1]>0)&(chosen[:,1]==0)).sum()),
            "gained_first_at_precision_cutoff":int(((base[:,1]==0)&(chosen[:,1]>0)).sum()),
            "lost_relevant_at_precision_cutoff":int(reduced.sum()),
            "mean_AP_change_when_precision_falls_pp":float(100*(chosen[reduced,0]-base[reduced,0]).mean()) if reduced.any() else None}


@torch.inference_mode()
def run(w,workers):
    start = time.monotonic()
    cfg,*inputs = global_inputs(w,"sketchy")
    words=read_words(cfg.vocabulary)
    if len(words)!=20931:
        raise ValueError("Different public name bank")
    source,target=(x.cuda() for x in load_words(cfg,words))
    _,scale=bare_words(cfg,words)
    samples={s:stratified_sample(labels,36,63241) for s,(_,labels) in inputs[1].items()}
    data,positions,source_record=prepare(w,SOURCE_CACHE,"sketchy",cfg,inputs,source,target,scale,samples)
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:
        raise ValueError("Changed KD-SBIR evaluator")
    root=w/"results"/FAMILY
    spec={"revision":revision(),"checkpoint":CHECKPOINT,"vocabulary":checksum(cfg.vocabulary),"source":source_record["signature"],
          "actions":ACTIONS,"population_count":64,"cost_exponents":POWERS,"penalties":PENALTIES,"caps":CAPS,
          "priors":PRIORS,"per_class":36,"seed":63241,"evaluator":KD_FUNCTION_SHA,"KD_replay":"every query and action"}
    signature=digest(spec)
    if (root/"protocol.json").exists() and read(root/"protocol.json")["sha256"]!=signature:
        raise ValueError("Study revision changed")
    atomic_json(root/"protocol.json",spec|{"sha256":signature})
    tables={}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for s,(pack,y,g) in data.items():
            dest=root/("pilot-"+s+".pt")
            if dest.is_file() and dest.with_suffix(".json").is_file():
                rec=read(dest.with_suffix(".json"))
                if rec["signature"]!=signature or checksum(dest)!=rec["per_query_sha256"]:
                    raise ValueError("Stale split evidence")
                tables[s]=rec["values"]
                continue
            pi=inputs[3][s][0]
            result,check=measure(s,pack,y,g,samples[s],positions[s],inputs[5][pi].cuda().float(),source,target,scale,kd,pool)
            by_name={k:follow_choice(result,ids).mean(0).tolist() for k,ids in policies(result["measures"])}
            best=result["action_metrics"][:,:,0].argmax(1)
            ceiling=follow_choice(result,best).mean(0).tolist()
            save_tensor(result,dest)
            atomic_json(dest.with_suffix(".json"),{"queries":len(best),"signature":signature,"values":by_name,"gallery":len(g),
                        "action_oracle_for_diagnosis":ceiling,"checks":check,"per_query_sha256":checksum(dest)})
            tables[s]=by_name
    means={k:[sum(t[k][i] for t in tables.values())/len(tables) for i in (0,1)] for k in tables["sketchy_1"]}
    primary=max((k for k in means if k.startswith("semantic/both/") and float(k.split("/")[3])>0),key=means.get)
    _,prior,cost,risk,cap=primary.split("/")
    homologues=[primary,"regional","/".join(("semantic",prior,cost,"0",cap)),primary.replace("semantic/","identity/"),
                primary.replace("semantic/","shuffled/"),primary.replace("/both/","/average/"),
                primary.replace("/both/","/lexical/"),primary.replace("/both/","/visual/")]
    fixed=max((k for k in means if k=="regional" or k.startswith("fixed/")),key=means.get)
    nop=max((k for k in means if k.startswith("semantic/") and k.split("/")[3]=="0"),key=means.get)
    identity=max((k for k in means if k.startswith("identity/")),key=means.get)
    random=max((k for k in means if k.startswith("shuffled/")),key=means.get)
    retained=list(dict.fromkeys([*homologues, fixed,nop,identity,random]))
    info,paired={},{}
    for s in tables:
        stored=torch.load(root/("pilot-"+s+".pt"),mmap=True,weights_only=True)
        decisions={k:v for k,v in policies(stored["measures"]) if k in retained}
        after=follow_choice(stored,decisions[primary])
        info[s]={k:decision_details(stored,v) for k,v in decisions.items()}
        paired[s]={k:paired_delta(after,follow_choice(stored,v),stored["query_labels"]) for k,v in decisions.items()}
    atomic_json(root/"pilot.json",{"nominee":primary,"matched":homologues,"best_fixed":fixed,"best_no_penalty":nop,
                "best_identity":identity,"best_null":random,"means":means,"splits":tables,"decisions":info,
                "paired":paired,"device":torch.cuda.get_device_name(),"elapsed_seconds":time.monotonic()-start})
    print("selected",primary,means[primary],"base",means["regional"],"fixed",fixed,means[fixed],flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--workers",type=int,default=6)
    a=p.parse_args()
    run(a.workspace,a.workers)
