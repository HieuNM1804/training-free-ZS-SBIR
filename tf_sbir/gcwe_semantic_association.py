"""Exact no-association law of the geometry estimator and factorial retrieval consequences."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
from pathlib import Path

import torch
from torch.nn import functional as F

from .concept_benchmark import atomic_json
from .crossdomain_benchmark import revision
from .depiction_path_diagnostic import class_interval
from .encoding import save_tensor
from .gallery_panel_resume import semantic_inputs
from .gcwe_configuration_report import cached
from .gcwe_semantic_configuration import FAMILY as PARENT, distance_bins, laws
from .geometry_word_flat import KD_FUNCTION_SHA, PrefixReplay, load_kd, make_word_bank, score_rows


FAMILY="gcwe-semantic-association"


def reference_models(bank, hist, ref, background):
    measured,_=laws(bank,hist,ref,background)
    mass=bank.new_zeros(bank.shape[1])
    for b in ref.split(512):
        mass+=bank[b].sum(0)
    average=hist[ref.cpu()].float().mean(0).cuda()/(49*24)
    null=(mass[:,None]*average+background)/(mass+1)[:,None]
    torch.testing.assert_close(null.sum(1),torch.ones_like(mass),rtol=2e-5,atol=1e-6)
    return measured.double().clamp_min(1e-30).log(),null.double().clamp_min(1e-30).log(),average.double()


def responses(prior,obs,loglaw,lognull,average):
    base=prior.double().clamp_min(1e-30).log()
    obs=obs.double()
    fixed=(base+16*(average@lognull.T)).softmax(1)
    null=16*(obs@lognull.T)
    real=16*(obs@loglaw.T)
    association=16*(obs@(loglaw-lognull).T)
    residual=(real-null-association).abs().max().item()
    if residual>1e-9:
        raise ArithmeticError(f"Log-evidence factorization drift {residual}")
    results={"null":(base+null).softmax(1),"fixed_null":fixed,"association":(base+association).softmax(1),
             "joint":(base+real).softmax(1)}
    return {k:v.to(prior) for k,v in results.items()},residual


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    w=Path("/content/sbir")
    root=w/"results"/FAMILY
    _,qp,qs,pp,gs,qf,pf,src,tgt,scale=semantic_inputs(w,"sketchy")
    reference=w/"results"/PARENT
    records={k:torch.load(reference/(k+"-pilot.pt"),weights_only=True,mmap=True) for k in qs}
    all_ids=torch.cat([r["qi"] for r in records.values()]).unique()
    cache=w/"cache"/PARENT
    query_obs=cached([qp[q] for q in all_ids.tolist()],cache)
    ph=cached(pp,cache)
    _,uniform=distance_bins()
    kd,_=load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    for split,r in records.items():
        ids,labels,pi=r["qi"],r["labels"],r["pi"]
        assert torch.equal(pi,gs[split][0])
        gl=gs[split][1]
        bank=make_word_bank(F.normalize(pf[pi].cuda().float(),dim=-1),tgt,scale,192)
        folds=[v.cuda() for v in torch.randperm(len(pi),generator=torch.Generator().manual_seed(42)).tensor_split(2)]
        models=[reference_models(bank,ph[pi],ref,uniform) for ref in reversed(folds)]
        evaluator=PrefixReplay(kd,gl,split)
        rows={}
        residual=0.
        with ThreadPoolExecutor(max_workers=2) as pool:
            for b in torch.arange(len(ids)).split(48):
                prior=(scale*(F.normalize(qf[ids[b]].cuda().float(),dim=-1)@src.T)).softmax(1)
                obs=query_obs[torch.searchsorted(all_ids,ids[b])].cuda().float()/(49*24)
                scores={"word":prior@bank.T}
                for held,model in zip(folds,models):
                    updates,error=responses(prior,obs,*model)
                    residual=max(residual,error)
                    table=bank[held]
                    for arm,post in updates.items():
                        if arm not in scores:
                            scores[arm]=torch.empty_like(scores["word"])
                        scores[arm][:,held]=post@table.T
                for arm,v in scores.items():
                    rows.setdefault(arm,[]).append(score_rows(v,labels[b],gl.cuda(),split,evaluator,pool)[:,:2])
        raw={k:torch.cat(v).double() for k,v in rows.items()}
        drift=100*(raw["joint"]-r["rows"]["joint_count16"])
        assert drift.abs().mean(0).max()<.01
        effects={}
        for name,delta in {
            "joint minus word":raw["joint"]-raw["word"],
            "null minus word":raw["null"]-raw["word"],
            "association minus word":raw["association"]-raw["word"],
            "joint minus null":raw["joint"]-raw["null"],
            "null minus fixed_null":raw["null"]-raw["fixed_null"],
            "rank interaction":raw["joint"]-raw["association"]-raw["null"]+raw["word"],
        }.items():
            effects[name]=dict(points=(100*delta.mean(0)).tolist(),classes=class_interval(100*delta[:,0],labels,seed=42))
        metrics={k:(100*v.mean(0)).tolist() for k,v in raw.items()}
        save_tensor(dict(qi=ids,labels=labels,pi=pi,rows=raw,replay=drift),root/(split+".pt"))
        atomic_json(root/(split+".json"),dict(commit=revision(),seed=42,gpu=torch.cuda.get_device_name(),
                    evaluator=KD_FUNCTION_SHA,queries=len(ids),gallery=len(pi),metrics=metrics,contrasts=effects,
                    maximum_factorization_residual=residual,replay_mean_absolute_points=drift.abs().mean(0).tolist()))
        print(split,metrics,flush=True)
        del bank,models,model,table


if __name__=="__main__":
    run()
