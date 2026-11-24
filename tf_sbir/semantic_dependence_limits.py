"""Sharp dependence-uncertainty intervals given a photo's observable one-region semantics."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import multiprocessing
import os
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
import torch
from torch.nn import functional as F
from .benchmark import merge_inputs
from .concept_benchmark import atomic_json
from .crossdomain_benchmark import revision
from .crossdomain_inputs import runtime
from .data import SKETCHY_SPLITS
from .depiction_path_diagnostic import class_interval
from .encoding import load_frozen_clip,save_tensor
from .foundation import load_words
from .semantic import read_words
from .semantic_common_depiction import query_observations,readings


@lru_cache(None)
def constraints(n):
    return csr_matrix(np.vstack((np.kron(np.eye(n),np.ones((1,n))),
                                 np.kron(np.ones((1,n)),np.eye(n))[:-1])))


def minimum(cost,rho):
    n=len(rho)
    result=linprog(cost.ravel(),A_eq=constraints(n),b_eq=np.r_[rho,rho[:-1]],
        bounds=(0,None),method='highs-ds',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not result.success:raise ValueError(result.message)
    plan=result.x.reshape(n,n)
    assert max(np.abs(plan.sum(0)-rho).max(),np.abs(plan.sum(1)-rho).max())<2e-8
    u,v=np.split(np.r_[result.eqlin.marginals,0.],[n])
    # Shift to a feasible dual, so even the small solver tolerance cannot overstate coverage.
    shift=max(0.,(u[:,None]+v[None,:]-cost).max())+1e-12*(1+np.abs(cost).max())
    dual=rho@(u-shift+v)
    assert (u[:,None]-shift+v[None,:]-cost).max()<1e-10
    gap=float(result.fun-dual)
    if not -2e-8<gap<2e-6:raise ValueError(f'Non-tight transport certificate: {gap}')
    return float(dual),gap


def solve(task):
    cost,rho=task
    rho=rho/rho.sum()
    lower,g1=minimum(cost,rho)
    neg,g2=minimum(-cost,rho)
    assert lower>0 and lower<=1.000001 and -neg>=.999999
    return np.array([np.log(lower),np.log(-neg),max(g1,g2)])


def validate():
    rho=np.array([.3,.7])
    c=np.array([[1.6,.7],[.7,2.]])
    c/=rho@c@rho
    out=solve((c,rho))
    lo=np.array([[0.,.3],[.3,.4]])
    hi=np.diag(rho)
    np.testing.assert_allclose(np.exp(out[:2]),[(lo*c).sum(),(hi*c).sum()],atol=1e-8)
    np.testing.assert_allclose(solve((np.ones_like(c),rho))[:2],0.,atol=1e-8)


def costs(q1,q2,bank):
    # Float64 at the boundary of the finite transport problem.
    x=bank.double()
    marginal=x[:,-1]
    shared=x/x.sum(1,keepdim=True).clamp_min(1e-300)*marginal[:,None]
    rho=shared.sum(-1)
    values=shared/rho[:,:,None]
    a=q1.double()@values.transpose(1,2)
    b=q2.double()@values.transpose(1,2)
    c=(a.transpose(1,2)@b)/a.size(1)
    c=.5*(c+c.transpose(1,2))
    indep=(rho[:,None]@c@rho[:,:,None]).flatten()
    diag=(c.diagonal(dim1=1,dim2=2)*rho).sum(1)
    return c/indep[:,None,None],rho,indep.log(),diag.log()


def assess(meta,limits,base,common):
    first_min=limits[:,0,0]-limits[:,1,1]
    first_max=limits[:,0,1]-limits[:,1,0]
    first=meta[:,3].bool()
    lo=torch.where(first,first_min,-first_max)
    hi=torch.where(first,first_max,-first_min)
    fixed=(lo>1e-7)|(hi < -1e-7)
    correct=(base[:,0]-base[:,1])*(2*first.long()-1)>0
    oracle=hi>=-1e-7
    same=(common[:,0]-common[:,1])*(2*first.long()-1)>0
    count=int(fixed.sum())
    order=(base[:,0]-base[:,1]).abs().argsort(descending=True,stable=True)
    u,n=meta[:,0].unique(return_counts=True)
    both=(meta[:,0,None]==u[n==2][None]).T
    balanced=lambda x:(both.double()@x.double())/2
    diff=100*balanced(oracle.double()-correct.double())
    labels=meta[both.long().argmax(1),5]
    assert ((~correct&fixed)==(hi< -1e-7)).all()
    return dict(pairs=len(meta),both=len(both),fixed=count,correct_fixed=int((correct&fixed).sum()),
        errors_independent=int((~correct).sum()),unfixable_errors=int((hi< -1e-7).sum()),
        errors_with_some_repair=int((oracle&~correct).sum()),uncertain=int((~fixed).sum()),
        shared_changed=int((same!=correct).sum()),mean_log_width=float((limits[:,:,1]-limits[:,:,0]).mean()),
        identified_accuracy=float(correct[fixed].double().mean()) if count else None,
        matched_coverage_accuracy=float(correct[order[:count]].double().mean()) if count else None,
        balanced_independent=float(balanced(correct).mean()),balanced_oracle_upper=float(balanced(oracle).mean()),
        maximum_balanced_gain_points=float(diff.mean()),class_gain=class_interval(diff,labels,seed=42))


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    validate()
    w=Path('/content/sbir')
    root=w/'results/semantic-dependence-limits'
    root.mkdir(parents=True,exist_ok=True)
    cfg=runtime(w,'sketchy')
    paths,qs=merge_inputs(cfg.root,SKETCHY_SPLITS,'sketch')
    ph,gs=merge_inputs(cfg.root,SKETCHY_SPLITS,'photo')
    model=load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda'))
    scale=model.logit_scale.exp().float()
    src,tgt=[v.cuda() for v in load_words(cfg,read_words(cfg.vocabulary))]
    anchors=torch.randperm(len(src),generator=torch.Generator().manual_seed(42))[:2]
    fold=torch.zeros(len(src),device='cuda',dtype=torch.bool)
    fold[(tgt@(tgt[anchors[0]]-tgt[anchors[1]])).argsort(stable=True).tensor_split(2)[0]]=True
    ref={s:torch.load(w/'results/semantic-common-depiction'/(s+'.pt'),weights_only=True) for s in qs}
    qu=torch.cat([r['meta'][:,0] for r in ref.values()]).unique()
    im=torch.cat([r['meta'][:,1:3].flatten() for r in ref.values()]).unique()
    sketches=readings(cfg,model.visual,[paths[i] for i in qu.tolist()],'sketch',{})
    old={}
    for f in (w/'cache/regional-semantic-interactions').glob('*.pt'):
        d=torch.load(f,weights_only=True)
        if Path(d['paths'][0]).parent.parent.name=='photo':old.update(zip(d['paths'],d['features']))
    photos=readings(cfg,model.visual,[ph[i] for i in im.tolist()],'photo',old)
    with ProcessPoolExecutor(max_workers=min(2,len(os.sched_getaffinity(0))),mp_context=multiprocessing.get_context('spawn')) as pool:
        for s,r in ref.items():
            meta=r['meta']
            assert torch.equal(r['pi'],gs[s][0]) and torch.isin(meta[:,0],r['qi']).all()
            matrices,probs,baselines,diagonals=[],[],[],[]
            for b in meta.split(20):
                a,bb=query_observations(sketches[torch.searchsorted(qu,b[:,0].contiguous())].cuda(),src,scale,fold)
                endpoints=[]
                for i in (1,2):
                    p=photos[torch.searchsorted(im,b[:,i].contiguous())].cuda().float()
                    bank=(scale*(F.normalize(p,dim=-1)@tgt.T)).softmax(-1)
                    endpoints.append(costs(a,bb,bank))
                for dest,v in zip((matrices,probs,baselines,diagonals),zip(*endpoints)):
                    dest.append(torch.stack(v,1).cpu())
            c,rho,base,diag=[torch.cat(t) for t in (matrices,probs,baselines,diagonals)]
            torch.testing.assert_close(base,r['scores'][:,1],atol=4e-5,rtol=1e-6)
            torch.testing.assert_close(diag,r['scores'][:,0],atol=4e-5,rtol=1e-6)
            jobs=zip(c.flatten(0,1).numpy(),rho.flatten(0,1).numpy())
            bounds=torch.from_numpy(np.stack(list(pool.map(solve,jobs,chunksize=12)))).reshape(len(meta),2,3)
            limits=bounds[:,:,:2]+base[:,:,None]
            assert ((limits[:,:,0]<=diag+1e-7)&(limits[:,:,1]>=diag-1e-7)).all()
            report=assess(meta,limits,base,diag)
            save_tensor(dict(meta=meta,qi=r['qi'],pi=r['pi'],log_limits=limits,independent=base,
                common=diag,relative_duality_gap=bounds[:,:,2]),root/(s+'.pt'))
            atomic_json(root/(s+'.json'),dict(commit=revision(),gpu=torch.cuda.get_device_name(),seed=42,
                maximum_relative_duality_gap=float(bounds[:,:,2].max()),assessment=report))
            print(s,report,flush=True)


if __name__=='__main__':
    run()
