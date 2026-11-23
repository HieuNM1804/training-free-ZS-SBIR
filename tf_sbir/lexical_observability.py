"""Does CLIP retain class evidence in directions poorly measured by a sketch's own word distribution?"""
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .branch_view_study import PrefixReplay
from .gallery_context_reversals import coordinates
from .gallery_region_panels import CONFIG,Index
from .positive_foil import raw_ranks
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .standardized_context_audit import audit_inputs
from .strong_pilot_scores import location


def low_variance_part(x,prob,terms):
    # C is the posterior-weighted covariance of normalized text embeddings.
    # Compute lambda (C+lambda I)^-1 x without materializing a matrix per sketch.
    mean=prob@terms
    trace=prob@terms.square().sum(1)-mean.square().sum(1)
    lam=(trace/x.size(1)).clamp_min(1e-9)[:,None]
    def operator(v):
        centered=v@terms.T-(v*mean).sum(1,keepdim=True)
        weighted=prob*centered
        return weighted@terms-weighted.sum(1,keepdim=True)*mean+lam*v
    rhs=x*lam
    z=torch.zeros_like(x)
    err=rhs.clone()
    search=err.clone()
    sq=err.square().sum(1,keepdim=True)
    rhs_sq=sq.clone()
    for _ in range(96):
        active=sq>rhs_sq*1e-10
        if not active.any():break
        grad=operator(search)
        step=torch.where(active,sq/(search*grad).sum(1,keepdim=True).clamp_min(1e-30),0.)
        z+=step*search
        err-=step*grad
        new=err.square().sum(1,keepdim=True)
        search=err+torch.where(active,new/sq.clamp_min(1e-30),0.)*search
        sq=new
    rel=(operator(z)-rhs).norm(dim=1)/rhs.norm(dim=1).clamp_min(1e-12)
    if float(rel.max())>.002:raise ValueError(f'Unresolved shift-invert system {float(rel.max())}')
    return z,rel


def normalize_scores(x):
    return (x-x.mean(1,keepdim=True))/x.std(1,keepdim=True,unbiased=False).clamp_min(1e-12)


def symmetric_kl(base,direction,terms,scale):
    # Finite perturbations with equal feature norms and opposite signs.
    v=F.normalize(direction)
    a=(scale*(F.normalize(base+.05*v)@terms.T)).log_softmax(1)
    b=(scale*(F.normalize(base-.05*v)@terms.T)).log_softmax(1)
    return ((a.exp()-b.exp())*(a-b)).sum(1)/2


@torch.inference_mode()
def run():
    w=Path('/content/sbir')
    qs,gs,encoded,_,q,p,flip,src,tgt,scale=audit_inputs(w,768)
    kd,_=load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    dimension=src.size(1)
    rng=torch.Generator().manual_seed(82917)
    rot=torch.randperm(dimension,generator=rng).cuda()
    signs=(torch.randint(2,(dimension,),generator=rng).cuda()*2-1).float()
    rotated=src[:,rot]*signs
    centered=src-src.mean(0)
    covariance=centered.T@centered/len(src)
    reg=covariance.trace()/dimension
    system=covariance+reg*torch.eye(dimension,device=src.device)
    folder=w/'results/lexical-observability-residual'
    folder.mkdir(exist_ok=True,parents=True)
    results={}
    for s,(qi,y) in qs.items():
        rows=torch.randperm(len(qi),generator=torch.Generator().manual_seed(87243))[:768].sort().values
        where=coordinates(encoded,qi[rows])
        pi,gl=gs[s]
        pack=torch.load(location(w,s,rows),mmap=True,weights_only=True)
        assert torch.equal(qi[rows],pack['qi']) and torch.equal(pi,pack['pi'])
        g=F.normalize(p[pi].cuda())
        bank=torch.cat([(scale*(v@tgt.T)).softmax(1) for v in g.split(256)])
        graph=Index(SimpleNamespace(p=g),rgb_only=True)
        replay=PrefixReplay(kd,gl,s)
        vals={}
        diag=[]
        exact_error=0.
        for start in range(0,len(rows),48):
            ix=where[start:start+48]
            sl=slice(start,start+len(ix))
            actual=y[rows[sl]]
            base=pack['score'][sl].cuda()
            a=q[ix].cuda()
            b=flip[ix].float().cuda()
            x=F.normalize((a+b)/2)
            posterior=(scale*(x@src.T)).softmax(1)
            lo,err=low_variance_part(x,posterior,src)
            hi=x-lo
            sham,_=low_variance_part(x,posterior,rotated)
            global_lo=torch.linalg.solve(system,reg*x.T).T
            evidence=((scale*(a@src.T)).softmax(1)+(scale*(b@src.T)).softmax(1))@bank.T/2
            native=graph.score(CONFIG,evidence).clamp_min(1e-30).log()
            logits_std=base.std(1,keepdim=True,unbiased=False)
            def record(key,score):
                nonlocal exact_error
                rank,fast=raw_ranks(score,actual,gl.cuda(),s)
                precise=replay.chunk((rank.cpu(),actual))
                exact_error=max(exact_error,float((precise-fast).abs().max()))
                if exact_error>2e-6:raise ValueError('KD mismatch')
                hit=gl.cuda()[rank[:,:200]]==actual.cuda()[:,None]
                vals.setdefault(key,[]).append(torch.cat([precise,hit.any(1).cpu()[:,None],hit[:,:1].cpu()],1))
            record('strong',base)
            record('native_words',native)
            dirs={'raw':x,'weak':lo,'sensitive':hi,'rotated':sham,'uniform_words':global_lo}
            diag.append(torch.stack([lo.norm(dim=1),hi.norm(dim=1),F.cosine_similarity(lo,x),
                  symmetric_kl(x,lo,src,scale),symmetric_kl(x,hi,src,scale),
                  symmetric_kl(x,sham,src,scale),err],1).cpu())
            for name,v in dirs.items():
                visual=F.normalize(v)@g.T
                record('only/'+name,visual)
                visual_z=normalize_scores(visual)
                # A log-unit of standardized visual evidence is allowed to modulate
                # donor agreement on the same photo. The separate-expectation
                # control has identical donor and marginal visual measurements.
                likelihood=visual_z.exp()
                weighted=graph.score(CONFIG,evidence*likelihood).clamp_min(1e-30).log()-native
                separate=graph.score(CONFIG,likelihood).clamp_min(1e-30).log()
                for typ,delta in [('late',visual),('same_donor',weighted),('separate',separate)]:
                    gain=normalize_scores(delta)*logits_std
                    for strength in (.03125,.125,.5):
                        record(f'{name}/{typ}/{strength:g}',base+strength*gain)
        z={k:torch.cat(v).double() for k,v in vals.items()}
        reference=z['strong']
        if (reference[:,:2]-pack['kd'].double()).abs().max()>2e-6:raise ValueError('Strong replay changed')
        report={}
        for key,values in z.items():
            dif=values[:,2:]-reference[:,2:]
            report[key]={'metrics':(100*values[:,:2].mean(0)).tolist(),
                         'top200_recovered_lost':[int((dif[:,0]>0).sum()),int((dif[:,0]<0).sum())],
                         'top1_fixed_damaged':[int((dif[:,1]>0).sum()),int((dif[:,1]<0).sum())]}
        probe=torch.cat(diag)
        results[s]={'queries':len(rows),'gallery':len(gl),'rows':report,
                    'strong_top200_misses':int((reference[:,2]==0).sum()),'max_kd_discrepancy':exact_error,
                    'mean_low_high_norm_cosine_to_raw_and_finite_kl_weak_sensitive_rotated':probe[:,:6].mean(0).tolist(),
                    'max_solver_relative_residual':float(probe[:,6].max())}
        torch.save({'labels':y[rows],'rows':rows,'per_query':z,'probe':probe},folder/(s+'.pt'))
        print(s,'strong',report['strong'],'weak',report['weak/same_donor/0.125'],flush=True)
        del graph,bank,pack
    (folder/'sketchy-pilot.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':
    run()
