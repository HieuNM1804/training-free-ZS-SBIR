"""Is directional circulation of lexical evidence useful for independent cross-domain queries?"""
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .branch_view_study import PrefixReplay
from .finite_gallery_response import retrieval
from .gallery_context_reversals import coordinates
from .gallery_region_panels import CONFIG, Index
from .public_context_standardization import endpoint_matrix
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .standardized_context_audit import audit_inputs


def stationary(kernel):
    n=len(kernel)
    pi=kernel.new_full((n,),1/n)
    for step in range(2400):
        after=pi@kernel
        after/=after.sum()
        residual=float(((after-pi).abs()/pi.clamp_min(1e-30)).max())
        pi=after
        if residual<3e-6:
            break
    error=float(((pi@kernel)/pi-1).abs().max())
    if error>3e-5:
        raise ValueError(f'Effective gallery kernel lacks a sufficiently precise stationary distribution: {error}')
    return pi,step+1,error


def decompose_flow(kernel,pi):
    n=len(pi)
    imbalance=0.
    energy=0.
    for start in range(0,n,128):
        sl=slice(start,start+128)
        forward=pi[sl,None]*kernel[sl]
        backward=pi[None,:]*kernel[:,sl].T
        imbalance+=float((forward-backward).abs().sum())
        energy+=float(((forward-backward).square()/(forward+backward).clamp_min(1e-30)).sum())
    return {'stationary_flow_tv':imbalance/2,'stationary_triangular_discrimination':energy}


@torch.inference_mode()
def run():
    w=Path('/content/sbir')
    qs,gs,encoded,cp,q,p,flip,src,tgt,scale=audit_inputs(w,768)
    fixture=torch.load(w/'results/gallery-context-reversals/fixed-boundaries.pt',weights_only=True)
    if not torch.equal(cp,fixture['photos']):raise ValueError('Gallery identity changed')
    shared=fixture['queries']
    fixed_rows,jl,jr,sign=[fixture[k] for k in ('rows','left','right','orientation')]
    kd,_=load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    result={}
    pair_margins={}
    for s,(qi,labels) in qs.items():
        sample=torch.randperm(len(qi),generator=torch.Generator().manual_seed(87243))[:768].sort().values
        requested=torch.cat([shared,qi[sample]]).unique(sorted=True)
        where=coordinates(encoded,requested)
        eval_rows=coordinates(requested,qi[sample])
        shared_rows=coordinates(requested,shared)
        pi_ids,gold=gs[s]
        cols=coordinates(pi_ids,cp).cuda()
        g=F.normalize(p[pi_ids].cuda())
        bank=torch.cat([(scale*(part@tgt.T)).softmax(1) for part in g.split(256)])
        graph=Index(SimpleNamespace(p=g),rgb_only=True)
        k=endpoint_matrix(graph)
        del graph
        pi,iterations,error=stationary(k)
        debug=decompose_flow(k,pi)
        debug.update({'stationarity_max_relative_error':error,'iterations':iterations,
                      'effective_gallery_size_fraction':float(1/(pi.square().sum()*len(pi)))})
        print(s,'flow',debug,flush=True)
        marginal=k.sum(0)
        values={}
        for idx in where.split(96):
            a=(scale*(q[idx].cuda()@src.T)).softmax(1)
            b=(scale*(flip[idx].float().cuda()@src.T)).softmax(1)
            e=((a+b)/2)@bank.T
            fwd=e@k.T
            # Time reversal is well-defined under the gallery's own stationary
            # measure. This calculation avoids materializing another n by n matrix.
            rev=((e*pi)@k)/pi
            naive=(e@k)/marginal
            sym=(fwd+rev)/2
            # Reassign the asymmetry's magnitude to its stationary counterpart
            # only in the reversible treatment. The transpose control uses equal
            # initial photo mass; row-wise symmetrization keeps the arithmetic form.
            for name,v in [('forward',fwd),('reversible',sym),('reverse',rev),
                           ('uniform_reverse',naive),('uniform_symmetric',(fwd+naive)/2)]:
                if not v.isfinite().all():raise ValueError(name)
                values.setdefault(name,[]).append(v.clamp_min(1e-30).log().cpu())
        replay=PrefixReplay(kd,gold,s)
        per_query={}
        pair_margins[s]={}
        top={}
        for name,batches in values.items():
            score=torch.cat(batches)
            chosen=score[eval_rows].cuda()
            per_query[name]=retrieval(chosen,labels[sample],gold,s,replay)
            top[name]=gold[chosen.argmax(1).cpu()]==labels[sample]
            head=score[shared_rows][:,cols.cpu()]
            pair_margins[s][name]=(head[fixed_rows,jl]-head[fixed_rows,jr])*sign
        original=per_query['forward']
        y=labels[sample]
        controls={}
        for name,z in per_query.items():
            d=(z-original)*100
            macro=torch.stack([d[y==cl].mean(0) for cl in y.unique()])
            boot=torch.randint(len(macro),(10000,len(macro)),generator=torch.Generator().manual_seed(287))
            controls[name]={'metrics':(100*z.mean(0)).tolist(),'class_mean_diff':macro.mean(0).tolist(),
                'class_ci95':macro[boot].mean(1).quantile(torch.tensor([.025,.975],dtype=macro.dtype),dim=0).tolist(),
                'top1_fixed_damaged':[int((~top['forward']&top[name]).sum()),int((top['forward']&~top[name]).sum())]}
        result[s]={'graph':debug,'rows':controls,'gallery':len(g),'queries':len(sample)}
        print(s,'official',controls['reversible']['metrics'],flush=True)
        del k,bank,values
    a,b=pair_margins.values()
    result['same_inputs']={}
    for name in a:
        result['same_inputs'][name]={'pairs':len(a[name]),'flips':int((a[name]*b[name]<0).sum()),
           'ties':int((a[name]*b[name]==0).sum()),'correct':[int((a[name]>0).sum()),int((b[name]>0).sum())]}
    da=a['forward']-a['reversible']
    db=b['forward']-b['reversible']
    total=a['forward']-b['forward']
    odd=da-db
    result['same_inputs']['circulation_change']={'rms_total_margin_drift':float(total.square().mean().sqrt()),
        'rms_odd_part_change_on_log_score_scale':float(odd.square().mean().sqrt()),
        'proportion_with_same_drift_direction':float((total*odd>0).float().mean())}
    out=w/'results/gallery-flow-direction'
    out.mkdir(exist_ok=True,parents=True)
    (out/'sketchy.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':
    run()
