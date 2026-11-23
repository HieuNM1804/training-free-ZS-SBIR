"""Does the visual component supported by noun embeddings add class evidence to the noun scores?"""
import argparse
import hashlib
import inspect
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F

from .basic_isolated_channels import assess_pair,comparison,harris,rescale,summarize,uniform_ranks
from .branch_view_study import PrefixReplay
from .encoding import save_tensor
from .gallery_panel_resume import semantic_inputs
from .locked_geometry_words import score_rows
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd


@torch.inference_mode()
def run(ds,pilot):
    w=Path('/content/sbir')
    root=w/'results/basic-isolated-channels'
    cfg,_,qs,_,gs,q,p,src,tgt,scale=semantic_inputs(w,ds)
    kd,_=load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    mid=(src+tgt)/2;zero=mid-mid.mean(0);delta=(src-tgt)/2
    cov=zero.T@zero/len(mid)
    val,eig=torch.linalg.eigh(.8*cov-.2*(delta.T@delta/len(mid)))
    rank=min(250,int((val>0).sum()))
    uv=eig[:,-rank:]
    pc=torch.linalg.eigh(cov).eigenvectors[:,-rank:]
    rnd=torch.linalg.qr(torch.randn(512,rank,generator=torch.Generator().manual_seed(6632)).cuda()).Q
    tag='pilot' if pilot else 'full'
    out={}
    for s,(qi,y) in qs.items():
        chosen=torch.randperm(len(qi),generator=torch.Generator().manual_seed(87243))[:768].sort().values if pilot else torch.arange(len(qi))
        if s=='tuberlin' and (pilot or len(y.unique())!=30):raise ValueError('Only complete TU')
        pi,gl=gs[s];cl=gl.cuda()
        g=F.normalize(p[pi].cuda());gm=g.mean(0);centered=g-gm
        bank=torch.cat([(scale*(p_@tgt.T)).softmax(1) for p_ in g.split(160)])
        replay=PrefixReplay(kd,gl,s)
        result,diagnostics={},{}
        identity=0.
        with ThreadPoolExecutor(max_workers=min(2,len(os.sched_getaffinity(0)))) as pool:
            for i in range(0,len(chosen),64):
                rows=chosen[i:i+64]
                x=F.normalize(q[qi[rows]].cuda())
                z=x-gm
                post=(scale*(x@src.T)).softmax(1)
                sem=post@bank.T
                text=(F.normalize(post@tgt)-tgt.mean(0))@centered.T
                align=(z@uv)@(centered@uv).T
                remaining=(z-(z@uv)@uv.T)@centered.T
                identity=max(identity,float((z@centered.T-align-remaining).abs().max()))
                v,sn,st=rescale(align),rescale(sem.clamp_min(1e-30).log()),rescale(text)
                vr,sr=uniform_ranks(align),uniform_ranks(sem)
                scores={'noun_overlap':sem,'center_text':text,'projected':align,'complement':remaining,
                    'sum_overlap':(v+sn)/2,'product_overlap':v*sn,'harris_overlap':harris(v,sn),
                    'sum_text':(v+st)/2,'harris_text':harris(v,st),
                    'remaining_harris':harris(rescale(remaining),sn),
                    'rank_harris':harris(vr,sr),'rank_linear75':.25*vr+.75*sr}
                if pilot:
                    scores.update({'random_subspace':(z@rnd)@(centered@rnd).T,'noun_pca':(z@pc)@(centered@pc).T,
                        'linear75':.25*v+.75*sn,'linear90':.1*v+.9*sn,'remaining_only_sum':(rescale(remaining)+sn)/2})
                for key,value in scores.items():
                    result.setdefault(key,[]).append(score_rows(value,y[rows],cl,s,replay,pool))
                for n,a,b in [('p_word',align,sem),('r_word',remaining,sem),('p_text',align,text)]:
                    diagnostics.setdefault(n,[]).append(assess_pair(a,b,y[rows].cuda(),cl))
                if i%4096==0:print(s,i,len(chosen),flush=True)
        if identity>5e-5:raise ValueError('Projected parts do not close')
        raw={k:torch.cat(v) for k,v in result.items()}
        dd={k:torch.cat(v) for k,v in diagnostics.items()}
        report={'queries':len(chosen),'gallery':len(g),'rank':rank,'closure_max_error':identity,
            'rows':summarize(raw,y[chosen],'noun_overlap'),
            'complementarity':{k:comparison(raw,dd[k],a,b,y[chosen]) for k,a,b in
                [('p_word','projected','noun_overlap'),('r_word','complement','noun_overlap'),('p_text','projected','center_text')]}}
        out[s]=report
        save_tensor({'ids':qi[chosen],'labels':y[chosen],'rows':raw,'diagnostics':dd},root/(tag+'-projection-'+s+'.pt'))
        (root/(tag+'-projection-'+s+'.json')).write_text(json.dumps(report,indent=2))
        print(s,report['rows']['harris_overlap']['metrics'],flush=True)
    (root/(tag+'-projection-'+ds+'.json')).write_text(json.dumps(out,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',required=True,choices=['sketchy','tuberlin'])
    ap.add_argument('--pilot',action='store_true')
    args=ap.parse_args()
    run(args.dataset,args.pilot)
