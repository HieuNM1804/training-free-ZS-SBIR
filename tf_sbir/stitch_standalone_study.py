"""Do multiple possible sketch nouns match distinct gallery views, or is simple averaging sufficient?"""
import argparse
import hashlib
import inspect
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F

from .basic_isolated_channels import summarize
from .branch_view_study import PrefixReplay
from .encoding import save_tensor
from .gallery_panel_resume import semantic_inputs
from .locked_geometry_words import score_rows
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .semantic_transport import conditional_similarity
from .stitch_single_views import load_views

TAU=.07


def set_scores(c,checks=False):
    # c: query, image, public noun, RGB view. Uniform nouns are competing readings.
    row=c.mean(-1,keepdim=True)
    col=c.mean(-2,keepdim=True)
    mean=row.mean(-2,keepdim=True)
    overall=mean.squeeze(-1).squeeze(-1)
    fw=((c/TAU).softmax(-1)*c).sum(-1).mean(-1)
    bw=((c/TAU).softmax(-2)*c).sum(-2).mean(-1)
    ct=(fw+bw)/2
    # Additive matrix has the exact same per-noun and per-view means as c.
    # Its transport can be evaluated without materializing another 4-D matrix.
    cm=col.squeeze(-2)
    rm=row.squeeze(-1)
    null=.5*((cm/TAU).softmax(-1)*cm).sum(-1)+.5*((rm/TAU).softmax(-1)*rm).sum(-1)
    shared=TAU*(torch.logsumexp(rm/TAU,-1)-math.log(rm.size(-1)))
    independent=TAU*(torch.logsumexp(c/TAU,-2)-math.log(c.size(-2))).mean(-1)
    out={'top1_original':c[...,0,0],'top1_multiview':c[...,0,:].mean(-1),
        'five_nouns_original':c[...,0].mean(-1),'set_mean':overall,
        'noun_to_views':fw,'views_to_nouns':bw,'bidirectional':ct,
        'additive_transport':null,'interaction_plus_mean':overall+ct-null,
        'shared_noun':shared,'independent_noun':independent}
    if checks:
        matches=(conditional_similarity(c,TAU)-ct).abs().max()
        additive=(conditional_similarity(row+col-mean,TAU)-null).abs().max()
        post_star=(rm/TAU).log_softmax(-1)
        post_view=(c/TAU).log_softmax(-2)
        gap=TAU*((post_star.exp()[...,None]*(post_star[...,None]-post_view)).sum(-2)).mean(-1)
        errors=[float(matches),float(additive),float((independent-shared-gap).abs().max())]
        if max(errors)>2e-5:raise ValueError('Set alignment identity failed')
        return out,errors
    return out,None


@torch.inference_mode()
def run(ds,pilot):
    w=Path('/content/sbir')
    root=w/'results/stitch-isolated-alignment'
    root.mkdir(parents=True,exist_ok=True)
    _,_,qs,photos,gs,q,p,src,tgt,_=semantic_inputs(w,ds)
    images=load_views(w,ds,photos)
    kd,_=load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    if ds=='tuberlin' and pilot:raise ValueError('Full TU transfer only')
    reports={}
    tag='pilot' if pilot else 'full'
    for s,(qi,y) in qs.items():
        rows=torch.randperm(len(qi),generator=torch.Generator().manual_seed(87243))[:768].sort().values if pilot else torch.arange(len(qi))
        pi,gl=gs[s];cl=gl.cuda()
        bank=F.normalize(images[pi].cuda().float(),dim=-1)
        aligned=(1-(bank[:,0]*F.normalize(p[pi].cuda())).sum(-1)).abs().max().item()
        if aligned>.002:raise ValueError(f'Original view changed {aligned}')
        replay=PrefixReplay(kd,gl,s)
        collected={}
        errors=[0.,0.,0.]
        signature='v1:5-uniform-source-nouns:10-rgb-views-v2:tau=0.07'
        cache=root/(tag+'-'+s+'-shards');cache.mkdir(exist_ok=True)
        with ThreadPoolExecutor(max_workers=min(2,len(os.sched_getaffinity(0)))) as pool:
            for begin in range(0,len(rows),384):
                end=min(begin+384,len(rows))
                file=cache/f'{begin:06d}.pt'
                if file.exists():
                    part=torch.load(file,weights_only=True)
                    assert part['spec']==signature and torch.equal(part['qi'],qi[rows[begin:end]])
                else:
                    measured={}
                    for start in range(begin,end,48):
                        select=rows[start:min(end,start+48)]
                        sketch=F.normalize(q[qi[select]].cuda())
                        nouns=(sketch@src.T).topk(5).indices
                        lex=tgt[nouns].flatten(0,1)
                        pieces={}
                        for first in range(0,len(bank),512):
                            z=bank[first:first+512]
                            hard=(lex@z.flatten(0,1).T).view(len(select),5,len(z),10).transpose(1,2)
                            scored,err=set_scores(hard,checks=(start==begin and first==0))
                            if err:errors=[max(a,b) for a,b in zip(errors,err)]
                            for key,value in scored.items():
                                pieces.setdefault(key,[]).append(value)
                        for key,piece in pieces.items():
                            val=score_rows(torch.cat(piece,1),y[select],cl,s,replay,pool)
                            measured.setdefault(key,[]).append(val)
                    part={'spec':signature,'qi':qi[rows[begin:end]],'rows':{k:torch.cat(v) for k,v in measured.items()},'checks':errors.copy()}
                    save_tensor(part,file)
                for k,v in part['rows'].items():collected.setdefault(k,[]).append(v)
                errors=[max(a,b) for a,b in zip(errors,part['checks'])]
                print(s,end,len(rows),flush=True)
        raw={k:torch.cat(v) for k,v in collected.items()}
        report={'queries':len(rows),'gallery':len(gl),'classes':len(cl.unique()),'tau':TAU,'original_view_max_error':aligned,
            'max_ct_null_kl_errors':errors,'rows':summarize(raw,y[rows],'set_mean')}
        reports[s]=report
        save_tensor({'qi':qi[rows],'labels':y[rows],'rows':raw},root/(tag+'-'+s+'.pt'))
        (root/(tag+'-'+s+'.json')).write_text(json.dumps(report,indent=2))
    (root/(tag+'-'+ds+'-summary.json')).write_text(json.dumps(reports,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['sketchy','tuberlin'],required=True);ap.add_argument('--pilot',action='store_true')
    args=ap.parse_args()
    run(args.dataset,args.pilot)
