"""Infer transfer of a candidate's semantic contrasts from evidence outside the withheld region."""
from pathlib import Path
from itertools import product

from .semantic_observation_channel import (CORNERS,atomic_json,cluster_effect,complete_windows,load_frozen_clip,
    read_corners,revision,save_tensor,scene_bank,semantic_inputs)
import torch
import numpy as np


NAMES=('persistent','fixed','anchor','visibility','swapped','context_mean','previous_visibility')


def anchor_forecast(q,b,anchor,target):
    seen=[k for k in range(4) if k not in (anchor,target)]
    context=q[:,anchor,None,None]
    h0=context.expand(-1,b.size(1),4,-1)
    h1=(context.clamp_min(1e-30).log()+b-b[:,:,anchor,None]).softmax(-1)
    def event(forecast):
        return (forecast[:,:,seen]*q[:,None,seen]).sum(-1).clamp_min(1e-30).log().sum(2)
    support=(event(h1)-event(h0)).sigmoid()
    return h0[:,:,target],h1[:,:,target],support


def forecasts(q,prob,target):
    lb=prob.clamp_min(1e-30).log()
    seen=[j for j in range(4) if j!=target]
    outcomes=[]
    support=[]
    for a in seen:
        h0,h1,weight=anchor_forecast(q,lb,a,target)
        mixed=torch.lerp(h0,h1,weight[:,:,None])
        wrong=torch.lerp(h0,h1,weight.flip(1)[:,:,None])
        outcomes.append(torch.stack((mixed,(h0+h1)/2,h0,h1,wrong),2))
        support.append(weight)
    neutral=q[:,seen].clamp_min(1e-30).log().mean(1)
    remaining=(neutral[:,None]+lb[:,:,target]-lb[:,:,seen].mean(2)).softmax(-1)
    return (torch.cat((torch.stack(outcomes).mean(0),
        neutral.softmax(-1)[:,None,None].expand(-1,prob.size(1),1,-1),remaining[:,:,None]),2),
        torch.stack(support).mean(0))


def check():
    g=torch.Generator().manual_seed(42)
    q=torch.randn(3,4,5,generator=g,dtype=torch.double).softmax(-1)
    p=torch.randn(3,2,4,5,generator=g,dtype=torch.double).softmax(-1)
    x,w=forecasts(q,p,3)
    altered=q.clone()
    altered[:,3]=altered[:,3].roll(1,-1)
    torch.testing.assert_close(x,forecasts(altered,p,3)[0])
    torch.testing.assert_close(x.sum(-1),torch.ones_like(x[...,0]))
    torch.testing.assert_close(x,forecasts(q,(p.log()+torch.rand(3,2,1,5,generator=g)).softmax(-1),3)[0])
    s=q[0];b=p[0,0];h0=s[0].expand_as(b)
    h1=(s[0].log()+b.log()-b[0].log()).softmax(-1)
    exact=[]
    for h in (h0,h1):
        exact.append(sum(float(h[1,u]*h[2,v]*s[1,u]*s[2,v]) for u,v in product(range(5),repeat=2)))
    got=anchor_forecast(q,p.log(),0,3)[2][0,0]
    assert abs(float(got)-exact[1]/sum(exact))<1e-12


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    check()
    w=Path('/content/sbir')
    root=w/'results/semantic-reporting-regime'
    root.mkdir(parents=True,exist_ok=True)
    inputs=semantic_inputs(w,'sketchy')
    cfg,_,qs,pp,gs,_,p,src,tgt,scale=inputs
    views=complete_windows(w,'sketchy',inputs)
    vision=load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda')).visual
    corners=read_corners(w,pp)
    for split,(qi,labels) in qs.items():
        pi,_=gs[split]
        rec=torch.load(w/'results/semantic-observation-channel'/(split+'.pt'),weights_only=True)
        assert torch.equal(qi,rec['qi']) and torch.equal(pi,rec['pi'])
        tr=rec['trials']
        used=tr[:,1:3].unique(sorted=True)
        heads,_=scene_bank(w,split,pp,pi,p,tgt,scale,vision)
        bank=(scale*(corners[pi[used]].cuda().float() @ tgt[heads].T)).softmax(-1).mean(2)
        qread={}
        allpred=[]
        weights=[]
        for batch in tr.split(96):
            ix=batch[:,0]
            obs=(scale*(views['regions'][qi[ix]][:,CORNERS].cuda().float() @ src[heads].T)).softmax(-1).mean(2)
            for i,v in zip(ix.tolist(),obs.cpu()):
                qread[i]=v
            photo=bank[torch.searchsorted(used,batch[:,1:3].contiguous()).cuda()]
            result=[forecasts(obs,photo,t) for t in range(4)]
            allpred.append(torch.stack([v[0] for v in result],1).cpu())
            weights.append(torch.stack([v[1] for v in result],1).cpu())
        # Build targets only after every candidate forecast is frozen.
        preds=torch.cat(allpred).clamp_min(1e-30).log()
        del allpred
        scores=[]
        for target_name in ('own','redraw'):
            records=tr[tr[:,3]==0]
            groups={int(s):records[records[:,1]==s,0].tolist() for s in records[:,1].unique()}
            truths=[]
            for j,s in tr[:,:2].tolist():
                truths.append(qread[j] if target_name=='own' else torch.stack([qread[m] for m in groups[s] if m!=j]).mean(0))
            truth=torch.stack(truths)
            scores.append((preds*truth[:,:,None,None]).sum(-1).mean(1).transpose(1,2))
        vals=torch.stack(scores,1).double()
        # The all-visible-region controls reconstruct independently scored own-sketch rows.
        torch.testing.assert_close(vals[:,0,5],rec['scores'][:,7],atol=2e-5,rtol=2e-6)
        torch.testing.assert_close(vals[:,0,6],rec['scores'][:,5],atol=2e-5,rtol=2e-6)
        report={}
        for m,mode in enumerate(('same','different')):
            sel=tr[:,3]==m
            trial=tr[sel].numpy()
            for j,name in enumerate(('own','redraw')):
                v=vals[sel,j].numpy()
                d=v[:,:,0]-v[:,:,1]
                summaries={}
                for k in range(1,len(NAMES)):
                    change=v[:,0]-v[:,k]
                    summaries[NAMES[k]]=dict(true=cluster_effect(change[:,0],trial[:,4]),
                        foil=cluster_effect(change[:,1],trial[:,4]),
                        contrast=cluster_effect(d[:,0]-d[:,k],trial[:,4]))
                decisions=[]
                for pid in np.unique(trial[:,4]):
                    ins=trial[:,4]==pid
                    a,b=np.unique(trial[ins,1])
                    margin=d[ins&(trial[:,1]==a),None]+d[ins&(trial[:,1]==b)][None]
                    decisions.append(((margin>0)+.5*(margin==0)).mean((0,1)))
                report[mode+'/'+name]=dict(n=len(v),differences=summaries,
                    mean_log_scores=v.mean(0).tolist(),assignment=np.mean(decisions,0).tolist())
        save_tensor(dict(qi=qi,pi=pi,trials=tr,scores=vals,weights=torch.cat(weights)),root/(split+'.pt'))
        atomic_json(root/(split+'.json'),dict(commit=revision(),gpu=torch.cuda.get_device_name(),seed=42,
            methods=list(NAMES),assessment=report))
        print(split,report['same/redraw']['differences']['fixed'],flush=True)


if __name__=='__main__':
    run()
