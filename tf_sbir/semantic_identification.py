"""Sharp pairwise decision bounds over source word distributions consistent with a single sketch."""
from collections import defaultdict
from pathlib import Path
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
from numba import njit
from scipy.optimize import linprog
import torch
from torch.nn import functional as F

from .geometry_word_flat import atomic_json, make_word_bank, revision, save_tensor, semantic_inputs
from .regional_scale_position_contrast import complete_windows


@njit
def envelope_min(x,y,left,right):
    hull=np.empty(len(x),np.int64)
    size=0
    for i in range(len(x)):
        while size>=2:
            a,b=hull[size-2],hull[size-1]
            if (y[b]-y[a])*(x[i]-x[b]) < (y[i]-y[b])*(x[b]-x[a]):
                break
            size-=1
        hull[size]=i
        size+=1
    best=np.inf
    for j in range(size):
        a=hull[j]
        if left<=x[a]<=right:
            best=min(best,y[a])
        if j+1<size:
            b=hull[j+1]
            lo,hi=max(left,x[a]),min(right,x[b])
            if lo<=hi:
                cut=lo if y[b]>=y[a] else hi
                best=min(best,y[a]+(y[b]-y[a])*(cut-x[a])/(x[b]-x[a]))
    return best


@njit
def sharp_bounds(logits,diffs,lo,hi):
    batch,words=logits.shape
    result=np.empty((batch,4))
    for i in range(batch):
        x=np.empty(words)
        y=np.empty(words)
        used=0
        total=0.
        for j in range(words):
            total+=diffs[i,j]
            if j==words-1 or logits[i,j]!=logits[i,j+1]:
                x[used]=(j+2)/2.
                y[used]=total/(j+1)
                used+=1
        x,y=x[:used],y[:used]
        a,b=max(x[0],lo[i]),min(x[-1],hi[i])
        if a>b+1e-9:
            raise ValueError('Infeasible native rank interval')
        b=max(b,a)
        result[i,0]=envelope_min(x,y,a,b)
        result[i,1]=-envelope_min(x,-y,a,b)
        result[i,2]=y.min()
        result[i,3]=y.max()
    return result


def verify_bounds():
    rng=np.random.default_rng(42)
    for _ in range(20):
        logits=np.sort(rng.integers(0,8,16))[::-1].astype(float)
        d=rng.normal(size=len(logits))
        endpoints=np.flatnonzero(np.r_[np.diff(logits)!=0,True])
        k=endpoints+1
        x=(k+1)/2
        y=d.cumsum()[endpoints]/k
        a,b=np.sort(rng.uniform(x.min(),x.max(),2))
        constraints=dict(A_ub=np.stack([x,-x]),b_ub=[b,-a],A_eq=np.ones((1,len(x))),b_eq=[1],bounds=(0,None))
        exact=[linprog(z,**constraints) for z in (y,-y)]
        assert all(v.success for v in exact)
        f=lambda di:sharp_bounds(logits[None],di[None],np.array([a]),np.array([b]))[0]
        np.testing.assert_allclose(f(d),[exact[0].fun,-exact[1].fun,y.min(),y.max()],atol=1e-10)
        np.testing.assert_allclose(f(d),-f(-d)[[1,0,3,2]],atol=1e-12)
        np.testing.assert_allclose(f(np.zeros_like(d)),0)


def greedy_pairs(ids,features,labels=None):
    ids=np.asarray(sorted(ids))
    similarity=features[ids] @ features[ids].T
    a,b=np.triu_indices(len(ids),1)
    if labels is not None:
        ok=labels[ids[a]]!=labels[ids[b]]
        a,b=a[ok],b[ok]
    order=np.lexsort((ids[b],ids[a],-similarity[a,b].numpy()))
    paired=[]
    taken=set()
    for i in order:
        u,v=int(ids[a[i]]),int(ids[b[i]])
        if u not in taken and v not in taken:
            paired.append((u,v))
            taken.update((u,v))
    return paired


def reference_trials(qp,pp,qi,pi,labels,gl,features):
    keys=[(Path(pp[i]).parent.name,Path(pp[i]).stem) for i in pi.tolist()]
    assert len(set(keys))==len(keys)
    lookup={key:j for j,key in enumerate(keys)}
    redraws=defaultdict(list)
    for i,qid in enumerate(qi.tolist()):
        path=Path(qp[qid])
        j=lookup[(path.parent.name,path.stem.rsplit('-',1)[0])]
        assert labels[i]==gl[j]
        redraws[j].append(i)
    rng=np.random.default_rng(42)
    within=[]
    for cat in gl.unique().tolist():
        ids=[i for i,draws in redraws.items() if int(gl[i])==cat and len(draws)>=2]
        pairs=greedy_pairs(ids,features)
        within.extend(pairs[i] for i in rng.choice(len(pairs),12,replace=False))
    across=greedy_pairs([i for pair in within for i in pair],features,gl.numpy())
    trials=[]
    for mode,pairs in enumerate([within,across]):
        for pair_id,(a,b) in enumerate(pairs):
            for own,other in ((a,b),(b,a)):
                trials.extend([i,own,other,mode,pair_id] for i in redraws[own])
    return torch.tensor(trials)


def diagnostics(ids,own,classes,method,view_margins,bounds):
    margin=view_margins.mean(1)
    eps=1e-11
    lower,upper=bounds[:,0],bounds[:,1]
    sign=np.where(lower>eps,1,np.where(upper< -eps,-1,0))
    if method=='native':
        sign=np.sign(margin)
    if method=='two_views':
        sign=np.where(view_margins.min(1)>0,1,np.where(view_margins.max(1)<0,-1,0))
    chosen=sign!=0
    def summary(pred,mask):
        peer=[]
        indices=defaultdict(list)
        for i,group in enumerate(own):
            indices[int(group)].append(i)
        # Is the reported direction endorsed by an independent drawing's native semantic read?
        native=np.sign(margin)
        support=np.empty(len(own))
        for group in indices.values():
            values=native[group]
            for i in group:
                support[i]=((values==pred[i]).sum()-(native[i]==pred[i]))/(len(group)-1)
        for c in np.unique(classes):
            sel=mask&(classes==c)
            if sel.any():
                peer.append([int(c),int(sel.sum()),float((pred[sel]>0).mean()),float(support[sel].mean())])
        return dict(n=int(mask.sum()),accuracy=float((pred[mask]>0).mean()) if mask.any() else None,
            sibling_endorsement=float(support[mask].mean()) if mask.any() else None,classes=peer)
    return sign,chosen,summary(sign,chosen),summary


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    verify_bounds()
    w=Path('/content/sbir')
    out=w/'results/semantic-identification'
    out.mkdir(exist_ok=True,parents=True)
    inputs=semantic_inputs(w,'sketchy')
    _,qp,qs,pp,gs,q,p,src,tgt,scale=inputs
    views=complete_windows(w,'sketchy',inputs)
    for split,(qi,labels) in qs.items():
        pi,gl=gs[split]
        trials=reference_trials(qp,pp,qi,pi,labels,gl,F.normalize(p[pi].float()))
        bank=make_word_bank(p[pi].cuda(),tgt,scale,192)
        outputs=[]
        ranks=torch.arange(1,len(src)+1,device='cuda',dtype=torch.double)
        for rows in trials.split(48):
            ix,own,other=rows[:,:3].T
            ids=qi[ix]
            poses=torch.stack((q[ids],views['flipped'][ids].float()),1).cuda()
            logits=scale*(poses @ src.T)
            ordered,idx=logits.sort(dim=2,descending=True,stable=True)
            prob=ordered.double().softmax(2)
            m=(prob*ranks).sum(2)
            left,right=m.min(1).values.cpu().numpy(),m.max(1).values.cpu().numpy()
            photos=bank[own].double()
            rivals=bank[other].double()
            photos/=photos.sum(1,keepdim=True)
            rivals/=rivals.sum(1,keepdim=True)
            d=(photos-rivals)[:,None,:].expand_as(prob).gather(2,idx)
            overlap=(photos+rivals)[:,None,:].expand_as(prob).gather(2,idx)
            margins=(d*prob).sum(2)
            confidence=margins.mean(1).abs()/(overlap*prob).sum(2).mean(1).clamp_min(1e-30)
            both=[sharp_bounds(ordered[:,v].cpu().numpy(),d[:,v].cpu().numpy(),left,right) for v in (0,1)]
            envelope=np.stack([np.minimum(both[0][:,0],both[1][:,0]),np.maximum(both[0][:,1],both[1][:,1]),
                np.minimum(both[0][:,2],both[1][:,2]),np.maximum(both[0][:,3],both[1][:,3])],axis=1)
            values=margins.cpu().numpy()
            if np.any(envelope[:,0]>values.min(1)+1e-10) or np.any(envelope[:,1]<values.max(1)-1e-10):
                raise ValueError('Native semantic decisions escaped their own admissible set')
            outputs.append(np.column_stack([values,confidence.cpu().numpy(),envelope,left,right]))
        result=np.concatenate(outputs)
        report={}
        for mode,name in enumerate(('same_category','different_category')):
            which=trials[:,3].numpy()==mode
            sub=result[which]
            ix,own,_,_,pair_id=trials[which].numpy().T
            settings={}
            for method,span in [('identified',sub[:,3:5]),('unrestricted',sub[:,5:7]),('two_views',sub[:,3:5]),('native',sub[:,3:5])]:
                sign,mask,stats,summarize=diagnostics(ix,own,labels[ix].numpy(),method,sub[:,:2],span)
                matched=np.zeros(len(sub),dtype=bool)
                matched[np.argsort(-sub[:,2],kind='stable')[:mask.sum()]]=True
                stats['confidence_matched']=summarize(np.sign(sub[:,:2].mean(1)),matched)
                settings[method]=stats
            report[name]=dict(trials=len(sub),photos=len(np.unique(own)),pairs=len(np.unique(pair_id)),methods=settings)
        save_tensor(dict(qi=qi,pi=pi,labels=labels,trials=trials,values=torch.from_numpy(result)),out/(split+'.pt'))
        atomic_json(out/(split+'.json'),dict(commit=revision(),seed=42,device=torch.cuda.get_device_name(),split=split,
            columns=['first_margin','flip_margin','native_confidence','lower','upper','unrestricted_lower',
                     'unrestricted_upper','min_expected_rank','max_expected_rank'],assessment=report))
        print(split,{m:(v['trials'],v['methods']['identified']['n'],v['methods']['identified']['accuracy'])
                     for m,v in report.items()},flush=True)


if __name__=='__main__':
    run()
