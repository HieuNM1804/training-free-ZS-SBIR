"""Quantify the information available to a two-word placement decision on actual retrieval candidates."""
from pathlib import Path
import numpy as np
from .semantic_grounded_word_events import (CORNERS,PAIRS,atomic_json,complete_windows,local_posterior,make_word_bank,
    odds_contrast,read_corners,revision,semantic_inputs)
import torch


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    w=Path('/content/sbir')
    out=w/'results/semantic-grounded-word-events'
    inputs=semantic_inputs(w,'sketchy')
    _,_,qs,pp,gs,q,p,src,tgt,scale=inputs
    views=complete_windows(w,'sketchy',inputs)
    photo_corners=read_corners(w,pp)
    full_report={}
    names=['admitted_mass_squared','distinct_mass','source_second','photo_second','cauchy_bound','signed',
           'absolute','relative_alignment','multiplier']
    for split in qs:
        prev=torch.load(out/(split+'.pt'),weights_only=True)
        ids=prev['qi']
        pi,classes=gs[split]
        assert torch.equal(pi,prev['pi'])
        bank=make_word_bank(p[pi].cuda(),tgt,scale,128)
        values,heads=bank.topk(32)
        target=[]
        for first in range(0,len(pi),96):
            sl=slice(first,first+96)
            prob=local_posterior(photo_corners[pi[sl]].cuda(),tgt,scale)
            target.append(prob.gather(2,heads[sl,None,:].expand(-1,4,-1)).clamp_min(1e-30).log())
        photo=torch.cat(target)
        rng=np.random.default_rng(42)
        uniform=torch.from_numpy(np.stack([rng.choice(len(pi),200,replace=False) for _ in ids]))
        result=[]
        hit=[]
        for batch in torch.arange(len(ids)).split(8):
            qi=ids[batch]
            poses=torch.stack((q[qi],views['flipped'][qi].float()),1).cuda()
            native=local_posterior(poses,src,scale)
            global_score=native @ bank.T
            near=global_score.topk(200).indices
            chosen=torch.cat((near,uniform[batch].cuda()),1)
            hit.append((classes[chosen.cpu()]==prev['labels'][batch,None]))
            qread=local_posterior(views['regions'][qi][:,CORNERS].cuda(),src,scale).clamp_min(1e-30).log()
            head=heads[chosen]
            n,ng,k=head.shape
            h=native[:,None].expand(-1,ng,-1).gather(2,head)*values[chosen]/global_score.gather(1,chosen)[:,:,None]
            mass=h.sum(-1).square()
            distinct=mass-h.square().sum(-1)
            h2=h[:,:,:,None]*h[:,:,None,:]
            qlocal=qread[:,None].expand(-1,ng,-1,-1).gather(3,head[:,:,None].expand(-1,-1,4,-1))
            dlocal=photo[chosen]
            c=torch.zeros_like(mass)
            sq,sd=c.clone(),c.clone()
            for i,j in PAIRS:
                u=odds_contrast(qlocal,i,j)
                v=odds_contrast(dlocal,i,j)
                for total,term in ((c,u*v),(sq,u.square()),(sd,v.square())):
                    total+=(h2*term).sum((-1,-2))/len(PAIRS)
            bound=(sq*sd).sqrt()
            if (c.abs()-bound).max()>2e-6 or (bound-distinct).max()>2e-6:
                raise ValueError('Violation of identifiability bounds')
            cosine=torch.where(bound>1e-9,c.abs()/bound.clamp_min(1e-30),torch.nan)
            result.append(torch.stack((mass,distinct,sq,sd,bound,c,c.abs(),cosine,(1+c).sqrt()),-1).cpu())
        out_val=torch.cat(result).double().numpy()
        hits=torch.cat(hit).numpy()
        report={}
        near_mask=np.broadcast_to(np.arange(400)<200,hits.shape)
        for name,mask in [('native_top200',near_mask),('uniform',~near_mask),
                          ('native_relevant',near_mask&hits),('native_irrelevant',near_mask&~hits)]:
            v=out_val[mask]
            report[name]={'pairs':len(v),'means':dict(zip(names,np.nanmean(v,0).tolist())),
                'quantiles':dict(zip(names,np.nanquantile(v,[.1,.5,.9],axis=0).T.tolist())),
                'fraction_undefined_alignment':float(np.isnan(v[:,7]).mean()),
                'fraction_change_above_one_percent':float((np.abs(v[:,-1]-1)>.01).mean())}
        full_report[split]=report
        print(split,report['native_top200']['means'],flush=True)
    atomic_json(out/'identifiability.json',dict(commit=revision(),seed=42,assessment=full_report))


if __name__=='__main__':
    run()
