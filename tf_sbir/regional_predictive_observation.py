"""Predict a withheld semantic observation from a different source region and one candidate photo."""
from pathlib import Path
from hashlib import sha256
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
from .depiction_pivot_geometry import native_patches
from .geometry_word_flat import (CHECKPOINT, atomic_json, digest, index, load_frozen_clip, revision,
    save_tensor, semantic_inputs)
from .regional_interventions import windows
from .regional_scale_position_contrast import complete_windows


def scene_bank(w,split,pp,pi,p,tgt,scale,vision):
    paths=[str(pp[i]) for i in pi.tolist()]
    stamp=digest({'gallery':paths,'model':CHECKPOINT,'source':'regional-64/RGB16'})
    heads,_=index(w,split,p[pi].cuda(),tgt,scale,stamp[:16])
    signature=digest(dict(stamp=stamp,heads=heads.tolist(),recipe='native-grid/crop3-mean/post64/v1'))
    out=w/'cache'/'predictive-photo-words'/(signature+'.pt')
    if out.exists():
        record=torch.load(out,weights_only=True,mmap=True)
        assert torch.equal(record['pi'],pi) and torch.equal(record['heads'],heads.cpu())
        return heads,record['words'].cuda()
    patches=native_patches(paths,vision,w/'cache/depiction-pivot-geometry')
    coords=torch.arange(49).reshape(7,7)
    ids=torch.stack([coords[y:y+3,x:x+3].reshape(-1) for y,x in windows(3)]).cuda()
    rows=[]
    for tokens in patches.split(192):
        prob=(scale*(tokens.cuda().float() @ tgt[heads].T)).softmax(-1)
        rows.append(prob[:,ids].mean(2).cpu())
    region=torch.cat(rows)
    out.parent.mkdir(parents=True,exist_ok=True)
    save_tensor(dict(pi=pi,heads=heads.cpu(),words=region),out)
    return heads,region.cuda()


def observation_tasks():
    for y,x in ((0,0),(0,2),(2,0),(2,2)):
        anchor=3*y+x
        for dy,dx in ((0,2-2*x),(2-2*y,0)):
            target=(y+dy)*3+(x+dx)
            cells=[(i,j) for i in range(3) for j in range(3) if 0<=i+dy<3 and 0<=j+dx<3]
            source=[3*i+j for i,j in cells]
            dest=[3*(i+dy)+j+dx for i,j in cells]
            assert len(source)==3 and set(source).isdisjoint(dest)
            yield anchor,target,source,dest


def predictions(anchor,photo,source,destination):
    # anchor: batch × words, photo: batch × candidates × locations × words.
    context=photo[:,:,source]
    future=photo[:,:,destination]
    mass=(context*anchor[:,None,None]).sum(-1)
    mass=mass/mass.sum(-1,keepdim=True).clamp_min(1e-30)
    def marginal(weights):
        return (weights[:,:,:,None]*future).sum(2)
    return {'conditional':marginal(mass),'marginal':future.mean(2),
            'shuffle1':marginal(mass.roll(1,2)),'shuffle2':marginal(mass.roll(2,2))}


def cluster_effect(values,pair_ids):
    ids,idx=np.unique(pair_ids,return_inverse=True)
    counts=np.bincount(idx)
    summed=np.bincount(idx,weights=values)
    rng=np.random.default_rng(42)
    picks=rng.integers(0,len(ids),(4000,len(ids)))
    stat=summed[picks].sum(1)/counts[picks].sum(1)
    return {'mean':float(values.mean()),'bootstrap95':np.quantile(stat,[.025,.975]).tolist(),'pairs':len(ids)}


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    w=Path('/content/sbir')
    out=w/'results/regional-predictive-observation'
    out.mkdir(parents=True,exist_ok=True)
    inputs=semantic_inputs(w,'sketchy')
    cfg,_,qs,pp,gs,_,p,src,tgt,scale=inputs
    views=complete_windows(w,'sketchy',inputs)
    vision=load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda')).visual
    for split,(qi,labels) in qs.items():
        pi,_=gs[split]
        design=torch.load(w/'results/semantic-identification'/(split+'.pt'),mmap=True,weights_only=True)
        assert torch.equal(qi,design['qi']) and torch.equal(pi,design['pi'])
        trials=design['trials']
        heads,bank=scene_bank(w,split,pp,pi,p,tgt,scale,vision)
        all_scores=[]
        for batch in trials.split(96):
            ix,own,other=batch[:,:3].T
            obs=(scale*(views['regions'][qi[ix],:9].float().cuda() @ src[heads].T)).softmax(-1).mean(2)
            photos=bank[torch.stack((own,other),1)]
            scores=[]
            for a,b,loc,dest in observation_tasks():
                pred=predictions(obs[:,a],photos,loc,dest)
                def logscore(key):
                    return (obs[:,b,None,:]*pred[key].clamp_min(1e-30).log()).sum(-1)
                scores.append(torch.stack((logscore('conditional'),logscore('marginal'),
                              .5*(logscore('shuffle1')+logscore('shuffle2'))),1))
            all_scores.append(torch.stack(scores,1).mean(1).cpu())
        rows=torch.cat(all_scores).double()
        bounds=design['values'][:,3:5]
        unresolved=(bounds[:,0]<=1e-11)&(bounds[:,1]>=-1e-11)
        report={}
        for m,name in enumerate(('same_category','different_category')):
            for group,mask in [('all',trials[:,3]==m),('global_unresolved',(trials[:,3]==m)&unresolved)]:
                subset=rows[mask].numpy()
                pairs=trials[mask,4].numpy()
                effect={}
                for j,control in [(1,'marginal'),(2,'shuffle')]:
                    d=subset[:,0]-subset[:,j]
                    effect[control]={k:cluster_effect(v,pairs) for k,v in (
                        ('true_log_gain',d[:,0]),('foil_log_gain',d[:,1]),('difference',d[:,0]-d[:,1]))}
                report[name+'/'+group]=dict(n=len(subset),mean_scores=subset.mean(0).tolist(),contrasts=effect,
                    pair_accuracy=(subset[:,:,0]>subset[:,:,1]).mean(0).tolist())
        save_tensor(dict(qi=qi,pi=pi,trials=trials,scores=rows,heads=heads.cpu()),out/(split+'.pt'))
        atomic_json(out/(split+'.json'),dict(commit=revision(),device=torch.cuda.get_device_name(),seed=42,
            methods=['conditional','marginal','shuffle'],split=split,assessment=report))
        print(split,report['same_category/all']['contrasts'],flush=True)


if __name__=='__main__':
    run()
