"""Transport an unobserved sketch operation through a finite distribution over geometric maps."""
from itertools import combinations,permutations
from pathlib import Path
import os

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .benchmark import merge_inputs
from .concept_benchmark import atomic_json
from .crossdomain_benchmark import revision
from .crossdomain_inputs import runtime
from .data import Images,SKETCHY_SPLITS,configure_image_worker,normal_transform
from .encoding import load_frozen_clip,save_tensor
from .foundation import load_words
from .regional_predictive_observation import cluster_effect
from .semantic import read_words


PAIRS=list(combinations(range(4),2))
TEST=((0,5),(1,4),(2,3))
ARMS=('square','registered','unrestricted','uniform','known_frame')
FLIP=(1,0,3,2)


def structure(device):
    grid=torch.arange(49).reshape(7,7)
    corners=torch.stack([grid[y:y+3,x:x+3].flatten() for y in (0,4) for x in (0,4)])
    indices=[torch.arange(49)]
    for a,b in PAIRS:
        swap=torch.arange(49)
        swap[corners[a]],swap[corners[b]]=corners[b],corners[a]
        assert torch.equal(swap[swap],torch.arange(49))
        indices.append(swap)
    perm=torch.tensor(list(permutations(range(4))))
    d=torch.cdist(torch.tensor([[0,0],[0,1],[1,0],[1,1]],dtype=torch.float32),
                  torch.tensor([[0,0],[0,1],[1,0],[1,1]],dtype=torch.float32))
    square=torch.tensor([torch.equal(d,d[g][:,g]) for g in perm])
    action=torch.tensor([[PAIRS.index(tuple(sorted((int(g[a]),int(g[b]))))) for a,b in PAIRS] for g in perm])
    assert int(square.sum())==8
    for gi,g in enumerate(perm):
        for mask_id,(a,b) in enumerate(PAIRS):
            t=torch.arange(4)
            t[a],t[b]=torch.tensor(b),torch.tensor(a)
            transported=g[t][g.argsort()]
            assert sorted(torch.where(transported!=torch.arange(4))[0].tolist())==list(PAIRS[action[gi,mask_id]])
    return [v.to(device) for v in (corners,torch.stack(indices),perm,square,action)]


def encode(paths,mirror,model,structure):
    corners,order,*_=structure
    visual=model.visual
    loader=DataLoader(Images(paths,normal_transform()),batch_size=34,num_workers=min(2,len(os.sched_getaffinity(0))),
        pin_memory=True,worker_init_fn=configure_image_worker)
    embeddings,descriptors=[],[]
    replay=0.
    for pixels in loader:
        pixels=pixels.cuda(non_blocking=True).to(visual.conv1.weight.dtype)
        if mirror:pixels=pixels.flip([-1])
        blocks=pixels.reshape(-1,3,7,32,7,32).permute(0,2,4,1,3,5).reshape(-1,49,3,32,32)
        swapped=blocks[:,order].reshape(-1,7,7,3,32,32).permute(0,3,1,4,2,5).reshape(-1,3,224,224)
        if not embeddings:
            indexed=swapped.reshape(len(pixels),7,3,224,224)
            torch.testing.assert_close(indexed[:,0],pixels,atol=0,rtol=0)
            explicit=pixels.clone()
            explicit[:,:,:96,:96]=pixels[:,:,:96,128:]
            explicit[:,:,:96,128:]=pixels[:,:,:96,:96]
            torch.testing.assert_close(indexed[:,1],explicit,atol=0,rtol=0)
        states=visual.transformer(visual._embeds(swapped))
        z=F.normalize((visual.ln_post(states[:,0])@visual.proj).float(),dim=-1).reshape(len(pixels),7,512)
        patch=F.normalize((visual.ln_post(states[::7,1:])@visual.proj).float(),dim=-1)[:,corners.flatten()]
        if not embeddings:
            replay=float((z[:,0]-F.normalize(visual(pixels).float(),dim=-1)).abs().max())
            if replay>.0005: raise ValueError(f'Native readout discrepancy {replay}')
        embeddings.append(z.half().cpu())
        descriptors.append(patch.half().cpu())
    return torch.cat(embeddings),torch.cat(descriptors),replay


def posterior(q,p,scale,perm,square):
    sim=scale*(F.normalize(q.float(),dim=-1)@F.normalize(p.float(),dim=-1).transpose(1,2))
    def log_blocks(similarity):
        return similarity.softmax(2).reshape(-1,4,9,4,9).sum(4).mean(2).clamp_min(1e-30).log()
    correspondence=(log_blocks(sim)+log_blocks(sim.transpose(1,2)).transpose(1,2))*.5
    ll=correspondence[:,torch.arange(4,device=q.device),perm].sum(2)
    return ll.masked_fill(~square,-torch.inf).softmax(1),ll.softmax(1)


def outcomes(emb,src,scale,qids):
    logits=scale*(F.normalize(emb[:,1:].float(),dim=-1)@src.T)
    distribution=logits.softmax(-1)
    draws=[]
    for j,idx in enumerate(qids.tolist()):
        gen=torch.Generator(device=src.device).manual_seed(42+int(idx))
        draws.append(torch.multinomial(distribution[j],2048,replacement=True,generator=gen))
    return logits,torch.stack(draws)


def assign(logits,draws,action):
    # For each presented pair of words return the assignment probability under every correspondence.
    output=[]
    for a,b in TEST:
        contrast=logits[:,action[:,a]]-logits[:,action[:,b]]
        indices=lambda j:draws[:,j,None].expand(-1,len(action),-1)
        output.append((contrast.gather(2,indices(a))-contrast.gather(2,indices(b))).sigmoid())
    return torch.stack(output,1)


def risk(prob):
    return torch.stack(((1-prob).square().mean((1,2)),((prob>.5).float()+.5*(prob==.5)).mean((1,2))),1)


def summary(meta,raw,oracle):
    out={}
    for mode,name in enumerate(('same','different')):
        ok=meta[:,4]==mode
        ids=meta[ok,5].numpy()
        x=raw[ok,:,:,0].numpy()
        margin=x[:,:,1]-x[:,:,0]
        out[name]=dict(sources=len(x),pairs=len(np.unique(ids)),true_brier=x[:,:,0].mean(0).tolist(),
            source_accuracy=((margin>1e-7)+.5*(np.abs(margin)<=1e-7)).mean(0).tolist(),
            assignment_accuracy=raw[ok,:,0,1].mean(0).tolist(),oracle=oracle[ok].mean(0).tolist(),
            comparisons={k:dict(true=cluster_effect(x[:,j,0]-x[:,0,0],ids),
                contrast=cluster_effect(margin[:,0]-margin[:,j],ids)) for j,k in enumerate(ARMS) if j})
    return out


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    w=Path('/content/sbir')
    root=w/'results/gcwe-transported-intervention'
    root.mkdir(parents=True,exist_ok=True)
    cfg=runtime(w,'sketchy')
    paths,qs=merge_inputs(cfg.root,SKETCHY_SPLITS,'sketch')
    ph,gs=merge_inputs(cfg.root,SKETCHY_SPLITS,'photo')
    records={s:torch.load(w/'results/regional-semantic-interactions'/(s+'.pt'),weights_only=True) for s in qs}
    for s,r in records.items():
        assert torch.equal(r['qi'],qs[s][0]) and torch.equal(r['pi'],gs[s][0])
    qids=torch.cat([r['meta'][:,0] for r in records.values()]).unique()
    pids=torch.cat([r['meta'][:,2:4].flatten() for r in records.values()]).unique()
    src,tgt=[v.cuda() for v in load_words(cfg,read_words(cfg.vocabulary))]
    model=load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda'))
    scale=model.logit_scale.exp().float()
    parts=structure(src.device)
    _,_,perm,square,action=parts
    identity=int(torch.where((perm==torch.arange(4,device=src.device)).all(1))[0].item())
    flipped=int(torch.where((perm==torch.tensor(FLIP,device=src.device)).all(1))[0].item())
    photos=encode([ph[i] for i in pids.tolist()],False,model,parts)
    sketches=[encode([paths[i] for i in qids.tolist()],mirror,model,parts) for mirror in (False,True)]
    for s,rec in records.items():
        meta=rec['meta']
        stored,analysis={},{}
        for mirror,(qe,qd,qerror) in enumerate(sketches):
            rows,oracles,geometry=[],[],[]
            for b in meta.split(10):
                qpos=torch.searchsorted(qids,b[:,0].contiguous())
                logq,draws=outcomes(qe[qpos].cuda(),src,scale,b[:,0])
                oracles.append(risk(assign(logq,draws,action[identity:identity+1]).squeeze(2)).cpu())
                sc,gscores=[],[]
                for ids in (b[:,2],b[:,3]):
                    pos=torch.searchsorted(pids,ids.contiguous())
                    logp=scale*(F.normalize(photos[0][pos,1:].cuda().float(),dim=-1)@tgt.T)
                    post,all_post=posterior(qd[qpos].cuda(),photos[1][pos].cuda(),scale,perm,square)
                    predicted=assign(logp,draws,action)
                    uniform=predicted[:,:,square].mean(2)
                    torch.testing.assert_close(uniform,torch.full_like(uniform,.5),atol=2e-6,rtol=0)
                    neutral=torch.full_like(uniform,.5)
                    frame=flipped if mirror else identity
                    pr=[(predicted*post[:,None,:,None]).sum(2),predicted[:,:,identity],
                        (predicted*all_post[:,None,:,None]).sum(2),neutral,predicted[:,:,frame]]
                    sc.append(torch.stack([risk(z) for z in pr],1).cpu())
                    gscores.append(torch.stack((post[:,frame],post.amax(1),all_post[:,square].sum(1)),1).cpu())
                rows.append(torch.stack(sc,2))
                geometry.append(torch.stack(gscores,2))
            raw,oracle,geo=torch.cat(rows).double(),torch.cat(oracles).double(),torch.cat(geometry).double()
            key=('native','flipped')[mirror]
            stored[key]=dict(scores=raw,oracle=oracle,geometry=geo)
            analysis[key]=summary(meta,raw,oracle)
            analysis[key]['geometry_means']=geo.mean(0).tolist()
        save_tensor(dict(meta=meta,qi=qs[s][0],pi=gs[s][0],rows=stored),root/(s+'.pt'))
        atomic_json(root/(s+'.json'),dict(commit=revision(),gpu=torch.cuda.get_device_name(),seed=42,arms=ARMS,
            replay_error=[photos[2],sketches[0][2],sketches[1][2]],geometry_columns=['known_frame_mass','peak_mass','unrestricted_square_mass'],
            assessment=analysis))
        print(s,analysis['native']['same']['true_brier'],flush=True)


if __name__=='__main__':
    run()
