"""Test a shared intervention frame at natural category boundaries without using the drawing's source."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .data import Images,configure_image_worker,normal_transform
from .gcwe_transported_intervention import assign,encode,outcomes,posterior,structure
from .geometry_word_flat import (CONFIG,Index,KD_FUNCTION_SHA,PrefixReplay,atomic_json,class_interval,
    load_frozen_clip,load_kd,make_word_bank,revision,save_tensor,score_rows,semantic_inputs)


ARMS=('shared','candidate','registered','geometric_peak','global')


def cohort(qi,pi,paths,ph,excluded):
    lookup={(Path(ph[i]).parent.name,Path(ph[i]).stem):i for i in pi.tolist()}
    seen=set(excluded)
    selected=[]
    for j in torch.randperm(len(qi),generator=torch.Generator().manual_seed(42)).tolist():
        p=Path(paths[qi[j]])
        ref=lookup[(p.parent.name,p.stem.rsplit('-',1)[0])]
        if ref not in seen:
            selected.append((j,ref))
            seen.add(ref)
    return torch.tensor(sorted(selected)).T.contiguous()


def contenders(rank,qid,truth,pi,gl,source,excluded):
    keep=[]
    for native_first in (True,False):
        options=[]
        for j,(a,b) in enumerate(rank.reshape(-1,2).tolist()):
            x,y=int(pi[a]),int(pi[b])
            if source in (x,y) or x in excluded or y in excluded:continue
            if (int(gl[a])==truth)==native_first and (int(gl[b])==truth)!=native_first:
                options.append((qid,x,y,int(native_first),2*j+1,truth))
        if options:
            g=torch.Generator().manual_seed(42+int(qid)*2+int(native_first))
            keep.append(options[int(torch.randint(len(options),(1,),generator=g))])
    return keep


def summarize_pairs(meta,score):
    # Lower first-column loss favors the native first item.
    choice=score[:,:,1]-score[:,:,0]
    correct=choice*torch.where(meta[:,3:4].bool(),1.,-1.)
    hit=(correct>1e-7).double()+.5*(correct.abs()<=1e-7)
    unique,counts=meta[:,0].unique(return_counts=True)
    paired=counts==2
    indices=(meta[:,0,None]==unique[paired][None]).T
    balanced=(indices.double()@hit)/2
    labels=meta[indices.long().argmax(1),5]
    native=meta[:,3].bool()
    out={}
    for i,name in enumerate(ARMS):
        out[name]=dict(balanced=float(balanced[:,i].mean()),class_above_chance=class_interval(100*(balanced[:,i]-.5),labels,seed=42),
            class_vs_candidate=class_interval(100*(balanced[:,i]-balanced[:,1]),labels,seed=42),
            balanced_gain_vs_candidate=float((balanced[:,i]-balanced[:,1]).mean()),
            rescued=float(hit[~native,i].sum()),damaged=float((1-hit[native,i]).sum()),
            correct_margin=float(correct[:,i].mean()))
    return dict(eligible_queries=len(unique),both=int(paired.sum()),sampled_pairs=len(meta),native_correct=int(native.sum()),
                native_wrong=int((~native).sum()),arms=out)


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    w=Path('/content/sbir')
    root=w/'results/gcwe-action-category-transfer'
    root.mkdir(parents=True,exist_ok=True)
    cfg,paths,qs,ph,gs,q,p,src,tgt,scale=semantic_inputs(w,'sketchy')
    model=load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda'))
    kd,_=load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    refs={s:torch.load(w/'results/gcwe-transported-intervention'/(s+'.pt'),weights_only=True) for s in qs}
    excluded=set(torch.cat([r['meta'][:,2:4].flatten() for r in refs.values()]).unique().tolist())
    prepared={}
    for s,(qi,labels) in qs.items():
        pi,gl=gs[s]
        assert torch.equal(refs[s]['qi'],qi) and torch.equal(refs[s]['pi'],pi)
        pos,source=cohort(qi,pi,paths,ph,excluded)
        ids=qi[pos]
        loader=DataLoader(Images([paths[i] for i in ids.tolist()],normal_transform()),batch_size=224,num_workers=2,
                          pin_memory=True,worker_init_fn=configure_image_worker)
        flip=torch.cat([F.normalize(model.visual(v.cuda().flip((-1,)).to(model.visual.conv1.weight.dtype)).float()).cpu() for v in loader])
        gallery=p[pi].cuda()
        graph=Index(SimpleNamespace(p=gallery),rgb_only=True)
        bank=make_word_bank(gallery,tgt,scale,192)
        replay=PrefixReplay(kd,gl,s)
        trials,rows=[],[]
        with ThreadPoolExecutor(max_workers=2) as pool:
            for b in torch.arange(len(ids)).split(64):
                poses=torch.stack((q[ids[b]],flip[b]),1).cuda()
                score=graph.score(CONFIG,(scale*(poses@src.T)).softmax(2).mean(1)@bank.T)
                rank=score.topk(200).indices.cpu()
                rows.append(score_rows(score,labels[pos[b]],gl.cuda(),s,replay,pool)[:,:2])
                for j in range(len(b)):
                    i=int(b[j])
                    trials.extend(contenders(rank[j],int(ids[i]),int(labels[pos[i]]),pi,gl,int(source[i]),excluded))
        meta=torch.tensor(trials)
        assert len(meta)>0 and not set(meta[:,1:3].flatten().tolist())&excluded
        prepared[s]=(ids,source,meta,torch.cat(rows))
        print(s,'source-disjoint queries and sampled pairs',len(ids),len(meta),flush=True)
        del graph,gallery,bank
    parts=structure(src.device)
    _,_,perm,square,action=parts
    action=action[square]
    identity=int(torch.where((perm[square]==torch.arange(4,device=src.device)).all(1))[0].item())
    query_ids=torch.cat([a[2][:,0] for a in prepared.values()]).unique()
    photo_ids=torch.cat([a[2][:,1:3].flatten() for a in prepared.values()]).unique()
    qe,qd,_=encode([paths[i] for i in query_ids.tolist()],False,model,parts)
    pe,pd,_=encode([ph[i] for i in photo_ids.tolist()],False,model,parts)
    for s,(ids,source,meta,rows) in prepared.items():
        losses=[]
        for b in meta.split(10):
            iq=torch.searchsorted(query_ids,b[:,0].contiguous())
            _,draws=outcomes(qe[iq].cuda(),src,scale,b[:,0])
            weights,results=[],[]
            for c in (1,2):
                ip=torch.searchsorted(photo_ids,b[:,c].contiguous())
                post,_=posterior(qd[iq].cuda(),pd[ip].cuda(),scale,perm,square)
                weights.append(post[:,square])
                logp=scale*(F.normalize(pe[ip,1:].cuda().float(),dim=-1)@tgt.T)
                results.append(assign(logp,draws,action))
            shared=.5*(weights[0]+weights[1])
            candidates=[]
            for i,(r,wt) in enumerate(zip(results,weights)):
                error=lambda prob:(1-prob).square().mean((1,2))
                candidates.append(torch.stack((error((r*shared[:,None,:,None]).sum(2)),
                    error((r*wt[:,None,:,None]).sum(2)),error(r[:,:,identity]),-wt.amax(1),
                    torch.full_like(wt[:,0],float(i))),1))
            losses.append(torch.stack(candidates,2).cpu())
        loss=torch.cat(losses).double()
        report=summarize_pairs(meta,loss)
        save_tensor(dict(qi=ids,source=source,pi=gs[s][0],meta=meta,native_metrics=rows,loss=loss),root/(s+'.pt'))
        atomic_json(root/(s+'.json'),dict(commit=revision(),gpu=torch.cuda.get_device_name(),seed=42,evaluator=KD_FUNCTION_SHA,
            queries=len(ids),gallery=len(gs[s][0]),native_metrics=(100*rows.double().mean(0)).tolist(),
            excluded_discovery_photos=len(excluded),assessment=report))
        print(s,'balanced',report['arms']['shared']['balanced'],flush=True)


if __name__=='__main__':
    run()
