"""Infer a full-vocabulary scene posterior when a withheld sketch region is revealed."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import os
from pathlib import Path
from types import SimpleNamespace

from .gcwe_semantic_prediction import log_score
from .data import Images, configure_image_worker, normal_transform
from .geometry_word_flat import (CONFIG, COUNTS, Index, KD_FUNCTION_SHA, PrefixReplay, atomic_json, class_interval,
    load_frozen_clip, load_kd, make_word_bank, revision, save_tensor, score_rows, semantic_inputs)
from .regional_predictive_observation import observation_tasks, predictions, scene_bank
from .regional_scale_position_contrast import complete_windows
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


CORNERS=(0,2,6,8)
DENSE=('prior','updated','shifted','misassigned','native')
SPARSE=('restricted_prior','restricted_updated')


def posterior(a,donors,h,y):
    # donors[w,d] and h[d,u] are both row-stochastic conditional distributions.
    weights=(a[:,None] @ donors).squeeze(1)
    forecast=(weights[:,None] @ h).squeeze(1)
    response=(h @ (y/forecast.clamp_min(1e-30))[:,:,None])
    result=a*(donors @ response).squeeze(-1)
    if not torch.allclose(result.sum(1),torch.ones_like(a[:,0]),atol=2e-5):
        raise ValueError('Inferred scene posterior lost mass')
    return result,forecast


def check():
    torch.manual_seed(42)
    a=torch.randn(4,11,dtype=torch.double).softmax(-1)
    d=torch.randn(4,11,7,dtype=torch.double).softmax(-1)
    h=torch.randn(4,7,5,dtype=torch.double).softmax(-1)
    y=torch.randn(4,5,dtype=torch.double).softmax(-1)
    updated,forecast=posterior(a,d,h,y)
    joint=a[:,:,None,None]*d[:,:,:,None]*h[:,None]
    expected=joint.sum((1,2))
    exact=(joint*(y/expected)[:,None,None]).sum((2,3))
    torch.testing.assert_close(updated,exact)
    torch.testing.assert_close(forecast,expected)
    torch.testing.assert_close(posterior(a,d,h,expected)[0],a)


def source_refs(qp,pp,qi,pi):
    photos={(Path(pp[i]).parent.name,Path(pp[i]).stem):j for j,i in enumerate(pi.tolist())}
    assert len(photos)==len(pi)
    groups=defaultdict(list)
    references=[]
    for i,qid in enumerate(qi.tolist()):
        p=Path(qp[qid])
        reference=photos[(p.parent.name,p.stem.rsplit('-',1)[0])]
        groups[reference].append(i)
        references.append(reference)
    return torch.tensor(references),groups


def encode_masked(visual,paths):
    loader=DataLoader(Images(paths,normal_transform(),'rgb'),batch_size=24,num_workers=min(2,len(os.sched_getaffinity(0))),
        pin_memory=True,worker_init_fn=configure_image_worker,generator=torch.Generator().manual_seed(42))
    rows=[]
    for pixels in loader:
        pixels=pixels.cuda(non_blocking=True)
        embeddings=[]
        for corner in CORNERS:
            r,c=divmod(corner,3)
            masked=pixels.clone()
            masked[:,:,r*64:r*64+96,c*64:c*64+96]=0
            poses=torch.stack((masked,masked.flip((-1,))),1).flatten(0,1)
            embeddings.append(F.normalize(visual(poses.to(visual.conv1.weight.dtype)).float()).view(len(pixels),2,-1))
        rows.append(torch.stack(embeddings,1).cpu())
    return torch.cat(rows)


def geometry_forecast(obs,b,photos):
    native,shuffled=[],[]
    for a,dest,source,targets in observation_tasks():
        if dest!=b:
            continue
        options=predictions(obs[:,a],photos,source,targets)
        native.append(options['conditional'])
        shuffled.extend((options['shuffle1'],options['shuffle2']))
    assert len(native)==2
    return torch.stack(native).mean(0),torch.stack(shuffled).mean(0)


def infer(masked,local,bank,photo_regions,keys,parts,src,scale,permutation):
    n=len(local)
    folds=[]
    supports=[]
    for t in (0,1):
        avg={k:bank.new_zeros(n,len(src)) for k in DENSE+SPARSE if k!='native'}
        indices=[]
        for j,b in enumerate(CORNERS):
            context=masked[:,j]
            a=(scale*(context @ src.T)).softmax(-1).mean(1)
            near=parts[1-t][(F.normalize(context.mean(1)) @ keys[1-t]).topk(100).indices]
            desc=bank[near].transpose(1,2).contiguous()
            donors=desc/desc.sum(2,keepdim=True).clamp_min(torch.finfo(desc.dtype).tiny)
            h,sh=geometry_forecast(local,b,photo_regions[near])
            y=local[:,b]
            revised,_=posterior(a,donors,h,y)
            wrong,_=posterior(a,donors,sh,y)
            mismatch,_=posterior(a,donors[:,:,permutation],h,y)
            mass,ix=a.topk(64)
            narrow=torch.zeros_like(a).scatter_(1,ix,mass/mass.sum(1,keepdim=True))
            narrowed,_=posterior(narrow,donors,h,y)
            for k,v in zip(('prior','updated','shifted','misassigned',*SPARSE),(a,revised,wrong,mismatch,narrow,narrowed)):
                avg[k]+=v/len(CORNERS)
            indices.append(ix)
        folds.append(avg)
        supports.append(torch.stack(indices,1))
    return {k:torch.stack([v[k] for v in folds],1) for k in folds[0]},supports[0]


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    check()
    w=Path('/content/sbir')
    out=w/'results/gcwe-semantic-reveal'
    out.mkdir(parents=True,exist_ok=True)
    inputs=semantic_inputs(w,'sketchy')
    cfg,qp,qs,pp,gs,q,p,src,tgt,scale=inputs
    views=complete_windows(w,'sketchy',inputs)
    vision=load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda')).visual
    kd,_=load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    previous=torch.cat([torch.load(w/'results/gcwe-semantic-prediction'/(s+'.pt'),weights_only=True)['qi'] for s in qs])
    # Shuffle descriptions locally in blocks of four adjacent nomination ranks.
    permutation=torch.arange(100,device='cuda').view(-1,4).roll(1,1).flatten()
    for split,(qi,labels) in qs.items():
        pi,gl=gs[split]
        assert (len(qi),len(pi),len(gl.unique()))==COUNTS[split]
        references,groups=source_refs(qp,pp,qi,pi)
        eligible=~torch.isin(qi,previous)&torch.tensor([len(groups[int(ref)])>1 for ref in references])
        perm=torch.randperm(len(qi),generator=torch.Generator().manual_seed(42))[5376:]
        ix=perm[eligible[perm]][:768].sort().values
        assert len(ix)==768
        masked=encode_masked(vision,[qp[i] for i in qi[ix].tolist()])
        parts=torch.randperm(len(pi),generator=torch.Generator().manual_seed(42)).cuda().tensor_split(2)
        fold=torch.zeros(len(pi),dtype=torch.long,device='cuda')
        fold[parts[1]]=1
        gallery=p[pi].cuda()
        keys=[gallery[part].T.contiguous() for part in parts]
        heads,regions=scene_bank(w,split,pp,pi,p,tgt,scale,vision)
        bank=make_word_bank(gallery,tgt,scale,128)
        endpoints=[bank[part].T.contiguous() for part in parts]
        graph=Index(SimpleNamespace(p=gallery),rgb_only=True)
        replay=PrefixReplay(kd,gl,split)
        classes=gl.cuda()
        # Accumulate the forecasts before reading any of the sibling targets.
        predictions_saved={k:[] for k in DENSE}
        supports,metrics=[],{}
        with ThreadPoolExecutor(max_workers=min(2,len(os.sched_getaffinity(0)))) as pool:
            for pos in torch.arange(len(ix)).split(8):
                batch=ix[pos]
                crop=views['regions'][qi[batch],:9].float().cuda()
                local=(scale*(crop @ src[heads].T)).softmax(-1).mean(2)
                updated,support=infer(masked[pos].cuda(),local,bank,regions,keys,parts,src,scale,permutation)
                poses=torch.stack((q[qi[batch]],views['flipped'][qi[batch]].float()),1).cuda()
                native=(scale*(poses @ src.T)).softmax(-1).mean(1)
                updated['native']=native[:,None].expand(-1,2,-1)
                supports.append(support.cpu())
                for k,val in updated.items():
                    score=gallery.new_empty(len(batch),len(pi))
                    for t,part in enumerate(parts):
                        score[:,part]=val[:,t] @ endpoints[t]
                    if k in predictions_saved:
                        predictions_saved[k].append(val[torch.arange(len(batch),device='cuda'),
                            fold[references[batch]]].cpu())
                    official=score_rows(graph.score(CONFIG,score),labels[batch],classes,split,replay,pool)
                    metrics.setdefault(k,[]).append(official[:,:2])
        by_class=defaultdict(list)
        for ref in sorted(groups):
            by_class[int(gl[ref])].append(ref)
        foil={a:b for refs in by_class.values() for a,b in zip(refs,refs[1:]+refs[:1])}
        def reading(row_ids):
            ids=qi[torch.as_tensor(row_ids)]
            poses=torch.stack((q[ids],views['flipped'][ids].float()),1).cuda()
            return (scale*(poses @ src.T)).softmax(-1).mean((0,1)).cpu()
        # Compute class and instance checks only on the already-frozen predictions for the reference's target fold.
        metrics={k:torch.cat(v).double() for k,v in metrics.items()}
        forecasts={k:torch.cat(v) for k,v in predictions_saved.items()}
        supports=torch.cat(supports)
        measurements={k:[] for k in DENSE}
        retained=[]
        for i,b in enumerate(ix.tolist()):
            ref=int(references[b])
            sib=reading([s for s in groups[ref] if s!=b])
            other=reading(groups[foil[ref]])
            truth=torch.stack([sib,bank[ref].cpu(),other])
            retained.append(sib[supports[i]].sum(1).mean())
            for k in DENSE:
                measurements[k].append(log_score(forecasts[k][i],truth))
        measurements={k:torch.stack(v).double() for k,v in measurements.items()}
        primary=measurements['updated']
        contrasts={}
        for k,v in measurements.items():
            delta=primary-v
            contrasts[k]=dict(points=delta.mean(0).tolist(),sibling_class_interval=class_interval(delta[:,0],labels[ix],seed=42),
                specificity=float((delta[:,0]-delta[:,2]).mean()))
        record=dict(commit=revision(),seed=42,device=torch.cuda.get_device_name(),n=len(ix),gallery=len(pi),
            evaluator=KD_FUNCTION_SHA,forecast_targets=['separate_redraw','reference_photo','same_class_other_source'],
            expected_log_scores={k:v.mean(0).tolist() for k,v in measurements.items()},updated_minus=contrasts,
            masked_top64_mass_on_redraw=float(torch.stack(retained).mean()),
            retrieval_percent={k:(100*v.mean(0)).tolist() for k,v in metrics.items()})
        save_tensor(dict(qi=qi[ix],pi=pi,labels=labels[ix],reference_ids=references[ix],rows=metrics,
                    predictions=measurements,retained=torch.stack(retained)),out/(split+'.pt'))
        atomic_json(out/(split+'.json'),record)
        print(split,record['retrieval_percent'],flush=True)
        del graph,bank,endpoints,forecasts,predictions_saved


if __name__=='__main__':
    run()
