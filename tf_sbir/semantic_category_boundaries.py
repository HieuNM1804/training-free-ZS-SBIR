"""Test conditional semantic forecasts at the actual boundary between relevant and irrelevant photos."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import os
from pathlib import Path
from types import SimpleNamespace

from .gcwe_relational_completion import CORNERS
from .geometry_word_flat import (CHECKPOINT, CONFIG, COUNTS, Index, KD_FUNCTION_SHA, PrefixReplay, atomic_json,
    class_interval,digest,index,load_kd,make_word_bank,regional_readout,revision,save_tensor,score_rows,semantic_inputs)
from .gallery_corner_observations import read_corners
from .regional_scale_position_contrast import complete_windows
import torch


def local_scores(q,context,bank):
    n,ng=len(q),len(bank)
    score=q.new_zeros(n,ng)
    for target in range(4):
        contrast=(4./3)*(bank[:,target]-bank.mean(1))
        # Normalize the same 64-word predictive distribution as the independent-crop pair audit.
        logits=context[:,target,None,:]+contrast[None]
        score+=(q[:,target] @ contrast.T)+(q[:,target]*context[:,target]).sum(1,keepdim=True)-logits.logsumexp(2)
    return score/4


def boundaries(order,official,classes,truth):
    ids=order[:,:200].reshape(len(truth),100,2)
    hit=classes[ids]==truth.cuda()[:,None,None]
    mixed=hit[:,:,0]^hit[:,:,1]
    first_correct=mixed&hit[:,:,0]
    first_wrong=mixed&hit[:,:,1]
    outputs={}
    def within(values,mask):
        count=mask.sum(1)
        return torch.where(count>0,(values*mask).sum(1)/count,torch.nan)
    for name,scores in official.items():
        pairs=scores.gather(1,ids.flatten(1)).reshape_as(ids)
        preference=(pairs[:,:,0]-pairs[:,:,1])*torch.where(hit[:,:,0],1,-1)
        good=(preference>0).float()+.5*(preference==0).float()
        retain=within(good,first_correct)
        rescue=within(good,first_wrong)
        balanced=.5*(retain+rescue)
        outputs[name]=torch.stack([within(good,mixed),retain,rescue,balanced,
            (good*first_wrong).sum(1),((1-good)*first_correct).sum(1)],1).cpu()
    return outputs,torch.stack([mixed.sum(1),first_correct.sum(1),first_wrong.sum(1)],1).cpu()


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    w=Path('/content/sbir')
    root=w/'results/semantic-category-boundaries'
    root.mkdir(parents=True,exist_ok=True)
    inputs=semantic_inputs(w,'sketchy')
    _,_,qs,pp,gs,q,p,src,tgt,scale=inputs
    views=complete_windows(w,'sketchy',inputs)
    corners=read_corners(w,pp)
    kd,_=load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    excluded=torch.cat([torch.load(w/'results'/fam/(s+'.pt'),weights_only=True)['qi'] for s in qs for fam in
       ('semantic-grounded-word-events','gcwe-relational-completion','gcwe-semantic-reveal','gcwe-semantic-prediction')]).unique()
    for split,(qi,labels) in qs.items():
        pi,gl=gs[split]
        assert (len(qi),len(pi),len(gl.unique()))==COUNTS[split]
        perm=torch.randperm(len(qi),generator=torch.Generator().manual_seed(42))[9216:]
        ix=perm[~torch.isin(qi[perm],excluded)][:768].sort().values
        assert len(ix)==768
        gallery=p[pi].cuda()
        stamp=digest(dict(gallery=[str(pp[i]) for i in pi.tolist()],model=CHECKPOINT,source='regional-64/RGB16'))
        heads,compact=index(w,split,gallery,tgt,scale,stamp[:16])
        bank=make_word_bank(gallery,tgt,scale,192)
        cropped=torch.cat([(scale*(corners[b].cuda().float() @ tgt[heads].T)).softmax(-1).mean(2) for b in pi.split(192)])
        logb=cropped.clamp_min(1e-30).log()
        neutral=cropped.mean(1).clamp_min(1e-30).log()
        graph=Index(SimpleNamespace(p=gallery),rgb_only=True)
        replay=PrefixReplay(kd,gl,split)
        classes=gl.cuda()
        metrics,assess,counts={},{},[]
        with ThreadPoolExecutor(max_workers=min(2,len(os.sched_getaffinity(0)))) as pool:
            for batch in ix.split(24):
                ids=qi[batch]
                poses=torch.stack((q[ids],views['flipped'][ids].float()),1).cuda()
                windows=views['regions'][ids].cuda()
                prob=(scale*(windows[:,CORNERS].float() @ src[heads].T)).softmax(-1).mean(2)
                logs=prob.clamp_min(1e-30).log()
                context=(logs.sum(1,keepdim=True)-logs)/3
                native=(scale*(poses @ src.T)).softmax(-1).mean(1) @ bank.T
                scores={'semantic':native}
                original=regional_readout(poses,windows,src,scale,heads,compact,native)
                logscores={'unregistered':prob.mean(1) @ neutral.T,
                    'raw_registered':(prob.flatten(1) @ logb.flatten(1).T)/4}
                visible,rotated=[],[]
                for gallery_chunk in logb.split(192):
                    visible.append(local_scores(prob,context,gallery_chunk))
                    rotated.append(local_scores(prob,context,gallery_chunk[:,[2,0,3,1]]))
                logscores.update(visibility=torch.cat(visible,1),turned=torch.cat(rotated,1))
                for k,v in logscores.items():
                    scores[k]=(v-v.amax(1,keepdim=True)).exp()
                evaluated={'regional_original':original}
                for k,v in scores.items():
                    evaluated[k+'/direct']=v
                    evaluated[k+'/diffused']=graph.score(CONFIG,v)
                for k,v in evaluated.items():
                    values=score_rows(v,labels[batch],classes,split,replay,pool)[:,:2]
                    metrics.setdefault(k,[]).append(values)
                rank=evaluated['semantic/diffused'].topk(200).indices
                values,cnt=boundaries(rank,evaluated,classes,labels[batch])
                for k,v in values.items():
                    assess.setdefault(k,[]).append(v)
                counts.append(cnt)
        count=torch.cat(counts)
        rows={k:torch.cat(v).double() for k,v in metrics.items()}
        region={k:torch.cat(v).double() for k,v in assess.items()}
        assessment={}
        for k,v in region.items():
            assessment[k]=dict(available_means=v[:,:4].nanmean(0).tolist(),rescued=float(v[:,4].sum()),
                damaged=float(v[:,5].sum()))
        ok=torch.isfinite(region['visibility/direct'][:,3])
        controls={}
        for k in ['visibility/direct','visibility/diffused']:
            suffix=k.split('/')[1]
            primary=region[k][ok,3]
            contrasts={'chance':primary-.5,'turned':primary-region['turned/'+suffix][ok,3],
                       'raw_registered':primary-region['raw_registered/'+suffix][ok,3],
                       'unregistered':primary-region['unregistered/'+suffix][ok,3]}
            controls[k]={c:dict(mean=float(v.mean()),classes=class_interval(100*v,labels[ix][ok],seed=42))
                         for c,v in contrasts.items()}
        save_tensor(dict(qi=qi[ix],pi=pi,labels=labels[ix],rows=rows,conditional=region,counts=count),root/(split+'.pt'))
        atomic_json(root/(split+'.json'),dict(commit=revision(),seed=42,gpu=torch.cuda.get_device_name(),evaluator=KD_FUNCTION_SHA,
            queries=len(ix),gallery=len(pi),queries_with_mixed=int((count[:,0]>0).sum()),both_directions=int(ok.sum()),
            pair_counts=count.sum(0).tolist(),columns=['mixed','global_correct','global_wrong','balanced'],
            official={k:(100*v.mean(0)).tolist() for k,v in rows.items()},conditional=assessment,paired=controls))
        print(split,int(ok.sum()),controls['visibility/direct']['chance']['mean'],flush=True)
        del bank,graph


if __name__=='__main__':
    run()
