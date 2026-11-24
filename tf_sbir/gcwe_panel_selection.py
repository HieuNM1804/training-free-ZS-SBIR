"""Condition on a disjoint observation to separate nomination information from semantic update information."""
from pathlib import Path
import torch
from torch.nn import functional as F
from .gcwe_relational_completion import (AXES,CORNERS,compose,frames,moments,read_corners,log_score,
    semantic_inputs,complete_windows,load_frozen_clip,scene_bank,atomic_json,class_interval,revision,save_tensor)


PANEL=('original','other_corner','blind')
MODELS=('distinct','tree','literal','sham','prior','log_evidence_ratio')


def run_panels(local,visual,bank,keys,eligible,groups,blind):
    best=eligible[(visual @ keys).topk(64).indices]
    out=local.new_zeros(len(local),len(PANEL),len(MODELS),dtype=torch.double)
    for a,b,c in AXES:
        o=next(j for j in range(4) if j not in (a,b,c))
        first,second=best[:,a,:32],best[:,b,:32]
        union=torch.cat((first,second),1)
        ok=torch.cat((torch.ones_like(first,dtype=torch.bool),~(second[:,:,None]==first[:,None]).any(2)),1)
        count=ok.sum(1,keepdim=True)
        keep=torch.arange(64,device=bank.device)[None]<count
        for j,(ids,valid) in enumerate(((union,ok),(best[:,o],keep),(blind,keep))):
            photo=bank[ids]
            ma=(photo*local[:,None,a,None]).sum(-1)
            mb=(photo*local[:,None,b,None]).sum(-1)
            stats=moments(photo,ma,mb,(a,b,c),groups[0])
            wrong=moments(photo,ma,mb,(a,b,c),groups[1])
            role=compose(stats,valid)[0]
            sham=compose(wrong,valid)[0][:,0]
            r,u,v,lit=(z*valid[:,:,None] for z in stats)
            mass=lit.sum((1,2))
            tree_mass=(u.sum(1)*v.sum(1)/r.sum(1).clamp_min(1e-300)).sum(1)
            forecasts=torch.stack((role[:,0],role[:,1],role[:,3],sham,role[:,4]),1)
            out[:,j,:5]+=log_score(forecasts,local[:,None,c])/len(AXES)
            out[:,j,5]+=(mass/tree_mass).log()/len(AXES)
    return out


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    w=Path('/content/sbir')
    root=w/'results/gcwe-panel-selection'
    root.mkdir(parents=True,exist_ok=True)
    inputs=semantic_inputs(w,'sketchy')
    cfg,_,qs,pp,gs,_,p,src,tgt,scale=inputs
    views=complete_windows(w,'sketchy',inputs)
    vision=load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda')).visual
    gallery_crops=read_corners(w,pp)
    for split,(qi,labels) in qs.items():
        ref=torch.load(w/'results/gcwe-relational-completion'/(split+'.pt'),weights_only=True)
        ids=ref['qi']
        pi,_=gs[split]
        assert torch.equal(ref['pi'],pi)
        parts=torch.randperm(len(pi),generator=torch.Generator().manual_seed(42)).cuda().tensor_split(2)
        folds=torch.zeros(len(pi),device='cuda',dtype=torch.long)
        folds[parts[1]]=1
        gallery=p[pi].cuda()
        keys=[gallery[part].T.contiguous() for part in parts]
        heads,_=scene_bank(w,split,pp,pi,p,tgt,scale,vision)
        bank=torch.cat([(scale*(gallery_crops[j].cuda().float() @ tgt[heads].T)).softmax(-1).mean(2) for j in pi.split(192)])
        groups=frames('cuda')
        results=[]
        for batch in torch.arange(len(ids)).split(16):
            crops=views['regions'][ids[batch]][:,CORNERS].cuda().float()
            obs=(scale*(crops @ src[heads].T)).softmax(-1).mean(2)
            vis=F.normalize(crops.mean(2),dim=-1)
            side=folds[ref['reference_ids'][batch]]
            out=obs.new_empty(len(batch),len(PANEL),len(MODELS),dtype=torch.double)
            for t in (0,1):
                mask=side==t
                if not mask.any():
                    continue
                eligible=parts[1-t]
                seeds=[42+2*int(i)+t for i in ids[batch][mask.cpu()]]
                blind=eligible[torch.stack([torch.randperm(len(eligible),generator=
                    torch.Generator().manual_seed(s))[:64] for s in seeds]).cuda()]
                out[mask]=run_panels(obs[mask],vis[mask],bank,keys[1-t],eligible,groups,blind)
            results.append(out.cpu())
        rows=torch.cat(results)
        prior=ref['local'][:,0,[0,1,3,4,5]]
        gap=(rows[:,0,:5]-prior).abs()
        if gap.mean()>5e-5 or gap.max()>0.004:
            raise ValueError(f'Original panel failed replay {gap.mean()}, {gap.max()}')
        deltas={}
        gain=rows[:,:,0]-rows[:,:,4]
        for j,panel in enumerate(PANEL):
            metrics={'distinct_minus_prior':gain[:,j],'gain_minus_original_gain':gain[:,j]-gain[:,0],
                'absolute_minus_original':rows[:,j,0]-rows[:,0,0],'distinct_minus_literal':rows[:,j,0]-rows[:,j,2],
                'distinct_minus_sham':rows[:,j,0]-rows[:,j,3],'literal_to_tree_log_evidence_ratio':rows[:,j,5]}
            deltas[panel]={k:dict(mean=float(v.mean()),classes=class_interval(v,ref['labels'],seed=42)) for k,v in metrics.items()}
        save_tensor(dict(qi=ids,pi=pi,labels=ref['labels'],rows=rows),root/(split+'.pt'))
        atomic_json(root/(split+'.json'),dict(commit=revision(),seed=42,gpu=torch.cuda.get_device_name(),
            panels=list(PANEL),measures=list(MODELS),queries=len(ids),results=deltas,log_scores=rows.mean(0).tolist(),
            replay_mean_absolute_error=float(gap.mean()),replay_max_absolute_error=float(gap.max())))
        print(split,[(k,v['distinct_minus_prior']['mean']) for k,v in deltas.items()],flush=True)


if __name__=='__main__':
    run()
