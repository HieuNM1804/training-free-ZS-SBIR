"""Cross the source depiction with the candidate's regional measurement to localize a failed forecast."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from .cross_view_region_cache import two_pose_regions
from .data import Images,configure_image_worker,normal_transform
from .geometry_word_flat import atomic_json,load_frozen_clip,revision,save_tensor,semantic_inputs
from .regional_predictive_observation import cluster_effect,observation_tasks,predictions,scene_bank
from .regional_source_assignment import half_correct
from .regional_scale_position_contrast import complete_windows


def crop_features(vision,paths):
    batches=DataLoader(Images(paths,normal_transform(),'rgb'),batch_size=24,num_workers=2,pin_memory=True,
        worker_init_fn=configure_image_worker,generator=torch.Generator().manual_seed(42))
    rows=[]
    for images in batches:
        _,regions=two_pose_regions(vision,images.cuda(non_blocking=True))
        rows.append(regions[:,:9].cpu())
    return torch.cat(rows).cuda().float()


def score(obs,photo):
    results=[]
    for a,b,loc,dest in observation_tasks():
        pred=predictions(obs[:,a],photo,loc,dest)
        def logs(f):
            return (obs[:,b,None,:]*f.clamp_min(1e-30).log()).sum(-1)
        results.append(torch.stack([logs(pred['conditional']),logs(pred['marginal']),
            .5*(logs(pred['shuffle1'])+logs(pred['shuffle2'])),logs(photo[:,:,b])],1))
    return torch.stack(results).mean(0)


@torch.inference_mode()
def run():
    import os
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    w=Path('/content/sbir')
    root=w/'results/regional-observation-control'
    root.mkdir(parents=True,exist_ok=True)
    inputs=semantic_inputs(w,'sketchy')
    cfg,_,qs,pp,gs,_,p,src,tgt,scale=inputs
    views=complete_windows(w,'sketchy',inputs)
    vision=load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda')).visual
    for split,(qi,labels) in qs.items():
        pi,_=gs[split]
        ref=torch.load(w/'results/regional-predictive-observation'/(split+'.pt'),weights_only=True)
        assert torch.equal(qi,ref['qi']) and torch.equal(pi,ref['pi'])
        trials=ref['trials']
        used=trials[:,1:3].unique(sorted=True)
        heads,context_bank=scene_bank(w,split,pp,pi,p,tgt,scale,vision)
        assert torch.equal(heads.cpu(),ref['heads'])
        crops=crop_features(vision,[pp[i] for i in pi[used].tolist()])
        photo_query=(scale*(crops @ src[heads].T)).softmax(-1).mean(2)
        photo_bank=(scale*(crops @ tgt[heads].T)).softmax(-1).mean(2)
        caches={}
        for batch in trials.split(96):
            ix=batch[:,0]
            originals=batch[:,1:3].cuda()
            compact=torch.searchsorted(used.cuda(),originals.contiguous())
            sketch=(scale*(views['regions'][qi[ix],:9].float().cuda() @ src[heads].T)).softmax(-1).mean(2)
            for s,obs in [('sketch',sketch),('self_photo',photo_query[compact[:,0]])]:
                for t,bank,ids in [('patch',context_bank,originals),('crop',photo_bank,compact)]:
                    caches.setdefault(s+'/'+t,[]).append(score(obs,bank[ids]).cpu())
        caches={k:torch.cat(v).double() for k,v in caches.items()}
        torch.testing.assert_close(caches['sketch/patch'][:,:3],ref['scores'],atol=5e-6,rtol=1e-6)
        report={}
        for name,rows in caches.items():
            for mode,label in enumerate(('same','different')):
                mask=(trials[:,3]==mode)
                # Self-photo controls would otherwise duplicate exactly the same prediction for every redraw.
                if name.startswith('self_photo'):
                    first={}
                    for i in mask.nonzero().flatten().tolist():
                        first.setdefault(int(trials[i,1]),i)
                    mask=torch.zeros_like(mask)
                    mask[list(first.values())]=True
                tr=trials[mask].numpy()
                val=rows[mask].numpy()
                d=val[:,:,0]-val[:,:,1]
                effects={}
                for method,j in [('conditional',0),('known_position',3)]:
                    for control,c in [('marginal',1),('shift',2)]:
                        diff=val[:,j]-val[:,c]
                        effects[method+'-'+control]=dict(true=cluster_effect(diff[:,0],tr[:,4]),
                            contrast=cluster_effect(diff[:,0]-diff[:,1],tr[:,4]))
                pair_scores=[]
                for pair in np.unique(tr[:,4]):
                    choose=tr[:,4]==pair
                    a,b=np.unique(tr[choose,1])
                    pair_scores.append(half_correct(d[choose&(tr[:,1]==a),None]+d[choose&(tr[:,1]==b)][None]).mean((0,1)))
                pair_scores=np.stack(pair_scores)
                report[name+'/'+label]=dict(queries=len(tr),pairs=len(pair_scores),source_assignment=[
                    cluster_effect(col,np.arange(len(col))) for col in pair_scores.T],effects=effects)
        save_tensor(dict(qi=qi,pi=pi,trials=trials,scores=caches),root/(split+'.pt'))
        atomic_json(root/(split+'.json'),dict(commit=revision(),seed=42,device=torch.cuda.get_device_name(),
            methods=['conditional','marginal','shift','known_position'],assessment=report))
        print(split,{k:v['source_assignment'][3]['mean'] for k,v in report.items()},flush=True)


if __name__=='__main__':
    run()
