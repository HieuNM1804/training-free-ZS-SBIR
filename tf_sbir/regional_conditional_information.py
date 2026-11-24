"""Test which locally resolved assignments survive removal of the shared coordinate frame."""
from pathlib import Path
import numpy as np
import torch

from .geometry_word_flat import atomic_json,load_frozen_clip,revision,semantic_inputs
from .regional_observation_control import crop_features
from .regional_layout_forecast import CORNERS,maps
from .regional_predictive_observation import scene_bank
from .regional_scale_position_contrast import complete_windows
from .regional_source_assignment import half_correct,summarize_pairs


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    w=Path('/content/sbir')
    root=w/'results/regional-layout-forecast'
    inputs=semantic_inputs(w,'sketchy')
    cfg,_,qs,pp,gs,q,p,src,tgt,scale=inputs
    views=complete_windows(w,'sketchy',inputs)
    vision=load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda')).visual
    orientations=maps('cuda')[0]
    identity=(orientations==torch.arange(4,device='cuda')).all(1)
    report={}
    for split,(qi,labels) in qs.items():
        pi,_=gs[split]
        ref=torch.load(root/(split+'.pt'),weights_only=True)
        trials=ref['trials']
        assert torch.equal(qi,ref['qi']) and torch.equal(pi,ref['pi'])
        used=trials[:,1:3].unique(sorted=True)
        heads,_=scene_bank(w,split,pp,pi,p,tgt,scale,vision)
        crops=crop_features(vision,[pp[i] for i in pi[used].tolist()])[:,CORNERS]
        gallery=(scale*(crops @ tgt[heads].T)).softmax(-1).clamp_min(1e-30)
        log_bank=gallery.mean(2).log()
        full_bank=(scale*(p[pi[used]].cuda() @ tgt.T)).softmax(-1)
        all_scores=[]
        for batch in trials.split(64):
            ix=batch[:,0]
            a,b=torch.searchsorted(used,batch[:,1:3].contiguous()).cuda().T
            views_q=torch.stack((q[qi[ix]],views['flipped'][qi[ix]].float()),1).cuda()
            words=(scale*(views_q @ src.T)).softmax(-1).mean(1)
            margin=(words*(full_bank[a]-full_bank[b])).sum(1)
            crop_q=views['regions'][qi[ix]][:,CORNERS].float().cuda()
            obs=(scale*(crop_q @ src[heads].T)).softmax(-1).mean(2)
            delta=((log_bank[a]-log_bank[b])[:,orientations]*obs[:,None]).sum(-1).mean(-1)
            null=(obs.mean(1)*(log_bank[a]-log_bank[b]).mean(1)).sum(-1)
            torch.testing.assert_close(null,delta.mean(1),atol=2e-6,rtol=2e-5)
            all_scores.append(torch.stack((margin,delta[:,identity].flatten(),null,delta[:,~identity].mean(1)),1).cpu())
        val=torch.cat(all_scores).double().numpy()
        prev=ref['scores'].numpy()
        native=prev[:,:,0]-prev[:,:,1]
        if abs(val[:,1]-native[:,-1]).max()>5e-5:
            raise ValueError('Registered scoring does not reproduce the stored scores')
        data=np.column_stack([val,native[:,0],native[:,1]])
        names=['global','registered','uniform_orientation','nonidentity','inferred','unconditioned']
        for mode,tag in enumerate(('same','different')):
            use=trials[:,3]==mode
            tr=trials[use].numpy()
            sc=data[use]
            groups={k:[] for k in ('all','global_wrong','global_correct')}
            counts=dict.fromkeys(groups,0)
            for pid in np.unique(tr[:,4]):
                mask=tr[:,4]==pid
                a,b=np.unique(tr[mask,1])
                pair=(sc[mask&(tr[:,1]==a),None]+sc[mask&(tr[:,1]==b)][None]).reshape(-1,len(names))
                accuracy=half_correct(pair)
                outcomes=np.column_stack([accuracy,accuracy[:,1]-accuracy[:,2],accuracy[:,4]-accuracy[:,2]])
                for k,sel in [('all',np.ones(len(pair),bool)),('global_wrong',pair[:,0]<0),('global_correct',pair[:,0]>0)]:
                    groups[k].append(outcomes[sel].mean(0) if sel.any() else np.full(outcomes.shape[1],np.nan))
                    counts[k]+=int(sel.sum())
            report[split+'/'+tag]=dict(counts=counts,single_query_accuracy=dict(zip(names,half_correct(sc).mean(0).tolist())),
                 conditional={k:summarize_pairs(v,names+['registered_minus_uniform','inferred_minus_uniform']) for k,v in groups.items()})
    atomic_json(root/'conditional-information.json',dict(commit=revision(),assessment=report))
    print({k:v['conditional']['global_wrong']['registered']['mean'] for k,v in report.items()})


if __name__=='__main__':
    run()
