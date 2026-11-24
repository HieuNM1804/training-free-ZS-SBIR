"""Test which fixed one-observation measurements account for errors beyond any view coupling."""
from pathlib import Path
import torch
from torch.nn import functional as F

from .benchmark import merge_inputs
from .concept_benchmark import atomic_json
from .crossdomain_benchmark import revision
from .crossdomain_inputs import runtime
from .data import SKETCHY_SPLITS
from .depiction_path_diagnostic import class_interval
from .encoding import load_frozen_clip,load_global_inputs,save_tensor
from .foundation import load_words
from .semantic import read_words
from .semantic_common_depiction import readings


ARMS=('split_image_split_words','whole_image_split_words','split_image_full_words',
      'whole_image_full_words','split_image_word_mass','whole_image_word_mass')


def scores(strip,whole,photo,fold):
    def table(x):
        first=x[:,:,fold]@photo[:,fold,None]
        second=x[:,:,~fold]@photo[:,~fold,None]
        a,b=first.squeeze(2).double(),second.squeeze(2).double()
        mass_a=x[:,:,fold].sum(2).double()
        mass_b=x[:,:,~fold].sum(2).double()
        def cross(a,b):
            return (a[:,0]*b[:,1]+b[:,0]*a[:,1]+a[:,2]*b[:,3]+b[:,2]*a[:,3])*.25
        normalized=cross(a/mass_a,b/mass_b)
        full=a+b
        all_words=.5*(full[:,0]*full[:,1]+full[:,2]*full[:,3])
        return normalized.log(),all_words.log(),cross(a,b).log()
    sp,sw=table(strip),table(whole[:,None,:].expand(-1,4,-1))
    return torch.stack((sp[0],sw[0],sp[1],sw[1],sp[2],sw[2]),1)


def summarize(meta,limits,score):
    is_first=meta[:,3].bool()
    mask=torch.where(is_first,limits[:,0,1]<limits[:,1,0]-1e-7,limits[:,1,1]<limits[:,0,0]-1e-7)
    margin=(score[:,:,0]-score[:,:,1])*(2*is_first[:,None].long()-1)
    hit=(margin>1e-7).double()+.5*(margin.abs()<=1e-7)
    assert (hit[mask,0]==0).all()
    unique,n=meta[:,0].unique(return_counts=True)
    both=(meta[:,0,None]==unique[n==2][None]).T
    bal=(both.double()@hit)/2
    labels=meta[both.long().argmax(1),5]
    rows={}
    for i,name in enumerate(ARMS):
        rows[name]=dict(fixed_errors_now_correct=float(hit[mask,i].sum()),
            total_rescued=float(hit[hit[:,0]==0,i].sum()),total_damaged=float((1-hit[hit[:,0]==1,i]).sum()),
            balanced=float(bal[:,i].mean()),class_change=class_interval(100*(bal[:,i]-bal[:,0]),labels,seed=42))
    change=(bal[:,3]-bal[:,2])-(bal[:,1]-bal[:,0])
    return dict(pairs=len(meta),both=len(bal),certified_errors=int(mask.sum()),arms=rows,
                interaction_points=float(100*change.mean()),class_interaction=class_interval(100*change,labels,seed=42)),mask


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    w=Path('/content/sbir')
    root=w/'results/semantic-marginal-error-audit'
    root.mkdir(parents=True,exist_ok=True)
    cfg=runtime(w,'sketchy')
    paths,qs=merge_inputs(cfg.root,SKETCHY_SPLITS,'sketch')
    ph,gs=merge_inputs(cfg.root,SKETCHY_SPLITS,'photo')
    model=load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda'))
    scale=model.logit_scale.exp().float()
    src,tgt=[v.cuda() for v in load_words(cfg,read_words(cfg.vocabulary))]
    anchor=torch.randperm(len(src),generator=torch.Generator().manual_seed(42))[:2]
    fold=torch.zeros(len(src),device='cuda',dtype=torch.bool)
    fold[(tgt@(tgt[anchor[0]]-tgt[anchor[1]])).argsort(stable=True).tensor_split(2)[0]]=True
    ref={s:torch.load(w/'results/semantic-dependence-limits'/(s+'.pt'),weights_only=True) for s in qs}
    qu=torch.cat([r['meta'][:,0] for r in ref.values()]).unique()
    im=torch.cat([r['meta'][:,1:3].flatten() for r in ref.values()]).unique()
    strip=readings(cfg,model.visual,[paths[i] for i in qu.tolist()],'sketch',{})
    old={}
    for f in (w/'cache/regional-semantic-interactions').glob('*.pt'):
        d=torch.load(f,weights_only=True)
        if Path(d['paths'][0]).parent.parent.name=='photo':old.update(zip(d['paths'],d['features']))
    complete=readings(cfg,model.visual,[ph[i] for i in im.tolist()],'photo',old)[:,-1].contiguous()
    q,_,_=load_global_inputs(cfg,paths,ph,False)
    for s,r in ref.items():
        meta=r['meta']
        assert torch.equal(r['pi'],gs[s][0]) and torch.isin(meta[:,0],r['qi']).all()
        rows=[]
        for batch in meta.split(32):
            ids=batch[:,0].contiguous()
            x=(scale*(F.normalize(strip[torch.searchsorted(qu,ids)].cuda().float(),dim=-1)@src.T)).softmax(-1)
            y=(scale*(q[ids].cuda().float()@src.T)).softmax(-1)
            outputs=[]
            for k in (1,2):
                location=torch.searchsorted(im,batch[:,k].contiguous())
                photo=(scale*(F.normalize(complete[location].cuda().float(),dim=-1)@tgt.T)).softmax(-1)
                outputs.append(scores(x,y,photo,fold))
            rows.append(torch.stack(outputs,-1).cpu())
        value=torch.cat(rows)
        error=(value[:,0]-r['independent']).abs()
        torch.testing.assert_close(value[:,0],r['independent'],atol=4e-5,rtol=1e-6)
        report,mask=summarize(meta,r['log_limits'],value)
        save_tensor(dict(meta=meta,qi=r['qi'],pi=r['pi'],scores=value,certified_error=mask),root/(s+'.pt'))
        atomic_json(root/(s+'.json'),dict(commit=revision(),gpu=torch.cuda.get_device_name(),seed=42,arms=ARMS,
            replay_absolute_log_score=[float(error.mean()),float(error.max())],assessment=report))
        print(s,report['certified_errors'],report['arms']['whole_image_full_words']['fixed_errors_now_correct'],flush=True)


if __name__=='__main__':
    run()
