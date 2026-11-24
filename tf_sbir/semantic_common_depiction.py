"""Compare two independent uncertain sketch observations under shared versus separate partial photo explanations."""
import hashlib
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .benchmark import merge_inputs
from .concept_benchmark import atomic_json
from .crossdomain_benchmark import revision
from .crossdomain_inputs import runtime
from .data import Images,SKETCHY_SPLITS,configure_image_worker,normal_transform
from .depiction_path_diagnostic import class_interval
from .encoding import load_frozen_clip,save_tensor
from .foundation import load_words
from .semantic import read_words
from .semantic_plausible_depiction import project,relabel


ARMS=('shared','independent','size_only','false_correspondence','forced_word')


def masks(which):
    y,x=torch.meshgrid(torch.arange(7),torch.arange(7),indexing='ij')
    if which=='sketch':
        regions=(x<3,x>3,y<3,y>3)
        assert not (regions[0]&regions[1]).any() and not (regions[2]&regions[3]).any()
    else:
        loc=2*(y>3).long()+(x>3).long()
        regions=[(y==3)|(x==3)|(((i>>loc)&1).bool()) for i in range(16)]
    return [torch.cat((torch.zeros(1,dtype=torch.long),v.flatten().nonzero().flatten()+1)).cuda() for v in regions]


def readings(cfg,vision,paths,which,previous):
    missing=[p for p in paths if p not in previous]
    if not missing:return torch.stack([previous[p] for p in paths])
    uid=hashlib.sha256(('native-regional-sets-v1/'+which+'\n'+'\n'.join(missing)).encode()).hexdigest()
    file=cfg.cache/('common-depiction-'+uid+'.pt')
    if file.exists():
        saved=torch.load(file,weights_only=True)
        assert saved['paths']==missing
        features=saved['features']
    else:
        parts=masks(which)
        batches=DataLoader(Images(missing,normal_transform()),batch_size=56,num_workers=2,
            pin_memory=True,worker_init_fn=configure_image_worker)
        out=[]
        for px in batches:
            px=px.cuda(non_blocking=True).to(vision.conv1.weight.dtype)
            tokens=vision._embeds(px)
            z=torch.empty((len(px),len(parts),512),device=px.device,dtype=torch.float16)
            for count in sorted({len(m) for m in parts}):
                ids=[i for i,p in enumerate(parts) if len(p)==count]
                subset=tokens[:,torch.stack([parts[i] for i in ids])].flatten(0,1)
                h=vision.transformer(subset)
                z[:,ids]=(vision.ln_post(h[:,0])@vision.proj).reshape(len(px),len(ids),-1).half()
            if which=='photo' and not out:
                torch.testing.assert_close(F.normalize(z[:,-1].float()),F.normalize(vision(px).float()),atol=5e-4,rtol=1e-4)
            out.append(z.cpu())
        features=torch.cat(out)
        save_tensor(dict(paths=missing,features=features),file)
    previous.update(zip(missing,features))
    return torch.stack([previous[p] for p in paths])


def query_observations(raw,src,scale,fold):
    logits=scale*(F.normalize(raw.float(),dim=-1)@src.T)
    q0=logits.masked_fill(~fold,-torch.inf).softmax(-1)
    q1=logits.masked_fill(fold,-torch.inf).softmax(-1)
    a=torch.stack((q0[:,0],q1[:,0],q0[:,2],q1[:,2]),1)
    b=torch.stack((q1[:,1],q0[:,1],q1[:,3],q0[:,3]),1)
    assert ((a*b)==0).all()
    return a,b


def likelihood(a,b,bank,size,shuffle):
    full=bank[:,-1]
    r=bank/bank.sum(1,keepdim=True).clamp_min(1e-30)
    h,rho=project(r,full)
    def components(h,rho):
        za=a@h.transpose(1,2)
        zb=b@h.transpose(1,2)
        first=(za*rho[:,None]).sum(-1)
        second=(zb*rho[:,None]).sum(-1)
        shared=(za*zb*rho[:,None]).sum(-1)
        return za,zb,first,second,shared
    za,zb,first,second,shared=components(h,rho)
    baseline=(a*full[:,None]).sum(-1)*(b*full[:,None]).sum(-1)
    tiny=1e-30
    torch.testing.assert_close((first*second).clamp_min(tiny).log(),baseline.clamp_min(tiny).log(),atol=5e-5,rtol=1e-6)
    pa=za*rho[:,None]/first[:,:,None].clamp_min(tiny)
    pb=zb*rho[:,None]/second[:,:,None].clamp_min(tiny)
    bf=(pa*pb/rho[:,None].clamp_min(tiny)).sum(-1)
    torch.testing.assert_close(bf,shared/baseline.clamp_min(tiny),atol=1e-5,rtol=1e-5)
    mass=rho@size
    ea=(za*rho[:,None])@size/mass[:,None]
    eb=(zb*rho[:,None])@size/mass[:,None]
    size_only=(ea*eb*rho[:,None]).sum(-1)
    fake,rf=project(r.gather(1,shuffle[None].expand_as(r)),full)
    bogus=components(fake,rf)[-1]
    forced_a=a@r.transpose(1,2)
    forced_b=b@r.transpose(1,2)
    force=.5*(first*(forced_a*zb).sum(-1)+second*(forced_b*za).sum(-1))
    return torch.stack([v.mean(1).clamp_min(tiny).log() for v in (shared,baseline,size_only,bogus,force)],1)


def validate():
    g=torch.Generator().manual_seed(42)
    b=torch.rand(3,16,12,generator=g)
    b/=b.sum(-1,keepdim=True)
    q=torch.rand(3,4,12,generator=g)
    q/=q.sum(-1,keepdim=True)
    t=q.flip([-1])
    counts=torch.tensor([i.bit_count() for i in range(16)])
    group=(counts[:,None]==counts[None]).float()
    shuffle=relabel(counts,12)
    torch.testing.assert_close(likelihood(q,t,b,group,shuffle),likelihood(t,q,b,group,shuffle))
    uniform=torch.full_like(q,1/12.)
    value=likelihood(uniform,t,b,group,shuffle)
    # Under neutral evidence on one side, shared and independent observations have the same likelihood.
    torch.testing.assert_close(value[:,0],value[:,1],atol=1e-6,rtol=1e-5)


def assess(meta,score):
    sign=2*meta[:,3:4]-1
    margin=sign*(score[:,:,0]-score[:,:,1])
    hit=(margin>1e-7).double()+.5*(margin.abs()<=1e-7)
    unique,count=meta[:,0].unique(return_counts=True)
    both=(meta[:,0,None]==unique[count==2][None]).T
    bal=both.double()@hit/2
    labels=meta[both.long().argmax(1),5]
    out={}
    for i,name in enumerate(ARMS):
        gains=100*(bal[:,0]-bal[:,i])
        dmargin=margin[:,0]-margin[:,i]
        out[name]=dict(balanced=float(bal[:,i].mean()),
            true_margin=float(margin[:,i].mean()),shared_advantage=float(dmargin.mean()),
            shared_minus_balanced=float(gains.mean()),class_balanced=class_interval(gains,labels,seed=42),
            class_margin=class_interval((both.double()@dmargin)/2,labels,seed=42),
            resolved=int(((hit[:,0]==1)&(hit[:,i]==0)).sum()),damaged=int(((hit[:,0]==0)&(hit[:,i]==1)).sum()),
            original_rescued=float(hit[meta[:,3]==0,i].sum()),original_damaged=float((1-hit[meta[:,3]==1,i]).sum()))
    return dict(both=len(bal),queries=len(unique),pairs=len(meta),arms=out)


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    validate()
    w=Path('/content/sbir')
    root=w/'results/semantic-common-depiction'
    root.mkdir(parents=True,exist_ok=True)
    cfg=runtime(w,'sketchy')
    paths,qs=merge_inputs(cfg.root,SKETCHY_SPLITS,'sketch')
    ph,gs=merge_inputs(cfg.root,SKETCHY_SPLITS,'photo')
    model=load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda'))
    scale=model.logit_scale.exp().float()
    words=read_words(cfg.vocabulary)
    src,tgt=[v.cuda() for v in load_words(cfg,words)]
    anchors=torch.randperm(len(words),generator=torch.Generator().manual_seed(42))[:2]
    order=(tgt@(tgt[anchors[0]]-tgt[anchors[1]])).argsort(stable=True)
    fold=torch.zeros(len(words),device='cuda',dtype=torch.bool)
    fold[order.tensor_split(2)[0]]=True
    counts=torch.tensor([i.bit_count() for i in range(16)])
    size=(counts[:,None]==counts[None]).float().cuda()
    shuffle=relabel(counts,len(words)).cuda()
    refs={s:torch.load(w/'results/gcwe-action-category-transfer'/(s+'.pt'),weights_only=True) for s in qs}
    qu=torch.cat([r['meta'][:,0] for r in refs.values()]).unique()
    im=torch.cat([r['meta'][:,1:3].flatten() for r in refs.values()]).unique()
    sketches=readings(cfg,model.visual,[paths[i] for i in qu.tolist()],'sketch',{})
    old={}
    for f in (w/'cache/regional-semantic-interactions').glob('*.pt'):
        d=torch.load(f,weights_only=True)
        if Path(d['paths'][0]).parent.parent.name=='photo':old.update(zip(d['paths'],d['features']))
    photos=readings(cfg,model.visual,[ph[i] for i in im.tolist()],'photo',old)
    for split,ref in refs.items():
        meta=ref['meta']
        assert torch.equal(ref['pi'],gs[split][0]) and torch.isin(meta[:,0],ref['qi']).all()
        rows=[]
        for b in meta.split(24):
            a,bb=query_observations(sketches[torch.searchsorted(qu,b[:,0].contiguous())].cuda(),src,scale,fold)
            sc=[]
            for i in (1,2):
                p=photos[torch.searchsorted(im,b[:,i].contiguous())].cuda().float()
                bank=(scale*(F.normalize(p,dim=-1)@tgt.T)).softmax(-1)
                sc.append(likelihood(a,bb,bank,size,shuffle))
            rows.append(torch.stack(sc,2).cpu())
        raw=torch.cat(rows).double()
        result=assess(meta,raw)
        save_tensor(dict(qi=ref['qi'],pi=ref['pi'],meta=meta,scores=raw),root/(split+'.pt'))
        atomic_json(root/(split+'.json'),dict(commit=revision(),seed=42,gpu=torch.cuda.get_device_name(),arms=ARMS,
            word_axis=anchors.tolist(),assessment=result))
        print(split,result['arms']['shared']['balanced'],flush=True)


if __name__=='__main__':
    run()
