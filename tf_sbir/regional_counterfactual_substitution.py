"""Measure counterfactual semantic binding above every pairwise marginal in a sketch/photo interchange."""
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .benchmark import merge_inputs
from .concept_benchmark import atomic_json
from .conditional_semantic_index import bare_words
from .crossdomain_benchmark import revision
from .crossdomain_inputs import runtime
from .data import Images,SKETCHY_SPLITS,configure_image_worker,normal_transform
from .depiction_path_diagnostic import class_interval
from .encoding import load_frozen_clip,save_tensor
from .semantic import read_words


NAMES=('query_interaction','pooled_interaction','lower_order','total','unmixed_js')


def binary_table(a,b,theta):
    # Use the positive-association root; swapping columns inverts a negative log odds.
    positive=theta>=0
    c=torch.where(positive,b,1-b)
    t=theta.abs().expm1()
    disc=1+2*t*(a+c-2*a*c)+(t*(a-c)).square()
    x=2*(1+t)*a*c/(1+t*(a+c)+disc.clamp_min(0).sqrt())
    return torch.where(positive,x,a-x)


def log_safe(x):
    return x.clamp_min(1e-300).log()


def connected(readout,query):
    readout=readout.double()
    marginal=readout.mean(1)
    p=readout/(4*marginal[:,None])
    a,b=p[:,2]+p[:,3],p[:,1]+p[:,3]
    target=torch.full_like(a[:,:1],.25)
    lo,hi=-32*torch.ones_like(target),32*torch.ones_like(target)
    def cell_sum(theta):
        return (marginal*binary_table(a,b,theta)).sum(1,keepdim=True)
    assert (cell_sum(lo)<=target+1e-9).all() and (cell_sum(hi)>=target-1e-9).all()
    for _ in range(34):
        mid=(lo+hi)*.5
        small=cell_sum(mid)<target
        lo,hi=torch.where(small,mid,lo),torch.where(small,hi,mid)
    theta=(lo+hi)*.5
    x=binary_table(a,b,theta)
    q=torch.stack((1-a-b+x,b-x,a-x,x),1).clamp_min(0)
    model=q*marginal[:,None]
    original=.25*readout
    torch.testing.assert_close(model.sum(2),original.sum(2),atol=8e-9,rtol=0)
    for i,j in ((0,1),(0,2),(2,3),(1,3)):
        torch.testing.assert_close(model[:,i]+model[:,j],original[:,i]+original[:,j],atol=2e-12,rtol=1e-9)
    supported=q.amin(1)>1e-7
    common=log_safe(q[:,0])+log_safe(q[:,3])-log_safe(q[:,1])-log_safe(q[:,2])
    if supported.any():assert ((common-theta).abs()[supported].max()<5e-7)
    local=(p*(log_safe(p)-log_safe(q))).sum(1)
    three=(marginal*local).sum(1)
    entropy_gap=(original*log_safe(original)-model*log_safe(model)).sum((1,2))
    torch.testing.assert_close(three,entropy_gap,atol=8e-8,rtol=1e-6)
    total=(original*(log_safe(p)+math.log(4))).sum((1,2))
    parents=readout[:,[0,3]]
    mean=parents.mean(1,keepdim=True)
    js=.5*(parents*(log_safe(parents)-log_safe(mean))).sum((1,2))
    return torch.stack(((query.double()*local).sum(1),three,total-three,total,js),1)


def validate():
    rng=torch.Generator().manual_seed(42)
    left=torch.randn(3,2,13,generator=rng,dtype=torch.double)
    right=torch.randn(3,2,13,generator=rng,dtype=torch.double)
    joint=(left[:,:,None]+right[:,None,:]).reshape(3,4,13).softmax(-1)
    query=joint[:,0]
    fitted=connected(joint,query)
    assert fitted[:,:2].abs().max()<1e-8
    joint=joint.clone()
    joint[:,3]=joint[:,3].roll(3,1)
    fitted=connected(joint,query)
    assert (fitted[:,:2]>0).all()
    torch.testing.assert_close(connected(joint[:,[3,2,1,0]],query),fitted,atol=1e-8,rtol=1e-6)


def input_tokens(paths,vision):
    loader=DataLoader(Images(paths,normal_transform()),batch_size=96,num_workers=2,pin_memory=True,
                      worker_init_fn=configure_image_worker)
    return torch.cat([vision._embeds(px.cuda(non_blocking=True).to(vision.conv1.weight.dtype)).cpu() for px in loader])


def axes(device):
    y,x=torch.meshgrid(torch.arange(7,device=device),torch.arange(7,device=device),indexing='ij')
    result=[]
    for pos in (x,y):
        ids=pos.flatten().ne(3).nonzero().flatten()
        left=torch.cat((torch.ones(1,dtype=torch.bool,device=device),pos.flatten()[ids]<3))
        result.append((torch.cat((torch.zeros(1,dtype=torch.long,device=device),ids+1)),left))
    return result


def exchanged(s,p,idx,left):
    return torch.stack([torch.where(left[None,:,None],a[:,idx],b[:,idx]) for a,b in ((s,s),(s,p),(p,s),(p,p))],1)


def lexical(tokens,vision,words,scale):
    z=vision.transformer(tokens)
    h=F.normalize((vision.ln_post(z[:,0])@vision.proj).float(),dim=-1)
    # Compute word probabilities in float64, so the conditional odds survive close to the simplex boundaries.
    return (scale*(h@words.T)).double().softmax(-1)


def assess(meta,raw):
    direction=(2*meta[:,3]-1)[:,None]
    margin=(raw[:,:,1]-raw[:,:,0])*direction
    success=(margin>1e-10).double()+.5*(margin.abs()<=1e-10)
    u,c=meta[:,0].unique(return_counts=True)
    supported=(meta[:,0,None]==u[c==2][None]).T
    balanced=supported.double()@success/2
    labels=meta[supported.long().argmax(1),5]
    result={}
    for i,name in enumerate(NAMES):
        diff=100*(balanced[:,0]-balanced[:,i])
        result[name]=dict(balanced=float(balanced[:,i].mean()),query_gain=float(diff.mean()),
            class_gain=class_interval(diff,labels,seed=42),mean_margin=float(margin[:,i].mean()),
            mean_relevant=float(torch.where(direction[:,0]>0,raw[:,i,0],raw[:,i,1]).mean()),
            mean_foil=float(torch.where(direction[:,0]>0,raw[:,i,1],raw[:,i,0]).mean()),
            rescued=float(success[meta[:,3]==0,i].sum()),damaged=float((1-success[meta[:,3]==1,i]).sum()))
    chance=class_interval(100*(balanced[:,0]-.5),labels,seed=42)
    return dict(pairs=len(meta),queries=len(u),both=len(balanced),arms=result,class_primary_minus_chance=chance)


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    validate()
    w=Path('/content/sbir')
    root=w/'results/regional-counterfactual-substitution'
    root.mkdir(parents=True,exist_ok=True)
    cfg=runtime(w,'sketchy')
    paths,qs=merge_inputs(cfg.root,SKETCHY_SPLITS,'sketch')
    photos,gs=merge_inputs(cfg.root,SKETCHY_SPLITS,'photo')
    vision=load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda')).visual
    words,scale=bare_words(cfg,read_words(cfg.vocabulary))
    ref={s:torch.load(w/'results/gcwe-action-category-transfer'/(s+'.pt'),weights_only=True) for s in qs}
    qu=torch.cat([r['meta'][:,0] for r in ref.values()]).unique()
    ph=torch.cat([r['meta'][:,1:3].flatten() for r in ref.values()]).unique()
    st=input_tokens([paths[i] for i in qu.tolist()],vision)
    pt=input_tokens([photos[i] for i in ph.tolist()],vision)
    tests=axes(words.device)
    sx=Images([paths[int(qu[0])]],normal_transform())[0][None].cuda().to(vision.conv1.weight.dtype)
    px=Images([photos[int(ph[0])]],normal_transform())[0][None].cuda().to(vision.conv1.weight.dtype)
    for d,(idx,left) in enumerate(tests):
        mixed=sx.clone()
        if d==0:mixed[:,:,:,128:]=px[:,:,:,128:]
        else:mixed[:,:,128:,:]=px[:,:,128:,:]
        joined=exchanged(vision._embeds(sx),vision._embeds(px),idx,left)
        torch.testing.assert_close(vision._embeds(mixed)[:,idx],joined[:,1],atol=0,rtol=0)
        z=lexical(joined.flatten(0,1),vision,words,scale).view(1,4,-1)
        z0=lexical(vision._embeds(sx)[:,idx],vision,words,scale)
        torch.testing.assert_close(z[:,0],z0,atol=5e-4,rtol=5e-3)
    for s,r in ref.items():
        meta=r['meta']
        assert torch.equal(r['pi'],gs[s][0]) and torch.isin(meta[:,0],r['qi']).all()
        rows=[]
        for batch in meta.split(20):
            q=st[torch.searchsorted(qu,batch[:,0].contiguous())].cuda()
            query=lexical(q,vision,words,scale)
            candidates=[]
            for c in (1,2):
                p=pt[torch.searchsorted(ph,batch[:,c].contiguous())].cuda()
                directions=[]
                for idx,left in tests:
                    pixels=exchanged(q,p,idx,left)
                    lex=lexical(pixels.flatten(0,1),vision,words,scale).reshape(len(q),4,-1)
                    directions.append(connected(lex,query))
                candidates.append(torch.stack(directions,1))
            rows.append(torch.stack(candidates,-1).cpu())
        per_axis=torch.cat(rows)
        raw=per_axis.mean(1)
        result=assess(meta,raw)
        save_tensor(dict(meta=meta,qi=r['qi'],pi=r['pi'],axes=per_axis,scores=raw),root/(s+'.pt'))
        atomic_json(root/(s+'.json'),dict(commit=revision(),seed=42,gpu=torch.cuda.get_device_name(),arms=NAMES,assessment=result))
        print(s,result['arms']['query_interaction'],flush=True)


if __name__=='__main__':
    run()
