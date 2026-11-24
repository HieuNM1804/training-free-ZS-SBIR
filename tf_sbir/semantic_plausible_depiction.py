"""Preserve a candidate's intact content in expectation and test a common latent depiction."""
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from .benchmark import merge_inputs
from .concept_benchmark import atomic_json
from .crossdomain_benchmark import revision
from .crossdomain_inputs import runtime
from .data import SKETCHY_SPLITS
from .encoding import load_frozen_clip,save_tensor
from .foundation import load_words
from .regional_predictive_observation import cluster_effect
from .semantic import read_words
from .semantic_latent_depiction import score_distinct


ARMS=('shared','independent_view','size_only','intact','false_correspondence')


def relabel(size,n):
    gen=torch.Generator().manual_seed(42)
    mapping=torch.arange(len(size))[:,None].expand(-1,n).clone()
    for count in size.unique():
        subset=torch.where(size==count)[0]
        mapping[subset]=subset[torch.rand((n,len(subset)),generator=gen).argsort(1).T]
    return mapping


def project(r,a):
    j=a[:,None]*r
    rho=j.sum(-1)
    h=j/rho[:,:,None]
    torch.testing.assert_close((h*rho[:,:,None]).sum(1),a,atol=3e-7,rtol=2e-6)
    return h,rho


def validate():
    g=torch.Generator().manual_seed(42)
    bank=torch.randn(2,5,11,dtype=torch.double,generator=g).softmax(-1)
    a=bank[:,-1]
    r=bank/bank.sum(1,keepdim=True)
    h,rho=project(r,a)
    perm=torch.tensor([3,0,4,1,2])
    hp,rp=project(r[:,perm],a)
    torch.testing.assert_close(hp,h[:,perm])
    torch.testing.assert_close(rp,rho[:,perm])
    bf=r.transpose(1,2)@(r/rho[:,:,None])
    torch.testing.assert_close((bf*a[:,None]).sum(-1),torch.ones_like(a))
    torch.testing.assert_close(bf,bf.transpose(1,2))
    alt=a[:,None]*torch.randn(2,5,11,generator=g,dtype=torch.double).softmax(1)
    j=r*a[:,None]
    def kl(p,q):
        return (p*(p.log()-q.log())).sum((1,2))
    torch.testing.assert_close(kl(alt,bank/5),kl(alt,j)+kl(j,bank/5))


def forecast(bank,u,targets,group,shuffle):
    a=bank[:,-1]
    marginal=bank.mean(1)
    r=bank/bank.sum(1,keepdim=True).clamp_min(1e-30)
    idx=u[:,None,:].expand(-1,bank.shape[1],-1)
    h,rho=project(r,a)
    posterior=r.gather(2,idx).transpose(1,2)
    fresh=rho[:,None].expand_as(posterior)
    size_posterior=posterior@group
    size_prior=rho@group
    size_only=size_posterior*rho[:,None]/size_prior[:,None]
    torch.testing.assert_close(size_only.sum(-1),torch.ones_like(size_only[:,:,0]),atol=2e-6,rtol=1e-6)
    same=score_distinct(h,u,targets,h.gather(2,idx).transpose(1,2),(posterior,fresh,size_only))
    intact=score_distinct(a[:,None],u,targets,a.gather(1,u)[:,:,None],(torch.ones_like(u[:,:,None]).float(),))
    false_r=r.gather(1,shuffle[None].expand_as(r))
    torch.testing.assert_close(false_r.sum(1),r.sum(1),atol=2e-6,rtol=1e-6)
    fake,prior=project(false_r,a)
    post=false_r.gather(2,idx).transpose(1,2)
    torch.testing.assert_close(post@group,size_posterior,atol=2e-6,rtol=1e-6)
    false=score_distinct(fake,u,targets,fake.gather(2,idx).transpose(1,2),(post,))
    conditional=torch.cat((same,intact,false),2)
    given=a.gather(1,u).clamp_min(1e-30).log().mean(1)
    joint=conditional+given[:,None,None]
    torch.testing.assert_close(joint-joint[:,:,:1],conditional-conditional[:,:,:1],atol=4e-6,rtol=1e-6)
    drift=(a*(a.clamp_min(1e-30).log()-marginal.clamp_min(1e-30).log())).sum(1)
    return conditional,joint,torch.stack((drift,given),1)


def analyze(meta,cond,joint,info):
    out={}
    for mode,name in enumerate(('same','different')):
        keep=meta[:,4]==mode
        ids=meta[keep,5].numpy()
        for i,target in enumerate(('own','redraw')):
            c=cond[keep,i].numpy()
            j=joint[keep,i].numpy()
            d=c[:,:,0]-c[:,:,1]
            jd=j[:,:,0]-j[:,:,1]
            out[name+'/'+target]=dict(n=len(c),pairs=len(np.unique(ids)),conditional=c.mean(0).tolist(),joint=j.mean(0).tolist(),
                info=info[keep].mean(0).tolist(),conditional_accuracy=((d>1e-6)+.5*(np.abs(d)<=1e-6)).mean(0).tolist(),
                joint_accuracy=((jd>1e-6)+.5*(np.abs(jd)<=1e-6)).mean(0).tolist(),
                contrasts={k:dict(true=cluster_effect(c[:,0,0]-c[:,ix,0],ids),
                    discrimination=cluster_effect(d[:,0]-d[:,ix],ids)) for ix,k in enumerate(ARMS) if ix})
    return out


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    validate()
    w=Path('/content/sbir')
    root=w/'results/semantic-plausible-depiction'
    root.mkdir(parents=True,exist_ok=True)
    cfg=runtime(w,'sketchy')
    paths,qs=merge_inputs(cfg.root,SKETCHY_SPLITS,'sketch')
    photos,gs=merge_inputs(cfg.root,SKETCHY_SPLITS,'photo')
    cache={}
    for f in (w/'cache/regional-semantic-interactions').glob('*.pt'):
        item=torch.load(f,weights_only=True)
        cache.update(zip(item['paths'],item['features']))
    words=read_words(cfg.vocabulary)
    src,tgt=[v.cuda() for v in load_words(cfg,words)]
    groups=torch.tensor([x.bit_count() for x in range(16)])
    size=(groups[:,None]==groups[None,:]).float().cuda()
    shuffle=relabel(groups,len(words)).cuda()
    scale=load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda')).logit_scale.exp().float()
    def get(ids,paths):
        return torch.stack([cache[paths[i]] for i in ids.tolist()]).cuda().float()
    for split in qs:
        ref=torch.load(w/'results/semantic-latent-depiction'/(split+'.pt'),weights_only=True)
        meta=ref['meta']
        assert torch.equal(qs[split][0],ref['qi']) and torch.equal(gs[split][0],ref['pi'])
        rows,joints,infos=[],[],[]
        for batch in meta.split(4):
            targets=torch.stack([(scale*(F.normalize(get(batch[:,i],paths)[:,-1],dim=-1)@src.T)).softmax(-1)
                                 for i in (0,1)],1)
            u=torch.stack([torch.multinomial(t,128,replacement=True,generator=torch.Generator(device='cuda').manual_seed(42+qid))
                          for t,qid in zip(targets[:,0],batch[:,0].tolist())])
            endpoints=[]
            for i in (2,3):
                bank=(scale*(F.normalize(get(batch[:,i],photos),dim=-1)@tgt.T)).softmax(-1)
                endpoints.append(forecast(bank,u,targets,size,shuffle))
            for store,items in zip((rows,joints,infos),zip(*endpoints)):
                store.append(torch.stack(items,-1).cpu())
        c,j,info=[torch.cat(v).double() for v in (rows,joints,infos)]
        torch.testing.assert_close(c[:,:,3],ref['scores'][:,:,4],atol=1e-5,rtol=1e-6)
        result=analyze(meta,c,j,info)
        save_tensor(dict(meta=meta,qi=ref['qi'],pi=ref['pi'],conditional=c,joint=j,info=info),root/(split+'.pt'))
        atomic_json(root/(split+'.json'),dict(commit=revision(),seed=42,gpu=torch.cuda.get_device_name(),arms=ARMS,
            diagnostics=['intact_to_unanchored_kl','observed_word_log_likelihood'],assessment=result))
        print(split,result['different/redraw']['contrasts']['size_only'],flush=True)


if __name__=='__main__':
    run()
