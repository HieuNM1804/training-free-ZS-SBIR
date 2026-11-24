"""Test cross-word predictive dependence from an unobserved partial view of a frozen gallery photo."""
from pathlib import Path
import math

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


ARMS=('partial_same','independent','same_size','wrong_same_size','full_photo')


def probabilities(bank,words,size):
    observed=bank.gather(2,words[:,None].expand(-1,bank.shape[1],-1)).transpose(1,2)
    latent=observed/observed.sum(-1,keepdim=True).clamp_min(1e-30)
    counts=size.sum(1)
    size_uniform=latent@(size/counts[:,None])
    distinct=(size-torch.eye(len(size),device=size.device)) / (counts-1).clamp_min(1)[:,None]
    distinct=distinct+torch.diag((counts==1).float())
    other=latent@distinct
    prior=torch.full_like(latent,1/len(size))
    full=torch.zeros_like(latent)
    full[:,:,-1]=1
    posteriors=(latent,prior,size_uniform,other,full)
    for a in posteriors:
        torch.testing.assert_close(a.sum(-1),torch.ones_like(a[:,:,0]),atol=2e-6,rtol=1e-6)
    within=(latent*(latent.clamp_min(1e-30).log()-size_uniform.clamp_min(1e-30).log())).sum(-1)
    total=(latent*(latent.clamp_min(1e-30).log()+math.log(len(size)))).sum(-1)
    area=(size_uniform*(size_uniform.clamp_min(1e-30).log()+math.log(len(size)))).sum(-1)
    torch.testing.assert_close(total,within+area,atol=2e-6,rtol=1e-6)
    info=torch.stack((total,area,within,latent[:,:,[0,-1]].sum(-1)),2).mean(1)
    return observed,posteriors,info


def evaluate(bank,words,size,targets):
    observed,weights,info=probabilities(bank,words,size)
    return score_distinct(bank,words,targets,observed,weights),info


def score_distinct(bank,words,targets,observed,weights):
    scores=[]
    for mix in weights:
        # The observed word is outside the target alphabet. Zero its log contribution below.
        pred=(mix/(1-observed).clamp_min(1e-12))@bank
        self_prob=pred.gather(2,words[:,:,None]).squeeze(2)
        torch.testing.assert_close(pred.sum(-1)-self_prob,torch.ones_like(self_prob),atol=2e-5,rtol=1e-5)
        logp=pred.clamp_min(1e-30).log()
        logp.scatter_(2,words[:,:,None],0.)
        cross=logp@targets.transpose(1,2)
        excluded=targets.gather(2,words[:,None].expand(-1,len(targets[0]),-1)).transpose(1,2)
        assert (excluded<1).all()
        scores.append((cross/(1-excluded)).mean(1))
    return torch.stack(scores,2)


def validate(size):
    g=torch.Generator().manual_seed(42)
    row=torch.randn(2,1,13,generator=g).softmax(-1)
    bank=row.expand(-1,16,-1).contiguous()
    u=torch.tensor([[0,2],[2,4]])
    obs,w,info=probabilities(bank,u,size)
    torch.testing.assert_close(w[0],w[1],atol=1e-6,rtol=1e-6)
    result,_=evaluate(bank,u,size,row.expand(-1,2,-1))
    torch.testing.assert_close(result,result[:,:,0:1].expand_as(result))
    assert info[:,:3].abs().max()<1e-6
    different=torch.randn(1,16,11,generator=g).softmax(-1)
    u=torch.tensor([[3]])
    a,_=evaluate(different,u,size,different[:,4:5])
    b=different[0]
    pw=b[:,3]/b[:,3].sum()
    target=b[4].clone()
    target[3]=0
    target/=target.sum()
    prediction=sum(pw[i]*b[i]/(1-b[i,3]) for i in range(16))
    prediction[3]=0
    torch.testing.assert_close(a[0,0,0],(target*prediction.clamp_min(1e-30).log()).sum())


def analyze(meta,scores,info):
    out={}
    for m,name in enumerate(('same','different')):
        keep=meta[:,4]==m
        ids=meta[keep,5].numpy()
        for vi,view in enumerate(('own','redraw')):
            x=scores[keep,vi].numpy()
            margin=x[:,:,0]-x[:,:,1]
            out[name+'/'+view]=dict(n=len(x),pairs=len(np.unique(ids)),mean_log_scores=x.mean(0).tolist(),
                source_accuracy=((margin>1e-6)+.5*(np.abs(margin)<=1e-6)).mean(0).tolist(),
                information=info[keep].mean(0).tolist(),comparisons={k:dict(
                    true=cluster_effect(x[:,0,0]-x[:,j,0],ids),
                    discrimination=cluster_effect(margin[:,0]-margin[:,j],ids))
                    for j,k in enumerate(ARMS) if j})
    return out


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    w=Path('/content/sbir')
    root=w/'results/semantic-latent-depiction'
    root.mkdir(parents=True,exist_ok=True)
    counts=torch.tensor([int(i).bit_count() for i in range(16)])
    size=(counts[:,None]==counts[None,:]).float()
    validate(size)
    size=size.cuda()
    cfg=runtime(w,'sketchy')
    paths,qs=merge_inputs(cfg.root,SKETCHY_SPLITS,'sketch')
    photos,gs=merge_inputs(cfg.root,SKETCHY_SPLITS,'photo')
    cache={}
    for f in (w/'cache/regional-semantic-interactions').glob('*.pt'):
        item=torch.load(f,weights_only=True)
        cache.update(zip(item['paths'],item['features']))
    words=read_words(cfg.vocabulary)
    assert len(words)==20931
    src,tgt=[v.cuda() for v in load_words(cfg,words)]
    scale=load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda')).logit_scale.exp().float()
    def get(ids,p):
        return torch.stack([cache[p[i]] for i in ids.tolist()]).cuda().float()
    for split in qs:
        ref=torch.load(w/'results/regional-semantic-interactions'/(split+'.pt'),weights_only=True)
        assert torch.equal(qs[split][0],ref['qi']) and torch.equal(gs[split][0],ref['pi'])
        meta=ref['meta']
        rows,infos=[],[]
        for batch in meta.split(4):
            targets=torch.stack([(scale*(F.normalize(get(batch[:,j],paths)[:,-1],dim=-1)@src.T)).softmax(-1)
                                 for j in (0,1)],1)
            sampled=torch.stack([torch.multinomial(prob,128,replacement=True,generator=torch.Generator(device='cuda').manual_seed(42+qid))
                           for prob,qid in zip(targets[:,0],batch[:,0].tolist())])
            s,i=[],[]
            for j in (2,3):
                bank=(scale*(F.normalize(get(batch[:,j],photos),dim=-1)@tgt.T)).softmax(-1)
                score,inf=evaluate(bank,sampled,size,targets)
                s.append(score)
                i.append(inf)
            rows.append(torch.stack(s,-1).cpu())
            infos.append(torch.stack(i,-1).cpu())
        raw,inf=torch.cat(rows).double(),torch.cat(infos).double()
        result=analyze(meta,raw,inf)
        save_tensor(dict(meta=meta,qi=ref['qi'],pi=ref['pi'],scores=raw,information=inf),root/(split+'.pt'))
        atomic_json(root/(split+'.json'),dict(commit=revision(),gpu=torch.cuda.get_device_name(),seed=42,
            sample_words=128,vocabulary=len(words),arms=ARMS,info_columns=['total_kl','size_kl','within_size_kl','extreme_size_mass'],
            assessment=result))
        print(split,result['same/redraw']['comparisons']['same_size'],flush=True)


if __name__=='__main__':
    run()
