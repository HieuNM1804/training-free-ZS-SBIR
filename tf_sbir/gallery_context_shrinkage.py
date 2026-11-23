"""Shrink the gallery contribution to a pair margin using single-gallery perturbations."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .branch_view_policy import outputs
from .branch_view_study import PrefixReplay
from .gallery_panel_resume import region_views,semantic_inputs
from .gallery_region_panels import CONFIG,Index
from .positive_foil import base_parts,raw_ranks,STRONG
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd


def propagate(images,evidence):
    index=Index(SimpleNamespace(p=images),rgb_only=True)
    return torch.cat([index.score(CONFIG,x.cuda()).clamp_min(1e-30).log().cpu() for x in evidence.split(192)])


def score_pairs(direct,native,perturbed,folds,anchors):
    # Evaluate each item's margins against the same uniform panel of reference
    # photos. A common shift of all scores within a context cancels in a margin.
    residual=perturbed-native[:,None,:]
    gain=native-direct
    diff_gain=gain[:,:,None]-gain[:,None,anchors]
    same=(folds[:,:,None]==folds[:,None,anchors]).float()
    count=same.sum(0)
    common=count.clamp_min(1)
    joint=torch.zeros_like(diff_gain)
    second=torch.zeros_like(joint)
    mean_i=torch.zeros_like(joint)
    mean_a=torch.zeros_like(joint)
    for j in range(len(folds)):
        u=residual[:,j,:,None]
        v=residual[:,j,None,anchors]
        joint+=(u-v).square()*same[j]
        second+=(u.square()+v.square())*same[j]
        mean_i+=u*same[j]
        mean_a+=v*same[j]
    shared=joint/common
    independent=(second/common-2*mean_i*mean_a/common.square()).clamp_min(0)
    present=(count>0)
    def fill_missing(var):
        return torch.where(present,var,(var*present).sum((1,2),keepdim=True)/present.sum())
    variance=fill_missing(shared)
    # Use the spread in contextual correction across the current query's
    # candidate pairs to estimate the signal variance of a centered normal prior.
    signal=(diff_gain.square().mean((1,2),keepdim=True)-variance.mean((1,2),keepdim=True)).clamp_min(0)
    def trust(noise):
        denom=signal+noise
        return torch.where(denom>1e-12,signal/denom.clamp_min(1e-12),torch.ones_like(denom))
    weight=trust(variance)
    avg=weight.mean((1,2),keepdim=True)
    offset=direct+gain[:,anchors].mean(1,keepdim=True)
    def estimates(weights):
        return offset+(diff_gain*weights).mean(2)
    return {'baseline':native,'joint':estimates(weight),'same_mean_trust':estimates(avg),
            'independent':estimates(trust(fill_missing(independent)))},avg.flatten()


@torch.inference_mode()
def run(w,ds,whole):
    cfg,paths,qs,_,gs,q,p,src,tgt,scale=semantic_inputs(w,ds)
    kd,_=load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    if ds=='tuberlin' and not whole:raise ValueError('Only frozen full transfer')
    result={}
    for s,(qi,labels) in qs.items():
        pi,gold=gs[s]
        if ds=='tuberlin' and (len(gold.unique())!=30 or len(labels.unique())!=30):raise ValueError('Incomplete TU transfer')
        rows=torch.arange(len(qi)) if whole else torch.randperm(len(qi),generator=torch.Generator().manual_seed(87243))[:768].sort().values
        g=F.normalize(p[pi].cuda())
        bank=torch.cat([(scale*(part@tgt.T)).softmax(1) for part in g.split(256)])
        views,_=region_views(w,s,cfg,paths,qi,q)
        mass=[]
        single=[]
        for ix in rows.split(128):
            a=(scale*(q[qi[ix]].cuda()@src.T)).softmax(1)
            b=(scale*(views['flipped'][ix].float().cuda()@src.T)).softmax(1)
            mass.append((((a+b)/2)@bank.T).cpu())
            single.append((a@bank.T).cpu())
        del bank
        mass=torch.cat(mass)
        native=propagate(g,mass)
        original=propagate(g,torch.cat(single))
        del single
        estimates=[]
        membership=[]
        for j in range(8):
            perm=torch.randperm(len(pi),generator=torch.Generator().manual_seed(69201+j))
            record=torch.empty_like(native)
            fold_map=torch.empty(len(pi),dtype=torch.long)
            for i,idx in enumerate(perm.tensor_split(2)):
                record[:,idx]=propagate(g[idx],mass[:,idx])
                fold_map[idx]=i
            estimates.append(record)
            membership.append(fold_map)
        folds=torch.stack(membership).cuda()
        anchors=torch.randperm(len(pi),generator=torch.Generator().manual_seed(1862))[:64].cuda()
        direct=mass.clamp_min(1e-30).log()
        del mass
        folder,tree,regional=base_parts(w,s)
        replay=PrefixReplay(kd,gold,s)
        values={}
        fixed={}
        trusts=[]
        for start in range(0,len(qi),192):
            local=(rows>=start)&(rows<start+192)
            if not local.any():continue
            pack=torch.load(folder/'full'/f'{start:06d}.pt',mmap=True,weights_only=True)
            for positions in local.nonzero().flatten().split(6):
                ix=rows[positions]
                off=ix-start
                assert torch.equal(pack['ids'][off],ix)
                reg=regional[ix].cuda()
                _,old,_=next(outputs(tree,pack['densities'][off].cuda(),reg,reg,[STRONG]))
                cell=(old-reg)/reg.std(1,keepdim=True)
                readings=torch.stack([sample[positions] for sample in estimates],1).cuda()
                choices,weight=score_pairs(direct[positions].cuda(),native[positions].cuda(),readings,folds,anchors)
                trusts.append(weight.cpu())
                prior=None
                for name,v in choices.items():
                    updated=reg+.65*(v-original[positions].cuda())
                    score=updated+updated.std(1,keepdim=True)*cell
                    rank,quick=raw_ranks(score,labels[ix],gold.cuda(),s)
                    kd_values=replay.chunk((rank.cpu(),labels[ix]))
                    assert (kd_values-quick).abs().max()<2e-6
                    values.setdefault(name,[]).append(kd_values)
                    correct=gold[rank[:,0].cpu()]==labels[ix]
                    if name=='baseline':prior=correct
                    fixed.setdefault(name,[]).append(torch.stack([~prior&correct,prior&~correct],1))
        z={k:torch.cat(v).double() for k,v in values.items()}
        # The sorted rows and monotonically visited chunks preserve query identity.
        y=labels[rows]
        signed={}
        for control in ('baseline','same_mean_trust','independent'):
            diff=100*(z['joint']-z[control])
            by_class=torch.stack([diff[y==j].mean(0) for j in y.unique()])
            boot=torch.randint(len(by_class),(10000,len(by_class)),generator=torch.Generator().manual_seed(1862))
            signed[control]={'class_mean':by_class.mean(0).tolist(),
                 'class_ci95':by_class[boot].mean(1).quantile(torch.tensor([.025,.975],dtype=torch.double),dim=0).tolist()}
        result[s]={'queries':len(rows),'gallery':len(gold),'official_all_sampled':True,
                   'query_selection':'all' if whole else '768 uniform, seed 87243',
                   'metrics':{k:(100*v.mean(0)).tolist() for k,v in z.items()},
                   'top1_fixed_damaged':{k:torch.cat(v).sum(0).tolist() for k,v in fixed.items()},
                   'mean_trust':float(torch.cat(trusts).mean()),'joint_minus_control':signed}
        print(s,json.dumps(result[s]),flush=True)
        del estimates,record,native,original,direct
    out=w/'results/gallery-context-margin-shrinkage'
    out.mkdir(parents=True,exist_ok=True)
    (out/(ds+('-full' if whole else '-pilot')+'.json')).write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',default='sketchy',choices=['sketchy','tuberlin'])
    parser.add_argument('--full',action='store_true')
    args=parser.parse_args()
    run(Path('/content/sbir'),args.dataset,args.full)
