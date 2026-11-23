"""Forecast native-gallery order reversals from perturbations of one gallery alone."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .gallery_context_reversals import coordinates
from .gallery_panel_resume import region_views,semantic_inputs
from .gallery_region_panels import CONFIG,Index
from .data import UNSEEN_CLASSES


def diffusion(image, evidence, ids):
    graph=Index(SimpleNamespace(p=image),rgb_only=True)
    return torch.cat([graph.score(CONFIG,e).clamp_min(1e-30).log()[:,ids].cpu()
                      for e in evidence.split(192)])


def auc_influence(y,x):
    pos=x[y].sort().values
    neg=x[~y].sort().values
    left=torch.searchsorted(neg,x[y],right=False)
    right=torch.searchsorted(neg,x[y],right=True)
    positive=(left+right).double()/(2*len(neg))
    left=torch.searchsorted(pos,x[~y],right=False)
    right=torch.searchsorted(pos,x[~y],right=True)
    negative=1-(left+right).double()/(2*len(pos))
    value=positive.mean()
    gradient=torch.zeros(len(x),dtype=torch.double)
    gradient[y]=(positive-value)/len(pos)
    gradient[~y]=(negative-value)/len(neg)
    return value,gradient


def summarize(risks, flips, group, objects):
    def auc(y,x):
        return float(roc_auc_score(y.numpy(),x.numpy())) if y.any() and (~y).any() else None
    by_reader={}
    for name,risk in risks.items():
        by_reader[name]={'roc_auc':auc(flips,risk),'class_aucs':{}}
        for c in group.unique():
            mask=group==c
            value=auc(flips[mask],risk[mask])
            by_reader[name]['class_aucs'][int(c)]=value
        top=risk.argsort(descending=True,stable=True)[:max(1,len(risk)//10)]
        by_reader[name]['flip_fraction_top_tenth']=float(flips[top].double().mean())
    # Evaluate variation at comparable native margins and within category.
    matched=[]
    strata=[]
    for c in group.unique():
        ids=(group==c).nonzero().flatten()
        for part in ids[risks['small_native_margin'][ids].argsort()].tensor_split(10):
            if not len(part):continue
            vals=[auc(flips[part],risks[name][part]) for name in risks]
            if all(v is not None for v in vals):
                matched.append((len(part),vals))
                strata.append(part)
    by_reader['within_class_and_margin_decile']={
        'n_bins_with_both_outcomes':len(matched),
        'pairs_covered':sum(v[0] for v in matched),
        'weighted_auc':dict(zip(risks,np.average([v[1] for v in matched],axis=0,weights=[v[0] for v in matched]).tolist()))}
    contrast={}
    for title,blocks in [('pooled',[torch.arange(len(flips))]),('within_class_and_margin_decile',strata)]:
        grad=torch.zeros(len(flips),dtype=torch.double)
        total=sum(map(len,blocks))
        delta=0.
        for ix in blocks:
            prop,phi=auc_influence(flips[ix],risks['variation_relative_to_native_margin'][ix])
            ctrl,psi=auc_influence(flips[ix],risks['small_native_margin'][ix])
            grad[ix]=len(ix)/total*(phi-psi)
            delta+=float((prop-ctrl)*len(ix)/total)
        _,groups=objects.unique(return_inverse=True)
        summed=torch.zeros(int(groups.max())+1,dtype=grad.dtype).scatter_add_(0,groups,grad)
        rng=torch.Generator().manual_seed(61382)
        signs=torch.randint(2,(10000,len(summed)),generator=rng).double()*2-1
        ci=(delta+signs@summed).quantile(torch.tensor([.025,.975],dtype=grad.dtype))
        contrast[title]={'gain_over_small_margin':delta,'object_cluster_multiplier_ci95':ci.tolist(),
                         'sketch_source_objects':len(summed)}
    by_reader['paired_auc_contrast']=contrast
    return by_reader


@torch.inference_mode()
def run():
    w=Path('/content/sbir')
    cfg,paths,qs,_,gs,q,p,src,tgt,scale=semantic_inputs(w,'sketchy')
    ss=('sketchy_1','sketchy_2')
    common=lambda z:torch.tensor(sorted(set(z[ss[0]][0].tolist())&set(z[ss[1]][0].tolist())))
    cq,cp=common(qs),common(gs)
    observations={}
    for s in ss:
        qi,labels=qs[s]
        pi,gold=gs[s]
        qloc,ploc=coordinates(qi,cq),coordinates(pi,cp)
        views,_=region_views(w,s,cfg,paths,qi,q)
        g=F.normalize(p[pi].cuda())
        bank=torch.cat([(scale*(part@tgt.T)).softmax(1) for part in g.split(256)])
        evidence=[]
        for batch in qloc.split(128):
            a=(scale*(q[qi[batch]].cuda()@src.T)).softmax(1)
            b=(scale*(views['flipped'][batch].float().cuda()@src.T)).softmax(1)
            evidence.append(((a+b)/2)@bank.T)
        evidence=torch.cat(evidence)
        orig=diffusion(g,evidence,ploc)
        trials=[]
        assignments=[]
        for draw in range(8):
            shuffled=torch.randperm(len(pi),generator=torch.Generator().manual_seed(69201+draw))
            folds=shuffled.tensor_split(2)
            scores=torch.empty_like(orig)
            membership=torch.full((len(ploc),),-1,dtype=torch.long)
            for i,fold in enumerate(folds):
                # Keep candidates in their native fold. Each comparison later
                # uses only draws in which both of its items survived together.
                positions=torch.full((len(pi),),-1,dtype=torch.long)
                positions[fold]=torch.arange(len(fold))
                at=positions[ploc]
                included=at>=0
                scores[:,included]=diffusion(g[fold],evidence[:,fold],at[included])
                membership[included]=i
            assert (membership>=0).all()
            assignments.append(membership)
            trials.append(scores)
        observations[s]=(orig,trials,torch.stack(assignments),evidence[:,ploc].cpu())
        del evidence,bank
        print(s,'8 native half-gallery repeats',flush=True)
    # From here onward join independently made predictions on identical photo
    # and sketch IDs, and reveal labels only for the evaluation.
    qloc,ploc=coordinates(qs[ss[0]][0],cq),coordinates(gs[ss[0]][0],cp)
    label=qs[ss[0]][1][qloc]
    truth=label[:,None]==gs[ss[0]][1][ploc][None,:]
    order=observations[ss[0]][3].topk(200).indices
    good=truth.gather(1,order)
    ii,jj=(good[:,:-1]^good[:,1:]).nonzero(as_tuple=True)
    left,right=order[ii,jj],order[ii,jj+1]
    positive=good[ii,jj].float()*2-1
    object_keys=[(Path(paths[k]).parent.name,Path(paths[k]).stem.rsplit('-',1)[0]) for k in cq]
    object_ids={key:i for i,key in enumerate(sorted(set(object_keys)))}
    object_groups=torch.tensor([object_ids[key] for key in object_keys])
    result={}
    for i,s in enumerate(ss):
        orig,trials,assignment,_=observations[s]
        other=observations[ss[1-i]][0]
        native=(orig[ii,left]-orig[ii,right])*positive
        external=(other[ii,left]-other[ii,right])*positive
        inside=assignment[:,left]==assignment[:,right]
        count=inside.sum(0)
        trial=torch.stack([m[ii,left]-m[ii,right] for m in trials])*positive
        delta=trial-native
        rms=((delta.square()*inside).sum(0)/count.clamp_min(1)).sqrt()
        usable=(count>=2)&(native!=0)&(external!=0)
        crossed=native*external<0
        risks={'small_native_margin':-native.abs(),'thinning_variation':rms,
               'variation_relative_to_native_margin':rms/native.abs().clamp_min(1e-9),
               'smoothed_fraction_of_thinned_reversals':(((trial*native<0)&inside).sum(0)+.5)/(count+1)}
        selected_label=label[ii][usable]
        result[s]={'eligible':int(usable.sum()),'all_adjacent_pairs':len(ii),
                   'sketches_covered':int(ii[usable].unique().numel()),'genuine_flips':int(crossed[usable].sum()),
                   'classes':{int(c):UNSEEN_CLASSES[ss[0]][c] for c in selected_label.unique()},
                   'risk':summarize({k:x[usable] for k,x in risks.items()},crossed[usable],selected_label,object_groups[ii[usable]])}
    out=w/'results/gallery-context-reversals'
    out.mkdir(exist_ok=True,parents=True)
    (out/'native-forecast.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    run()
