"""Test the predicted mechanism on exactly the same queries, endpoints and pair IDs."""
import json
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .gallery_context_reversals import coordinates
from .benchmark import merge_inputs
from .crossdomain_inputs import runtime
from .data import Images, SKETCHY_SPLITS, configure_image_worker, normal_transform
from .encoding import load_frozen_clip, load_global_inputs, save_tensor
from .foundation import load_words
from .semantic import read_words
from .gallery_region_panels import Index
from .public_context_standardization import SoftStandardizedDonors,endpoint_matrix


def audit_inputs(w,additional_samples=0,*,all_sketches=False,dataset='sketchy'):
    workers=min(4,len(os.sched_getaffinity(0)))
    cfg=runtime(w,dataset,batch=384,workers=workers)
    splits=SKETCHY_SPLITS if dataset=='sketchy' else ('tuberlin',)
    qp,qs=merge_inputs(cfg.root,splits,'sketch')
    pp,gs=merge_inputs(cfg.root,splits,'photo')
    def common(groups):
        return torch.tensor(sorted(set(groups['sketchy_1'][0].tolist()) & set(groups['sketchy_2'][0].tolist())))
    if dataset=='sketchy':
        cq,cp=common(qs),common(gs)
        assert (len(cq),len(cp))==(2231,2205)
    else:
        if not all_sketches or len(qs['tuberlin'][1].unique())!=30 or len(gs['tuberlin'][1].unique())!=30:
            raise ValueError('Only the full 30-class TU transfer is supported')
        cq,cp=qs['tuberlin'][0],gs['tuberlin'][0]
    if all_sketches:
        cq=torch.arange(len(qp))
    elif additional_samples:
        studies=[all_ids[torch.randperm(len(all_ids),generator=torch.Generator().manual_seed(87243))[:additional_samples]]
                 for all_ids,_ in qs.values()]
        cq=torch.cat([cq,*studies]).unique(sorted=True)
    paths=[qp[k] for k in cq]
    q,p,_=load_global_inputs(cfg,paths,pp,False)
    words=read_words(cfg.vocabulary)
    assert len(words)==20931
    src,tgt=(v.cuda() for v in load_words(cfg,words))
    model=load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda'))
    scale=float(model.logit_scale.exp())
    dl=DataLoader(Images(paths,normal_transform(),'rgb'),batch_size=cfg.batch_size,
                  num_workers=workers,pin_memory=True,worker_init_fn=configure_image_worker)
    flips=[]
    for pixels in dl:
        x=pixels.cuda(non_blocking=True).to(model.visual.conv1.weight.dtype)
        flips.append(F.normalize(model.visual(x.flip((-1,))).float()).half().cpu())
    return qs,gs,cq,cp,q,p,torch.cat(flips),src,tgt,scale


def fixed_boundaries(w,cq,cp,direct,truth):
    path=w/'results/gallery-context-reversals/fixed-boundaries.pt'
    if path.is_file():
        fixed=torch.load(path,weights_only=True)
        assert torch.equal(cq,fixed['queries']) and torch.equal(cp,fixed['photos'])
        return fixed,'restored'
    top=direct.topk(200).indices
    hit=truth.gather(1,top)
    rows,j=(hit[:,:-1]^hit[:,1:]).nonzero(as_tuple=True)
    fixed={'queries':cq,'photos':cp,'rows':rows,'left':top[rows,j],'right':top[rows,j+1],
           'orientation':hit[rows,j].float()*2-1}
    save_tensor(fixed,path)
    return fixed,'reconstructed from direct two-view word scores'


@torch.inference_mode()
def run():
    w=Path('/content/sbir')
    qs,gs,cq,cp,q,p,flipped,src,tgt,scale=audit_inputs(w)
    outcomes={}
    shared_mass={}
    for s in ('sketchy_1','sketchy_2'):
        qi,_=qs[s]
        pi,_=gs[s]
        cols=coordinates(pi,cp)
        g=F.normalize(p[pi].cuda())
        bank=torch.cat([(scale*(part@tgt.T)).softmax(1) for part in g.split(256)])
        index=Index(SimpleNamespace(p=g),rgb_only=True)
        native=endpoint_matrix(index)[cols]
        del index
        estimator=SoftStandardizedDonors(native,bank,cols.to(g.device))
        mats={'baseline':native,**estimator.kernels}
        # Identify shared endpoints only after compiling all score matrices. These
        # column sums describe exposure and never enter the retrieval weights.
        shared_mass[s]={k:v[:,cols].sum(1).cpu() for k,v in mats.items()}
        scores={k:[] for k in mats}
        direct=[]
        for ix in torch.arange(len(q)).split(48):
            a=(scale*(q[ix].cuda()@src.T)).softmax(1)
            b=(scale*(flipped[ix].float().cuda()@src.T)).softmax(1)
            e=((a+b)/2)@bank.T
            direct.append(e[:,cols].clamp_min(1e-30).log().cpu())
            for name,weights in mats.items():
                scores[name].append((e@weights.T).cpu())
        outcomes[s]={k:torch.cat(v) for k,v in scores.items()}
        outcomes[s]['direct']=torch.cat(direct)
        del estimator,mats,bank
    # Reveal class labels and the category-boundary fixture solely for evaluation.
    first,second=outcomes.values()
    a,b=shared_mass.values()
    qi,labels=qs['sketchy_1']
    pi,gold=gs['sketchy_1']
    y=labels[coordinates(qi,cq)]
    categories=gold[coordinates(pi,cp)]
    truth=y[:,None]==categories[None]
    fixed,source=fixed_boundaries(w,cq,cp,first['direct'],truth)
    i,l,r,ori=[fixed[k] for k in ('rows','left','right','orientation')]
    assert (truth[i,l]^truth[i,r]).all()
    assert torch.equal(ori,truth[i,l].float()*2-1)
    result={}
    for name in first:
        m1=(first[name][i,l]-first[name][i,r])*ori
        m2=(second[name][i,l]-second[name][i,r])*ori
        h1=categories[first[name].argmax(1)]==y
        h2=categories[second[name].argmax(1)]==y
        result[name]={'pairs':len(i),'flips':int((m1*m2<0).sum()),'ties':int((m1*m2==0).sum()),
                      'pair_correct':[int((m1>0).sum()),int((m2>0).sum())],
                      'shared_top1_correct':[int(h1.sum()),int(h2.sum())],
                      'shared_queries':len(y),'shared_top1_correctness_changed':int((h1^h2).sum()),
                      'mean_shared_mass':[float(a[name].mean()),float(b[name].mean())] if name in a else None,
                      'mean_abs_shared_mass_drift':float((a[name]-b[name]).abs().mean()) if name in a else None}
    boundary_sha=hashlib.sha256()
    for key in ('queries','photos','rows','left','right','orientation'):
        boundary_sha.update(fixed[key].contiguous().numpy().tobytes())
    result['protocol']={'device':torch.cuda.get_device_name(),'boundary_source':source,
                        'boundary_arrays_sha256':boundary_sha.hexdigest(),'shared_photos':len(cp)}
    root=w/'results/public-context-standardization'
    root.mkdir(parents=True,exist_ok=True)
    (root/'same-input-audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    run()
