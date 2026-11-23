"""Test the predicted mechanism on exactly the same queries, endpoints and pair IDs."""
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .gallery_context_reversals import coordinates
from .gallery_panel_resume import region_views,semantic_inputs
from .gallery_region_panels import Index
from .public_context_standardization import SoftStandardizedDonors,endpoint_matrix


@torch.inference_mode()
def run():
    w=Path('/content/sbir')
    cfg,paths,qs,_,gs,q,p,src,tgt,scale=semantic_inputs(w,'sketchy')
    fixed=torch.load(w/'results/gallery-context-reversals/fixed-boundaries.pt',weights_only=True)
    cq,cp=fixed['queries'],fixed['photos']
    outcomes={}
    shared_mass={}
    for s in ('sketchy_1','sketchy_2'):
        qi,_=qs[s]
        pi,_=gs[s]
        rows,cols=coordinates(qi,cq),coordinates(pi,cp)
        views,_=region_views(w,s,cfg,paths,qi,q)
        g=F.normalize(p[pi].cuda())
        bank=torch.cat([(scale*(part@tgt.T)).softmax(1) for part in g.split(256)])
        index=Index(SimpleNamespace(p=g),rgb_only=True)
        estimator=SoftStandardizedDonors(endpoint_matrix(index),bank)
        del index
        mats={'baseline':estimator.w[cols],**{k:v[cols] for k,v in estimator.kernels.items()}}
        # Identify shared endpoints only after compiling all score matrices. These
        # column sums describe exposure and never enter the retrieval weights.
        shared_mass[s]={k:v[:,cols].sum(1).cpu() for k,v in mats.items()}
        scores={k:[] for k in mats}
        for ix in rows.split(128):
            a=(scale*(q[qi[ix]].cuda()@src.T)).softmax(1)
            b=(scale*(views['flipped'][ix].float().cuda()@src.T)).softmax(1)
            e=((a+b)/2)@bank.T
            for name,weights in mats.items():
                scores[name].append((e@weights.T).cpu())
        outcomes[s]={k:torch.cat(v) for k,v in scores.items()}
        del estimator,mats,bank
    # Reveal class labels and the category-boundary fixture solely for evaluation.
    i,l,r,ori=[fixed[k] for k in ('rows','left','right','orientation')]
    first,second=outcomes.values()
    a,b=shared_mass.values()
    qi,labels=qs['sketchy_1']
    pi,gold=gs['sketchy_1']
    y=labels[coordinates(qi,cq)]
    categories=gold[coordinates(pi,cp)]
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
                      'mean_shared_mass':[float(a[name].mean()),float(b[name].mean())],
                      'mean_abs_shared_mass_drift':float((a[name]-b[name]).abs().mean())}
    root=w/'results/public-context-standardization'
    root.mkdir(parents=True,exist_ok=True)
    (root/'same-input-audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    run()
