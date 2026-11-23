"""Frozen horizontal-flip control, evaluated through original KD metrics by rank replay."""
from pathlib import Path
import json
import torch
from .gallery_panel_resume import semantic_inputs, region_views
from .retrieval_evidence import load_kd

@torch.inference_mode()
def run():
    w = Path('/content/sbir')
    cfg, paths, splits, _, galleries, sketches, photos, src, tgt, scale = semantic_inputs(w,'tuberlin')
    kd,_ = load_kd(w)
    for split,(qi,y) in splits.items():
        pi,cl = galleries[split]
        assert cl.unique().numel() == y.unique().numel() == 30
        views,_ = region_views(w,split,cfg,paths,qi,sketches)
        bank = torch.cat([(scale*(v@tgt.T)).softmax(1) for v in photos[pi].cuda().split(256)])
        # Distinct unit vectors yield strictly descending KD cosine for the supplied ranking.
        x = torch.linspace(.9,-.9,len(pi))
        gallery = torch.stack((x,(1-x.square()).sqrt()),1)
        query = torch.tensor([[1.,0.]])
        result=[[],[]]
        for start in range(0,len(y),128):
            sl=slice(start,start+128)
            a=(scale*(sketches[qi[sl]].cuda()@src.T)).softmax(1)
            b=(scale*(views['flipped'][sl].float().cuda()@src.T)).softmax(1)
            for i,posterior in enumerate([a,(a+b)/2]):
                ranks=(posterior@bank.T).argsort(dim=1,descending=True).cpu()
                for row,idx in enumerate(ranks):
                    ap,prec,_,_=kd(query,gallery,y[start+row:start+row+1],cl[idx],split)
                    result[i].append([float(ap),float(prec)])
        print(json.dumps({'split':split,'queries':len(y),'gallery':len(pi),'classes':30,'original_and_mean':[(torch.tensor(z).double().mean(0)*100).tolist() for z in result]}),flush=True)

if __name__=='__main__':
    run()
