from pathlib import Path
import torch
from .gallery_panel_resume import semantic_inputs

w = Path('/content/sbir')

def paired_map_probe():
 cfg,_,qs,_,gs,q,p,src,tgt,scale=semantic_inputs(w,'sketchy');rng=torch.Generator().manual_seed(802);order=torch.randperm(len(src),generator=rng).cuda();fit,test=order[::2],order[1::2];norm=torch.nn.functional.normalize;out={}
 for kind in ['identity','translation','paired','shuffled']:
  x=src[fit].double();y=tgt[fit].double();mx=x.mean(0);my=y.mean(0)
  if kind=='shuffled':y=y[torch.randperm(len(y),generator=rng).cuda()]
  u,_,vh=torch.linalg.svd((x-mx).T@(y-my));rot=(u@vh).float()
  def move(v):
   if kind=='identity':return v
   if kind=='translation':return norm(v-mx.float()+my.float())
   return norm((v-mx.float())@rot+my.float())
  report={'wordcos':float((move(src[test])*tgt[test]).sum(1).mean())}
  for s,(qi,labels) in qs.items():
   pi,gl=gs[s];gg=p[pi].cuda();cl=gl.cuda();scores=[]
   for j in range(0,len(qi),128):
    sl=slice(j,j+128);vv=move(q[qi[sl]].cuda());n=200 if s=='sketchy_2' else len(gg);hit=cl[(vv@gg.T).topk(n).indices]==labels[sl,None].cuda();rank=torch.arange(1,n+1,device=gg.device);ap=(hit*hit.cumsum(1)/rank).sum(1)/hit.sum(1).clamp_min(1);pr=hit[:,:200 if s=='sketchy_2' else 100].float().mean(1);scores.append(torch.stack([ap,pr],1).cpu())
   report[s]=(torch.cat(scores).double().mean(0)*100).tolist()
  out[kind]=report
 return out

if __name__ == '__main__':
    print(paired_map_probe())
