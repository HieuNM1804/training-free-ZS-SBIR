from pathlib import Path
import torch
from .gallery_panel_resume import semantic_inputs

w = Path('/content/sbir')

def smooth_controls():
 cfg,_,qs,_,gs,q,p,src,tgt,scale=semantic_inputs(w,'sketchy'); near=[]
 for i in range(0,len(src),384):
  sim=src[i:i+384]@src.T;idx=torch.arange(len(sim),device=src.device);sim[idx,idx+i]=-torch.inf;near.append(sim.topk(8).indices)
 nn=torch.cat(near);perm=torch.randperm(len(src),generator=torch.Generator().manual_seed(732)).cuda();local=src[nn].mean(1);fake=src[perm[nn]].mean(1);out={}
 for s,(qi,y) in qs.items():
  pi,gl=gs[s];cl=gl.cuda();bank=torch.cat([(scale*(v@tgt.T)).softmax(1) for v in p[pi].cuda().split(256)]);acc=[]
  for j in range(0,len(y),128):
   sl=slice(j,j+128);v=q[qi[sl]].cuda();l=scale*(v@src.T);h=scale*(v@local.T);f=scale*(v@fake.T);obs=[]
   for z in (l,1.25*l,l+.25*(l-h),l+.25*(l-f),l-.25*(l-h),.75*l,.75*l+.25*f,.75*l+.25*h):
    sc=z.softmax(1)@bank.T;n=200 if s=='sketchy_2' else len(pi);hit=cl[sc.topk(n).indices]==y[sl,None].cuda();ap=(hit.cumsum(1)*hit/torch.arange(1,n+1,device=cl.device)).sum(1)/hit.sum(1).clamp_min(1);pr=hit[:,:200 if s=='sketchy_2' else 100].float().mean(1);obs.append(torch.stack([ap,pr],1))
   acc.append(torch.stack(obs,1).cpu())
  out[s]=(torch.cat(acc).double().mean(0)*100).tolist()
 return out

if __name__ == '__main__':
    print(smooth_controls())
