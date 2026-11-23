from pathlib import Path
import torch
from .gallery_panel_resume import semantic_inputs

w=Path('/content/sbir')

def all_redraws():
 cfg,paths,qs,pics,gs,q,p,src,tgt,_=semantic_inputs(w,'sketchy');out={}
 for s,(qi,y) in qs.items():
  pi,gl=gs[s];lookup={(Path(pics[i]).parent.name,Path(pics[i]).stem):j for j,i in enumerate(pi.tolist())};ids=torch.tensor([lookup[(Path(paths[i]).parent.name,Path(paths[i]).stem.rsplit('-',1)[0])] for i in qi.tolist()]);im,inv=ids.unique(return_inverse=True);terms=(p[pi[im]].cuda()@tgt.T).topk(8).indices;top=torch.cat([(z@src.T).topk(32).indices for z in q[qi].cuda().split(256)]);out[s]={}
  for c in y.unique():
   ix=torch.where(y==c)[0];sources,which=inv[ix].unique(return_inverse=True);which=which.cuda();n=torch.bincount(which);counts=torch.zeros(len(sources),len(src),device=src.device);counts.index_put_((which[:,None].expand(-1,32),top[ix]),torch.ones_like(top[ix],dtype=torch.float32),accumulate=True);t=terms[sources];h=counts.gather(1,t);other=(counts.sum(0)[t]-h)/(len(ix)-n[:,None]);freq=torch.bincount(t.flatten(),minlength=len(src))[t];vals=[]
   for mask in [torch.ones_like(t,dtype=torch.bool),freq==1,freq>=10]:
    missing=(n[:,None]-h)*mask*(n[:,None]>1);den=missing.sum();vals.append([float(den),float((missing*h/(n[:,None]-1).clamp_min(1)).sum()),float((missing*other).sum())])
   out[s][int(c)]=vals
 return out

if __name__=='__main__':
    print(all_redraws())
