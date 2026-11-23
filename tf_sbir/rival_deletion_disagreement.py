"""Bidirectional deletion test with equal-area windows and label-free rival selection."""
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .query_fragment_reference_inputs import dataset, Images, normal_transform, to_tokens, load_frozen_clip, stratified_sample

@torch.inference_mode()
def run():
    w=Path('/content/sbir')
    cfg,paths,qs,_,gs,q,p=dataset(w,'sketchy')
    v=load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda')).visual
    report={}
    for s,(qi,y) in qs.items():
        ix=stratified_sample(y,36,63241)
        folder=w/'cache/gallery-calibrated-window-extremes'/s
        group=torch.load(folder/'structure.pt',weights_only=True)['groups']
        ref=torch.load(w/'cache/gallery-stable-region-evidence'/(s+'-regional-reference.pt'),weights_only=True,mmap=True)[ix]
        pi,cl=gs[s]
        top=ref.argmax(1)
        rival=(q[qi[ix]].cuda()@p[pi].cuda().T).argmax(1).cpu()
        axis=(p[pi[top]]-p[pi[rival]]).cuda()
        loader=DataLoader(Images([paths[i] for i in qi[ix].tolist()],normal_transform(),'rgb'),batch_size=48,num_workers=6)
        obs=[]
        offset=0
        for batch in loader:
            x=v.conv1(batch.cuda().to(v.conv1.weight.dtype))
            sl=slice(offset,offset+len(x))
            direction=axis[sl]
            full=(to_tokens(v,x)*direction).sum(1)
            assert (1-(to_tokens(v,x)*q[qi[ix[sl]]].cuda()).sum(1)).abs().max()<.002
            direction=direction*full.sign()[:,None]
            fill=x.mean((-1,-2),keepdim=True)
            keep=[]; delete=[]
            for row in (0,2,4):
                for col in (0,2,4):
                    mask=torch.zeros((7,7),device=x.device,dtype=torch.bool)
                    mask[row:row+3,col:col+3]=True
                    keep.append((to_tokens(v,torch.where(mask,x,fill))*direction).sum(1))
                    delete.append((to_tokens(v,torch.where(mask,fill,x))*direction).sum(1))
            kept=torch.stack(keep,1); removed=torch.stack(delete,1)
            ids=kept.argmax(1); ar=torch.arange(len(ids),device=x.device)
            selected=removed[ar,ids]; other=(removed.sum(1)-selected)/8
            obs.append(torch.stack([full.abs(),selected,other,(removed<0).float().sum(1),(selected<0).float()],1).cpu())
            offset+=len(x)
        z=torch.cat(obs)
        gold=y[ix]
        info=(cl[top]==gold)^(cl[rival]==gold)
        prefer=cl[rival]==gold
        disagree=top!=rival
        margin=z[:,0]; effect=z[:,2]-z[:,1]
        from sklearn.metrics import roc_auc_score
        res={}
        for label,vv in [('cosine',margin),('deletion',effect)]:
            idxs=torch.where(disagree)[0]
            switch=idxs[vv[idxs].argsort(descending=True)[:max(1,len(idxs)//10)]]
            res[label]=[float(roc_auc_score(prefer[info],vv[info])),int((prefer[switch]&info[switch]).sum()),int((~prefer[switch]&info[switch]).sum())]
        report[s]={'queries':len(ix),'different':int(disagree.sum()),'native_only':int((prefer&info).sum()),'regional_only':int((~prefer&info).sum()),'tests':res}
    print(json.dumps(report))

if __name__=='__main__':
    run()
