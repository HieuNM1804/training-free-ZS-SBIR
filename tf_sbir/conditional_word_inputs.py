"""Nested photoword incidence and two independently observed conditional branching distributions."""
import argparse
from pathlib import Path

import torch

from .concept_benchmark import atomic_json,read
from .crossdomain_benchmark import digest,revision
from .encoding import save_tensor
from .gallery_panel_resume import semantic_inputs
from .query_partition_inputs import place
from .query_partition_tree import Hierarchy


FAMILY="conditional-word-persistence"


def bank_path(w,s):return w/"cache"/FAMILY/s


def tree(w,s,g):
    known=place(w,s)/"tree.pt"
    if known.exists():
        data=torch.load(known,mmap=True,weights_only=True)
        return Hierarchy([z.cuda() for z in data["ids"]],[z.cuda() for z in data["parents"]],data["sizes"])
    return Hierarchy.build(g)


def group_sum(x,idx,m):
    return x.new_zeros((m,x.size(1))).index_add_(0,idx,x)


@torch.inference_mode()
def prepare(w,ds):
    if ds=="tuberlin" and not (w/"results"/FAMILY/"freeze.json").is_file():raise ValueError("Freeze this method before TU")
    cfg,_,qs,paths,gs,_,p,src,tgt,scale=semantic_inputs(w,ds)
    torch.set_float32_matmul_precision("high")
    for s,(ix,_) in gs.items():
        root=bank_path(w,s);root.mkdir(parents=True,exist_ok=True)
        sig=digest({"gallery":[paths[i] for i in ix.tolist()],"word_count":len(src),"half_seed":87631,
                    "smoothing":"one parent-mean occurrence added to each child's half-sample mean v1","version":1})
        if (root/"done.json").is_file():
            if read(root/"done.json")["signature"]!=sig:raise ValueError("Different lexical hierarchy")
            continue
        g=p[ix].cuda()
        torch.set_float32_matmul_precision("highest")
        h=tree(w,s,g)
        torch.set_float32_matmul_precision("high")
        n=len(g);v=len(src)
        probs=g.new_empty((n,v))
        for start in range(0,n,256):probs[start:start+256]=(scale*(g[start:start+256]@tgt.T)).softmax(1)
        global_mean=probs.mean(0).clamp_min(1e-32)
        perm=torch.randperm(n,generator=torch.Generator().manual_seed(87631)).to(g.device)
        mask=torch.zeros(n,device=g.device,dtype=torch.bool);mask[perm[:n//2]]=True
        count=[torch.bincount(i,minlength=m).float() for i,m in zip(h.ids,h.sizes,strict=True)]
        ct=[[torch.bincount(i[sel],minlength=m).float() for i,m in zip(h.ids,h.sizes,strict=True)] for sel in (mask,~mask)]
        fout=[];parents=h.parents
        for l,up in enumerate(parents):
            pri=count[l]/count[l+1][up]
            full=torch.empty((h.sizes[l],v),dtype=torch.float32,device="cpu")
            ha=torch.empty_like(full,dtype=torch.bfloat16);hb=torch.empty_like(ha)
            for j in range(0,v,512):
                sl=slice(j,j+512)
                val=probs[:,sl]/global_mean[sl]
                c=group_sum(val,h.ids[l],h.sizes[l])
                par=group_sum(c,up,h.sizes[l+1])
                # Full conditional ratio divided by child size prior; a path telescopes across these ratios.
                full[:,sl]=(c/par[up].clamp_min(1e-30)/pri[:,None]).cpu()
                for sf,(sel,obs) in enumerate(((mask,ha),(~mask,hb))):
                    ca=group_sum(val[sel],h.ids[l][sel],h.sizes[l])
                    pa=group_sum(ca,up,h.sizes[l+1])/ct[sf][l+1][:,None].clamp_min(1)
                    pa=torch.where(ct[sf][l+1][:,None]>0,pa,torch.ones_like(pa))
                    child=(ca+pa[up])/(ct[sf][l][:,None]+1)
                    norm=group_sum(child*pri[:,None],up,h.sizes[l+1])
                    normalized=child/norm[up].clamp_min(1e-30)
                    obs[:,sl]=normalized.cpu().bfloat16()
            f=root/f"level-{l}.pt"
            save_tensor({"full":full,"A":ha,"B":hb,"prior":pri.cpu()},f)
            fout.append(f.name)
        # Independent single-state closed form for checking the path recursion at each level.
        base=[group_sum(probs/global_mean[None],h.ids[k],m)/count[k][:,None] for k,m in enumerate(h.sizes) if k<len(parents)]
        for i,z in enumerate(base):save_tensor(z.cpu(),root/f"marginal-{i}.pt")
        save_tensor({"ids":[z.cpu() for z in h.ids],"parents":[z.cpu() for z in h.parents],"count":[z.cpu() for z in count],
                     "sizes":h.sizes,"word_mean":global_mean.cpu()},root/"tree.pt")
        atomic_json(root/"done.json",{"signature":sig,"levels":fout,"code":revision(),"gallery":n,"words":v})
        print("ready",s,h.sizes,flush=True)
        del probs,base


def load(w,s):
    root=bank_path(w,s);meta=read(root/"done.json")
    get=lambda f:torch.load(root/f,mmap=True,weights_only=True)
    h=get("tree.pt")
    ht=Hierarchy([i.cuda() for i in h["ids"]],[i.cuda() for i in h["parents"]],h["sizes"])
    levels=[{k:v.cuda() for k,v in get(f).items()} for f in meta["levels"]]
    closed=[get(f"marginal-{i}.pt").cuda() for i in range(len(levels))]
    return ht,levels,closed,h["word_mean"].cuda()


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    ap.add_argument("--dataset",choices=("sketchy","tuberlin"),default="sketchy")
    args=ap.parse_args();prepare(args.workspace,args.dataset)
