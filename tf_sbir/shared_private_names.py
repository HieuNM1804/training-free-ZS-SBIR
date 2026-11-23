"""Cross-fitted public-name posteriors for private and group-shared explanations of a candidate photo."""
import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from .concept_benchmark import atomic_json,read
from .conditional_word_inputs import tree
from .crossdomain_benchmark import digest,revision
from .encoding import save_tensor
from .gallery_panel_resume import semantic_inputs
from .semantic import read_words


FAMILY="shared-private-name-evidence"
FOLD_SEED=27179
DEPTHS=(1,2,3)
# Arithmetic peer pooling and ordinary normalized prototype softmax are controls for sharing a latent name.
MODELS=("arithmetic","prototype","fixed-r25-p25","fixed-r75-p25","fixed-r75-p1","fixed-r1-p1",
        "tree-r25-p25","tree-r75-p25","tree-r75-p1","tree-r1-p1")


def path(w,s):return w/"cache"/FAMILY/s


def sum_by(x,idx,n):
    return x.new_zeros(n,x.size(1)).index_add_(0,idx,x)


def atoms(probs,g,ht,donor):
    base=ht.ids[0][donor];n=ht.sizes[0]
    cnt=torch.bincount(base,minlength=n).float()
    sums=sum_by(probs[donor],base,n)
    visual=sum_by(g[donor],base,n)
    return cnt,sums,visual


def complete_means(sums,counts,parents,prior):
    out=[None]*len(sums)
    out[-1]=prior.reshape(1,-1).expand_as(sums[-1])
    for j in reversed(range(len(out))):
        default=prior[None] if j==len(out)-1 else out[j+1][parents[j]]
        out[j]=torch.where(counts[j][:,None]>0,sums[j]/counts[j][:,None].clamp_min(1),default)
    return out


def levels_from(x,ht):
    values=[x]
    for up,n in zip(ht.parents,ht.sizes[1:],strict=True):values.append(sum_by(values[-1],up,n))
    return values


def catalog(ht,probs,g,target,scale,donor):
    count,total,visual=atoms(probs,g,ht,donor)
    counts=[x[:,0] for x in levels_from(count[:,None],ht)]
    prior=(total.sum(0)/len(donor)).clamp_min(1e-32);prior/=prior.sum()
    means=complete_means(levels_from(total,ht),counts,ht.parents,prior)
    vg=complete_means(levels_from(visual,ht),counts,ht.parents,g[donor].mean(0))
    for lev in DEPTHS:
        yield ("arithmetic",lev),means[lev]
        proto=torch.nn.functional.normalize(vg[lev],dim=1)
        yield ("prototype",lev),(scale*(proto@target.T)).softmax(1)
    ratio=means[0]/prior
    for name,eta,tau in (("r25-p25",.25,.25),("r75-p25",.75,.25),("r75-p1",.75,1.),("r1-p1",1.,1.)):
        loglik=torch.log(((1-eta)+eta*ratio).clamp_min(1e-30))
        loglik=loglik*(count>0)[:,None]
        prec=levels_from(loglik,ht)
        pp=[];stop=[]
        ll=None
        for j,v in enumerate(prec):
            logits=prior.log()[None]+tau*v
            marg=logits.logsumexp(1)
            if j==0:
                ll=marg;stay=torch.ones_like(marg)
            else:
                split=ll.new_zeros(ht.sizes[j]).index_add_(0,ht.parents[j-1],ll)
                ll=torch.logaddexp(marg,split)-0.6931471805599453
                stay=(marg-split).sigmoid()
            pp.append(logits.softmax(1));stop.append(stay)
        for depth in DEPTHS:
            yield ("fixed-"+name,depth),pp[depth]
            mixture=pp[-1];remaining=1-stop[-1]
            pooled=mixture*stop[-1][:,None]
            for j in reversed(range(depth,len(pp)-1)):
                up=ht.parents[j];rem=remaining[up]
                fraction=torch.ones_like(rem) if j==depth else stop[j]
                pooled=pooled[up]+(rem*fraction)[:,None]*pp[j]
                remaining=rem*(1-fraction)
            if (pooled.sum(1)-1).abs().amax()>2e-4:raise ValueError("Cut prediction lost probability mass")
            yield ("tree-"+name,depth),pooled


def segment_shuffle(groups,indices,seed):
    perm=torch.randperm(len(indices),generator=torch.Generator().manual_seed(seed)).to(indices.device)
    order=perm[groups[perm].argsort(stable=True)]
    group=groups[order];cuts=torch.cat((group.new_tensor([0]),torch.where(group[1:]!=group[:-1])[0]+1,group.new_tensor([len(group)])))
    shuffled=order.clone()
    for a,b in zip(cuts[:-1].tolist(),cuts[1:].tolist(),strict=True):
        shuffled[order[a:b]]=order[a:b].roll(1)
    return shuffled


@torch.inference_mode()
def prepare(w,ds):
    if ds=="tuberlin" and not (w/"results"/FAMILY/"freeze.json").is_file():raise ValueError("Method must be frozen before TU")
    cfg,_,qs,paths,gs,_,p,src,target,scale=semantic_inputs(w,ds)
    for s,(pi,gl) in gs.items():
        root=path(w,s);root.mkdir(parents=True,exist_ok=True)
        sig=digest({"gallery":[paths[i] for i in pi.tolist()],"vocab":len(src),"seed":FOLD_SEED,"models":MODELS,"version":"candidate-specific private/public posterior v1"})
        if (root/"done.json").exists():
            if read(root/"done.json")["signature"]!=sig:raise ValueError("Incompatible name bank")
            continue
        g=p[pi].cuda()
        torch.set_float32_matmul_precision("highest");ht=tree(w,s,g)
        torch.set_float32_matmul_precision("high")
        probs=g.new_empty((len(g),len(src)))
        for i in range(0,len(g),384):probs[i:i+384]=(scale*(g[i:i+384]@target.T)).softmax(1)
        save_tensor(probs.cpu(),root/"photo-words.pt")
        seq=torch.randperm(len(g),generator=torch.Generator().manual_seed(FOLD_SEED)).to(g.device)
        groups=[seq[::2].sort().values,seq[1::2].sort().values]
        priors=[]
        for fold in (0,1):
            donor=groups[1-fold]
            prior=probs[donor].mean(0).clamp_min(1e-32);prior=prior/prior.sum();priors.append(prior.cpu())
            for (name,level),pred in catalog(ht,probs,g,target,scale,donor):
                if not torch.isfinite(pred).all():raise ValueError("Invalid group name posterior")
                save_tensor(pred.cpu(),root/f"words-{fold}-{name}-{level}.pt")
            print(s,fold,"estimated",len(MODELS)*len(DEPTHS),flush=True)
        save_tensor({"folds":[a.cpu() for a in groups],"groups":[q.cpu() for q in ht.ids],"sizes":ht.sizes,"priors":priors},root/"geometry.pt")
        atomic_json(root/"done.json",{"signature":sig,"code":revision(),"vocabulary":len(src),"gallery":len(g),"sizes":ht.sizes})
        print("ready",s,flush=True);del probs,g


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    p.add_argument("--dataset",choices=("sketchy","tuberlin"),default="sketchy")
    a=p.parse_args();prepare(a.workspace,a.dataset)
