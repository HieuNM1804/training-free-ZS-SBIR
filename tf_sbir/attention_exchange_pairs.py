"""Score the same two CLS queries on an image's own keys and on an unrelated set of keys."""
import argparse
from collections import defaultdict
from pathlib import Path

import torch

from .attention_exchange_inputs import FAMILY,LAYERS,layer
from .concept_benchmark import atomic_json,checksum,read
from .crossdomain_benchmark import digest,revision
from .encoding import save_tensor
from .query_fragment_reference_inputs import dataset,load as fragment_bank
from .recoverable_patch_study import locate
from .retrieval_diagnosis import stratified_sample


SEEDS=("native","visual","regional")
K=384
RANDOM=128
KINDS=("to_photo","to_sketch","symmetric","to_photo_frag","to_sketch_frag","symmetric_frag","symmetric_empty","symmetric_wrong_keys",
       "symmetric_wrong_observer","symmetric_uniform_observer","euclidean","euclidean_frag","cosine_frag","fragment_gain","blank_gain",
       "gain_linear_part","gain_log_normalizer","gain_wrong_keys","gain_wrong_observer","photo_uncertainty")
MOVED=torch.randperm(49,generator=torch.Generator().manual_seed(13627))


def key_statistics(keys,native):
    # Compute all distributions in FP32. The native Q/K projections were stored in their original precision.
    results=defaultdict(list)
    for i in range(0,len(keys),128):
        k=keys[i:i+128].float();q=native[i:i+128].float()
        logits=(q[:,:,None]*k).sum(-1)/8.;p=logits.softmax(-1)
        entropy=-(p*logits.log_softmax(-1)).sum(-1)
        for name,x in (("mean",(p[:,:,:,None]*k).sum(2)),("shift",(p[:,:,MOVED.to(k.device),None]*k).sum(2)),
                       ("uniform",k.mean(2)),("entropy",entropy)):
            results[name].append(x)
    return {name:torch.cat(parts) for name,parts in results.items()}


def indexed(stats,ids):
    return {k:v[ids] for k,v in stats.items()}


def cross(q,keys,stats):
    logits=torch.einsum("bhd,bhjd->bhj",q.float(),keys.float())/8.
    logz=logits.logsumexp(-1);mu=(q.float()*stats["mean"]).sum(-1)/8.
    entropy=stats["entropy"]
    return {"real":mu+entropy-logz,
            "shift":(q.float()*stats["shift"]).sum(-1)/8.+entropy-logz,
            "uniform":(q.float()*stats["uniform"]).sum(-1)/8.+3.8918202981106265-logz,
            "linear":mu,"logz":logz}


def pair_values(g,q,global_query,sp,sq,wrongp,wrongq,pg,sx,qi,pi):
    x=q["q"][qi].float();y=g["q"][pi].float();f=q["part"][qi].float();blank=q["empty"][qi].float()
    photo=g["k"][pi];sketch=q["k"][qi];pst=indexed(sp,pi);sst=indexed(sq,qi)
    into=cross(x,photo,pst);back=cross(y,sketch,sst)
    uf=x-f+global_query;ur=y-global_query+f
    cf=cross(uf,photo,pst);cr=cross(ur,sketch,sst)
    blf=cross(x-blank+global_query,photo,pst);blr=cross(y-global_query+blank,sketch,sst)
    nf=cross(f,photo,pst);nb=cross(blank,photo,pst)
    wk=g["k"][pg[pi]];wr=q["k"][sx[qi]]
    # For a wrong receiver set the native observation still comes from the *same* native CLS query.
    fw=cross(uf,wk,indexed(wrongp,pi));bw=cross(ur,wr,indexed(wrongq,qi))
    fw0=cross(x,wk,indexed(wrongp,pi));fw1=cross(f,wk,indexed(wrongp,pi))
    eu=-((x-y)**2).mean(-1);cal=-((uf-y)**2).mean(-1)
    qcos=torch.nn.functional.cosine_similarity(x-f,y-global_query,dim=-1)
    vals=(into["real"],back["real"],.5*(into["real"]+back["real"]),cf["real"],cr["real"],.5*(cf["real"]+cr["real"]),
          .5*(blf["real"]+blr["real"]),.5*(fw["real"]+bw["real"]),.5*(cf["shift"]+cr["shift"]),.5*(cf["uniform"]+cr["uniform"]),
          eu,cal,qcos,into["real"]-nf["real"],into["real"]-nb["real"],into["linear"]-nf["linear"],nf["logz"]-into["logz"],
          fw0["real"]-fw1["real"],into["shift"]-nf["shift"],-pst["entropy"])
    return torch.stack([t.mean(1) for t in vals],1)


def make_panel(c,w,ds,s,stage):
    cfg,qp,qs,pp,gs,q,p=c
    qi,y=qs[s];pi,l=gs[s]
    ix=stratified_sample(y,36,63241) if stage=="pilot" else torch.arange(len(qi))
    f=fragment_bank(w,ds,"full")
    residual=q[qi[ix]]-.8*f["modes"]["appearance"][qi[ix]].float().mean(1)
    gallery=p[pi].cuda()
    matrices={"native":q[qi[ix]].cuda()@gallery.T,"visual":residual.cuda()@gallery.T,
              "regional":torch.load(w/"cache/gallery-stable-region-evidence"/(s+"-regional-reference.pt"),mmap=True,weights_only=True)[ix].cuda()}
    plans={k:v.argsort(dim=1,descending=True,stable=True).to(torch.int32).cpu() for k,v in matrices.items()}
    random=torch.randperm(len(pi),generator=torch.Generator().manual_seed(88477))[:RANDOM].int()[None].expand(len(ix),-1)
    selected=torch.cat((*[plans[k][:,:K] for k in SEEDS],random),1)
    seedvals=torch.stack([matrices[k].gather(1,plans[k][:,:K].long().cuda()).cpu() for k in SEEDS],1)
    return {"rows":ix,"qi":qi[ix],"pi":pi,"labels":y[ix],"gl":l,"plans":plans,"selected":selected,"seedvals":seedvals}


@torch.inference_mode()
def run(w,ds,stage):
    if ds=="tuberlin" and stage=="pilot":raise ValueError("No TU tuning")
    c=dataset(w,ds)
    root=w/"cache"/FAMILY/"panel";root.mkdir(parents=True,exist_ok=True)
    for s in c[2]:
        file=root/(stage+"-"+s+".pt")
        if not file.exists():save_tensor(make_panel(c,w,ds,s,stage),file)
    for depth in LAYERS:
        pa,g,gcheck=layer(w,ds,"photos",depth)
        qa,q,qcheck=layer(w,ds,stage,depth)
        for s in c[2]:
            code=stage+"-"+s+f"-L{depth}"
            dest=root/(code+".pt");meta=root/(code+".json")
            if meta.exists() and dest.exists():
                if checksum(dest)!=read(meta)["sha256"]:raise ValueError("Stale pair scores")
                continue
            inp=torch.load(root/(stage+"-"+s+".pt"),mmap=True,weights_only=True)
            pi=locate(pa,inp["pi"]).cuda();ids=locate(qa,inp["qi"]).cuda()
            gg={k:v[pi] for k,v in g.items()}
            # Diagnostics on wrong sketch keys must only use this sketch and a gallery key set.
            # Use unrelated gallery keys for the reverse control too. No other sketch is used in scoring.
            n=len(pi);device=pi.device
            shuffled=torch.randperm(n,generator=torch.Generator().manual_seed(38003)).to(device)
            pg=torch.empty_like(shuffled);pg[shuffled]=shuffled.roll(1)
            summary=key_statistics(gg["k"],gg["q"]);wrongp=key_statistics(gg["k"][pg],gg["q"])
            sq=key_statistics(q["k"][ids],q["q"][ids])
            qq={k:v[ids] for k,v in q.items()}
            # One query-independent gallery receiver panel for the reverse negative control.
            # Use the identical external key set for every sketch to preserve single-query inference.
            donor=int(shuffled[0]);dummy=gg["k"][donor:donor+1]
            external=dummy.expand(len(ids),-1,-1,-1)
            wrongs=key_statistics(external,qq["q"])
            global_q=gg["q"].float().mean(0)
            selected=inp["selected"];b,lenp=selected.shape
            values=torch.empty((b,lenp,len(KINDS)))
            sqp={k:v for k,v in qq.items()}
            # The last slot is a fixed photo's key set; indices of real sketch keys always refer to the current query.
            sqp["k"]=torch.cat((qq["k"],dummy),0)
            sx=torch.full((len(ids),),len(ids),device=device,dtype=torch.long)
            flat=selected.flatten()
            for i in range(0,len(flat),192):
                rr=torch.arange(i,min(len(flat),i+192),device=device)
                val=pair_values(gg,sqp,global_q,summary,sq,wrongp,wrongs,pg,sx,rr//lenp,flat[i:i+192].long().cuda())
                if not torch.isfinite(val).all():raise ValueError("Invalid attention divergence")
                values.view(-1,len(KINDS))[i:i+len(val)]=val.cpu()
            gap=float((values[:,:,KINDS.index("fragment_gain")]-(values[:,:,KINDS.index("gain_linear_part")]+values[:,:,KINDS.index("gain_log_normalizer")])).abs().amax())
            if gap>2e-5:raise ValueError("Conditional likelihood ratio does not decompose")
            sample=gg["q"][:12].float()
            self_error=float(cross(sample,gg["k"][:12],{k:v[:12] for k,v in summary.items()})["real"].abs().amax())
            if self_error>1e-4:raise ValueError("Native self-KL differs from zero")
            save_tensor(values,dest)
            atomic_json(meta,{"sha256":checksum(dest),"code":revision(),"layer":depth,"shape":list(values.shape),"kinds":KINDS,
                       "self_kl":self_error,"decomposition":gap,"native_attention_cache_checks":torch.stack((gcheck,qcheck)).amax(0).tolist()})
            print("saved",code,values.shape,self_error,gap,flush=True)
            del gg,qq,sqp,values
        del g,q


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    ap.add_argument("--dataset",choices=("sketchy","tuberlin"),default="sketchy")
    ap.add_argument("--stage",choices=("pilot","full"),default="pilot")
    a=ap.parse_args();run(a.workspace,a.dataset,a.stage)
