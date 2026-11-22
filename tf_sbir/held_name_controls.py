"""Compare per-image-pair depiction shifts with global prompt weights and with omitted semantic neighbors."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import inspect
from pathlib import Path

import torch
from torch.nn import functional as F

from .concept_benchmark import atomic_json, checksum, read
from .conditional_agreement_study import Replay
from .crossdomain_benchmark import digest, revision
from .encoding import save_tensor
from .gallery_panel_resume import semantic_inputs
from .map_diagnosis import map_values
from .predictive_depiction_distance import FAMILY, GAINS, center, fixed_distance, predict, similarity
from .predictive_depiction_study import baseline, candidate_axes, parse
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd, paired_delta


SHIFTS = (-1., 0., .5, 1., 2.)


def removals(src,tgt,names,count,semantic):
    b,k=names.shape
    own=torch.arange(k,device=names.device).reshape(1,k,1).expand(b,-1,-1)
    if count==1:return own
    if semantic:
        emb=F.normalize(src[names]+tgt[names],dim=-1)
        sim=emb@emb.transpose(1,2)
        sim.diagonal(dim1=1,dim2=2).fill_(-torch.inf)
        others=sim.topk(count-1,dim=2).indices
    else:
        offsets=torch.randperm(k-1,generator=torch.Generator().manual_seed(6421))[:count-1].to(names.device)+1
        others=(own+offsets)%k
    return torch.cat((own,others),dim=-1)


def missing_residual(r,u,v,exclude,ridge):
    """The held-out coordinate and its neighbors never enter this pair's fitted coefficients."""
    b,_,k=r.shape
    hidden=exclude.shape[-1]
    left=k-hidden
    flat=exclude.flatten(1).unsqueeze(1)
    def sufficient(z):
        subset=z.gather(2,flat.expand(-1,z.shape[1],-1)).reshape(b,z.shape[1],k,hidden).sum(3)
        return z.sum(2,keepdim=True)-subset
    sr,su,sv=[sufficient(z) for z in (r,u,v)]
    aa=sufficient(u.square())-su.square()/left+k*ridge
    bb=sufficient(v.square())-sv.square()/left+k*ridge
    ab=sufficient(u*v)-su*sv/left
    ar=sufficient(u*r)-su*sr/left
    br=sufficient(v*r)-sv*sr/left
    det=(aa*bb-ab.square()).clamp_min(1e-15)
    wa=(bb*ar-ab*br)/det
    wb=(aa*br-ab*ar)/det
    return (r-sr/left)-wa*(u-su/left)-wb*(v-sv/left)


def global_scores(moments,a,b):
    rr,ru,rv,uu,uv,vv=moments
    mse=rr+2*a*ru-2*b*rv+a*a*uu+b*b*vv-2*a*b*uv
    return similarity(mse)


def fixed_moments(r,u,v):
    # The held-out intercept error for fixed slopes includes the same K/(K-1) scale.
    k=r.size(-1)
    x,a,b=(center(z)*(k/(k-1)) for z in (r,u,v))
    return tuple(v.mean(2) for v in (x.square(),x*a,x*b,a.square(),a*b,b.square()))


def sanity(device,ridge):
    gen=torch.Generator(device=device).manual_seed(331)
    def sample(shape):
        return torch.randn(shape,device=device,generator=gen,dtype=torch.float64)
    x=sample((2,1,19))
    y=sample((2,5,19))
    u=sample((2,1,19))
    v=sample((2,5,19))
    r=x-y
    own=torch.arange(19,device=device)[None,:,None].expand(2,-1,-1)
    err1=float((missing_residual(r,u,v,own,ridge)-predict(r,u,v,ridge)[2]).abs().max())
    excluded=torch.cat((own,(own+5)%19,(own+1)%19),-1)
    observed=missing_residual(r,u,v,excluded,ridge)
    design=torch.stack((torch.ones_like(r),u.expand_as(r),v),-1)
    keep=torch.arange(19,device=device)
    penalty=torch.diag(torch.tensor([0,19*ridge,19*ridge],device=device,dtype=r.dtype))
    err2=0.
    for j in (0,3,8,15):
        mask=~torch.isin(keep,excluded[0,j])
        xx=design[:,:,mask]
        beta=torch.linalg.solve(xx.transpose(2,3)@xx+penalty,xx.transpose(2,3)@r[:,:,mask,None])
        expected=r[:,:,j]-(design[:,:,j,None]@beta).squeeze(-1).squeeze(-1)
        err2=max(err2,float((observed[:,:,j]-expected).abs().max()))
    m=fixed_moments(r,u,v)
    err3=float((global_scores(m,1.,-.5)-fixed_distance(r+u+.5*v)).abs().max())
    if max(err1,err2,err3)>1e-8:
        raise ValueError("Grouped omission or fixed-slope identity failed")
    return [err1,err2,err3]


def rows():
    return ([f"fixed/{a:g}/{b:g}/{readout}/{gain:g}" for a in SHIFTS for b in SHIFTS for readout in ("distance","gain") for gain in GAINS]
          + [f"omit/{count}/{choice}/{readout}/{gain:g}" for count in (1,2,4) for choice in (("solo",) if count==1 else ("semantic","shuffled"))
             for readout in ("distance","gain") for gain in GAINS])


def split_eval(s,inputs,ids,base,selected,nominee,kd,workers):
    _,_,qs,_,gs,qraw,pr,src,tgt,scale=inputs
    k,mode,ridge,gain=parse(nominee)
    if mode not in ("gain","predict"):raise ValueError("Unexpected original candidate")
    all_rows=sorted(selected)
    fixed,holds=defaultdict(list),defaultdict(list)
    for code in all_rows:
        family,a,b,readout,g=code.split("/")
        if family=="fixed":fixed[float(a),float(b)].append((code,readout,float(g)))
        else:holds[int(a),b].append((code,readout,float(g)))
    qi,yl=qs[s]
    pi,gl=gs[s]
    photo=pr[pi].cuda()
    truth=gl.cuda()
    data=defaultdict(list)
    errs={}
    evaluator=Replay(kd,gl,s) if kd else None
    with (ThreadPoolExecutor(max_workers=workers) if kd else nullcontext()) as pool:
        for off in range(0,len(ids),12):
            sl=slice(off,off+12)
            q=qraw[qi[ids[sl]]].cuda()
            x,y,u,v=candidate_axes(q,photo,src,tgt,scale)
            x,y,u,v=[z[...,:k] for z in (x,y,u,v)]
            r=x-y
            start=base[sl].cuda()
            names=(scale*(q@src.T)).topk(64,1).indices[:,:k]
            native=fixed_distance(r)
            yf=yl[ids[sl]]
            def record(name,matrix):
                calc,order,_,_=map_values(matrix,yf.cuda(),truth,s)
                val=calc.cpu()
                if evaluator:
                    exact=evaluator.score(order,yf,pool)
                    err=float((exact-val).abs().max())
                    if err>2e-6:raise ValueError("Original KD score differs")
                    errs[name]=max(errs.get(name,0.),err)
                    val=exact
                data[name].append(val)
            record("regional",start)
            stats=fixed_moments(r,u,v)
            for (ca,cb),methods in fixed.items():
                value=global_scores(stats,ca,cb)
                for name,readout,g in methods:
                    record(name,start+g*(value-native if readout=="gain" else value))
            for (count,choice),methods in holds.items():
                exc=removals(src,tgt,names,count,choice=="semantic")
                if count==1:
                    held=predict(r,u,v,ridge)[0]
                else:
                    chunks=[]
                    for left in range(0,r.size(1),384):
                        section=slice(left,left+384)
                        residual=missing_residual(r[:,section],u,v[:,section],exc,ridge)
                        chunks.append(residual.square().mean(-1))
                    held=torch.cat(chunks,1)
                value=similarity(held)
                for name,readout,g in methods:
                    record(name,start+g*(value-native if readout=="gain" else value))
            if off%240==0:print("official" if evaluator else "controlled",s,off,len(ids),flush=True)
    return {k:torch.cat(p) for k,p in data.items()},errs


@torch.inference_mode()
def run(w,workers):
    folder=w/"results"/FAMILY
    pilot=read(folder/"pilot.json")
    main=pilot["nominee"]
    k,mode,ridge,gain=parse(main)
    same=f"omit/1/solo/{'gain' if mode=='gain' else 'distance'}/{gain:g}"
    candidates=rows()
    kd,_=load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:raise ValueError("Wrong KD source")
    inputs=semantic_inputs(w,"sketchy")
    checks=sanity(torch.device("cuda"),ridge)
    ids,bases,cache,values={},{},{},{}
    for s in inputs[2]:
        original=torch.load(folder/("pilot-"+s+".pt"),weights_only=True,mmap=True)
        ids[s]=original["query_ids"]
        own=w/"cache"/FAMILY/(s+"-regional-"+digest(ids[s].tolist())[:16]+".pt")
        bases[s]=(torch.load(own,weights_only=True,mmap=True) if own.is_file() else
                  baseline(w,s,inputs,ids[s]))
        met,_=split_eval(s,inputs,ids[s],bases[s],candidates,main,None,workers)
        same_error=float((met[same]-original["metrics"][main]).abs().max())
        if same_error>2e-6:raise ValueError(f"Changed original pilot {same_error}")
        cache[s]=met
        values[s]={k:v.mean(0).tolist() for k,v in met.items()}
    means={k:[sum(x[k][j] for x in values.values())/2 for j in (0,1)] for k in ["regional",*candidates]}
    def best(check):
        return max((k for k in means if check(k)),key=means.get)
    choices={"original":same,"fixed":best(lambda k:k.startswith("fixed/")),"semantic":best(lambda k:"/semantic/" in k),
             "shuffled":best(lambda k:"/shuffled/" in k),"fixed_distance":best(lambda k:k.startswith("fixed/") and "/distance/" in k),
             "fixed_gain":best(lambda k:k.startswith("fixed/") and "/gain/" in k)}
    selected=sorted({*choices.values(), *(choices["semantic"].replace("/semantic/","/"+g+"/") for g in ("semantic","shuffled"))})
    verified,paired,errors={},{},{}
    for s in ids:
        met,err=split_eval(s,inputs,ids[s],bases[s],selected,main,kd,workers)
        worst=max(float((v-cache[s][k]).abs().max()) for k,v in met.items())
        if worst>2e-6:raise ValueError("Controlled test differs from original KD")
        verified[s]={k:v.mean(0).tolist() for k,v in met.items()}
        paired[s]={k:paired_delta(met[same],v,inputs[2][s][1][ids[s]]) for k,v in met.items()}
        errors[s]={"queries":len(ids[s]),"max_KD_error":max(err.values()),"reference_error":worst}
    dest=folder/"held-name-controls.pt"
    save_tensor({s:{"query_ids":ids[s],"metrics":cache[s]} for s in ids},dest)
    atomic_json(dest.with_suffix(".json"),{"revision":revision(),"original":main,"matched_count":k,"ridge":ridge,
                "pilot_hash":checksum(folder/"pilot.json"),"algebra_error":checks,"choice":choices,"scores":verified,
                "means":means,"paired":paired,"KD":errors,"GPU":torch.cuda.get_device_name(),
                "per_query_sha256":checksum(dest)})
    print("controlled scores",choices,{k:means[v] for k,v in choices.items()},flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args()
    run(args.workspace,args.workers)
