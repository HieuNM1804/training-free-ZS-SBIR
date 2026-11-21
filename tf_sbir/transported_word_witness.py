"""Transport repeatable gallery corrections along each pair of CLIP word prompts."""

import torch
import torch.nn.functional as F


def minimal_rotation(points,source,target):
    # Every row uses the minimum-angle ambient rotation mapping one unit word
    # direction to its paired word in the other domain.
    dot=(source*target).sum(1,keepdim=True).clamp(-1,1)
    a=(points*source).sum(1,keepdim=True)
    b=(points*target).sum(1,keepdim=True)
    den=(1+dot).clamp_min(1e-8)
    out=points+target*a-source*b+(target*(dot*a-b)+source*(dot*b-a))/den
    if ((1+dot)<1e-7).any():
        raise ValueError("Ambiguous antipodal prompt pair")
    return F.normalize(out)


def discover(photos,source,target,scale,mle_prior,count,rounds,anchor,seed):
    rng=torch.Generator(device=photos.device).manual_seed(seed)
    half_ids=torch.randperm(len(photos),generator=rng,device=photos.device).tensor_split(2)
    named=mle_prior.topk(count).indices
    t=target[named]
    s=source[named]
    centered=F.normalize(photos-photos.mean(0))
    original=photos@t.T*scale
    prior=mle_prior[named]
    logprior=(prior/prior.sum()).clamp_min(1e-30).log()
    fitted=[]
    for group in half_ids:
        x=centered[group]
        probs=(original[group]+logprior).softmax(1)
        init=F.normalize(probs.T@x)
        mu=init
        for _ in range(rounds):
            probs=(scale*(x@mu.T)+logprior).softmax(1)
            direction=probs.T@x/probs.sum(0)[:,None].clamp_min(1e-8)
            mu=F.normalize(direction+anchor*init)
        raw=F.normalize(probs.T@photos[group])
        fitted.append((mu,raw,probs.sum(0)))
    a,raw_a,ma=fitted[0]
    b,raw_b,mb=fitted[1]
    # Identify drift against a fixed competing word, without reassigning identities.
    similarity=a@b.T
    diag=similarity.diag()
    rival=similarity.clone()
    rival.fill_diagonal_(-1.)
    margins=torch.minimum(diag-rival.amax(1),diag-rival.amax(0))
    gate=(scale*margins).sigmoid()
    gate=(2*gate-1).clamp_min(0)
    visual=F.normalize(a+b)
    photo_name=F.normalize(raw_a+raw_b)
    centerword=(raw_a*raw_b).sum(1)
    resp_a=(scale*(centered@a.T)+logprior).softmax(1)
    resp_b=(scale*(centered@b.T)+logprior).softmax(1)
    # Candidate assignments come from the opposite reference partition.
    resp=torch.empty_like(resp_a)
    resp[half_ids[0]]=resp_b[half_ids[0]]
    resp[half_ids[1]]=resp_a[half_ids[1]]
    disp=photo_name-t
    mean_disp=(disp*(ma+mb)[:,None]).sum(0)/(ma+mb).sum()
    return {"named":named,"src":s,"target":t,"visual":visual,"raw":photo_name,"resp":resp,
            "fitted_same_source":(scale*(centered@visual.T)+logprior).softmax(1),"prior":prior,
            "gate":gate,"margins":margins,"agree":centerword,"mean_disp":mean_disp,
            "disp":disp,"halves":half_ids,"effective":1/resp.square().sum(1).mean()}


def queries_and_scores(query,bank,scale,config):
    _,_,_,strength,mode,beta,prior_power=config
    src,t,raw=bank["src"],bank["target"],bank["raw"]
    g=bank["gate"][:,None]
    if mode=="none":
        proto=src
    else:
        adjusted=F.normalize(t+strength*(bank["disp"]-bank["mean_disp"]))
        if mode=="raw_displacement":
            pred=F.normalize(src+strength*(bank["disp"]-bank["mean_disp"]))
        elif mode=="permuted_displacement":
            r=torch.randperm(len(src),generator=torch.Generator().manual_seed(81817)).to(src.device)
            pred=minimal_rotation(F.normalize(t+strength*(bank["disp"][r]-bank["mean_disp"])),t,src)
        elif mode=="visual_only":
            pred=raw
        else:
            pred=minimal_rotation(adjusted,t,src)
        proto=pred if mode in ("ungated","visual_only") else F.normalize(src+g*(pred-src))
    logits=query@proto.T*(scale*beta)
    l=(logits-logits.amax(1,keepdim=True)).exp()
    pw=bank["prior"].clamp_min(1e-30).pow(-prior_power)
    gamma=bank["resp"] if mode!="same_fold" else bank["fitted_same_source"]
    return (l*pw)@gamma.T


def verify_rotation(bank):
    a,b=bank["target"].double(),bank["src"].double()
    x=bank["raw"].double()
    a,b,x=map(F.normalize,(a,b,x))
    y=minimal_rotation(x,a,b)
    error1=float((minimal_rotation(a,a,b)-b).abs().max())
    error2=float(((x*a).sum(1)-(y*b).sum(1)).abs().max())
    error3=float((minimal_rotation(y,b,a)-x).abs().max())
    if max(error1,error2,error3)>2e-6:
        raise ValueError("Unreliable paired-word rotation")
    return {"map_prompt":error1,"dot_isometry":error2,"round_trip":error3}
