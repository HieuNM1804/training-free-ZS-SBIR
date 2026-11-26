"""Minimax naming under a bounded change of latent population odds, conditional on the actual observation."""

import torch
import triton
import triton.language as tl


@triton.jit
def _envelope(theta, posterior, likelihood, word_obs, dest, N:tl.constexpr, H:tl.constexpr, V:tl.constexpr,
              ODDS:tl.constexpr, SOURCE:tl.constexpr, HS:tl.constexpr, VS:tl.constexpr):
    i=tl.program_id(0)
    h=tl.arange(0,HS)
    w=tl.program_id(1)*VS+tl.arange(0,VS)
    mu=tl.load(posterior+i*H+h,mask=h<H,other=0.)
    e=tl.load(theta+h[:,None]*V+w[None,:],mask=(h[:,None]<H)&(w[None,:]<V),other=0.)
    if SOURCE:
        obs=tl.load(word_obs+i*V+w,mask=w<V,other=0.)
        l=tl.load(likelihood+i*H+h,mask=h<H,other=1.)
        e=e*(obs[None,:]/tl.maximum(l[:,None],1e-28))
    mass=mu[:,None]*e
    base=tl.sum(mass,axis=0)
    r=base
    if ODDS>1.:
        # Dinkelbach iterations maximize a linear-fractional function over [1,Gamma]^H.
        for _ in tl.static_range(8):
            active=e>r[None,:]
            top=tl.sum(tl.where(active,mass,0.),axis=0)
            occ=tl.sum(tl.where(active,mu[:,None],0.),axis=0)
            r=(base+(ODDS-1.)*top)/(1.+(ODDS-1.)*occ)
    tl.store(dest+i*V+w,r,mask=w<V)


def naming_law(theta, post, odds, *, likelihood=None, observation=None):
    source=likelihood is not None
    theta,post=theta.contiguous(),post.contiguous()
    n,h=post.shape
    v=theta.size(1)
    if odds==1:
        if source:
            out=((post/likelihood.clamp_min(1e-28))@theta)*observation
        else:
            out=post@theta
        return out/out.sum(1,keepdim=True).clamp_min(1e-28), out.sum(1)
    out=post.new_empty(n,v)
    _envelope[(n,triton.cdiv(v,8))](theta,post,likelihood.contiguous() if source else post,
                                    observation.contiguous() if source else post,out,n,h,v,float(odds),
                                    source,triton.next_power_of_2(h),8,num_warps=4)
    z=out.sum(1)
    if not torch.isfinite(out).all() or z.min()<.9997:
        raise ValueError("Invalid conditional regret envelope")
    return out/z[:,None],z


def source_likelihood(split, text, temperature):
    return tuple((temperature*(x.float()@text.T)).softmax(1) for x in (split.sketch,split.mirror))


def source_code(inputs, theta, base, odds):
    answers=[]
    normalizers=[]
    for obs in inputs:
        likelihood=obs@theta.T
        raw=likelihood*base
        post=raw/raw.sum(1,keepdim=True).clamp_min(1e-28)
        law,z=naming_law(theta,post,odds,likelihood=likelihood,observation=obs)
        answers.append(law)
        normalizers.append(z)
    return sum(answers)/len(answers),torch.stack(normalizers).mean(0)


def population_parameters(model,law):
    theta=model.observed if law=="reports" else model.lexical_mle
    base=model.sizes/model.sizes.sum()
    # Remove round-off from iteration and make all rows proper laws.
    theta=theta/theta.sum(1,keepdim=True).clamp_min(1e-28)
    return theta,base


def target_code(model,theta,base,scale,odds):
    post=(scale*model.visual+base.log()).softmax(1)
    return naming_law(theta,post,odds)


def pair_codes(split,source_text,models,law,source_temperature,target_temperature,odds_source,odds_target):
    # A test photo is always compared with a query predicted from the same out-of-fold family.
    # Use two disjoint code blocks so the query never needs to see other sketches.
    obs=source_likelihood(split,source_text,source_temperature)
    qrows=[]
    ybank=obs[0].new_zeros(len(split.photo),2*obs[0].size(1))
    normalizers=[]
    for fold,model in enumerate(models):
        theta,base=population_parameters(model,law)
        q,zq=source_code(obs,theta,base,odds_source)
        y,zy=target_code(model,theta,base,target_temperature,odds_target)
        qrows.append(q)
        v=theta.size(1)
        ybank[model.rows,fold*v:(fold+1)*v]=y
        normalizers.append({"source_mean_log_z":zq.log().mean().item(),"target_mean_log_z":zy.log().mean().item()})
    return torch.cat(qrows,1),ybank,normalizers


def fixture(device):
    import itertools
    theta=torch.tensor([[.75,.20,.05],[.1,.8,.1],[.05,.25,.70]],device=device)
    obs=torch.tensor([[.25,.6,.15],[.6,.15,.25]],device=device)
    base=torch.tensor([.2,.3,.5],device=device)
    likelihood=obs@theta.T
    posterior=torch.nn.functional.normalize(likelihood*base,p=1,dim=1)
    law,z=naming_law(theta,posterior,8,likelihood=likelihood,observation=obs)
    cases=[]
    for mult in itertools.product((1.,8.),repeat=3):
        prior=base*torch.tensor(mult,device=device)
        prob=(prior@theta)*obs
        cases.append(prob/prob.sum(1,keepdim=True))
    exact=torch.stack(cases).amax(0)
    error=(law*z[:,None]-exact).abs().max().item()
    if error>2e-6:raise ValueError(f"Conditional envelope vs enumerated priors: {error}")
    return {"max_error":error,"base_prior":base.tolist(),"conditional_normalizers":z.tolist()}
