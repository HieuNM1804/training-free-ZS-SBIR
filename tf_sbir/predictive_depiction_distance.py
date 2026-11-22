"""A pairwise, held-out-name distance from two frozen depictions of each public name."""

import torch


FAMILY = "predictive-depiction-distance"
NAMES = (16, 32, 64)
RIDGES = (.15, 3.)
GAINS = (.2, .7, 2., 5.)
FIXED = ("native", "middle")
ESTIMATES = ("predict", "gain", "inside", "inside_gain", "source", "target", "wrong")


def center(a):
    return a-a.mean(-1,keepdim=True)


def similarity(mse):
    return -.5*mse.clamp_min(1e-12).log()


def predict(residual, u, v, ridge):
    """PRESS for a free intercept and two ridge-penalized, image-specific directions."""
    k = residual.size(-1)
    r, a, b = center(residual), center(u), center(v)
    aa = a.square().sum(-1,keepdim=True)+k*ridge
    bb = b.square().sum(-1,keepdim=True)+k*ridge
    ab = (a*b).sum(-1,keepdim=True)
    ar = (a*r).sum(-1,keepdim=True)
    br = (b*r).sum(-1,keepdim=True)
    determinant = (aa*bb-ab.square()).clamp_min(1e-20)
    wa = (bb*ar-ab*br)/determinant
    wb = (aa*br-ab*ar)/determinant
    e = r-wa*a-wb*b
    h = 1/k+(bb*a.square()-2*ab*a*b+aa*b.square())/determinant
    deleted = e/(1-h).clamp_min(1e-7)
    return deleted.square().mean(-1), e.square().mean(-1), deleted


def fixed_distance(r):
    k=r.size(-1)
    return similarity((center(r)*(k/(k-1))).square().mean(-1))


def estimates(x, y, u, v, ridge, modes):
    residual=x-y
    native=fixed_distance(residual)
    out={}
    if "native" in modes:
        out["native"]=native
    if "middle" in modes:
        out["middle"]=fixed_distance(residual+.5*(u-v))
    if modes.intersection(("predict","gain","inside","inside_gain")):
        held,inside,_=predict(residual,u,v,ridge)
        full=similarity(held)
        train=similarity(inside)
        if "predict" in modes:out["predict"]=full
        if "gain" in modes:out["gain"]=full-native
        if "inside" in modes:out["inside"]=train
        if "inside_gain" in modes:out["inside_gain"]=train-native
    if "source" in modes:
        out["source"]=similarity(predict(residual,u,torch.zeros_like(v),ridge)[0])
    if "target" in modes:
        out["target"]=similarity(predict(residual,torch.zeros_like(u),v,ridge)[0])
    if "wrong" in modes:
        perm=torch.randperm(v.size(-1),generator=torch.Generator().manual_seed(92741)).to(v.device)
        out["wrong"]=similarity(predict(residual,u,v.index_select(-1,perm),ridge)[0])
    return out


def algebra(device):
    rng=torch.Generator(device=device).manual_seed(3701)
    def random():
        return torch.randn(4,13,device=device,generator=rng,dtype=torch.float64)
    a,b,r=random(),random(),random()
    a[0]=4
    b[1]=2*a[1]
    a[2]=0
    b[2]=0
    rows=[]
    for ridge in RIDGES:
        k=r.size(1)
        truth=[]
        design=torch.stack((torch.ones_like(r),a,b),-1)
        penalty=torch.diag(torch.tensor([0.,k*ridge,k*ridge],device=device,dtype=r.dtype))
        for j in range(k):
            mask=torch.arange(k,device=device)!=j
            xx=design[:,mask]
            coef=torch.linalg.solve(xx.transpose(1,2)@xx+penalty,xx.transpose(1,2)@r[:,mask,None])
            truth.append(r[:,j]-(design[:,j,None]@coef).flatten())
        deleted=predict(r,a,b,ridge)[2]
        rows.append(float((torch.stack(truth,1)-deleted).abs().max()))
        rows.append(float((predict(r,a,b,ridge)[0]-predict(-r,b,a,ridge)[0]).abs().max()))
    if max(rows)>1e-8:
        raise ValueError(f"Deleted-name predictions failed explicit refits {rows}")
    return rows
