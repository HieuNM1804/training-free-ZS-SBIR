"""Discover and cross-check gallery spectral evidence from independent photos and concepts."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from threadpoolctl import threadpool_limits
import torch
import torch.nn.functional as F

from .encoding import save_tensor
from .gallery_diffusion import nearest_gallery


def read_basis(folder, split, photos, posterior, target, view, neighbors, rank):
    # Index features are frozen. The enclosing experiment verifies split/order,
    # checkpoint and word hash at the caller before this gallery-only operation.
    file=folder/f"{split}-{view}-k{neighbors}-r{rank}.pt"
    if file.exists():
        return torch.load(file,weights_only=True,map_location="cpu")
    if view=="rgb":
        features=F.normalize(photos)
    else:
        sem=posterior@target
        sem=F.normalize(sem-sem.mean(0))
        rgb=F.normalize(photos-photos.mean(0))
        features=sem if view=="semantic" else F.normalize(torch.cat((rgb,sem),1))
    ids,vals=nearest_gallery(features,neighbors)
    distance=(2-2*vals).clamp_min(1e-8)
    sigma=distance[:,-1].sqrt()
    weights=(-distance/(sigma[:,None]*sigma[ids]).clamp_min(1e-8)).exp()
    n=len(features)
    rows=np.repeat(np.arange(n,dtype=np.int32),neighbors)
    directed=sparse.csr_matrix((weights.cpu().numpy().ravel(),(rows,ids.cpu().numpy().ravel())),shape=(n,n))
    graph=(directed+directed.T)*.5
    degree=np.asarray(graph.sum(1)).ravel()
    inv=1/np.sqrt(degree)
    transition=graph.multiply(inv[:,None]).multiply(inv[None,:]).tocsr()
    rng=np.random.default_rng(117)
    with threadpool_limits(limits=1):
        ev,u=eigsh(transition,k=rank+1,which="LA",tol=2e-5,v0=rng.normal(size=n).astype(np.float32))
    order=ev.argsort()[::-1]
    ev,u=ev[order].copy(),u[:,order].copy()
    residual=np.linalg.norm(transition@u-u*ev,axis=0)
    if residual.max()>5e-4 or not np.isfinite(u).all():
        raise ValueError(f"Inaccurate gallery eigensystem {residual.max()}")
    result={"eigenvalues":torch.from_numpy(ev),"vectors":torch.from_numpy(u),
            "degree":torch.from_numpy(degree),"max_residual":float(residual.max()),
            "neighbors":ids.cpu().int()}
    save_tensor(result,file)
    return result


class CrossedSpectrum:
    def __init__(self,basis,profiles):
        self.p=profiles
        degree=basis["degree"].to(profiles)
        mass=degree/degree.sum()
        u=basis["vectors"].to(profiles)
        # The stationary signal is an intercept. Disconnected component contrasts,
        # if present, are handled by the explicit weighted centering below.
        self.eigenvalues=basis["eigenvalues"].to(profiles)[1:]
        self.read=u[:,1:]*degree.sqrt()[:,None]
        self.write=u[:,1:]/degree.sqrt()[:,None]
        self.mean=mass@profiles
        self.table=profiles.new_empty((len(self.eigenvalues),profiles.size(1)))
        self.diagonal=torch.empty_like(self.table)
        r2=self.read.square().T.contiguous()
        for start in range(0,profiles.size(1),512):
            sl=slice(start,start+512)
            center=profiles[:,sl]-self.mean[sl]
            self.table[:,sl]=self.read.T@center
            self.diagonal[:,sl]=r2@center.square()

    def estimates(self,l):
        e=l@self.p.T
        centered=e-(l@self.mean)[:,None]
        c=l@self.table.T
        vb=l.square()@self.table.square().T
        va=centered.square()@self.read.square()
        overlap=l.square()@self.diagonal.T
        return e,c,va,vb,overlap

    def scores(self,stats,params):
        e,c,va,vb,overlap=stats
        rank,width,mode,power,residual=params
        c=c[:,:rank]
        a=c.square()
        if mode=="fixed":
            gain=1.
        else:
            count=(rank+width-1)//width
            def pool(x):
                return F.pad(x[:,:rank],(0,count*width-rank)).view(len(x),count,width).sum(2)
            energy=pool(a)
            if mode=="cross":
                noise=pool(va+vb-overlap)
            elif mode=="photos":
                noise=pool(va)
            elif mode=="words":
                noise=pool(vb)
            else:
                raise ValueError(mode)
            g=((energy-noise)/energy.clamp_min(1e-30)).clamp(0,1)
            gain=g.repeat_interleave(width,dim=1)[:,:rank]
        coeff=c*gain*self.eigenvalues[:rank].clamp_min(0).pow(power)
        return coeff@self.write[:,:rank].T+residual*e


def verify_crossed_formula(model,l):
    # The exact two-way identity for one graph coordinate on a small grid of
    # real photo/word contributions, checked by enumerating every donor pair.
    k=min(4,len(model.eigenvalues)-1)
    photos=torch.linspace(0,len(model.p)-1,13,device=l.device).long()
    words=l[0].topk(11).indices
    x=(model.read[photos,k,None]*(model.p[photos][:,words]-model.mean[words])*l[0,words]).double()
    sumsq=x.sum().square()-x.sum(1).square().sum()-x.sum(0).square().sum()+x.square().sum()
    rr=torch.arange(len(photos),device=x.device)
    cc=torch.arange(len(words),device=x.device)
    mask=(rr[:,None,None,None]!=rr[None,None,:,None])&(cc[None,:,None,None]!=cc[None,None,None,:])
    explicit=(x[:,:,None,None]*x[None,None,:,:]*mask).sum()
    error=float((explicit-sumsq).abs()/x.abs().sum().square().clamp_min(1e-30))
    if error>1e-9:
        raise ValueError("Distinct-photo and distinct-word pairs were counted incorrectly")
    return error
