"""Project a query's observable moment toward operators that share latent categorical structure."""

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F


FAMILY="shared-semantic-operators"
DIMENSIONS=(8,16,32)
PENALTIES=(.15,1.,6.)
BLENDS=(.15,.5,1.)
GAINS=(.35,1.,3.)


def sym(x):
    return (x+x.transpose(-1,-2))*.5


def traceless(x):
    d=x.size(-1)
    eye=torch.eye(d,device=x.device,dtype=x.dtype)
    return x-x.diagonal(dim1=-2,dim2=-1).mean(-1)[...,None,None]*eye


def distinct_mean(x):
    r=x.size(1)
    total=x.sum(1)
    return sym((total.T@total-x.flatten(0,1).T@x.flatten(0,1))/(len(x)*r*(r-1)))


def per_photo(x):
    r=x.size(1)
    total=x.sum(1)
    return sym((total[:,:,None]*total[:,None,:]-x.transpose(1,2)@x)/(r*(r-1)))


@dataclass
class SymmetricBasis:
    row: torch.Tensor
    col: torch.Tensor
    scale: torch.Tensor
    size: int

    @classmethod
    def create(cls,size,device,dtype):
        row,col=torch.triu_indices(size,size,device=device)
        scale=1+(row!=col).to(dtype)*(math.sqrt(2)-1)
        return cls(row,col,scale,size)

    def pack(self,x):
        return x[...,self.row,self.col]*self.scale

    def unpack(self,x):
        m=x.new_zeros((*x.shape[:-1],self.size,self.size))
        vals=x/self.scale
        m[...,self.row,self.col]=vals
        m[...,self.col,self.row]=vals
        return m


def random_basis(d,device,dtype,seed):
    draw=torch.randn(d,d,generator=torch.Generator().manual_seed(seed),dtype=dtype).to(device)
    q,r=torch.linalg.qr(draw)
    return q*r.diagonal().sign()


class CommonOperators:
    def __init__(self,operators):
        d=operators.shape[-1]
        self.basis=SymmetricBasis.create(d,operators.device,operators.dtype)
        # Trace cannot contribute to a commutator. Give each named measurement equal energy.
        a=traceless(operators)
        a=a/a.square().sum((-1,-2),keepdim=True).sqrt().clamp_min(1e-12)
        self.probes=a
        n=len(self.basis.row)
        basis=self.basis.unpack(torch.eye(n,device=a.device,dtype=a.dtype))
        aa=(a@a).mean(0)
        gram=aa@basis+basis@aa
        for visited in a.split(8):
            for start in range(0,n,64):
                loc=slice(start,start+64)
                gram[loc]-=2*((visited[:,None]@basis[None,loc])@visited[:,None]).sum(0)/len(a)
        matrix=sym(self.basis.pack(gram))
        self.unit=matrix.trace().clamp_min(1e-10)/max(1,n-1)
        self.values,self.axes=torch.linalg.eigh(matrix.double())
        if self.values.min() < -1e-5*float(self.unit)-1e-7:
            raise ValueError("Commutator quadratic form is not positive semidefinite")
        self.values=(self.values.clamp_min(0)/self.unit).to(a.dtype)
        self.axes=self.axes.to(a.dtype)
        self.rot=random_basis(d,a.device,a.dtype,39107+d)

    def project(self,observed,penalty):
        part=traceless(observed)
        coef=self.basis.pack(part)@self.axes
        out=self.basis.unpack((coef/(1+penalty*self.values))@self.axes.T)
        return observed-part+traceless(out)

    def energy(self,observed):
        centered=traceless(observed)
        coords=self.basis.pack(centered)@self.axes
        return (coords.square()*self.values).sum(1)/coords.square().sum(1).clamp_min(1e-16)

    def alternatives(self,observed,penalty):
        out=self.project(observed,penalty)
        rot=self.rot
        rotated=rot@self.project(rot.T@observed@rot,penalty)@rot.T
        b=traceless(observed)
        factor=traceless(out).square().sum((-1,-2),keepdim=True).sqrt()/b.square().sum((-1,-2),keepdim=True).sqrt().clamp_min(1e-16)
        return {"shared":out,"matched":observed-b+b*factor,"rotated":rotated}


@dataclass
class GalleryFold:
    read_ids: torch.Tensor
    target_ids: torch.Tensor
    train_moments: torch.Tensor
    candidates: torch.Tensor
    linear_map: tuple
    common: CommonOperators
    diagnostics: dict

    def read(self,score,penalties):
        h=score[:,self.read_ids]
        shift=h.mean(1,keepdim=True)
        hc=h-shift
        q=self.common.basis.unpack((hc@self.train_moments)/h.size(1))
        def project(x):
            return shift+self.common.basis.pack(x)@self.candidates.T
        raw=project(q)
        out={"raw":raw,"linear":shift+hc@self.linear_map[0]@self.linear_map[1]}
        for strength in penalties:
            for kind,estimate in self.common.alternatives(q,strength).items():
                out[kind,strength]=project(estimate)
        return out


def probes_from_names(p,names,scale,count=64):
    prior=p.new_zeros(len(names))
    for batch in p.split(192):
        prior+=(scale*(batch@names.T)).softmax(1).sum(0)
    return prior.topk(count).indices


def make_folds(views,photo,tgt,scale,dimensions=DIMENSIONS):
    device=photo.device
    permutation=torch.randperm(len(photo),generator=torch.Generator().manual_seed(59731))
    parts=[j.to(device) for j in permutation.tensor_split(3)]
    result={d:[] for d in dimensions}
    for turn in range(3):
        ref,train,test=parts[turn],parts[(turn+1)%3],parts[(turn+2)%3]
        ref_views=views[ref]
        m=distinct_mean(ref_views)
        eigen,axes=torch.linalg.eigh(m.double())
        eigen=eigen.flip(0).to(m.dtype)
        axes=axes.flip(1).to(m.dtype)
        valid=int((eigen>eigen[0]*1e-6).sum())
        if valid<max(dimensions):
            raise ValueError("Too few positive moment directions")
        selected=probes_from_names(photo[ref],tgt,scale)
        href=photo[ref]@tgt[selected].T
        htest=photo[train]@tgt[selected].T
        for dim in dimensions:
            white=axes[:,:dim]/eigen[:dim].sqrt()
            vx=views@white
            bas=SymmetricBasis.create(dim,device,views.dtype)
            xref=bas.pack(per_photo(vx[ref]))
            xtrain=bas.pack(per_photo(vx[train]))
            probes=bas.unpack((href-href.mean(0)).T@xref/len(ref))
            common=CommonOperators(probes)
            checked=bas.unpack((htest-htest.mean(0)).T@xtrain/len(train))
            control=common.rot.T@checked@common.rot
            energy=common.energy(checked)
            null=common.energy(control)
            z=vx.mean(1)
            zn=F.normalize(z[test],dim=1)
            candidate=bas.pack(zn[:,:,None]*zn[:,None,:])
            cx=z[train]-z[train].mean(0)
            cy=z[test]-z[train].mean(0)
            linear=(cx@torch.linalg.pinv((cx.T@cx).double(),hermitian=True).to(cx.dtype),cy.T)
            rec=bas.unpack(xref.mean(0))
            eye=torch.eye(dim,device=device)
            identity_error=float((rec-eye).abs().max())
            if identity_error>.002:raise ValueError(f"Distinct-view whitening failed {identity_error}")
            info={"whiten_max_error":identity_error,"positive_directions":valid,
                  "relative_eigen_at_cutoff":float(eigen[dim-1]/eigen[0]),
                  "reference_energy_of_held_names":float(energy.mean()),"common_rotation_null_energy":float(null.mean()),
                  "held_names_more_commuting":float((energy<null).float().mean()),
                  "reference_photo_count":len(ref),"read_photo_count":len(train),"candidate_count":len(test)}
            result[dim].append(GalleryFold(train,test,xtrain,candidate,linear,common,info))
        print("compiled gallery",turn,len(ref),{d:round(result[d][-1].diagnostics["reference_energy_of_held_names"],3) for d in dimensions},flush=True)
    return result


def algebra(device):
    dtype=torch.float64
    rng=torch.Generator(device=device).manual_seed(7061)
    modes=5
    rot=random_basis(modes,device,dtype,834)
    probes=rot@torch.diag_embed(torch.randn(12,modes,device=device,generator=rng,dtype=dtype))@rot.T
    fit=CommonOperators(probes)
    true=rot@torch.diag_embed(torch.randn(4,modes,device=device,generator=rng,dtype=dtype))@rot.T
    other=fit.rot.T@true@fit.rot
    x=fit.project(other,1.6)
    energy_before=fit.energy(other)
    energy_after_unscaled=(fit.basis.pack(traceless(x))@fit.axes).square()@fit.values
    energy_before_unscaled=(fit.basis.pack(traceless(other))@fit.axes).square()@fit.values
    fidelity=float((fit.project(true,3)-true).abs().max())
    trace=float((x.diagonal(dim1=-2,dim2=-1).sum(1)-other.diagonal(dim1=-2,dim2=-1).sum(1)).abs().max())
    source=torch.randn(8,5,6,device=device,generator=rng,dtype=dtype)
    exact=sum(source[:,i,:,None]*source[:,j,None,:] for i in range(5) for j in range(5) if i!=j)/20
    identity=float((per_photo(source)-exact).abs().max())
    if max(fidelity,trace,identity)>1e-8 or (energy_after_unscaled>energy_before_unscaled+1e-8).any():
        raise ValueError("Ideal mixture, trace, smoothing or distinct-view identities failed")
    return {"unchanged_exact_shared_states":fidelity,"trace":trace,"exact_distinct_region_pairs":identity,
            "rotated_noncommutation":float(energy_before.mean())}
