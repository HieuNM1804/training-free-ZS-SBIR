"""Basis-independent observable moments from the whole unlabeled visual measurement space."""

import torch
from torch.nn import functional as F

from .shared_semantic_operators import (CommonOperators, GalleryFold, SymmetricBasis, distinct_mean,
                                     per_photo, probes_from_names, random_basis, sym, traceless)


REFS=("public","visual","white","one","visual16")
DIMS=(24,48,64)
STRENGTHS=(.5,3.,16.)
BLENDS=(.06,.15,.35,.65)
GAINS=(.35,1.25)
CONDITIONS=("real","matched","rotated")


class IntegratedOperators(CommonOperators):
    def __init__(self,ops,unit_probes=False):
        d=ops.shape[-1]
        self.basis=SymmetricBasis.create(d,ops.device,ops.dtype)
        a=traceless(ops)
        if unit_probes:a=a/a.square().sum((-1,-2),keepdim=True).sqrt().clamp_min(1e-12)
        # A single overall scale has no effect on the normalized normal equation.
        a=a/(a.square().sum((1,2)).mean().sqrt().clamp_min(1e-16))
        self.probes=a
        n=len(self.basis.row)
        gram=(a.flatten(1).T@a.flatten(1)/len(a)).reshape(d,d,d,d)
        square=(a@a).mean(0)
        r,c=self.basis.row,self.basis.col
        matrix=a.new_empty((n,n))
        k,l=r[None,:],c[None,:]
        for pos in range(0,n,96):
            block=slice(pos,pos+96)
            i,j=r[block,None],c[block,None]
            inner=gram[i,k,l,j]+(k!=l)*gram[i,l,k,j]
            outer=(square[i,k]*(j==l)+(i==k)*square[l,j]
                  +(k!=l)*(square[i,l]*(j==k)+(i==l)*square[k,j]))
            matrix[block]=(outer-2*inner)*(self.basis.scale[block,None]/self.basis.scale[None,:])
        matrix=sym(matrix)
        self.unit=matrix.trace().clamp_min(1e-10)/(n-1)
        self.values,self.axes=torch.linalg.eigh(matrix.double())
        if self.values.min() < -1e-4*float(self.unit)-1e-7:
            raise ValueError("Integrated commutator Gram matrix is not positive semidefinite")
        self.values=(self.values.clamp_min(0)/self.unit).to(a.dtype)
        self.axes=self.axes.to(a.dtype)
        self.rot=random_basis(d,a.device,a.dtype,39107+d)


def measurements(photo,cal,probe,tgt,scale):
    z=photo[cal]-photo[cal].mean(0)
    b=photo[probe]-photo[probe].mean(0)
    names=probes_from_names(photo[cal],tgt,scale)
    vals,vectors=torch.linalg.eigh((z.T@z/len(z)).double())
    keep=vals>1e-6*vals[-1]
    basis=vectors[:,keep].to(z.dtype)
    white=basis/vals[keep].sqrt().to(z.dtype)
    return {"public":(z@tgt[names].T,b@tgt[names].T),
            "visual":(z,b),"white":(z@white,b@white),
            "one":(z@basis[:,-1:],b@basis[:,-1:]),"visual16":(z@basis[:,-16:],b@basis[:,-16:])}


def build(views,photo,tgt,settings,seed=59731,scale=100.):
    device=photo.device
    folds=[a.to(device) for a in torch.randperm(len(photo),generator=torch.Generator().manual_seed(seed)).tensor_split(3)]
    fits={key:[] for key in settings}
    for idx in range(3):
        cal,probe,test=folds[idx],folds[(idx+1)%3],folds[(idx+2)%3]
        c=distinct_mean(views[cal])
        vals,axes=torch.linalg.eigh(c.double())
        vals=vals.flip(0).to(c.dtype)
        axes=axes.flip(1).to(c.dtype)
        measures=measurements(photo,cal,probe,tgt,scale)
        for d in sorted({k[0] for k in settings}):
            if vals[d-1]<=vals[0]*1e-6:raise ValueError("Requested dimension is numerically singular")
            z=views@(axes[:,:d]/vals[:d].sqrt())
            bas=SymmetricBasis.create(d,z.device,z.dtype)
            ref=bas.pack(per_photo(z[cal]))
            train=bas.pack(per_photo(z[probe]))
            mid=z.mean(1)
            tc=F.normalize(mid[test],dim=1)
            candidate=bas.pack(tc[:,:,None]*tc[:,None,:])
            x=mid[probe]-mid[probe].mean(0)
            linear=(x@torch.linalg.pinv((x.T@x).double(),hermitian=True).to(x.dtype),(mid[test]-mid[probe].mean(0)).T)
            for refname in sorted(name for dim,name in settings if dim==d):
                a,b=measures[refname]
                fitted=IntegratedOperators(bas.unpack(a.T@ref/len(a)),refname=="public")
                # Compare exactly the same public measurement residuals for every basis.
                checked=bas.unpack(measures["public"][1].T@train/len(probe))
                real=fitted.energy(checked)
                rotated=fitted.energy(fitted.rot.T@checked@fitted.rot)
                info={"held_public_energy":float(real.mean()),"rotated_energy":float(rotated.mean()),
                      "fraction_below_rotated":float((real<rotated).float().mean()),"probe_axes":a.size(1),
                      "relative_smallest_moment":float(vals[d-1]/vals[0])}
                fits[d,refname].append(GalleryFold(probe,test,train,candidate,linear,fitted,info))
        print("measurement folds",idx,len(cal),"dimensions",sorted({k[0] for k in settings}),flush=True)
    return fits


def population_checks(device):
    gen=torch.Generator(device=device).manual_seed(1735)
    a=sym(torch.randn(14,8,8,device=device,generator=gen,dtype=torch.float64))
    b=sym(torch.randn(5,8,8,device=device,generator=gen,dtype=torch.float64))
    h=random_basis(len(a),device,a.dtype,802)
    first=IntegratedOperators(a)
    transformed=IntegratedOperators((h@a.flatten(1)).reshape_as(a))
    exact=CommonOperators(a)
    faster=IntegratedOperators(a,True)
    err1=float((first.project(b,4)-transformed.project(b,4)).abs().max())
    err2=float((faster.project(b,4)-exact.project(b,4)).abs().max())
    if max(err1,err2)>1e-8:
        raise ValueError(f"Integrated-operator identities failed {err1}, {err2}")
    return {"visual_probe_basis_invariance":err1,"fast_vs_explicit_public_commutators":err2}
