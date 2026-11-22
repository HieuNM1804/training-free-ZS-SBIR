"""Word-specific matching after removing the scalar evidence that the word already supplies."""

from dataclasses import dataclass

import torch


def tiles(parts, resolution):
    if resolution=="heads":
        return parts[:,:48]
    if resolution=="blocks":
        return parts[:,:48].float().unflatten(1,(4,12)).sum(2)
    raise ValueError(resolution)


def quadratic(text,covariance):
    return ((text@covariance)*text).sum(1)


def covariance_words(a,b,covariance):
    return ((a@covariance)*b).sum(1)


def inverse_scale(variance,reference=None):
    if reference is None:
        reference=variance
    return torch.where(variance>reference.clamp_min(1e-12)*1e-6, variance.clamp_min(1e-16).rsqrt(),0.)


@dataclass
class Moments:
    means:torch.Tensor
    global_mean:torch.Tensor
    slope:torch.Tensor
    raw_inverse:torch.Tensor
    inverse:torch.Tensor
    global_inverse:torch.Tensor


class ConditionalHeads:
    """All gallery statistics and matrix products respect independent per-sketch inference."""

    def __init__(self,parts,source,target,resolution):
        # The complete logit is the sum over all stored slots, even when only
        # the final heads (or their block sums) are used to check a word.
        full=parts.sum(1,dtype=torch.float32)
        self.global_center=full.mean(0)
        centered=full-self.global_center
        n=len(parts)
        total_cov=centered.T@centered/n
        xs=quadratic(source,total_cov)
        xt=quadratic(target,total_cov)
        cross_full=covariance_words(source,target,total_cov)
        items=tiles(parts,resolution)
        self.h=items.size(1)
        self.mean=items.mean(0,dtype=torch.float32)
        self.scale=[]
        src_tables=[[],[],[],[]]
        tgt_tables=[[],[],[],[]]
        rho=[]
        # Each stored descriptor is standardized for mixed-precision gallery multiplication.
        features=torch.empty((n,self.h+1,full.size(1)),device=parts.device,dtype=torch.float16)
        for j in range(self.h):
            h=items[:,j].float()-self.mean[j]
            cov_h=h.T@h/n
            cov_x=h.T@centered/n
            weight=h.square().mean().sqrt().clamp_min(1e-12)
            self.scale.append(weight)
            features[:,j]=(h/weight).half()
            bs=quadratic(source,cov_x)/xs.clamp_min(1e-16)
            bt=quadratic(target,cov_x)/xt.clamp_min(1e-16)
            vs=quadratic(source,cov_h)
            vt=quadratic(target,cov_h)
            es=inverse_scale((vs-bs.square()*xs).clamp_min(0),vs)
            et=inverse_scale((vt-bt.square()*xt).clamp_min(0),vt)
            num=(covariance_words(source,target,cov_h)
                 -bt*covariance_words(source,target,cov_x)
                 -bs*covariance_words(source,target,cov_x.T)
                 +bs*bt*cross_full)
            rho.append((num*es*et).clamp(-1,1))
            for table,text,b,v,e in ((src_tables,source,bs,vs,es),(tgt_tables,target,bt,vt,et)):
                for part,row in zip(table,(text@self.mean[j],b,inverse_scale(v),e)):
                    part.append(row)
        tw=centered.square().mean().sqrt().clamp_min(1e-12)
        self.scale=torch.cat((torch.stack(self.scale),tw.reshape(1)))
        features[:,-1]=(centered/tw).half()
        self.index=features.flatten(1)
        self.rho=torch.stack(rho,1)
        self.source,self.target=source,target
        self.resolution=resolution
        self.permutation=(torch.arange(self.h,device=parts.device).reshape(-1,12).roll(1,1).flatten() if resolution=="heads"
                          else torch.arange(self.h,device=parts.device).roll(1))
        def store(table,text,variance):
            means,slope,raw_inverse,inv=(torch.stack(x,1) for x in table)
            return Moments(means,text@self.global_center,slope,raw_inverse,inv,inverse_scale(variance))
        self.s=store(src_tables,source,xs)
        self.t=store(tgt_tables,target,xt)
        # Audit empirical conditional uncorrelatedness directly from held image rows.
        sample_terms=torch.arange(0,len(source),max(1,len(source)//24),device=parts.device)[:24]
        scalar=(full@source[sample_terms].T)-self.s.global_mean[sample_terms]
        err=[]
        for j in range(self.h):
            h=items[:,j].float()
            raw=h@source[sample_terms].T-self.s.means[sample_terms,j]
            residual=(raw-scalar*self.s.slope[sample_terms,j])*self.s.inverse[sample_terms,j]
            err.append((residual*scalar*self.s.global_inverse[sample_terms]).mean(0).abs().max())
        self.orthogonality_error=float(torch.stack(err).max())
        if self.orthogonality_error>.003:
            raise ValueError(f"Unexpected residual/full-score correlation {self.orthogonality_error}")

    def query(self,parts,terms):
        q=tiles(parts,self.resolution).float()
        total=parts.sum(1,dtype=torch.float32)
        source=self.source[terms]
        x=torch.bmm(source,q.transpose(1,2))
        z=(source*total[:,None]).sum(-1)-self.s.global_mean[terms]
        x=x-self.s.means[terms]
        res=(x-z[:,:,None]*self.s.slope[terms])*self.s.inverse[terms]
        return x*self.s.raw_inverse[terms],res,z*self.s.global_inverse[terms]

    def read(self,terms,prob,measure,mode):
        raw,res,scalar=measure
        if mode=="scalar":
            vec=((prob*scalar*self.t.global_inverse[terms])[:,:,None]*self.target[terms]).sum(1)
            target_x=self.index[:,-self.target.size(1):].float()*self.scale[-1]
            return vec@target_x.T
        tgt=self.target[terms]
        if mode=="raw":
            coeff=raw*self.t.raw_inverse[terms]*prob[:,:,None]/self.h
            slope=torch.zeros_like(coeff[:,:,0])
        else:
            shape=res
            if mode=="permuted":
                shape=shape.index_select(2,self.permutation)
            coeff=shape*self.t.inverse[terms]*prob[:,:,None]/self.h
            if mode in ("transport","permuted"):
                coeff=coeff*self.rho[terms]
            elif mode!="aligned":
                raise ValueError(mode)
            slope=-(coeff*self.t.slope[terms]).sum(2)
        vectors=torch.bmm(coeff.transpose(1,2),tgt)
        total=(slope[:,:,None]*tgt).sum(1,keepdim=True)
        v=(torch.cat((vectors,total),1)*self.scale[None,:,None]).flatten(1)
        # Keep one shared dynamic scaling per independent query. Cross-word
        # cancellations happen before this product, including the full-score subtraction.
        scale=v.abs().amax(1,keepdim=True).clamp_min(1e-15)
        return ((v/scale).half()@self.index.T).float()*scale

    def explicit_error(self,photo,query,terms,prob,measure):
        p=tiles(photo,self.resolution).float()
        total=photo.sum(1,dtype=torch.float32)
        qres=measure[1]
        explicit=[]
        for i in range(len(query)):
            ix=terms[i]
            t=self.target[ix]
            z=total@t.T-self.t.global_mean[ix]
            h=torch.einsum("nhd,kd->nkh",p,t)-self.t.means[ix]
            r=(h-z[:,:,None]*self.t.slope[ix])*self.t.inverse[ix]
            explicit.append((r*qres[i]*self.rho[ix]*prob[i,:,None]).sum((1,2))/self.h)
        calculated=self.read(terms,prob,measure,"transport")[:,:len(photo)]
        expected=torch.stack(explicit)
        denom=expected.square().mean().sqrt().clamp_min(1e-8)
        return float((expected-calculated).square().mean().sqrt()/denom)
