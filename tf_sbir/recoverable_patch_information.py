"""Measure information from a real image in an exactly invisible achromatic patch subspace."""

import torch
from torch.nn import functional as F

from .data import CLIP_MEAN, CLIP_STD


FAMILY="recoverable-patch-information"
# Four phases of two crossed half-patch lattices, symmetrized around the original image.
OFFSETS=((-8,-8),(-8,8),(8,-8),(8,8))
AMOUNTS=(-8.,-2.,.5,2.,8.,24.)
MODES=("hidden","rotated_hidden","misplaced","visible_matched","all_matched","all_unscaled")


def to_tokens(v,features):
    patch=features.flatten(2).transpose(1,2).to(dtype=v.conv1.weight.dtype)
    cls=v.class_embedding.to(patch).expand(len(patch),1,-1)
    state=torch.cat((cls,patch),1)+v.positional_embedding.to(patch)
    hidden=v.transformer(v.ln_pre(v.patch_dropout(state)))
    pooled,_=v._pool(hidden)
    return F.normalize((pooled@v.proj).float())


def reshape_pixels(profile,side=32):
    n=profile.size(0)
    return profile.reshape(n,7,7,side,side).permute(0,1,3,2,4).reshape(n,1,7*side,7*side)


def back_to_native(tokens,dy,dx):
    out=tokens
    for delta,axis in ((dy,-2),(dx,-1)):
        if delta:
            weight=abs(delta)/32
            opposite=-1 if delta>0 else 1
            out=(1-weight)*out+weight*torch.roll(out,opposite,axis)
    return out


def sample_norm(x):
    return x.flatten(1).norm(dim=1)


class PatchInformation:
    def __init__(self,v):
        weights=v.conv1.weight.detach().float()
        if weights.shape!=(768,3,32,32) or v.conv1.stride!=(32,32) or v.conv1.groups!=1:
            raise ValueError("The method is defined for the original OpenAI CLIP ViT-B/32 patch tokenizer")
        self.v=v
        self.w=weights
        self.axis=F.normalize(torch.tensor(CLIP_STD,device=weights.device,dtype=torch.float64).reciprocal(),dim=0)
        # Equal physical RGB increments map to this direction after CLIP normalization.
        luminance=(weights.double()*self.axis[None,:,None,None]).sum(1).flatten(1)
        dc=torch.ones((1,1024),device=weights.device,dtype=torch.float64)/32
        rows=torch.cat((luminance,dc))
        _,sing,vh=torch.linalg.svd(rows,full_matrices=True,driver="gesvd")
        cutoff=float(sing[0])*max(rows.shape)*torch.finfo(rows.dtype).eps
        rank=int((sing>cutoff).sum())
        self.null=vh[rank:].T.contiguous().float()
        k=self.null.size(1)
        if k==0:raise ValueError("No achromatic native-null direction")
        rng=torch.Generator().manual_seed(37291)
        self.perm=torch.randperm(k,generator=rng).to(weights.device)
        self.sign=(2*torch.randint(2,(k,),generator=rng)-1).to(weights.device,self.null.dtype)
        self.info={"input_channels":3,"patch_side":32,"embedding_dimension":768,
                   "achromatic_patch_directions":1024,"rank_including_patch_mean":rank,"invisible_dimension":k,
                   "relative_smallest_nonzero_singular":float(sing[rank-1]/sing[0]),
                   "operator_leak":float((rows@self.null.double()).abs().max()),
                   "zero_mean_error":float(self.null.double().mean(0).abs().max())}
        if self.info["operator_leak"]>2e-5:raise ValueError("The native patch kernel is not numerically null")
        self.axis=self.axis.float()

    def conv(self,pixels):
        return F.conv2d(pixels,self.w,stride=32)

    def phases(self,images):
        # The normalizer scores the average at the native spatial location of each token.
        total=0.
        for dy,dx in OFFSETS:
            z=self.conv(torch.roll(images,(dy,dx),(-2,-1)))
            total=total+back_to_native(z,dy,dx)
        return total/len(OFFSETS)

    def parts(self,px):
        n=len(px)
        scalar=(px*self.axis[None,:,None,None]).sum(1,keepdim=True)
        row=F.unfold(scalar,kernel_size=32,stride=32).transpose(1,2)
        row=row-row.mean(2,keepdim=True)
        coef=row@self.null
        hidden=reshape_pixels(coef@self.null.T)*self.axis[None,:,None,None]
        placebo=reshape_pixels((coef[:,:,self.perm]*self.sign)@self.null.T)*self.axis[None,:,None,None]
        tokens=self.conv(px)
        blind=self.conv(hidden)
        shifted=self.phases(torch.cat((px,hidden,placebo)))
        full,d,w=shifted.split(n)
        whole=full-tokens
        # A finite precision centered term ensures the recovery is zero when the sampled grids coincide.
        d=d-blind
        w=w-self.conv(placebo)
        visible=whole-d
        hnorm=sample_norm(d)
        def match(z):
            ratio=hnorm/sample_norm(z).clamp_min(1e-20)
            return ratio[:,None,None,None]*z
        outputs={"hidden":d,"rotated_hidden":match(w),"misplaced":torch.roll(d,(1,1),(-2,-1)),
                 "visible_matched":match(visible),"all_matched":match(whole),"all_unscaled":whole}
        energy=coef.square().sum((1,2))/row.square().sum((1,2)).clamp_min(1e-16)
        inrel=sample_norm(blind)/sample_norm(tokens).clamp_min(1e-16)
        observed=hnorm/sample_norm(tokens).clamp_min(1e-16)
        # The counterfactual diagnostic may leave the ordinary pixel range. No image is clipped or saved.
        mean=px.new_tensor(CLIP_MEAN)[None,:,None,None]
        std=px.new_tensor(CLIP_STD)[None,:,None,None]
        counter=(px-2*hidden)*std+mean
        out=(counter.lt(-1e-4)|counter.gt(1.0001)).float().mean((1,2,3))
        rms=(2*hidden*std).square().mean((1,2,3)).sqrt()
        diag=torch.stack((energy,inrel,observed,out,rms,
                          (hnorm-sample_norm(outputs["rotated_hidden"])).abs()/hnorm.clamp_min(1e-16)),1)
        return tokens,outputs,diag,blind,hidden

    def measure(self,px,rows=None,check=False):
        rows=rows if rows is not None else [(m,a) for m in MODES for a in AMOUNTS]
        linear_native,delta,diag,blind,hidden=self.parts(px)
        # Anchor all variants to the exact convolution used by the frozen encoder, including its rounding.
        native=self.v.conv1(px.to(self.v.conv1.weight.dtype)).float()
        z=to_tokens(self.v,native)
        if check:
            mirror=self.conv(px-2*hidden)
            err=float((linear_native-mirror).abs().max()/linear_native.abs().max().clamp_min(1))
            refl=to_tokens(self.v,native+(mirror-linear_native))
            ref=F.normalize(self.v(px.to(self.v.conv1.weight.dtype)).float())
            half_refl=F.normalize(self.v((px-2*hidden).to(self.v.conv1.weight.dtype)).float())
            # Direct test of the geometric remapping against exact integration of the shifted token field.
            tiny=native[:1,:4]
            shifts=[(dy,dx) for dy,dx in OFFSETS]
            es=[]
            for dy,dx in shifts:
                dense=torch.roll(F.interpolate(tiny,scale_factor=32,mode="nearest"),(-dy,-dx),(-2,-1))
                es.append((F.avg_pool2d(dense,32)-back_to_native(tiny,dy,dx)).abs().max())
            checkvals=[err,float((1-(refl*z).sum(1)).abs().max()),float((1-(ref*z).sum(1)).abs().max()),
                       float(torch.stack(es).max()/tiny.abs().max().clamp_min(1)),
                       float((1-(half_refl*ref).sum(1)).abs().max())]
            if checkvals[0]>2e-4 or max(checkvals[1:3])>.002 or checkvals[3]>.001:
                raise ValueError(f"Token null, native readout or phase adjoint invalid {checkvals}")
        else:checkvals=[]
        matrix={}
        for i in range(0,len(rows),4):
            group=rows[i:i+4]
            tokens=torch.cat([native+alpha*delta[m] for m,alpha in group])
            values=to_tokens(self.v,tokens).split(len(px))
            matrix.update(zip(group,values,strict=True))
        return z,matrix,diag,checkvals
