"""Add a permutation-centered fine-detail message under native coarse attention."""

import torch
import torch.nn.functional as F


def tokens(v, pixels, interpolate=False):
    patches=v.conv1(pixels.to(v.conv1.weight.dtype)).flatten(2).transpose(1,2)
    pos=v.positional_embedding
    if interpolate:
        c=pos.shape[1]
        lattice=F.interpolate(pos[1:].T.reshape(1,c,7,7).float(),size=(14,14),mode="bicubic",align_corners=False)
        pos=torch.cat((pos[:1],lattice.reshape(c,196).T.to(pos)),0)
    cls=v.class_embedding.to(patches).expand(len(patches),1,-1)
    return v.ln_pre(torch.cat((cls,patches),1)+pos.to(patches))


def pool(v, state):
    pooled,_=v._pool(state)
    return F.normalize((pooled@v.proj).float())


def child_grid(tile_tokens):
    """Four 7x7 quadrant lattices -> 49 parents with four spatially correct children."""
    b=tile_tokens.size(0)//4
    c=tile_tokens.size(-1)
    lattice=tile_tokens[:,1:].reshape(b,2,2,7,7,c).permute(0,1,3,2,4,5).reshape(b,14,14,c)
    return lattice.reshape(b,7,2,7,2,c).permute(0,1,3,2,4,5).reshape(b,49,4,c)


def heads(x, n):
    return x.unflatten(-1,(n,-1)).movedim(-2,1)


def messages(block, coarse, tiles, modes, validate):
    attn=block.attn
    q,k,value=F.linear(block.ln_1(coarse),attn.in_proj_weight,attn.in_proj_bias).chunk(3,-1)
    q,k,value=(heads(x,attn.num_heads) for x in (q,k,value))
    fine=block.ln_1(child_grid(tiles))
    _,fk,fv=F.linear(fine,attn.in_proj_weight,attn.in_proj_bias).chunk(3,-1)
    fk,fv=(heads(x,attn.num_heads) for x in (fk,fv))
    factor=q.size(-1)**-.5
    coarse_weights=((q@k.transpose(-1,-2)).float()*factor).softmax(-1)
    a=coarse_weights[:,:,:,1:]
    local=(q@fk.flatten(2,3).transpose(-1,-2)).float().unflatten(-1,(49,4))*factor
    child=local.softmax(-1)
    mean=fv.mean(3)
    centered=fv-mean.unsqueeze(3)
    grouped=a[...,None]*child
    correction={}
    check={}
    if validate:
        # Inspect a subset in float64 independently of production matrix products.
        weights=grouped[0,:2,:3].double()
        cvals=fv[0,:2].double()
        average=cvals-cvals.mean(2,keepdim=True)
        null=sum(torch.einsum("himr,hmrd->hid",weights,average.roll(j,2)) for j in range(4))/4
        check={"native_parent_mass_error":float((grouped.sum(-1)-a).abs().max()),
               "finite_child_reassignment_mean":float(null.abs().max())}
        if check["native_parent_mass_error"]>2e-6 or check["finite_child_reassignment_mean"]>1e-8:
            raise ValueError(f"Broken resolution identity {check}")
    def read(w,values):
        return w.flatten(-2).to(values.dtype)@values.flatten(2,3)
    for mode in modes:
        if mode=="parent":
            d=read(grouped,centered)
        elif mode=="wrong_key":
            d=read(grouped,centered.roll(1,3))
        elif mode=="wrong_parent":
            d=read(a.roll(11,3)[...,None]*child,centered)
        elif mode=="free":
            unrestrained=local.flatten(-2).softmax(-1).unflatten(-1,(49,4))
            d=read((1-coarse_weights[:,:,:,:1])[...,None]*unrestrained,centered)
        elif mode=="mean":
            d=a.to(mean.dtype)@(mean-value[:,:,1:])
        elif mode=="direct":
            d=read(grouped,centered)+a.to(mean.dtype)@(mean-value[:,:,1:])
        else: raise ValueError(mode)
        d=d.transpose(1,2).flatten(2)
        correction[mode]=F.linear(d,attn.out_proj.weight,bias=None)
    return correction,check


def features(v, pixels, variants, check=False):
    full=pixels.cuda(non_blocking=True)
    enlarged=F.interpolate(full,size=(448,448),mode="bilinear",align_corners=False)
    tiles=enlarged.reshape(len(full),3,2,224,2,224).permute(0,2,4,1,3,5).reshape(-1,3,224,224)
    c=tokens(v,full)
    fine=tokens(v,tiles)
    out={}
    checks={}
    all_blocks=v.transformer.resblocks
    if any(item[0]=="large" for item in variants):
        high=tokens(v,enlarged,interpolate=True)
        out["large",0,0.]=pool(v,v.transformer(high))
    for j,block in enumerate(all_blocks):
        at=[item for item in variants if item[1]==j+1 and item[0] not in ("native","tile","large")]
        if at:
            modes={item[0] for item in at}
            terms,diagnostics=messages(block,c,fine,modes,check)
            if check:checks[str(j+1)]=diagnostics
            original=block.attention(q_x=block.ln_1(c))
            for mode,layer,amount in at:
                active=c+block.ls_1(original+amount*terms[mode])
                active=active+block.ls_2(block.mlp(block.ln_2(active)))
                for nxt in list(all_blocks)[j+1:]:
                    active=nxt(active)
                out[mode,layer,amount]=pool(v,active)
        c=block(c)
        fine=block(fine)
    out["native",0,0.]=pool(v,c)
    if any(x[0]=="tile" for x in variants):
        out["tile",0,0.]=F.normalize(pool(v,fine).reshape(len(full),4,-1).mean(1))
    return out,checks
