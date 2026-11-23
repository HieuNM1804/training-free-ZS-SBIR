"""Fused, full-vocabulary expectation of the symmetric term in pairwise choice likelihoods."""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _symmetric_kernel(logu, joint, independent, out, out_i,
                      N:tl.constexpr, V:tl.constexpr, BLOCK_I:tl.constexpr, BLOCK_J:tl.constexpr,
                      BLOCK_V:tl.constexpr, PANELS:tl.constexpr):
    panel = tl.program_id(0)
    vb = tl.program_id(1)
    batch = tl.program_id(2)
    i = panel*BLOCK_I+tl.arange(0,BLOCK_I)
    j = tl.arange(0,BLOCK_J)
    v = vb*BLOCK_V+tl.arange(0,BLOCK_V)
    first = tl.load(logu+(batch*N+i[:,None])*V+v[None,:],mask=(i[:,None]<N)&(v[None,:]<V),other=0.)
    second = tl.load(logu+(batch*N+j[:,None])*V+v[None,:],mask=(j[:,None]<N)&(v[None,:]<V),other=0.)
    diff = tl.abs(first[:,None,:]-second[None,:,:])
    # log cosh(d/2) is symmetric and nonnegative; the antisymmetric d/2
    # has an exact expectation determined entirely by the two winner marginals.
    penalty = .5*diff+tl.log(1.+tl.exp(-diff))-.6931471805599453
    mask = (i[:,None]<N)&(j[None,:]<N)
    off = (batch*N+i[:,None])*N+j[None,:]
    a = tl.load(joint+off,mask=mask,other=0.)
    b = tl.load(independent+off,mask=mask,other=0.)
    expected = tl.sum(tl.sum(penalty*a[:,:,None],axis=1),axis=0)
    null = tl.sum(tl.sum(penalty*b[:,:,None],axis=1),axis=0)
    pos = (batch*PANELS+panel)*V+v
    tl.store(out+pos,expected,mask=v<V)
    tl.store(out_i+pos,null,mask=v<V)


def penalties(logu,joint,independent):
    logu=logu.contiguous()
    joint=joint.contiguous()
    independent=independent.contiguous()
    b,n,v=logu.shape
    panels=triton.cdiv(n,4)
    mu=logu.new_empty((b,panels,v))
    nu=torch.empty_like(mu)
    _symmetric_kernel[(panels,triton.cdiv(v,32),b)](logu,joint,independent,mu,nu,
        n,v,4,triton.next_power_of_2(n),32,panels,num_warps=4)
    return mu.sum(1),nu.sum(1)


def check_penalty_kernel():
    r=torch.Generator(device='cuda').manual_seed(550)
    logu=torch.randn((2,41,111),device='cuda',generator=r)*3
    j=torch.rand((2,41,41),device='cuda',generator=r)
    j=j/j.sum((1,2),keepdim=True)
    ind=j.sum(2)[:,:,None]*j.sum(1)[:,None,:]
    mu,nu=penalties(logu,j,ind)
    delta=logu.double()[:,:,None,:]-logu.double()[:,None,:,:]
    value=.5*delta.abs()+torch.log1p((-delta.abs()).exp())-math.log(2)
    error=max(float(((j.double()[:,:,:,None]*value).sum((1,2))-mu).abs().max()),
              float(((ind.double()[:,:,:,None]*value).sum((1,2))-nu).abs().max()))
    if error>5e-6:raise ValueError(f'Inaccurate pairwise likelihood kernel: {error}')
    return error
