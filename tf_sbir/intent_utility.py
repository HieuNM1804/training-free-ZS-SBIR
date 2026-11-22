"""Closed-form full AP of a ranked action for a latent name and a latent population cutoff."""

import torch


def normalized_membership(u, sizes):
    v, order = u.sort(dim=-1, descending=True, stable=True)
    population = sizes[order].cumsum(-1)
    next_value = torch.nn.functional.pad(v[..., 1:], (0,1))
    mass = (v - next_value) / population
    suffix = mass.flip((-1,)).cumsum(-1).flip((-1,))
    return torch.zeros_like(u).scatter_(-1, order, suffix)


def exposure(group_order, count):
    """Every list has exactly sum_g L_g*1 = number of photos."""
    batch, length = group_order.shape
    cursor = torch.arange(1,length+1,dtype=torch.float32,device=group_order.device).reciprocal()
    cells = torch.zeros((batch,length,count),dtype=torch.float32,device=group_order.device)
    cells.scatter_(2,group_order[:,:,None],1.)
    cells = torch.cumsum(cells,1)
    cells.mul_(cursor[None,:,None])
    pair = cells.new_zeros((batch,count,count))
    pair.scatter_add_(1,group_order[:,:,None].expand(-1,-1,count),cells)
    first = cells.new_zeros((batch,count))
    first.scatter_add_(1,group_order,cursor.expand(batch,-1))
    return pair,first


def co_relevance(probability, f):
    mean = (probability[:,:,None]*f).sum(1)
    joint = mean.new_zeros((len(mean),mean.size(1),mean.size(1)))
    for start in range(0,f.size(1),8):
        block=f[:,start:start+8]
        joint += (probability[:,start:start+8,None,None] * torch.minimum(block[:,:,:,None],block[:,:,None,:])).sum(1)
    collapsed=torch.minimum(mean[:,:,None],mean[:,None,:])
    if not torch.allclose(joint.diagonal(dim1=1,dim2=2),mean,rtol=2e-5,atol=1e-8):
        raise ValueError("A unary marginal changed during co-relevance construction")
    return joint,collapsed,mean


def utility(pair, first, model, weights):
    if model=="first":
        return (first*weights[:,None]).sum(-1)
    return (pair*weights[:,None]).sum((-1,-2))


def verify_identity(device):
    groups=torch.tensor([0,2,1,0,1,0,2],device=device)
    size=groups.bincount()
    membership=torch.tensor([[.7,1.,.1],[.9,.3,1.]],device=device)
    weight=torch.tensor([[.35,.65]],device=device)
    f=normalized_membership(membership,size)
    pair,first=exposure(groups[None],len(size))
    joint,_,_=co_relevance(weight,f[None])
    predicted=utility(pair[:,None],first[:,None],"joint",joint)[0,0]
    explicit=0.
    rank=torch.arange(1,len(groups)+1,device=device)
    for h,row in enumerate(membership):
        val=row.sort(descending=True).values
        next_level=torch.nn.functional.pad(val[1:],(0,1))
        for z,nxt in zip(val,next_level):
            relevant=row[groups]>=z
            ap=((relevant.cumsum(0)/rank)*relevant).sum()/relevant.sum()
            explicit += weight[0,h]*(z-nxt)*ap
    error=float((predicted-explicit).abs())
    if error>2e-6 or not torch.allclose((f*size).sum(-1),torch.ones(2,device=device)):
        raise ValueError("The closed-form utility did not reproduce explicit enumeration")
    return error
