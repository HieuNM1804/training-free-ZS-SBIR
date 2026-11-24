"""Does a second, disjoint frozen CLIP patch correspondence confirm the first?"""
from functools import partial

from .gcwe_displacement_consensus import score_update
from .gcwe_conditioned_donor_test import run
import torch


def collision(c,offset):
    shared_pair=c.square().sum((-1,-2))
    h=c.flatten(-2) @ offset
    repeated_offset=h.square().sum(-1)
    a=repeated_offset-shared_pair
    b=1-c.sum(-1).square().sum(-1)-c.sum(-2).square().sum(-1)+shared_pair
    if a.min() < -5e-6 or (a-b).max()>5e-6:
        raise ValueError('Joint offset agreement must be contained in distinct-endpoint trials')
    # Treat exact degeneracies as providing no distinct pair.
    conditional=torch.where(b>0,a.clamp_min(0)/b.clamp_min(torch.finfo(c.dtype).tiny),0.)
    return repeated_offset,a,conditional


def enumerate_check():
    coord=torch.tensor([[0,0],[0,1],[1,0],[1,1]])
    delta=(coord[None]-coord[:,None]).flatten(0,1)
    ids=torch.unique(delta,dim=0,return_inverse=True)[1]
    groups=torch.nn.functional.one_hot(ids).double()
    cells=torch.cartesian_prod(torch.arange(4),torch.arange(4))
    distinct=(cells[:,None]!=cells[None]).all(-1)
    same=(delta[:,None]==delta[None]).all(-1)
    for masses in (torch.arange(1,17).double(),torch.ones(16).double(),torch.eye(4).flatten().double()):
        c=(masses/masses.sum()).reshape(4,4)
        joint=c.reshape(-1,1)*c.reshape(1,-1)
        _,a,conditional=collision(c,groups)
        brute=joint[distinct&same].sum()
        torch.testing.assert_close(a,brute)
        torch.testing.assert_close(conditional,brute/joint[distinct].sum())
    single=torch.zeros(4,4,dtype=torch.double)
    single[0,0]=1
    assert collision(single,groups)[2].item()==0


def measures(coupling,geometry):
    results={}
    for rows,offset,sham in geometry:
        c=coupling.index_select(-2,rows)
        c=c/c.sum((-1,-2),keepdim=True)
        independent=c.sum(-1,keepdim=True)*c.sum(-2,keepdim=True)
        raw,unconditional,primary=collision(c,offset)
        residual=(c-independent).flatten(-2) @ offset
        values={'conditional':primary,'disjoint_joint':unconditional,'raw':raw,
                'same_partition':residual.square().sum(-1),'scrambled':collision(c,sham)[2],
                'independent':collision(independent,offset)[2]}
        for name,v in values.items():
            results[name]=results.get(name,0.)+.5*v
    # Whole-grid intervention uses the original unit-mass coupling and all 49 locations.
    # Group once by original rows and sum the vectors before taking their squared norms.
    residual=coupling-coupling.sum(-1,keepdim=True)*coupling.sum(-2,keepdim=True)
    whole=sum(residual.index_select(-2,rows).flatten(-2) @ offset for rows,offset,_ in geometry)
    wrong=sum(residual.index_select(-2,rows).flatten(-2) @ sham for rows,_,sham in geometry)
    results['whole_grid']=whole.square().sum(-1)
    results['whole_grid_sham']=wrong.square().sum(-1)
    return results


if __name__=='__main__':
    enumerate_check()
    run(update_fn=partial(score_update,measures),family='gcwe-distinct-offset-collisions',start=11520,count=1152,
        controls=False,comparisons=[('conditional',k) for k in ('original','same_partition','disjoint_joint',
            'raw','scrambled','independent','semantic_regional')]+[('whole_grid',k) for k in
            ('original','same_partition','whole_grid_sham')])
