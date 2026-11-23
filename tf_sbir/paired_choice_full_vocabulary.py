"""Exactly separate preference direction and symmetric dispersion for every public word."""
import torch

from .paired_depic_choice_law import winner_law
from .paired_choice_penalty import penalties


def infer_full(p,donors,real,blank,valid,prior):
    n=int(valid.sum(1).max())
    ids=valid.long().argsort(dim=1,descending=True,stable=True)[:,:n]
    keep=valid.gather(1,ids)
    j=winner_law(real.gather(1,ids),blank.gather(1,ids),keep)
    real_m,blank_m=j.sum(2),j.sum(1)
    ind=real_m[:,:,None]*blank_m[:,None,:]
    u=(donors.gather(1,ids[:,:,None].expand(-1,-1,donors.size(2)))+prior).clamp_min(1e-30)
    # Remove the common per-word log preference for numerical conditioning.
    logu=u.log()-prior.clamp_min(1e-30).log()
    def average(marginal,terms):
        return (marginal[:,None,:]@terms).squeeze(1)
    direction=.5*average(real_m-blank_m,logu)
    matched,independent=penalties(logu,j,ind)
    slots=(keep.sum(1)//8).clamp_min(1)[:,None]
    def posterior(gain):
        return (p.clamp_min(1e-30).log()+gain).softmax(1)
    post={'matched_log':posterior(slots*(direction-matched)),
          'independent_log':posterior(slots*(direction-independent)),
          'reverse_log':posterior(slots*(-direction-matched)),
          'direction_only':posterior(slots*direction),'direction_unit':posterior(direction),
          'penalty_only':posterior(-slots*matched),'matched_unit_log':posterior(direction-matched),
          'marginal_ratio':posterior(average(real_m,u).clamp_min(1e-30).log()-average(blank_m,u).clamp_min(1e-30).log())}
    diag={'retained':p.sum(1),'switch_mass':1-j.diagonal(dim1=1,dim2=2).sum(1),
          'separated_switch_mass':1-ind.diagonal(dim1=1,dim2=2).sum(1)}
    return post,diag
