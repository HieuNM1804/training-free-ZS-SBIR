"""Likelihood of the orientation of a same-slate real-query versus blank-query choice."""
import itertools

import torch
import torch.nn.functional as F


def winner_law(real, blank, valid, draws=8):
    """Exact joint distribution of winners for the same iid uniform sample.

    Break ties in the same candidate order in both readouts. Rows are real
    winners, columns are blank winners. All invalid nominated slots have zero mass.
    """
    def ordering(scores):
        ranks = scores.masked_fill(~valid, -torch.inf).argsort(dim=1, stable=True).argsort(dim=1)
        return (ranks[:, :, None] >= ranks[:, None, :]) & valid[:, None, :] & valid[:, :, None]
    real_order = ordering(real)
    blank_order = ordering(blank)
    count = real_order.float() @ blank_order.float().transpose(1, 2)
    count = count.double()
    total = valid.sum(1)[:, None, None]
    at_most = (count / total).pow(draws)
    one_out = ((count-1).clamp_min(0) / total).pow(draws)
    two_out = ((count-2).clamp_min(0) / total).pow(draws)
    compatible = real_order & blank_order.transpose(1, 2)
    diag = torch.eye(real.size(1), dtype=torch.bool, device=real.device)[None]
    joint = torch.where(diag, at_most-one_out, at_most-2*one_out+two_out)*compatible
    joint = joint.clamp_min(0)
    expected = ((real_order.sum(2)/total.squeeze(2)).double().pow(draws)
                - ((real_order.sum(2)-1).clamp_min(0)/total.squeeze(2)).double().pow(draws))
    discrepancy = (joint.sum(2)-expected).abs().max()
    if discrepancy>2e-6 or (joint.sum((1,2))-1).abs().max()>2e-8:
        raise ValueError('Invalid maximum choice law')
    return joint.float()


def check_choice_law():
    v = torch.tensor([[.5, 2., 3., 1., 3.]], device='cuda')
    b = torch.tensor([[3., 1., 2., .5, 1.]], device=v.device)
    valid = torch.tensor([[True, True, True, False, True]], device=v.device)
    j = winner_law(v,b,valid,draws=3)[0]
    truth = torch.zeros_like(j)
    eligible = valid[0].nonzero().flatten().tolist()
    for draw in itertools.product(eligible,repeat=3):
        first = max(draw,key=lambda i:(float(v[0,i]),i))
        second = max(draw,key=lambda i:(float(b[0,i]),i))
        truth[first,second] += 1/len(eligible)**3
    if not torch.allclose(j,truth,atol=1e-7):raise ValueError('Enumeration differs')
    if not torch.allclose(winner_law(v,v,valid).diagonal(dim1=1,dim2=2).sum(),v.new_tensor(1.)):
        raise ValueError('Matching orders must have the same winner')
    return float((j-truth).abs().max())


def collapsed_source(p,bank,ids,prior):
    noun_ids = ids[:,None].expand(-1,bank.size(1),-1)
    donor = bank.gather(2,noun_ids)
    first = p.gather(1,ids)
    background = prior.expand_as(p).gather(1,ids)
    def remainder(values):
        return torch.cat([values,(1-values.sum(-1,keepdim=True)).clamp_min(0)],-1)
    omitted = p.scatter(1,ids,0.).sum(1,keepdim=True)
    return torch.cat([first,omitted],1), (remainder(donor)+remainder(background)[:,None]).clamp_min(1e-30)


def infer(p, bank, real, blank, valid, prior, ids):
    # Compress out the other gallery fold to reduce the number of donor pairs.
    n = int(valid.sum(1).max())
    order = valid.long().argsort(dim=1,descending=True,stable=True)[:,:n]
    keep = valid.gather(1,order)
    bank = bank.gather(1,order[:,:,None].expand(-1,-1,bank.size(2)))
    joint = winner_law(real.gather(1,order),blank.gather(1,order),keep)
    original, utility = collapsed_source(p,bank,ids,prior)
    real_m,blank_m = joint.sum(2),joint.sum(1)
    independent = real_m[:,:,None]*blank_m[:,None,:]
    logu = utility.log()
    shape = original.shape
    logs = {name:torch.zeros(shape,device=p.device) for name in ('matched','independent','reverse','one','independent_one')}
    for start in range(0,n,8):
        sl = slice(start,min(n,start+8))
        preference = logu[:,sl,None,:]-logu[:,None,:,:]
        odds = F.logsigmoid(preference)+0.6931471805599453
        one = odds.exp()
        w = joint[:,sl].flatten(1,2)[:,None,:]
        alt = independent[:,sl].flatten(1,2)[:,None,:]
        rev = joint.transpose(1,2)[:,sl].flatten(1,2)[:,None,:]
        def average(weights,values):
            return (weights@values.flatten(1,2)).squeeze(1)
        for name,weights,data in [('matched',w,odds),('independent',alt,odds),('reverse',rev,odds),
                                  ('one',w,one),('independent_one',alt,one)]:
            logs[name] += average(weights,data)
    slots = (keep.sum(1)//8).clamp_min(1)[:,None]
    def mu(marginal):
        return (marginal[:,None]@utility).squeeze(1).clamp_min(1e-30)
    raw = {'matched_one':logs['one'].clamp_min(1e-30).log(),'independent_one':logs['independent_one'].clamp_min(1e-30).log(),
           'matched_log':slots*logs['matched'],'independent_log':slots*logs['independent'],'reverse_log':slots*logs['reverse'],
           'matched_unit_log':logs['matched'],'marginal_ratio':mu(real_m).log()-mu(blank_m).log()}
    def expand(log_bf):
        weighted = (original.clamp_min(1e-30).log()+log_bf).softmax(1)
        # Preserve relative mass of the omitted words within the explicit other category.
        result = p*(weighted[:,-1:]/original[:,-1:].clamp_min(1e-30))
        result.scatter_(1,ids,weighted[:,:-1])
        return F.normalize(result,p=1,dim=1)
    post = {k:expand(v) for k,v in raw.items()}
    meta = {'joint':joint,'utility':utility,'retained':original[:,:-1].sum(1),
            'switch_mass':1-joint.diagonal(dim1=1,dim2=2).sum(1),
            'separated_switch_mass':1-independent.diagonal(dim1=1,dim2=2).sum(1)}
    return post,meta


def held_logscore(meta,post,ids):
    """Predict which candidate the real query chose, using the other fold's words."""
    joint, utility = meta['joint'],meta['utility']
    logu = utility.log()
    reduced = {}
    for name,p in post.items():
        q = p.gather(1,ids)
        reduced[name] = torch.cat([q,(1-q.sum(1,keepdim=True)).clamp_min(0)],1)
    mix = torch.stack(list(reduced.values()),2)
    total = torch.zeros((len(joint),len(reduced)),device=joint.device)
    for start in range(0,joint.size(1),8):
        sl = slice(start,min(start+8,joint.size(1)))
        prob = (logu[:,sl,None,:]-logu[:,None,:,:]).sigmoid()
        predictive = (prob.flatten(1,2)@mix).clamp_min(1e-30)
        total += (joint[:,sl].flatten(1,2)[:,:,None]*(2*predictive).log()).sum(1)
    return {name:total[:,i] for i,name in enumerate(reduced)}
