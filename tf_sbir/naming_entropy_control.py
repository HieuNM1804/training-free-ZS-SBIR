"""Largest-entropy counterfactual switch table with fixed margins, diagonal, and allowed directions."""

from dataclasses import dataclass

import torch


@dataclass
class EntropyControl:
    levels: torch.Tensor
    from_log: torch.Tensor
    to_log: torch.Tensor

    def mass(self, u, v):
        allowed = self.levels[:, u] > self.levels[:, v]
        return torch.where(allowed, self.from_log[:, u] + self.to_log[:, v], -torch.inf).exp()


def maximum_entropy(model):
    source, target = model.source.double(), model.target.double()
    source /= source.sum(1, keepdim=True)
    target /= target.sum(1, keepdim=True)
    ratios, ids = (source/target).sort(dim=1, stable=True)
    inverse = ids.argsort(dim=1)
    a, b = source.gather(1, ids), target.gather(1, ids)
    zero = torch.zeros_like(a[:, :1])
    capacity = torch.cat((a[:, 1:].flip((1,)).cumsum(1).flip((1,)), zero), 1) + ratios*b.cumsum(1)
    diagonal = a/capacity
    row, col = (a-diagonal).clamp_min(0), (b-diagonal).clamp_min(0)
    new_level = torch.cat((torch.ones_like(zero, dtype=torch.bool), ratios[:,1:] != ratios[:,:-1]), dim=1)
    level = new_level.cumsum(1)-1
    supply = torch.zeros_like(row).scatter_add_(1, level, row)
    demand = torch.zeros_like(col).scatter_add_(1, level, col)
    available = torch.cat((zero, (demand-supply).cumsum(1)[:, :-1]), 1)
    if torch.any(supply > available+1e-10):
        raise ValueError("Prohibited transition would be necessary")
    fraction = torch.where(available > 0, supply/available.clamp_min(1e-300), 0.)
    # Only the last nonempty source exhausts all preceding capacity when
    # the strictly positive categorical margins induce at least three levels.
    log_left = torch.log1p(-fraction.clamp(0, 1 - 1e-15)).cumsum(1)
    before = torch.cat((zero, log_left[:, :-1]), 1)
    outgoing = row.log() - available.gather(1,level).clamp_min(1e-300).log() + before.gather(1,level)
    incoming = col.log() - log_left.gather(1,level)
    return EntropyControl(level.gather(1,inverse).short().contiguous(),
                          outgoing.gather(1,inverse).float().contiguous(),
                          incoming.gather(1,inverse).float().contiguous())


def audit_entropy(model, null, rows):
    v = model.source.size(1)
    u = torch.arange(v, device=model.source.device).repeat_interleave(v)
    w = torch.arange(v, device=model.source.device).repeat(v)
    margin_error, forbidden, divergences, counts = 0., 0., [], []
    for ids in rows.split(8):
        mask = null.levels[ids][:,u] > null.levels[ids][:,w]
        log_ref = (null.from_log[ids][:,u]+null.to_log[ids][:,w]).double()
        ref = torch.where(mask, log_ref, -torch.inf).exp().reshape(len(ids),v,v)
        alt = (model.source[ids][:,u]*model.target[ids][:,w] *
               (model.primitive[ids][:,u]-model.primitive[ids][:,w]).clamp_min(0)).double().reshape(len(ids),v,v)
        same = model.stable[ids]
        error = max(float((ref.sum(1)+same-model.target[ids]).abs().max()),
                    float((ref.sum(2)+same-model.source[ids]).abs().max()))
        margin_error = max(margin_error, error)
        forbidden = max(forbidden, float(ref.masked_fill(alt>0, 0.).max()),
                        float(alt.masked_fill(ref>0, 0.).max()))
        divergence = (alt*(alt.clamp_min(1e-300).log2()-ref.clamp_min(1e-300).log2())).sum((1,2))
        divergences.append(divergence)
        counts.append(alt.sum((1,2)))
    if margin_error > 2e-6 or forbidden > 2e-6:
        raise ValueError(f"Entropy control mismatches required invariants {margin_error=} {forbidden=}")
    kl, probability = torch.cat(divergences), torch.cat(counts)
    return {"images":len(rows),"max_margin_error":margin_error,"largest_mass_on_different_support":forbidden,
            "kl_bits_per_image":kl.mean().item(),"kl_bits_conditional_on_a_switch":(kl.sum()/probability.sum()).item()}


def fixture(device):
    from .counterfactual_naming import law
    a = torch.tensor([[.33,.21,.13,.15,.11,.07]], device=device)
    b = torch.tensor([[.09,.11,.08,.23,.28,.21]], device=device)
    model = law(a,b)
    null = maximum_entropy(model)
    u = torch.arange(6,device=device).repeat_interleave(6)
    v = torch.arange(6,device=device).repeat(6)
    measured = null.mass(u,v).reshape(6,6).double()
    # Independent reference by iterative proportional fitting on the permitted cells.
    support = ((a/b).T > a/b)
    matrix = support.double()
    source = (model.source-model.stable).double().clamp_min(0)[0]
    target = (model.target-model.stable).double().clamp_min(0)[0]
    for _ in range(200):
        matrix *= (source/matrix.sum(1).clamp_min(1e-300))[:,None]
        matrix *= (target/matrix.sum(0).clamp_min(1e-300))[None,:]
    error = float((matrix-measured).abs().max())
    if error > 2e-6:
        raise ValueError("Closed form differs from independent entropy fit")
    return {"margin_and_support":audit_entropy(model,null,torch.tensor([0],device=device)),
            "max_error_vs_iterative_fit":error}
