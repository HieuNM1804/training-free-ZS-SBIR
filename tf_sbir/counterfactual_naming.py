"""Name substitutions that can be reproduced by changing only the CLIP domain prompt."""

from dataclasses import dataclass

import numpy as np
from numba import njit
import torch


@dataclass
class NamingLaw:
    source: torch.Tensor
    target: torch.Tensor
    primitive: torch.Tensor
    stable: torch.Tensor
    source_begin: torch.Tensor
    source_end: torch.Tensor
    target_begin: torch.Tensor
    target_end: torch.Tensor

    def switched(self, u, v):
        return self.source[:, u] * self.target[:, v] * (self.primitive[:, u] - self.primitive[:, v]).clamp_min(0)

    def ordered_control(self, u, v):
        # Quantile transport between residual margins, in likelihood-ratio order.
        # The stable probability is outside this transport plan.
        left = torch.maximum(self.source_begin[:, u], self.target_begin[:, v])
        right = torch.minimum(self.source_end[:, u], self.target_end[:, v])
        return (right-left).clamp_min_(0)


def law(source, target):
    a, b = source.double(), target.double()
    if torch.any(a <= 0) or torch.any(b <= 0):
        raise ValueError("Domain posteriors must have full support")
    a, b = a/a.sum(1, keepdim=True), b/b.sum(1, keepdim=True)
    ratio, order = (a/b).sort(1)
    aa, bb = a.gather(1, order), b.gather(1, order)
    zeros = torch.zeros_like(aa[:, :1])
    tail_a = torch.cat((aa[:, 1:].flip((1,)).cumsum(1).flip((1,)), zeros), 1)
    capacity = tail_a + ratio * bb.cumsum(1)
    segments = ratio.diff(dim=1) / (capacity[:, 1:] * capacity[:, :-1])
    primitive = torch.cat((zeros, segments.cumsum(1)), dim=1)
    stable = aa/capacity
    origin = (aa-stable).clamp_min(0).cumsum(1)
    destination = (bb-stable).clamp_min(0).cumsum(1)
    start_a = torch.cat((zeros, origin[:, :-1]), 1)
    start_b = torch.cat((zeros, destination[:, :-1]), 1)
    if torch.any(origin > start_b + 1e-10):
        raise ValueError("Residual masses violate directed change feasibility")
    reverse = order.argsort(dim=1)
    fields = [x.gather(1, reverse).float().contiguous() for x in
              (primitive, stable, start_a, origin, start_b, destination)]
    return NamingLaw(a.float(), b.float(), *fields)


def matrices(x, y):
    """Exact exhaustive ranking scores by streaming arbitrary changed-name entries."""
    v = x.source.size(1)
    literal = x.source @ y.target.T
    changed, control = torch.zeros_like(literal), torch.zeros_like(literal)
    pairs = torch.triu_indices(v, v, offset=1, device=literal.device)
    for tile in pairs.split(2048, 1):
        i, j = tile
        for value, method in ((changed, "switched"), (control, "ordered_control")):
            left, right = getattr(x, method), getattr(y, method)
            a = torch.cat((left(i,j), left(j,i)), 1)
            b = torch.cat((right(i,j), right(j,i)), 1)
            value.addmm_(a, b.T)
            del a, b
    return literal, changed, control


def finite_agreement(x, y):
    """Exact agreement matrix for tiny batches only, used to check the efficient solver."""
    v = x.source.size(1)
    i = torch.arange(v, device=x.source.device).repeat_interleave(v)
    j = torch.arange(v, device=x.source.device).repeat(v)
    terms = x.switched(i,j) @ y.switched(i,j).T
    return x.source @ y.target.T + terms


@njit
def dominance(a, b, hx, ay, by, hy, order, ranks):
    # One point for each public name. A transition u -> v can contribute only
    # when v precedes u on both images' likelihood-ratio orders.
    v = len(order)
    tree = np.zeros((v+1, 4), np.float64)
    result = 0.
    for u in order:
        stop = ranks[u]
        total = np.zeros(4, np.float64)
        while stop > 0:
            total += tree[stop]
            stop -= stop & -stop
        gain = hx[u]*hy[u]*total[0] - hy[u]*total[1] - hx[u]*total[2] + total[3]
        result += a[u]*ay[u]*gain
        index = ranks[u]+1
        weight = b[u]*by[u]
        while index <= v:
            tree[index, 0] += weight
            tree[index, 1] += weight*hx[u]
            tree[index, 2] += weight*hy[u]
            tree[index, 3] += weight*hx[u]*hy[u]
            index += index & -index
    return result


def audit_law(model, count=32):
    # Compare whole dense tables with their intended row/column probabilities,
    # stable-name diagonal, and shared set of forbidden directions.
    v = model.source.size(1)
    i = torch.arange(v, device=model.source.device).repeat_interleave(v)
    j = torch.arange(v, device=model.source.device).repeat(v)
    small = NamingLaw(*(a[:count] for a in vars(model).values()))
    true, placebo = (f(i,j).reshape(-1, v, v) for f in (small.switched, small.ordered_control))
    errors = []
    for table in (true, placebo):
        errors.extend((float((table.sum(2)+small.stable-small.source).abs().max()),
                       float((table.sum(1)+small.stable-small.target).abs().max()),
                       float(table.diagonal(dim1=1, dim2=2).abs().max())))
    wrong_way = float(placebo.masked_fill(true > 0, 0.).max())
    if max(*errors, wrong_way) > 2e-6:
        raise ValueError(f"Joint laws are inconsistent with their controlled margins {errors=} {wrong_way=}")
    return {"images_checked": len(small.source), "maximum_margin_or_diagonal_error": max(errors),
            "largest_placebo_mass_outside_model_support": wrong_way}


def fixture(device):
    a = torch.tensor([[.48, .25, .20, .06, .01], [.20, .53, .12, .09, .06]], device=device)
    b = torch.tensor([[.13, .18, .19, .32, .18], [.08, .17, .14, .37, .24]], device=device)
    x = law(a, b)
    checked = audit_law(x)
    literal, changed, placebo = matrices(x, x)
    torch.testing.assert_close(literal+changed, finite_agreement(x, x))
    data = [t.double().cpu().numpy() for t in (x.source, x.target, x.primitive)]
    aa, bb, h = data
    exact = dominance(aa[0], bb[0], h[0], aa[1], bb[1], h[1], np.argsort(h[0]), np.argsort(np.argsort(h[1])))
    if abs(exact - float(changed[0,1])) > 2e-7 or float((changed-placebo).abs().max()) < 1e-5:
        raise ValueError("Fixture does not isolate the pairing of changed labels")
    same = law(a, a)
    torch.testing.assert_close(same.stable, a)
    if same.primitive.abs().max() > 1e-6:
        raise ValueError("Identical prompts should not change the name")
    # Inspect the two winners under the same exponential race.
    gen = torch.Generator(device=device).manual_seed(42)
    uniforms = torch.rand((160000, a.size(1)), device=device, generator=gen, dtype=torch.float64)
    clocks = -uniforms.clamp_min(1e-300).log()
    first = (clocks/a[0]).argmin(1)
    second = (clocks/b[0]).argmin(1)
    counts = torch.bincount(first*a.size(1)+second, minlength=a.size(1)**2).double()/len(first)
    ii = torch.arange(a.size(1), device=device).repeat_interleave(a.size(1))
    jj = torch.arange(a.size(1), device=device).repeat(a.size(1))
    expected = x.switched(ii,jj)[0].reshape(a.size(1),a.size(1)) + torch.diag(x.stable[0])
    mc_error = float((counts-expected.flatten()).abs().max())
    if mc_error > .006:
        raise ValueError("Analytic substitution law disagrees with shared-noise simulation")
    return {"joint_checks": checked, "sampled_pairs": len(first), "max_mc_error": mc_error,
            "changed_paths_01": exact, "same_margin_directional_placebo_01": float(placebo[0,1])}
