"""Exact top-K retrieval with compact per-image laws and two residual-mass bounds."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit
def _switch_sums(
    x_a, x_b, x_h, y_a, y_b, y_h, candidate, result, v: tl.constexpr,
    tile: tl.constexpr, all_words: tl.constexpr,
):
    row = tl.program_id(1)
    image = tl.load(candidate + row)
    u = tl.program_id(0)*tile + tl.arange(0, tile)
    w = tl.arange(0, all_words)
    ax = tl.load(x_a + u, u < v, 0.)
    ay = tl.load(y_a + image*v + u, u < v, 0.)
    bx = tl.load(x_b + w, w < v, 0.)
    by = tl.load(y_b + image*v + w, w < v, 0.)
    xu = tl.load(x_h + u, u < v, 0.)
    xw = tl.load(x_h + w, w < v, 0.)
    yu = tl.load(y_h + image*v + u, u < v, 0.)
    yw = tl.load(y_h + image*v + w, w < v, 0.)
    left = tl.maximum(xu[:,None]-xw[None,:], 0.)
    right = tl.maximum(yu[:,None]-yw[None,:], 0.)
    # All terms are nonnegative. Match the factorization used for the reference.
    tx = (left * ax[:,None])*bx[None,:]
    ty = (right * ay[:,None])*by[None,:]
    total = tl.sum(tx*ty, 1)
    tl.store(result + row*v + u, total, u < v)


def switch_scores(x, y, row, candidates):
    v = y.source.size(1)
    block = 8
    result = torch.empty((len(candidates), v),device=y.source.device)
    _switch_sums[(triton.cdiv(v,block),len(candidates))](
        x.source[row],x.target[row],x.primitive[row],y.source,y.target,y.primitive,
        candidates,result,v,block,triton.next_power_of_2(v),num_warps=4,
    )
    return result.sum(1)


@dataclass
class SwitchParameters:
    source: torch.Tensor
    target: torch.Tensor
    primitive: torch.Tensor


class NamingIndex:
    def __init__(self, gallery_law):
        self.law = SwitchParameters(gallery_law.source, gallery_law.target, gallery_law.primitive)
        self.source_switched = (gallery_law.source-gallery_law.stable).clamp_min(0)
        self.target_switched = (gallery_law.target-gallery_law.stable).clamp_min(0)

    def topk(self, query_law, row, k):
        if not 0 < k <= len(self.law.source):
            raise ValueError("k must be between one and the gallery size")
        q = query_law
        xa, xb = q.source[row], q.target[row]
        literal = self.law.target @ xa
        cap = torch.minimum(self.source_switched @ (xa-q.stable[row]).clamp_min(0),
                            self.target_switched @ (xb-q.stable[row]).clamp_min(0))
        # The analytic lower/upper bounds apply to exact nonnegative sums.
        # Round outward before screening in float32, including differences of
        # nearly equal binary32 marginal and diagonal probabilities.
        gamma = (xa.numel()+4)*torch.finfo(torch.float32).eps
        upper = (literal+cap)*(1+2*gamma)+2e-6
        cut = literal.topk(k).values[-1]*(1-2*gamma)-2e-6
        selected = torch.where(upper >= cut)[0]
        exact = literal[selected] + switch_scores(q,self.law,row,selected)
        order = exact.argsort(descending=True,stable=True)[:k]
        return selected[order], selected
