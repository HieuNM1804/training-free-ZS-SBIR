"""Equal-angle check of complementary residual directions in CLIP's unit sphere."""
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .gallery_context_reversals import coordinates
from .lexical_observability import low_variance_part
from .standardized_context_audit import audit_inputs


@torch.inference_mode()
def run():
    w = Path('/content/sbir')
    qs, _, enc, _, q, _, flip, src, _, scale = audit_inputs(w, 768)
    report = {}
    for split, (qi, _) in qs.items():
        rows = torch.randperm(len(qi), generator=torch.Generator().manual_seed(87243))[:768].sort().values
        vals = []
        for ix in coordinates(enc, qi[rows]).split(64):
            x = F.normalize((q[ix].cuda() + flip[ix].float().cuda())/2)
            weak, _ = low_variance_part(x, (scale*(x@src.T)).softmax(1), src)
            def unit_tangent(v):
                return F.normalize(v-(x*v).sum(1, keepdim=True)*x)
            def kl(t):
                a = (scale*((math.cos(.05)*x+math.sin(.05)*t)@src.T)).log_softmax(1)
                b = (scale*((math.cos(.05)*x-math.sin(.05)*t)@src.T)).log_softmax(1)
                return ((a.exp()-b.exp())*(a-b)).sum(1)/2
            lo, hi = unit_tangent(weak), unit_tangent(x-weak)
            d1, d2 = kl(lo), kl(hi)
            vals.append(torch.stack(((lo*hi).sum(1), d1, d2, (d1-d2).abs()),1).cpu())
        arr = torch.cat(vals)
        report[split] = {'queries': len(arr), 'angle_radians': .05, 'mean_cosine_tangents': float(arr[:,0].mean()),
                         'mean_symmetric_kl_weak': float(arr[:,1].mean()), 'mean_symmetric_kl_complement': float(arr[:,2].mean()),
                         'maximum_absolute_kl_difference': float(arr[:,3].max())}
    (w/'results/lexical-observability-residual/equal-angle.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    run()
