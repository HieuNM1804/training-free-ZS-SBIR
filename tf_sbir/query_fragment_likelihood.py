"""Likelihood of a complete query relative to its own incomplete and spatially permuted observations."""

from collections import defaultdict
from types import SimpleNamespace
import math

import torch
from torch.nn import functional as F

from .cross_diffusion_map import Config,Index


TEMPS=(8.,32.,64.)
POWERS=(.25,.55,.8,1.,1.3,1.7)
ORDERS=("direct","ratio_after_walk","walk_after_ratio")
REFS=("paired","appearance","area_shuffle","grid","geometric","scrambled_whole","scrambled_fragments",
      "empty_token","mean_colour","fixed_white","fixed_neutral","gallery_mean")
GRAPH=Config(neighbors=16,bandwidth=.36,iterations=2,mode="rgb")
BASES=("native","regional")


def key(ref,temp,power,order):
    return f"{ref}/{temp:g}/{power:g}/{order}"


def pure_grid():
    return ["native",*[f"walk/{t:g}" for t in TEMPS],"t4_source","t4_linear",
            *(key(r,t,g,o) for r in REFS for t in TEMPS for g in POWERS for o in ORDERS)]


def required_modes(tags):
    used=set()
    for tag in tags:
        spec=tag.split("|")[-1]
        ref=spec.split("/")[0]
        if ref in ("native","regional","walk","gallery_mean"):continue
        if ref in ("t4_source","t4_linear","appearance","geometric"):used.add("appearance")
        elif ref=="paired":
            used.update(("appearance","permuted_whole_A","permuted_whole_B","permuted_fragments_A","permuted_fragments_B"))
        elif ref=="scrambled_whole":used.update(("permuted_whole_A","permuted_whole_B"))
        elif ref=="scrambled_fragments":used.update(("permuted_fragments_A","permuted_fragments_B"))
        else:used.add(ref)
    return sorted(used)


def centered(x):
    return (x-x.mean(1,keepdim=True))/x.std(1,keepdim=True,unbiased=False).clamp_min(1e-8)


class GalleryWalk:
    def __init__(self,p):
        self.index=Index(SimpleNamespace(p=p),rgb_only=True)
        self.index.prepare(GRAPH)
        self.precision=torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("high")
            self.mass=self.index.path(p.new_ones((len(p),1)),0,GRAPH).clamp_min(1e-30)
        finally:
            torch.set_float32_matmul_precision(self.precision)

    def log(self,logits):
        # Row constants cancel in retrieval, and retaining them also makes the positive-kernel identity testable.
        maxval=logits.amax(1,keepdim=True)
        src=(logits-maxval).clamp_min(-80).exp()
        try:
            torch.set_float32_matmul_precision("high")
            z=self.index.path(src.T.contiguous(),0,GRAPH)/self.mass
        finally:
            torch.set_float32_matmul_precision(self.precision)
        return z.T.contiguous().clamp_min(1e-35).log()+maxval

    def check(self,x):
        ref=self.index.score(GRAPH,(x-x.amax(1,keepdim=True)).exp()).clamp_min(1e-35).log()
        exact=self.log(x)-x.amax(1,keepdim=True)
        err=float((ref-exact).abs().max())
        if err>2e-4:raise ValueError("The effective RGB graph differs from its original implementation")
        return err


def distribution(logits):
    return logits-logits.logsumexp(-1,keepdim=True)


def reference(ref,t,dots):
    logviews=lambda name:distribution(t*dots[name])
    def mixture(name):
        a=logviews(name)
        return a.logsumexp(1)-math.log(a.size(1))
    if ref=="geometric":return logviews("appearance").mean(1)
    if ref=="scrambled_whole":return .5*(mixture("permuted_whole_A")+mixture("permuted_whole_B"))
    if ref=="scrambled_fragments":return .5*(mixture("permuted_fragments_A")+mixture("permuted_fragments_B"))
    if ref=="paired":
        ratios=[mixture("permuted_whole_"+suffix)-mixture("permuted_fragments_"+suffix) for suffix in ("A","B")]
        return mixture("appearance")+.5*(ratios[0]+ratios[1])
    return mixture(ref)


def components(rows):
    recipes=defaultdict(lambda:defaultdict(lambda:defaultdict(set)))
    for tag in rows:
        if tag in ("native","regional","t4_source","t4_linear") or tag.startswith("walk/"):continue
        ref,t,g,order=tag.split("/")
        recipes[float(t)][ref][float(g)].add(order)
    return recipes


def read_scores(photos,full,view_tensors,rows,walk):
    """Yield a complete ranked-score matrix for each unique pure scorer."""
    b=len(full)
    native=full@photos.T
    if "native" in rows:yield "native",native
    req=required_modes(rows)
    dots={}
    for mode in req:
        feat=view_tensors[mode].float()
        dots[mode]=(feat.flatten(0,1)@photos.T).reshape(b,feat.size(1),len(photos))
    if any(tag.startswith("gallery_mean/") for tag in rows):
        mean=F.normalize(photos.mean(0),dim=0)
        dots["gallery_mean"]=(photos@mean)[None,None].expand(b,-1,-1)
    if "t4_source" in rows or "t4_linear" in rows:
        t=32.;fs=t*dots["appearance"]
        # Previous exploratory source-only contrast: exactly reproduce its querywise scaling, not the target side.
        for tag,negative in (("t4_source",fs.logsumexp(1)-math.log(fs.size(1))),("t4_linear",fs.mean(1))):
            if tag in rows:yield tag,native+1.6*native.std(1,keepdim=True,unbiased=False)*centered(t*native-negative)
    recipes=components(rows)
    temperatures=set(recipes)|{float(tag.split("/")[1]) for tag in rows if tag.startswith("walk/")}
    for t in sorted(temperatures):
        logu=distribution(t*native)
        want_walk=any("ratio_after_walk" in orders for powers in recipes[t].values() for orders in powers.values())
        native_key=f"walk/{t:g}"
        walked=walk.log(logu) if want_walk or native_key in rows else None
        if native_key in rows:yield native_key,walked
        for ref,groups in recipes[t].items():
            h=reference(ref,t,dots)
            gh=walk.log(h) if any("ratio_after_walk" in orders for orders in groups.values()) else None
            for power,orders in groups.items():
                direct=logu-power*h
                if "direct" in orders:yield key(ref,t,power,"direct"),direct
                if "ratio_after_walk" in orders:yield key(ref,t,power,"ratio_after_walk"),walked-power*gh
                if "walk_after_ratio" in orders:yield key(ref,t,power,"walk_after_ratio"),walk.log(direct)


def mathematical_checks():
    gen=torch.Generator().manual_seed(7328)
    g=torch.rand(9,9,generator=gen,dtype=torch.float64);g/=g.sum(1,keepdim=True)
    u=torch.rand(3,9,generator=gen,dtype=torch.float64)+.1
    h=torch.rand(3,9,generator=gen,dtype=torch.float64)+.1
    ratio=(u@g.T)/(h@g.T)
    calibrated=g[None]*h[:,None,:];calibrated/=calibrated.sum(-1,keepdim=True)
    identity=float((ratio-(calibrated*(u/h)[:,None]).sum(-1)).abs().max())
    constant=torch.full_like(h,.7)
    err=float(((u@g.T)/(constant@g.T)-(u/constant)@g.T).abs().max())
    if max(identity,err)>1e-12:raise ValueError("Invalid reference-conditioned propagation")
    return {"weighted_walk_identity_error":identity,"constant_reference_error":err}
