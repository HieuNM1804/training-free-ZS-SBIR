"""Fuse appearance and phase agreement through the identities of an unlabeled gallery."""

from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.nn import functional as F

from .contextual_rank import jaccard_seed, personalized_walk
from .gallery_semantic_balance import compact_union
from .global_methods import centered
from .paired_neighborhood_scale import scale_index
from .reflection_transfer import file_tensor
from .reranking import standardize
from .transfer_benchmark import write_result
from .union_density import candidate_union, density_rows, grow_density


@dataclass(frozen=True)
class Config:
    k: int = 192
    region: float = 0.35
    polarity: str = "both"
    iso: bool = True
    shuffled: bool = False


def name(c, mode):
    kind = "iso" if c.iso else "native"
    pairing = "-shuffled" if c.shuffled else ""
    return f"phase-k{c.k}-r{c.region:g}-{c.polarity}-{kind}{pairing}-{mode}"


def grid():
    base = Config()
    common = ("direct", "cross", "adaptive", "cross25", "cross75")
    groups = [
        (base, (*common, "rgb", "phase", "peak", "rgb-to-phase", "phase-to-rgb")),
        (Config(k=112), common),
        (Config(k=320), common),
        (Config(region=0.0), common),
        (Config(region=0.7), common),
        (Config(polarity="negative"), common),
        (Config(polarity="positive"), common),
        (Config(iso=False), common),
        (Config(shuffled=True), ("adaptive",)),
    ]
    return {name(c, m): (c, m) for c, modes in groups for m in modes}


def phase_features(native, views, transform, c):
    rgb = torch.cat((native[:, None], views[:, :4].float()), dim=1)
    # Two contrast response curves with dark/light renderings and both orientations.
    phases = views[:, 4:].float().unflatten(1, (2, 2, 2))
    if c.polarity == "both":
        p = phases.mean(2)
    else:
        p = phases[:, :, 0 if c.polarity == "negative" else 1]
    p = p.flatten(1, 2)
    if c.iso:
        rgb = rgb @ transform
        p = p @ transform
    return F.normalize(rgb, dim=-1), F.normalize(p, dim=-1)


def vote(scores, bank):
    vs = torch.stack([jaccard_seed(s, bank) for s in scores.unbind(1)], dim=1)
    mass = vs.sum(-1)
    w = (2 * mass.clamp_min(1e-8).log()).softmax(1)
    return (w[:, :, None] * vs).sum(1), (w * mass).sum(1), (standardize(scores) * w[:, :, None]).sum(1)


def fusion(mode, confidence):
    b = len(confidence)
    w = confidence.new_zeros((b, 4))
    singles = {"rgb": 0, "phase": 1, "rgb-to-phase": 2, "phase-to-rgb": 3}
    if mode in singles:
        w[:, singles[mode]] = 1
    elif mode == "adaptive":
        w = (confidence.clamp_min(1e-8).log() * 2).softmax(1)
    elif mode == "peak":
        w.scatter_(1, confidence.argmax(1, keepdim=True), 1.0)
    else:
        fraction = {"direct": 0.0, "cross": 1.0, "cross25": 0.25, "cross75": 0.75}[mode]
        w[:, :2] = (confidence[:, :2].clamp_min(1e-8).log() * 2).softmax(1) * (1 - fraction)
        w[:, 2:] = (confidence[:, 2:].clamp_min(1e-8).log() * 2).softmax(1) * fraction
    return w


def group_scores(qrgb, qphase, means, index, modes):
    gallery, structure, banks, graph, walk = index
    rgb_mean, phase_mean = means
    parts = [
        (qrgb, rgb_mean, gallery, banks[0]),
        (qphase, phase_mean, structure, banks[1]),
        (qrgb, phase_mean, structure, banks[1]),
        (qphase, rgb_mean, gallery, banks[0]),
    ]
    seeds = []
    confidence = []
    priors = []
    for queries, mean, targets, bank in parts:
        scores = centered(queries, mean) @ targets.T
        v, conf, pr = vote(scores, bank)
        seeds.append(v)
        confidence.append(conf)
        priors.append(pr)
    conf = torch.stack(confidence, 1)
    priors = torch.stack(priors, 1)
    seeds = torch.stack(seeds, 1)
    candidates = compact_union(*candidate_union(priors.unbind(1), min(1280, len(gallery))))
    # A shared tail permits comparison of top-rank fusion rules with identical low-rank scores.
    tail_weight = fusion("adaptive", conf)
    prior = (tail_weight[:, :, None] * priors).sum(1)
    seed = (tail_weight[:, :, None] * seeds).sum(1)
    total = seed.sum(1, keepdim=True)
    fallback = torch.zeros_like(seed).scatter_(1, prior.argmax(1, keepdim=True), 1.0)
    prob = torch.where(total > 0, seed / total.clamp_min(1e-8), fallback)
    stationary = personalized_walk(*walk, prob, 0.85)
    tail = standardize(prior) + 1.2 * standardize((stationary * len(gallery)).log1p())
    heads = {}
    for mode in modes:
        w = fusion(mode, conf)[:, :, None]
        voting = (w * seeds).sum(1)
        initial = (w * priors).sum(1)
        heads[mode] = grow_density(64 * voting, initial, *candidates, graph, min(256, len(gallery)))
    return tail, heads


def evaluate(args, split, raw, photo, query_views, photo_views, transform, catalog, labels, meta):
    groups = {}
    for n, (c, m) in catalog.items():
        groups.setdefault(c, []).append((n, m))
    ql, pl = labels
    n, g = len(ql), len(pl)
    for config, modes in groups.items():
        rgb, ph = phase_features(photo, photo_views, transform, config)
        qrgb, qphase = phase_features(raw, query_views, transform, config)
        target = F.normalize(rgb[:, 0].lerp(F.normalize(rgb[:, 1:].mean(1), dim=-1), config.region), dim=-1)
        phase = F.normalize(ph.mean(1), dim=-1)
        if config.shuffled:
            rng = torch.Generator(device=phase.device).manual_seed(145)
            phase = phase[torch.randperm(len(phase), device=phase.device, generator=rng)]
        means = target.mean(0), phase.mean(0)
        index = scale_index(photo, centered(target, means[0]), centered(phase, means[1]), min(config.k, g - 1))
        depth = min(256, g)
        with TemporaryDirectory(dir=args.output) as directory:
            folder = Path(directory)
            tail = file_tensor(folder, "tail", (n, g))
            heads = {
                m: torch.from_file(
                    str(folder / (m + ".i32")), shared=True, size=n * depth, dtype=torch.int32
                ).view(n, depth)
                for _, m in modes
            }
            for start in range(0, n, args.score_batch):
                sl = slice(start, start + args.score_batch)
                t, h = group_scores(qrgb[sl], qphase[sl], means, index, heads)
                tail[sl] = t.cpu()
                for m in heads:
                    heads[m][sl] = h[m].to("cpu", dtype=torch.int32)
            for method, mode in modes:

                def rows():
                    for a, b in zip(tail.split(256), heads[mode].split(256), strict=True):
                        yield from density_rows(a.cuda(), b.cuda().long(), len(a), None)

                write_result(args, split, method, rows(), ql, pl, meta, asdict(config) | {"fusion": mode})
