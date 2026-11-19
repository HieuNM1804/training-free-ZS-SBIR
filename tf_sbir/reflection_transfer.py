"""Compare independent view votes with paired votes across a complete set of horizontal reflections."""

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NamedTuple

from PIL import Image, ImageOps
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor, resize

from .contextual_rank import jaccard_seed
from .contour_query_expansion import pagerank_tail
from .global_methods import centered
from .mta import load_view_features
from .multiview_anchor_transfer import evidence_weights, predict_views
from .reranking import standardize
from .sketch_canvas_fusion import CanvasSketches, canvas_images
from .transfer_reconstruction import anchor_errors, gallery_index, nearest_anchors
from .union_density import candidate_union, density_rows, grow_density
from .view_hypotheses import ink_box_image, view_concepts


class ReflectedSketches(CanvasSketches):
    def __getitem__(self, index):
        with Image.open(self.paths[index]) as source:
            full = source.convert("RGB")
        medium, small, _ = canvas_images(ink_box_image(full))
        frames = (resize(full, [224, 224]), medium, small)
        return torch.stack([pil_to_tensor(ImageOps.mirror(frame)) for frame in frames])


def load_reflected_views(args, paths):
    return load_view_features(args, ReflectedSketches(paths), 3, "sketch", "reflected-full-medium-small-v1")["features"]


class PairSetting(NamedTuple):
    paired: bool
    power: float
    risk_power: float
    agreement_power: float = 0.
    intersection: float = 0.
    shuffle: bool = False
    lexical_fraction: float = .5
    separate_risk: bool = False
    risk_component: str = "both"
    shuffle_risk: bool = False


SETTINGS = {
    "flat": PairSetting(False, 0., 0.), "flat-risk2": PairSetting(False, 0., 2.),
    "flat-sharp": PairSetting(False, 4., 0.), "flat-sharp-risk4": PairSetting(False, 4., 4.),
    "pairs-risk2": PairSetting(True, 0., 2.), "pairs-sharp": PairSetting(True, 4., 2.),
    "pairs-agree1": PairSetting(True, 4., 2., 1.), "pairs-agree2": PairSetting(True, 4., 2., 2.),
    "pairs-intersect025": PairSetting(True, 4., 2., 1., .25),
    "pairs-intersect05": PairSetting(True, 4., 2., 1., .5),
    "pairs-balanced": PairSetting(True, 2., 2., .5, .25),
    "pairs-shuffled": PairSetting(True, 4., 2., 1., .25, True),
    "flat-visual-risk-l035": PairSetting(False, 0., 2., lexical_fraction=.35, separate_risk=True),
    "pairs-visual-risk-l025": PairSetting(True, 0., 2., 1., .25, lexical_fraction=.25, separate_risk=True),
    "pairs-visual-risk-l035": PairSetting(True, 0., 2., 1., .25, lexical_fraction=.35, separate_risk=True),
    "pairs-visual-risk-l05": PairSetting(True, 0., 2., 1., .25, separate_risk=True),
    "pairs-sharp-visual-l035": PairSetting(True, 4., 2., 1., .25, lexical_fraction=.35, separate_risk=True),
    "pairs-sharp-visual-l05": PairSetting(True, 4., 2., 1., .25, separate_risk=True),
    "pairs-visual-shuffled-l035": PairSetting(True, 0., 2., 1., .25, True, .35, True),
}


def matched_controls():
    base = SETTINGS["flat-visual-risk-l035"]
    sharp = SETTINGS["flat-sharp-risk4"]
    return {
        "reference": base, "none": base._replace(risk_power=0.),
        "source": base._replace(risk_component="source"), "target": base._replace(risk_component="target"),
        "shuffled": base._replace(shuffle_risk=True),
        "risk1": base._replace(risk_power=1.), "risk4": base._replace(risk_power=4.),
        "l02": base._replace(lexical_fraction=.2), "l0275": base._replace(lexical_fraction=.275),
        "l0425": base._replace(lexical_fraction=.425),
        "p1-l0275": base._replace(power=1., lexical_fraction=.275), "p1-l035": base._replace(power=1.),
        "sharp": sharp, "sharp-source": sharp._replace(risk_component="source"),
        "sharp-target": sharp._replace(risk_component="target"), "sharp-shuffled": sharp._replace(shuffle_risk=True),
    }


def select_view_risk(components, config):
    if config.risk_component == "source":
        risk = components[..., 0]
    elif config.risk_component == "target":
        risk = components[..., 1]
    else:
        risk = components[..., 0] + components[..., 1]
    if config.shuffle_risk:
        left, right = risk.chunk(2, dim=1)
        risk = torch.cat((left.roll(1, dims=1), right.roll(1, dims=1)), dim=1)
    return risk


def reflection_seed(votes, risk, config):
    prior = votes.new_tensor([.33, .22, .225, .225])
    left, right = votes.chunk(2, dim=1)
    a, b = risk.chunk(2, dim=1)
    if not config.paired:
        weights = evidence_weights(votes.sum(dim=-1), risk, prior.repeat(2) * .5, config.power, config.risk_power)
        wl, wr = weights.chunk(2, dim=1)
        return (left * wl[:, :, None] + right * wr[:, :, None]).sum(dim=1)
    if config.shuffle:
        right, b = right.roll(1, dims=1), b.roll(1, dims=1)
    low = torch.minimum(left, right)
    reliability = low.sum(dim=-1) / torch.maximum(left, right).sum(dim=-1).clamp_min(1e-8)
    average = (left + right) * .5
    weights = evidence_weights(average.sum(dim=-1), (a + b) * .5, prior, config.power, config.risk_power)
    if config.agreement_power:
        weights = weights * reliability.clamp_min(1e-6).pow(config.agreement_power)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-18)
    support = average.lerp(low, config.intersection)
    return (support * weights[:, :, None]).sum(dim=1)


def reflection_block(vectors, risk, concepts, index, config):
    gallery, vocabulary, banks, graph, walk = index
    visual = vectors @ gallery.T
    lexical = concepts @ vocabulary.T
    prior = vectors.new_tensor([.33, .22, .225, .225])[None, :, None]

    def symmetric_scores(scores):
        first, second = standardize(scores).chunk(2, dim=1)
        return ((first + second) * (.5 * prior)).sum(dim=1)

    appearance, language = symmetric_scores(visual), symmetric_scores(lexical)
    tied = standardize(appearance) + standardize(language)
    candidates = candidate_union((*visual.unbind(1), *lexical.unbind(1)), 1280)
    if config.separate_risk:
        shapes = torch.stack([jaccard_seed(v, banks[0]) for v in visual.unbind(1)], dim=1)
        words = torch.stack([jaccard_seed(v, banks[1]) for v in lexical.unbind(1)], dim=1)
        appearance_seed = reflection_seed(shapes, risk, config)
        text_seed = reflection_seed(words, risk, config._replace(risk_power=0.))
        seed = 2. * appearance_seed.lerp(text_seed, config.lexical_fraction)
    else:
        votes = torch.stack([jaccard_seed(a, banks[0]) + jaccard_seed(b, banks[1])
                             for a, b in zip(visual.unbind(1), lexical.unbind(1), strict=True)], dim=1)
        seed = reflection_seed(votes, risk, config)
    prefix = grow_density(32 * seed, tied, *candidates, graph, 256)
    tail = pagerank_tail(appearance, language, tied, banks, *walk, len(vectors))
    return tail, prefix


def file_tensor(folder, name, shape):
    path = str(folder / (name + ".f32"))
    return torch.from_file(path, shared=True, size=math.prod(shape), dtype=torch.float32).view(shape)


def evaluate_views(args, folder, split, query_ids, q_labels, p_labels, data, index, source, target,
                   source_center, transform, sketch_words, photo_words, metadata, record, five_rows):
    original, extras, reflected = data
    n, d = len(query_ids), target.size(1)
    vectors = file_tensor(folder, "vectors", (n, 8, d))
    risks = file_tensor(folder, "risks", (n, 8, 2))
    concepts = file_tensor(folder, "concepts", (n, 8, photo_words.size(1)))
    errors = anchor_errors(source, target)
    device = source.device
    for start in range(0, n, 64):
        ids = query_ids[start:start + 64]
        cropped = F.normalize(extras[ids].to(device).float(), dim=-1)
        mirror = F.normalize(reflected[ids].to(device).float(), dim=-1)
        raw = torch.cat((cropped[:, :1], original[ids, None].to(device), cropped[:, 1:3], cropped[:, 3:4], mirror), dim=1)
        q = centered(F.normalize(raw @ transform, dim=-1), source_center)
        neighbors = nearest_anchors(q.flatten(0, 1), source, 64)
        mapped, uncertainty = predict_views(q, source, target, errors, neighbors, count=64, penalty=1., carry=1., risk_components=True)
        loc = slice(start, start + len(ids))
        vectors[loc], risks[loc] = mapped.cpu(), uncertainty.cpu()
        concepts[loc] = view_concepts(raw, sketch_words, photo_words).cpu()
    del raw, mapped, q, neighbors, cropped, mirror, uncertainty, errors
    torch.cuda.empty_cache()

    controls = args.stage == "reflection-controls"
    configs = matched_controls() if controls else SETTINGS

    def rows(mode):
        for start in range(0, n, args.score_batch):
            loc = slice(start, start + args.score_batch)
            vec, components, words = (data[loc].to(device) for data in (vectors, risks, concepts))
            if mode not in configs:
                risk = components[..., 0] + components[..., 1]
                yield from five_rows(vec[:, :5], risk[:, :5], words[:, :5], index, mode, args.score_batch)
            else:
                config = configs[mode]
                tail, prefix = reflection_block(vec, select_view_risk(components, config), words, index, config)
                yield from density_rows(tail, prefix, args.score_batch, None)

    modes = tuple(configs) if controls else ("risk4", "risk-only", *configs)
    for mode in modes:
        name = ("reflect-control-" if controls else "reflect-") + (mode if mode in configs else "five-" + mode)
        params = {"neighbors": 64, "penalty": 1., "carry": 1., "fusion": mode, "views": 8 if mode in configs else 5}
        if mode in configs:
            params.update(configs[mode]._asdict())
        record(args, split, name, rows(mode), q_labels, p_labels, metadata, params)

    ids = torch.linspace(0, n-1, 32).long()
    vec, components, words = (data[ids].to(device) for data in (vectors, risks, concepts))
    audit = {}
    audited = ("reference", "source", "target", "shuffled") if controls else ("flat-risk2", "pairs-balanced", "pairs-intersect025", "pairs-visual-risk-l035")
    for mode in audited:
        risk = select_view_risk(components, configs[mode])
        first, head = reflection_block(vec, risk, words, index, configs[mode])
        other, swapped = reflection_block(vec.roll(4, 1), risk.roll(4, 1), words.roll(4, 1), index, configs[mode])
        audit[mode] = {"queries": len(ids), "prefix_positions_equal": int(head.eq(swapped).sum()),
                       "prefix_positions": head.numel(), "tail_max_abs_difference": (first-other).abs().max().item()}
    (args.output / f"{split}-flip-check.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


def score_reflections(args, split, q_ids, q_labels, p_labels, query_data, photos, contours,
                      sketch_words, photo_words, transform, metadata, record, five_rows):
    proxy = F.normalize(F.normalize(contours @ transform, dim=-1).mean(dim=1), dim=-1)
    source_center = proxy.mean(dim=0)
    source = centered(proxy, source_center)
    target = F.normalize(photos @ transform, dim=-1)
    target = centered(target, target.mean(dim=0))
    index = gallery_index(photos, target, photo_words)
    torch.cuda.empty_cache()
    with TemporaryDirectory(dir=args.output) as temp:
        evaluate_views(args, Path(temp), split, q_ids, q_labels, p_labels, query_data, index, source, target,
                       source_center, transform, sketch_words, photo_words, metadata, record, five_rows)
