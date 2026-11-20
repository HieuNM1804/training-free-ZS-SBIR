"""Use each gallery photo's saliency and color representations as paired retrieval anchors."""

from dataclasses import asdict, dataclass, replace

import torch
from torch.nn import functional as F

from .domain_prompt_gallery import emit_group, local_readout, prompt_coordinates, source_descriptors
from .global_methods import centered
from .paired_neighborhood_scale import scale_index
from .transfer_benchmark import write_result
from .transfer_reconstruction import anchor_errors


@dataclass(frozen=True)
class Setting:
    family: str
    layout: float = 0.45
    region: float = 0.35
    share: float = 0.35
    support: int = 64
    gallery_k: int = 192
    carry: float = 1.0
    penalty: float = 1.0
    views: str = "all"
    paired: bool = True


def setting_id(c, fusion):
    return (
        f"sal-{c.family}-l{c.layout:g}-r{c.region:g}-s{c.share:g}-n{c.support}-k{c.gallery_k}"
        f"-c{c.carry:g}-{c.views}-{'paired' if c.paired else 'shuffled'}-{fusion}"
    )


FAMILIES = ("image", "public")


def grid():
    entries = {}
    for family in FAMILIES:
        base = Setting(family)
        groups = [
            base,
            *(replace(base, layout=x) for x in (0.0, 0.2, 0.75, 1.0)),
            *(replace(base, region=x) for x in (0.0, 0.75)),
            *(replace(base, share=x) for x in (0.0, 0.7)),
            replace(base, support=24),
            replace(base, gallery_k=112),
            replace(base, gallery_k=320),
            replace(base, carry=0.0),
            replace(base, views="core"),
            replace(base, views="full", layout=0.0, region=0.0),
            replace(base, paired=False),
        ]
        for c in groups:
            for fusion in ("joint", "separate"):
                entries[setting_id(c, fusion)] = (c, fusion)
    return entries


def refined(winners):
    settings = {}
    for c, fusion in winners.values():
        # A local two-parameter search around the Sketchy winner, plus support and graph scale.
        proposals = []
        for dl, dr in (
            (-0.15, 0),
            (0.15, 0),
            (0, -0.2),
            (0, 0.2),
            (-0.15, -0.2),
            (0.15, 0.2),
            (-0.15, 0.2),
            (0.15, -0.2),
        ):
            proposals.append(
                replace(
                    c,
                    layout=min(1.0, max(0.0, round(c.layout + dl, 2))),
                    region=min(1.0, max(0.0, round(c.region + dr, 2))),
                )
            )
        proposals.extend(
            (replace(c, support=40), replace(c, support=96), replace(c, gallery_k=256), replace(c, carry=1.5))
        )
        for config in proposals:
            settings[setting_id(config, fusion)] = (config, fusion)
    return {n: v for n, v in settings.items() if n not in grid()}


def normal(t):
    return F.normalize(t, dim=-1)


def view_components(native, regions, kind):
    """Identical visual quantities can be constructed for a query or a gallery entry."""
    color = normal(regions[:, :6].float())
    polarity = normal(regions[:, 6:10].float().unflatten(1, (2, 2)).mean(1))
    if kind == "full":
        return native[:, None], native[:, None]
    if kind == "core":
        return color[:, :2], polarity
    return torch.cat((native[:, None], color), 1), torch.cat(
        (normal(polarity.mean(1))[:, None], polarity.repeat(1, 3, 1)), 1
    )


def query_terms(color, source, target):
    shape = color.shape[:2]
    _, words, _ = prompt_coordinates(color.flatten(0, 1), source, target, 16)
    return normal(words).unflatten(0, shape)


def report_parameters(c, fusion, count):
    return asdict(c) | {"fusion": fusion, "query_views": count, "uses_isoclip": False}


@torch.inference_mode()
def evaluate_split(args, split, native, photo, crops, photo_crops, texts, requested, labels, metadata):
    """All statistics are derived from the unlabeled gallery or from one query at a time."""
    p_col, p_pattern = view_components(photo, photo_crops, "all")
    region_mean = normal(p_col[:, 1:].mean(1))
    shared_gallery = normal(p_pattern.mean(1))
    permutation = torch.randperm(
        len(photo), generator=torch.Generator(device=photo.device).manual_seed(209), device=photo.device
    )
    for family in FAMILIES:
        groups = {}
        for name, (c, mode) in requested.items():
            if c.family == family:
                groups.setdefault((c.views, c.region, c.gallery_k), {}).setdefault(c, []).append(
                    (name, c, mode)
                )
        queries = {}
        galleries = {}
        for (views, region, k), by_config in groups.items():
            if views not in queries:
                qcolor, qpattern = view_components(native, crops, views)
                qaux = query_terms(qcolor, *texts) if family == "public" else qpattern
                queries[views] = (qcolor, qpattern, qaux)
            q_color, q_pattern, q_aux = queries[views]
            region_key = (views == "full", region)
            if region_key not in galleries:
                target = normal(photo.lerp(region_mean, region))
                pattern = photo if views == "full" else shared_gallery
                if family == "public":
                    word_native, word_source, _ = prompt_coordinates(target, texts[1], texts[0], 32)
                    gallery_aux, source = normal(word_native), normal(word_source)
                else:
                    gallery_aux, source = pattern, target
                galleries[region_key] = (target, centered(target, target.mean(0)), pattern, gallery_aux, source)
            raw_target, target, pattern, p_aux, source = galleries[region_key]
            k = min(k, len(photo) - 1)
            index = scale_index(photo, target, p_aux, k)
            aux_mean = p_aux.mean(0)
            code_p, code_q = centered(p_aux, aux_mean), centered(q_aux, aux_mean)
            for c, group in by_config.items():
                proxy = normal(source.lerp(pattern, c.layout))
                qproxy = normal(q_color.lerp(q_pattern, c.layout))
                mean = proxy.mean(0)
                pvis, qvis = centered(proxy, mean), centered(qproxy, mean)
                pbank = source_descriptors(pvis, code_p, c.share)
                qbank = source_descriptors(qvis, code_q, c.share)
                target_ids = target if c.paired else target[permutation]
                errors = anchor_errors(pbank, target_ids)
                mapped, risk = local_readout(qbank, qvis, pbank, pvis, target_ids, errors, c)
                spec = {n: report_parameters(conf, mode, mapped.size(1)) for n, conf, mode in group}

                def record(arguments, split_name, method, rows, ql, pl, meta, _):
                    write_result(arguments, split_name, method, rows, ql, pl, meta, spec[method])

                emit_group(
                    args,
                    split,
                    group,
                    mapped,
                    risk,
                    q_aux,
                    index,
                    labels,
                    metadata | {"external_vocabulary": metadata["public_terms"] if family == "public" else 0},
                    record,
                )
