"""Build source/target gallery pairs from domain-prompt coordinates without source-domain images."""

from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.nn import functional as F

from .contextual_rank import jaccard_seed
from .contour_query_expansion import pagerank_tail
from .gallery_semantic_balance import compact_union
from .global_methods import centered
from .multiview_anchor_transfer import evidence_weights
from .paired_neighborhood_scale import scale_index
from .reflection_transfer import file_tensor
from .reranking import standardize
from .transfer_reconstruction import anchor_errors, nearest_anchors, reconstruction_weights
from .union_density import candidate_union, density_rows, grow_density
from .visual_paired_graph import image_index, visual_tail


@dataclass(frozen=True)
class PairConfig:
    retain: float = 0.5
    semantic_share: float = 0.35
    support: int = 64
    carry: float = 1.0
    penalty: float = 1.0
    prompts: str = "paired"
    pairing: str = "intact"


def settings():
    default = PairConfig()
    groups = [
        ("default", default),
        ("retain0", PairConfig(retain=0.0)),
        ("retain025", PairConfig(retain=0.25)),
        ("retain1", PairConfig(retain=1.0)),
        ("share0", PairConfig(semantic_share=0.0)),
        ("share07", PairConfig(semantic_share=0.7)),
        ("share1", PairConfig(semantic_share=1.0)),
        ("k16", PairConfig(support=16)),
        ("k128", PairConfig(support=128)),
        ("carry0", PairConfig(carry=0.0)),
        ("carry15", PairConfig(carry=1.5)),
        ("no-penalty", PairConfig(penalty=0.0)),
        ("neutral", PairConfig(prompts="neutral")),
        ("unpaired", PairConfig(pairing="shuffled")),
    ]
    out = {}
    for label, config in groups:
        modes = ["separate", "joint"]
        if label == "default":
            modes.extend(("one", "lex20", "lex50", "visual-votes"))
        for mode in modes:
            out["domain-" + label + "-" + mode] = (config, mode)
    return out


def prompt_coordinates(images, own_prompts, other_prompts, neighbors):
    """Transfer affine coordinates to paired prompts and keep the residual in CLIP's common axes."""
    indices = nearest_anchors(images, own_prompts, neighbors)
    native, converted, residual = [], [], []
    for x, ids in zip(images.split(256), indices.split(256), strict=True):
        w = reconstruction_weights(x, own_prompts[ids], 0.1).unsqueeze(1)
        own = (w @ own_prompts[ids]).squeeze(1)
        native.append(own)
        converted.append((w @ other_prompts[ids]).squeeze(1))
        residual.append(x - own)
    return torch.cat(native), torch.cat(converted), torch.cat(residual)


def source_descriptors(visual, semantic, share):
    if share == 0.0:
        return visual
    if share == 1.0:
        return semantic
    return torch.cat((visual * (1.0 - share) ** 0.5, semantic * share**0.5), dim=-1)


def local_readout(queries, visible, bank, visible_bank, target, errors, config):
    """Only the visual component's residual is shared with the target coordinate system."""
    shape = queries.shape[:2]
    source = queries.flatten(0, 1)
    hints = visible.flatten(0, 1)
    neighbors = nearest_anchors(source, bank, config.support)
    predictions, risks = [], []
    for x, v, ids in zip(source.split(192), hints.split(192), neighbors.split(192), strict=True):
        keys, err = bank[ids], errors[ids]
        w = reconstruction_weights(x, keys, 0.1, err, config.penalty).unsqueeze(1)
        offset = x - (w @ keys).squeeze(1)
        dispersion = (keys - x[:, None]).square().sum(-1).mean(1).clamp_min(1e-8)
        risk = offset.square().sum(-1) / dispersion + (w @ err).squeeze(1).square().sum(-1)
        visual_residual = v - (w @ visible_bank[ids]).squeeze(1)
        mapped = (w @ target[ids]).squeeze(1) + config.carry * visual_residual
        predictions.append(F.normalize(mapped, dim=-1))
        risks.append(risk)
    return torch.cat(predictions).unflatten(0, shape), torch.cat(risks).unflatten(0, shape)


def rank_block(mapped, risks, lexical, index, modes):
    gallery, vocabulary, banks, graph, walk = index
    visual = mapped @ gallery.T
    words = lexical @ vocabulary.T
    shape = torch.stack([jaccard_seed(s, banks[0]) for s in visual.unbind(1)], dim=1)
    tokens = torch.stack([jaccard_seed(s, banks[1]) for s in words.unbind(1)], dim=1)
    sets = compact_union(*candidate_union((*visual.unbind(1), *words.unbind(1)), min(1280, len(gallery))))
    appearance, text = standardize(visual).mean(1), standardize(words).mean(1)
    prior = standardize(appearance) + standardize(text)
    tail = pagerank_tail(appearance, text, prior, banks, *walk, len(prior))
    view_prior = risks.new_ones(risks.size(1)) / risks.size(1)
    shape_weights = evidence_weights(shape.sum(-1), risks, view_prior, 0.0, 2.0)
    visual_seed = (shape * shape_weights[:, :, None]).sum(1)
    language_seed = tokens.mean(1)
    heads = {}
    for mode in modes:
        if mode == "joint":
            joined = shape + tokens
            weights = evidence_weights(joined.sum(-1), risks, view_prior, 4.0, 2.0)
            seed = (joined * weights[:, :, None]).sum(1)
        else:
            fraction = {"separate": 0.35, "one": 0.35, "lex20": 0.2, "lex50": 0.5, "visual-votes": 0.0}[mode]
            seed = 2.0 * visual_seed.lerp(language_seed, fraction)
        heads[mode] = grow_density(32.0 * seed, prior, *sets, graph, min(256, len(gallery)))
    return tail, heads


def emit_group(args, split, names, mapped, risk, words, index, labels, metadata, record):
    q_labels, p_labels = labels
    device = mapped.device
    tags = [mode for _, _, mode in names]
    with TemporaryDirectory(dir=args.output) as folder:
        path = Path(folder)
        n, g = len(mapped), len(p_labels)
        background = file_tensor(path, "tail", (n, g))
        depth = min(256, g)
        heads = {
            mode: torch.from_file(str(path / (mode + ".i32")), shared=True, size=n * depth, dtype=torch.int32).view(
                n, depth
            )
            for mode in tags
        }
        for start in range(0, n, args.score_batch):
            loc = slice(start, start + args.score_batch)
            tail, head = rank_block(mapped[loc], risk[loc], words[loc], index, tags)
            background[loc] = tail.cpu()
            for mode in tags:
                heads[mode][loc] = head[mode].to(device="cpu", dtype=torch.int32)
        for name, config, mode in names:

            def rows():
                for tail, head in zip(background.split(256), heads[mode].split(256), strict=True):
                    yield from density_rows(
                        tail.to(device), head.to(device=device, dtype=torch.long), len(tail), None
                    )

            record(
                args,
                split,
                name,
                rows(),
                q_labels,
                p_labels,
                metadata,
                {**asdict(config), "fusion": mode, "views": mapped.size(1), "gallery_k": min(192, g - 1)},
            )


def semantic_rows(args, split, raw, photos, transform, source_terms, target_terms, requested, labels, meta, record):
    catalog = settings()
    projected = F.normalize(photos @ transform, dim=-1)
    target = centered(projected, projected.mean(0))
    # The gallery text representation stays in target-prompt coordinates for every configuration.
    native_words, source_words, image_residual = prompt_coordinates(photos, target_terms, source_terms, 32)
    vocabulary = F.normalize(native_words, dim=-1)
    word_mean = vocabulary.mean(0)
    centered_words = centered(vocabulary, word_mean)
    index = scale_index(photos, target, vocabulary, min(192, len(photos) - 1))
    query_visual = F.normalize(raw @ transform, dim=-1)
    shape = raw.shape[:2]
    rng = torch.Generator(device=raw.device).manual_seed(731)
    perm = torch.randperm(len(photos), device=raw.device, generator=rng)
    for neutral in (False, True):
        groups = {}
        for name in requested:
            config, mode = catalog[name]
            if (config.prompts == "neutral") == neutral:
                groups.setdefault(config, []).append((name, config, mode))
        if not groups:
            continue
        language = target_terms if neutral else source_terms
        _, readout, _ = prompt_coordinates(raw.flatten(0, 1), language, target_terms, 16)
        words = F.normalize(readout, dim=-1).unflatten(0, shape)
        semantic_q = centered(words, word_mean)
        for config, names in groups.items():
            print(split, [n[0] for n in names], flush=True)
            transferred = native_words if neutral else source_words
            proxy = F.normalize(
                F.normalize(transferred + config.retain * image_residual, dim=-1) @ transform, dim=-1
            )
            mean = proxy.mean(0)
            visible = centered(proxy, mean)
            query = centered(query_visual, mean)
            bank = source_descriptors(visible, centered_words, config.semantic_share)
            descriptors = source_descriptors(query, semantic_q, config.semantic_share)
            targets = target[perm] if config.pairing != "intact" else target
            errors = anchor_errors(bank, targets)
            mapped, risk = local_readout(descriptors, query, bank, visible, targets, errors, config)
            for one_view in (False, True):
                subset = [v for v in names if (v[2] == "one") == one_view]
                if subset:
                    top = 1 if one_view else raw.size(1)
                    emit_group(
                        args,
                        split,
                        subset,
                        mapped[:, :top],
                        risk[:, :top],
                        words[:, :top],
                        index,
                        labels,
                        meta,
                        record,
                    )


def visual_rows(args, split, raw, photos, transform, requested, labels, meta, record):
    target = F.normalize(photos @ transform, dim=-1)
    mean = target.mean(0)
    target = centered(target, mean)
    queries = centered(F.normalize(raw @ transform, dim=-1), mean)
    gallery_k = min(192, len(photos) - 1)
    smooth, bank, graph, walk = image_index(photos, target, gallery_k)
    q_labels, p_labels = labels
    for name in requested:
        v = 1 if name.endswith("-one") else raw.size(1)

        def rows():
            for q in queries[:, :v].split(args.score_batch):
                visual = q @ smooth.T
                prior, tail = visual_tail(standardize(visual).mean(1), bank, walk)
                votes = torch.stack([jaccard_seed(s, bank) for s in visual.unbind(1)], dim=1)
                base = votes.mean(1)
                if name.endswith("-agreement"):
                    weights = (votes.sum(-1).clamp_min(1e-8).log() * 4.0).softmax(1)
                    base = (votes * weights[:, :, None]).sum(1)
                candidates = compact_union(*candidate_union(visual.unbind(1), min(1280, len(photos))))
                order = grow_density(64.0 * base, prior, *candidates, graph, min(256, len(photos)))
                yield from density_rows(tail, order, len(q), None)

        record(
            args,
            split,
            name,
            rows(),
            q_labels,
            p_labels,
            meta,
            {"views": v, "gallery_k": gallery_k, "channels": "images"},
        )
