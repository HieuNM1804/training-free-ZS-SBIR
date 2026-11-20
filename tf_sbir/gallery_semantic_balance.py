"""Separate the contribution of semantic edges to the gallery index, retrieval graph and query."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.nn import functional as F

from .contextual_rank import column_stochastic, jaccard_graph, jaccard_seed
from .contour_graph_index import gallery_walk, row_transition
from .contour_query_expansion import gallery_word_codes, pagerank_tail
from .global_methods import centered
from .reflection_transfer import PairSetting, cache_reflection_features, file_tensor, reflection_seed
from .reranking import standardize
from .union_density import candidate_union, density_rows, grow_density, pack_affinity


# Six equally wide intervals make 2/6 the unmodified three-stream average.
GALLERY_FRACTIONS = {2: (2, 0), 0: (0, 2), 1: (1,), 3: (3,), 4: (4,)}
LEXICAL_FRACTIONS = {"text20": .2, "text35": .35, "text50": .5}


def edge_weights(three_graphs, sixths):
    if sixths == 2:
        return three_graphs
    fraction = sixths / 6.
    factors = (1.5 * (1. - fraction), 3. * fraction, 1.5 * (1. - fraction))
    return tuple((src, dst, weight * scale) for (src, dst, weight), scale in zip(three_graphs, factors, strict=True))


def symmetric_mean(scores):
    left, right = standardize(scores).chunk(2, dim=1)
    prior = scores.new_tensor([.33, .22, .225, .225])
    return ((left + right) * (.5 * prior[None, :, None])).sum(dim=1)


def compact_union(ids, unique):
    counts = unique.sum(dim=1)
    width = int(counts.max())
    positions = unique.cumsum(dim=1) - 1
    compact = ids.new_zeros((len(ids), width)).scatter_(1, positions, ids)
    return compact, torch.arange(width, device=ids.device)[None, :] < counts[:, None]


def query_evidence(mapped, uncertainty, concepts, index, *, modes=None, compact=False, blends=None):
    smooth, vocabulary, banks, graph, walk = index
    visual = mapped @ smooth.T
    lexical = concepts @ vocabulary.T
    appearance, words = symmetric_mean(visual), symmetric_mean(lexical)
    tied = standardize(appearance) + standardize(words)
    candidates = candidate_union((*visual.unbind(1), *lexical.unbind(1)), 1280)
    if compact:
        candidates = compact_union(*candidates)
    shapes = torch.stack([jaccard_seed(v, banks[0]) for v in visual.unbind(1)], dim=1)
    terms = torch.stack([jaccard_seed(v, banks[1]) for v in lexical.unbind(1)], dim=1)
    risk = uncertainty.sum(dim=-1)
    shape_seed = reflection_seed(shapes, risk, PairSetting(False, 0., 4.))
    word_seed = reflection_seed(terms, risk, PairSetting(False, 0., 0.))
    seeds = {name: 2. * shape_seed.lerp(word_seed, fraction) for name, fraction in LEXICAL_FRACTIONS.items()}
    seeds["sharp"] = reflection_seed(shapes + terms, uncertainty[..., 0], PairSetting(False, 4., 4.))
    if blends is None:
        head = {name: grow_density(32. * seed, tied, *candidates, graph, 256) for name, seed in seeds.items()
                if modes is None or name in modes}
    else:
        head = {}
        for name, (fraction, strength) in blends.items():
            if fraction == 0:
                seed = seeds["text35"]
            elif fraction == 1:
                seed = seeds["sharp"]
            else:
                seed = seeds["text35"].lerp(seeds["sharp"], fraction)
            head[name] = grow_density(strength * seed, tied, *candidates, graph, 256)
    return pagerank_tail(appearance, words, tied, banks, *walk, len(mapped)), head


def evaluate_pair(args, split, queries, labels, index, pre, post, metadata, record):
    q_labels, p_labels = labels
    names = [*LEXICAL_FRACTIONS, "sharp"]
    prefix = f"balance-index{pre}-graph{post}-"
    def saved(name):
        result = args.output / f"{split}-{prefix}{name}.json"
        per_query = args.output / "per-query" / f"{split}-{prefix}{name}.pt"
        if not (result.is_file() and per_query.is_file()):
            return False
        row = json.loads(result.read_text())
        return (row["revision"] == metadata["revision"] and row["queries"] == len(q_labels)
                and row["gallery"] == len(p_labels) and row["checkpoint_sha256"] == metadata["checkpoint_sha256"])
    if all(saved(name) for name in names):
        return
    n, device = len(q_labels), index[0].device
    with TemporaryDirectory(dir=args.output) as folder:
        path = Path(folder)
        tail = file_tensor(path, "tail", (n, len(p_labels)))
        heads = {name: torch.from_file(str(path / (name + ".i32")), shared=True, size=n * 256, dtype=torch.int32).view(n, 256)
                 for name in names}
        for start in range(0, n, args.score_batch):
            section = slice(start, start + args.score_batch)
            prior, ranks = query_evidence(*(part[section].to(device) for part in queries), index)
            tail[section] = prior.cpu()
            for name, ranking in ranks.items():
                heads[name][section] = ranking.to(device="cpu", dtype=torch.int32)
        def rows(head):
            for scores, selected in zip(tail.split(256), head.split(256), strict=True):
                yield from density_rows(scores.to(device), selected.to(device=device, dtype=torch.long), len(scores), None)
        for name in names:
            params = {"gallery_semantic_smoothing": pre / 6., "gallery_semantic_reranking": post / 6.,
                      "fusion": name, "query_lexical_fraction": LEXICAL_FRACTIONS.get(name, .5),
                      "views": 8, "neighbors": 64, "penalty": 1., "carry": 1.}
            record(args, split, prefix + name, rows(heads[name]), q_labels, p_labels, metadata, params)
        del heads, tail


def score_balanced_gallery(args, split, q_ids, q_labels, p_labels, data, photos, contours, sketch_words,
                           photo_words, transform, metadata, record):
    proxies = F.normalize(F.normalize(contours @ transform, dim=-1).mean(dim=1), dim=-1)
    center = proxies.mean(dim=0)
    source = centered(proxies, center)
    target = F.normalize(photos @ transform, dim=-1)
    target = centered(target, target.mean(dim=0))
    vocabulary = gallery_word_codes(photos, photo_words).half().float()
    _, *iso_edges = jaccard_graph(target, 192)
    words, *word_edges = jaccard_graph(vocabulary, 192)
    _, *rgb_edges = jaccard_graph(photos, 192)
    with TemporaryDirectory(dir=args.output) as folder:
        queries = cache_reflection_features(Path(folder), q_ids, data, source, target, center, transform, sketch_words, photo_words)
        for pre, graph_options in GALLERY_FRACTIONS.items():
            edges = edge_weights((iso_edges, word_edges, rgb_edges), pre)
            move, dangling = row_transition(len(photos), edges)
            smooth = gallery_walk(move, dangling, target, .8, 4)
            bank, *smooth_edges = jaccard_graph(smooth, 192)
            for post in graph_options:
                print(f"{split}: index {pre}/6, reranking {post}/6", flush=True)
                edges = edge_weights((smooth_edges, word_edges, rgb_edges), post)
                index = (smooth, vocabulary, (bank, words), pack_affinity(len(photos), edges), column_stochastic(len(photos), edges))
                evaluate_pair(args, split, queries, (q_labels, p_labels), index, pre, post, metadata, record)
                del index
        del queries
