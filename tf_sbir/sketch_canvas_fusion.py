"""Fuse complementary centered scales and reflected canvases for one sketch."""

from functools import partial

from PIL import Image, ImageOps
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from .conditional_view_affinity import consistency_fraction, shared_view_votes
from .contextual_rank import column_stochastic, jaccard_seed
from .contour_query_expansion import pagerank_tail
from .global_methods import prepare_methods
from .manifold import iterative_feedback
from .mta import load_view_features
from .reranking import standardize
from .sketch_view_graph_fusion import sketch_view_graph_inputs, source_votes
from .union_density import candidate_union, density_prefixes, density_rows, grow_density, pack_affinity
from .view_hypotheses import ink_box_image, view_concepts


CANVAS_NAMES = ("medium", "small", "mirror")


def canvas_images(ink):
    for margin in (20, 39):
        side = ink.width - 2 * margin
        reduced = ink.resize((side, side), Image.Resampling.BICUBIC)
        yield ImageOps.expand(reduced, border=margin, fill=(255, 255, 255))
    yield ImageOps.mirror(ink)


class CanvasSketches(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as source:
            ink = ink_box_image(source.convert("RGB"))
        return torch.stack([pil_to_tensor(im) for im in canvas_images(ink)])


def load_canvas_sketches(args, paths):
    return load_view_features(args, CanvasSketches(paths), len(CANVAS_NAMES), "sketch", "full-ink-canvases-v1")


def canvas_consensus(votes):
    strengths = votes.sum(dim=-1, keepdim=True)
    expectation = (votes * strengths).sum(dim=1) / strengths.sum(dim=1).clamp_min(1e-8)
    return expectation


def canvas_prefix_variants(visual, lexical, expansions, concepts, index, prior, banks, graph, score_batch):
    names = ("late-reference", *(f"only-{name}" for name in CANVAS_NAMES), "mix-medium", "mix-scale", "mix-mirror",
             "mix-all", "mix-all-base-candidates", "adaptive-all", "adaptive-all-sharp")
    output = {name: [] for name in names}
    inputs = (*visual, *lexical, expansions, concepts, prior)
    batches = (tensor.split(min(score_batch, 384)) for tensor in inputs)
    gallery, gallery_words = index
    for crop, full, crop_words, full_words, vectors, phrases, ties in zip(*batches, strict=True):
        original = (crop, full, crop_words, full_words)
        base_candidates = candidate_union(original, 1280)
        appearance = source_votes(crop, full, banks[0], 0.35, True)
        words = source_votes(crop_words, full_words, banks[1], 0.65, True)
        shared = shared_view_votes(crop, full, banks[0], 0.35) + shared_view_votes(crop_words, full_words, banks[1], 0.65)
        baseline = (appearance + words).lerp(shared, 0.5)
        output["late-reference"].append(grow_density(baseline * 32, ties, *base_candidates, graph, 256))

        local_visual = (vectors @ gallery.T).unbind(dim=1)
        local_words = (phrases @ gallery_words.T).unbind(dim=1)
        local_votes = torch.stack([jaccard_seed(a, banks[0]) + jaccard_seed(b, banks[1])
                                   for a, b in zip(local_visual, local_words, strict=True)], dim=1)
        for index_id, name in enumerate(CANVAS_NAMES):
            candidates = candidate_union((local_visual[index_id], local_words[index_id]), 1280)
            output[f"only-{name}"].append(grow_density(local_votes[:, index_id] * 32, ties, *candidates, graph, 256))

        all_candidates = candidate_union((*original, *local_visual, *local_words), 1280)
        consensus = canvas_consensus(local_votes)
        for name, indices in (("medium", (0,)), ("scale", (0, 1)), ("mirror", (2,)), ("all", (0, 1, 2))):
            extra_streams = tuple(local_visual[i] for i in indices) + tuple(local_words[i] for i in indices)
            candidates = all_candidates if name == "all" else candidate_union((*original, *extra_streams), 1280)
            votes = consensus if name == "all" else canvas_consensus(local_votes[:, indices])
            output[f"mix-{name}"].append(grow_density(baseline.lerp(votes, 0.35) * 32, ties, *candidates, graph, 256))
        output["mix-all-base-candidates"].append(grow_density(baseline.lerp(consensus, 0.35) * 32, ties, *base_candidates, graph, 256))
        strength = (baseline.sum(dim=1, keepdim=True), consensus.sum(dim=1, keepdim=True))
        for name, power in (("adaptive-all", 1), ("adaptive-all-sharp", 4)):
            fraction = consistency_fraction(*strength, power=power, prior=0.35)
            output[name].append(grow_density(baseline.lerp(consensus, fraction) * 32, ties, *all_candidates, graph, 256))
    return {name: torch.cat(rows) for name, rows in output.items()}


def sketch_canvas_fusion_settings(queries, photos, contours, sketch_words, photo_words, maps, ink, canvases, score_batch):
    visual, lexical, banks, edges, index = sketch_view_graph_inputs(
        queries, photos, contours, sketch_words, photo_words, maps, ink, include_index=True,
    )
    graph = pack_affinity(len(photos), edges)
    transition, dangling = column_stochastic(len(photos), edges)
    v0, v1 = (standardize(scores) for scores in visual)
    l0, l1 = (standardize(scores) for scores in lexical)
    combined = (v0.lerp(v1, .5), l0.lerp(l1, .5))
    prior = standardize(combined[0]) + standardize(combined[1])
    prefix = density_prefixes(*combined, combined, prior, banks, graph, 1280, score_batch, 256, 64.)
    tail = pagerank_tail(*combined, prior, banks, transition, dangling, score_batch)
    yield "canvas-fusion-early-reference", partial(density_rows, tail, prefix, score_batch, None)
    del combined, prior, prefix, tail
    combined = (v0.lerp(v1, .35), l0.lerp(l1, .65))
    del v0, v1, l0, l1
    prior = standardize(combined[0]) + standardize(combined[1])
    tail = pagerank_tail(*combined, prior, banks, transition, dangling, score_batch)
    del combined

    canvases = F.normalize(canvases.float(), dim=-1)
    transform = maps[(150, 50)]
    q, p, contour = (F.normalize(tensor @ transform, dim=-1) for tensor in (canvases, photos, contours))
    q, _ = prepare_methods(q.flatten(0, 1), p, contour, ("local_residual",))["local_residual"]
    expanded = iterative_feedback(q, index[0], 16, 2, 2.).unflatten(0, canvases.shape[:2])
    concepts = view_concepts(canvases, sketch_words, photo_words)
    for name, prefix in canvas_prefix_variants(visual, lexical, expanded, concepts, index, prior, banks, graph, score_batch).items():
        yield f"canvas-fusion-{name}", partial(density_rows, tail, prefix, score_batch, None)
