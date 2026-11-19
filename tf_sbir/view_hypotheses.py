"""Generate competing retrieval lists from individual sketch regions and select by MBR."""

from functools import partial

from PIL import Image, ImageOps
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from .isoclip import projected_feedback
from .mta import load_view_features
from .reranking import standardize
from .semantic import code_vectors, top_codes
from .union_density import density_blocks, density_prefixes, density_rows, select_density_rows


REGIONS = ("ink-box", "upper-left", "upper-right", "lower-left", "lower-right")


class RegionalSketches(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as source:
            im = source.convert("RGB")
        box = ImageOps.grayscale(im).point([255 if i < 204 else 0 for i in range(256)]).getbbox()
        if box is not None:
            left, top, right, bottom = box
            margin = round(0.05 * max(right - left, bottom - top))
            im = im.crop((max(0, left-margin), max(0, top-margin), min(im.width, right+margin), min(im.height, bottom+margin)))
        full = ImageOps.pad(im, (224, 224), method=Image.Resampling.BICUBIC, color=(255, 255, 255))
        # The second R-MAC scale gives four overlapping windows on a square input.
        side = 2 * full.width // 3
        offset = full.width - side
        patches = [full]
        for top in (0, offset):
            for left in (0, offset):
                patches.append(full.crop((left, top, left + side, top + side)).resize((224, 224), Image.Resampling.BICUBIC))
        return torch.stack([pil_to_tensor(patch) for patch in patches])


def load_regional_sketches(args, paths):
    return load_view_features(args, RegionalSketches(paths), len(REGIONS), "sketch", "rmac-regional-v1")


def view_concepts(views, sketch_words, photo_words):
    dtype = torch.float16 if views.is_cuda else torch.float32
    prompts = sketch_words.to(dtype)
    features = []
    for block in views.flatten(0, 1).split(512):
        ids, weights = top_codes((block.to(dtype) @ prompts.T).float(), 5)
        features.append(code_vectors(photo_words, ids, weights))
    return torch.cat(features).unflatten(0, views.shape[:2])


def head_prototypes(rankings, features, score_batch):
    batch = min(128, score_batch)
    depth = min(256, len(features))
    prototypes = []
    for initial, selected, weight in rankings:
        means = []
        for scores in density_blocks(initial, selected, batch, weight):
            ids = scores.topk(depth, dim=-1).indices
            means.append(F.normalize(features[ids].float().mean(dim=1), dim=-1))
        prototypes.append(torch.cat(means))
    return torch.stack(prototypes, dim=1)


def expected_utility(prototypes, posterior):
    """Compute expected cosine utility against a weighted set of candidate references."""
    consensus = prototypes.transpose(1, 2) @ posterior.unsqueeze(-1)
    return (prototypes @ consensus).squeeze(-1)


def view_hypotheses(
    rankings, queries, random_views, regional_views, photos, contours, sketch_words,
    photo_words, transform, gallery, words, banks, graph, score_batch,
):
    views = F.normalize(torch.cat((random_views, regional_views), dim=1).float(), dim=-1)
    embedded, _ = projected_feedback(views.flatten(0, 1), photos, contours, transform)
    embedded = embedded.unflatten(0, views.shape[:2])
    concepts = view_concepts(views, sketch_words, photo_words)
    names = [f"view-random-{i+1}" for i in range(random_views.size(1))]
    names += [f"view-regional-{name}" for name in REGIONS]
    rankings = list(rankings)
    tail = rankings[0][0]
    for name, inside, phrase in zip(names, embedded.unbind(dim=1), concepts.unbind(dim=1), strict=True):
        visual = inside @ gallery.T
        lexical = phrase @ words.T
        score = standardize(visual) + standardize(lexical)
        prefix = density_prefixes(visual, lexical, (visual, lexical), score, banks, graph, 1280, score_batch, 256, 16.)
        rankings.append((tail, prefix, None))
        yield name, partial(density_rows, tail, prefix, score_batch, None)
    del visual, lexical, score

    dtype = torch.float16 if photos.is_cuda else torch.float32
    features = torch.cat([
        F.normalize(view - view.mean(dim=0), dim=-1) for view in (photos, gallery, words)
    ], dim=-1).to(dtype)
    signal = head_prototypes(rankings, features, score_batch)
    similarity = (queries[:, None] * views).sum(dim=-1) / .07
    random_prior, region_prior = similarity.split((random_views.size(1), regional_views.size(1)), dim=1)
    leaders = similarity.new_full((len(queries), 4), .125)
    balanced = torch.cat((leaders, .25 * random_prior.softmax(dim=1), .25 * region_prior.softmax(dim=1)), dim=1)
    uniform = torch.full_like(balanced, 1. / balanced.size(1))
    yield "view-mbr-uniform", partial(select_density_rows, rankings, expected_utility(signal, uniform).argmax(dim=1), score_batch)
    confidence = expected_utility(signal, balanced)
    yield "view-mbr-balanced", partial(select_density_rows, rankings, confidence.argmax(dim=1), score_batch)
    for index, method in enumerate(("mta", "cds")):
        favored = standardize(confidence)
        favored[:, index] += .75
        yield f"view-mbr-guard-{method}", partial(select_density_rows, rankings, favored.argmax(dim=1), score_batch)
