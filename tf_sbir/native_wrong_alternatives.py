"""Compare the rivals predicted by independent subjects across depictions, conditional on the same choices."""

import argparse
import hashlib
import itertools
import json

import numpy as np
import torch
from torch.nn.functional import normalize

from .description_choice_blocks import inputs
from .description_gallery_menu import observers
from .description_gallery_transfer import OUT
from .description_menu_bias import assignment, subject_sum
from .description_visual_reference import centers
from .kaggle_unseen_features import DEFAULT_CACHE
from .native_block_probe import SEED, deterministic
from .photo_name_interaction import OUT as NAME_INDEX, PARTS, ordered_shards, record, verified
from .semantic_boxable import DEFAULT_PROMPTS
from .wordnet_renaming_pilot import SPLITS


CHANNELS = {
    "sketch_RGB_to_other_RGB": ("sketch_RGB", "other_RGB"),
    "sketch_drawing_to_other_RGB": ("sketch_drawing", "other_RGB"),
    "source_RGB_to_other_RGB": ("source_RGB", "other_RGB"),
    "sketch_drawing_to_other_drawing": ("sketch_drawing", "sketch_drawing"),
    "sketch_names_to_other_names": ("sketch_names", "other_names"),
    "source_names_to_other_names": ("source_names", "other_names"),
}
MEASURES = ("top_agreement", "pair_agreement", "top_tied", "pair_tied")
VALUES = ("same", "different", "gap")


def signed_pairs(scores, left, right):
    return (scores[:, left] - scores[:, right]).sign()


class ObserverClass:
    def __init__(self, scores, subjects):
        order = subjects.argsort(stable=True)
        subjects, counts = subjects[order].unique_consecutive(return_counts=True)
        values = scores[order].double().contiguous()
        self.n = len(subjects)
        self.counts = counts.cuda()
        self.weights = self.counts.double().reciprocal().repeat_interleave(self.counts)
        self.scores = values.cuda()
        self.order = self.scores.argsort(1, descending=True, stable=True)
        left, right = torch.triu_indices(values.shape[1], values.shape[1], offset=1)
        self.left = left.cuda()
        self.right = right.cuda()
        signed = signed_pairs(self.scores, self.left, self.right)
        by_subject = self.group_mean(signed)
        self.mean_sign = by_subject.mean(0)
        self.mean_squared_sign = by_subject.square().mean(0)
        self.pair_ties = (self.weights[:, None] * (signed == 0)).sum(0) / self.n

    def group_mean(self, x):
        ends = self.counts.cumsum(0) - 1
        prefix = x.cumsum(0)[ends]
        return torch.diff(prefix, dim=0, prepend=torch.zeros_like(prefix[:1])) / self.counts[:, None]

    def summary(self, choices):
        pairs = choices[self.left] & choices[self.right]
        if int(pairs.sum()) == 0:
            raise ValueError("Need at least two genuine rivals")
        cut = choices[self.order].to(torch.int8)
        first = cut.argmax(1)
        winner = self.order.gather(1, first[:, None]).flatten()
        best = self.scores.gather(1, winner[:, None])
        ties = ((self.scores == best) & choices).sum(1) > 1
        ballot = torch.nn.functional.one_hot(winner, self.scores.shape[1]).double()
        subjects = self.group_mean(ballot)
        return (subjects.mean(0), subjects.square().sum(1).mean(),
                self.mean_sign[pairs], self.mean_squared_sign[pairs],
                self.weights.dot(ties.double()) / self.n, self.pair_ties[pairs].mean(), self.n)


def agreement(a, b, identical):
    top_a, self_top, sign_a, self_sign, tie_a, pair_tie_a, n = a
    top_b, _, sign_b, _, tie_b, pair_tie_b, _ = b
    top = top_a.dot(top_b)
    signed = sign_a * sign_b
    if identical:
        if a is not b or n < 2:
            raise ValueError("Excluding the same source requires one shared population with at least two subjects")
        top = (n * top - self_top) / (n - 1)
        signed = (n * signed - self_sign) / (n - 1)
    return torch.stack((top, (1 + signed.mean()) / 2, (tie_a + tie_b) / 2, (pair_tie_a + pair_tie_b) / 2))


def draw_contrasts(a, b, c, d, cache):
    ac, ad, bc, bd = (cache[k] for k in ((a,c), (a,d), (b,c), (b,d)))
    same = (agreement(ac, bc, a == b) + agreement(ad, bd, a == b)) / 2
    different = (agreement(ac, bd, False) + agreement(ad, bc, False)) / 2
    if a != b:
        top_identity = (ac[0] - ad[0]).dot(bc[0] - bd[0]) / 2
        pair_identity = ((ac[2] - ad[2]) * (bc[2] - bd[2])).mean() / 4
        if not torch.allclose((same - different)[:2], torch.stack((top_identity, pair_identity)), atol=1e-11, rtol=0):
            raise ValueError("Independent same-minus-foil contrasts failed exact decomposition")
    return torch.stack((same, different, same - different), dim=1)


def local_choices(pool, c, d, option_ids):
    active = torch.zeros(len(option_ids), dtype=torch.bool)
    used = {int(option_ids[c]), int(option_ids[d])}
    for k in pool:
        uid = int(option_ids[k])
        if uid not in used:
            active[k] = True
            used.add(uid)
    return active.cuda()


def hardest_candidates(scores, labels, source_counts, pool):
    weights = source_counts.double().reciprocal().cuda()
    selections = {}
    for c in pool:
        rivals = [r for r in pool if r != c]
        mask = labels == c
        predictions = scores[mask][:, rivals].argmax(1)
        ballot = torch.nn.functional.one_hot(predictions, len(rivals)).double()
        selections[c] = rivals[int((weights[mask, None] * ballot).sum(0).argmax())]
    return selections


def measure_panel(observers_by_role, pools, option_ids, train_scores, train_labels, counts):
    rows, hard = {}, {}
    for panel, pool in pools.items():
        hard[panel] = hardest_candidates(train_scores, train_labels, counts, pool)
        table = torch.full((len(option_ids), len(option_ids), len(CHANNELS), len(MEASURES), len(VALUES)), torch.nan,
                           dtype=torch.float64)
        for c, d in itertools.combinations(pool, 2):
            visual_choices = local_choices(pool, c, d, torch.arange(len(option_ids)))
            name_choices = local_choices(pool, c, d, option_ids)
            cache = {}
            for role, groups in observers_by_role.items():
                allowed = name_choices if role.endswith("_names") else visual_choices
                cache[(role,c)] = groups[c].summary(allowed)
                cache[(role,d)] = groups[d].summary(allowed)
            row = torch.stack([draw_contrasts(a, b, c, d, cache) for a, b in CHANNELS.values()]).cpu()
            table[c,d] = row
            table[d,c] = row
        rows[panel] = table
    return rows, hard


def indices_hash(ids):
    return hashlib.sha256(ids.numpy().astype("<i8", copy=False).tobytes()).hexdigest()


@torch.inference_mode()
def run(args):
    split, held = args.split, args.fold
    _, index, _ = inputs()
    prior = verified(OUT/f"{split}-source_membership.pt", args.table_revision)
    old = verified(OUT/f"visual-reference-{split}-{held}.pt", args.visual_revision)
    prior_info = json.loads((OUT/f"{split}-source_membership.json").read_bytes())
    other_info = json.loads((OUT/f"{next(s for s in SPLITS if s != split)}-source_membership.json").read_bytes())
    if prior_info["revision"] != args.table_revision or other_info["revision"] != args.table_revision:
        raise ValueError("Class-name lists changed")
    names = prior_info["classes"]
    n = len(names)
    qi, qclass, src = (prior[k] for k in ("query_ids","class_labels","source_ids"))
    source_ids, group, counts = src.unique(return_inverse=True, return_counts=True)
    subject_class, folds = assignment(split, group, source_ids, qclass)
    held_draw = folds[group] == held
    ref_subject = folds != held
    gi = old["gallery_ids"]
    target_class = prior["gallery_labels"][prior["part_positions"][1]]
    if not torch.equal(gi, prior["gallery_ids"][prior["part_positions"][1]]) or torch.isin(gi, source_ids).any():
        raise ValueError("Scoring gallery must contain only non-source photos")
    if not torch.equal(source_ids[ref_subject], old["reference_ids"]) or not torch.equal(qi[held_draw], old["query_ids"]):
        raise ValueError("The original fold identities changed")
    photos = ordered_shards([NAME_INDEX/f"names-{i}.pt" for i in range(PARTS)], args.name_revision,
                            torch.arange(len(index["photo"]["paths"])))["native"]
    obs = observers(qi, src, len(photos), args.native_revision, args.name_revision)
    rgb = normalize(photos[source_ids].float().cuda(), dim=1)
    gallery = normalize(photos[gi].float().cuda(), dim=1)
    drawings = normalize((subject_sum(obs["sketch_two"].double().cpu(), group, len(counts)) / counts[:,None]).float(), dim=1)
    picture_center = centers(rgb[ref_subject], subject_class[ref_subject], n)
    draw_center = centers(drawings[ref_subject].cuda(), subject_class[ref_subject], n)
    query = obs["sketch_two"][held_draw]
    picture = rgb[~ref_subject]
    option_ids = old["selected_texts"]["names"]
    text = torch.load(DEFAULT_CACHE/"text.pt",weights_only=True)["text"]
    t_draw, t_photo = [text[fmt][option_ids].float().cuda() for fmt in DEFAULT_PROMPTS]
    scores = {
        "sketch_RGB": query @ picture_center.T,
        "sketch_drawing": query @ draw_center.T,
        "source_RGB": picture @ picture_center.T,
        "other_RGB": gallery @ picture_center.T,
        "sketch_names": query @ t_draw.T,
        "source_names": picture @ t_photo.T,
        "other_names": gallery @ t_photo.T,
    }
    roles = {
        "sketch":(qclass[held_draw], src[held_draw]), "source":(subject_class[~ref_subject], source_ids[~ref_subject]),
        "other":(target_class, gi),
    }
    for key, expected in (("sketch_RGB","rgb_centroid"), ("sketch_drawing","drawing_centroid"), ("sketch_names","names")):
        if not torch.equal(scores[key].argmax(1).cpu(), old["predictions"][expected]):
            raise ValueError(f"Original unrestricted observer did not reproduce for {key}")
    objects = {}
    for key, z in scores.items():
        classes, ids = roles[key.partition("_")[0]]
        objects[key] = {int(c): ObserverClass(z[classes == c], ids[classes == c]) for c in classes.unique()}
    pools = {"all_classes": list(range(n)), "disjoint_classes": [c for c,w in enumerate(names) if w not in other_info["classes"]]}
    train_scores = obs["sketch_two"][~held_draw] @ picture_center.T
    tables, hard = measure_panel(objects, pools, option_ids, train_scores, qclass[~held_draw], counts[group][~held_draw])
    record(OUT/f"wrong-alternatives-{split}-{held}.pt", {
        "query_ids":qi[held_draw], "source_ids":src[held_draw], "gallery_ids":gi, "reference_ids":source_ids[ref_subject],
        "roles":roles, "scores":{k:v.cpu() for k,v in scores.items()}, "class_public_name_ids":option_ids,
        "pair_tables":tables, "hard_foils":hard,
    }, args.revision, {"split":split,"fold":held,"pools":pools,"queries":len(query),"scoring_photos":len(gi),
                       "classes":names,"channels":list(CHANNELS),"measures":MEASURES,"values":VALUES,
                       "query_id_hash":indices_hash(qi[held_draw]), "table_revision":args.table_revision,
                       "visual_revision":args.visual_revision,"native_revision":args.native_revision,"name_revision":args.name_revision})


def aggregate(split, revision):
    stores = [verified(OUT/f"wrong-alternatives-{split}-{i}.pt", revision) for i in (0,1)]
    meta = json.loads((OUT/f"wrong-alternatives-{split}-0.json").read_bytes())
    all_queries = torch.cat([z["query_ids"] for z in stores])
    if len(all_queries) != len(all_queries.unique()) or not torch.equal(stores[0]["gallery_ids"], stores[1]["gallery_ids"]):
        raise ValueError("The two folds must partition queries and share the scoring gallery")
    n = len(meta["classes"])
    class_ids = torch.arange(n)
    output, payload = {}, {}
    for panel, pool in meta["pools"].items():
        subset = torch.tensor(pool)
        output[panel] = {}
        for foils in ("all_foils", "hard_foil"):
            by_fold = []
            for store in stores:
                t = store["pair_tables"][panel]
                if foils == "hard_foil":
                    opp = torch.tensor([store["hard_foils"][panel][c] for c in pool])
                    by_fold.append(t[subset, opp])
                else:
                    by_fold.append(torch.stack([t[c, subset[subset != c]].mean(0) for c in pool]))
            class_rows = torch.stack(by_fold).mean(0)
            if not torch.isfinite(class_rows).all():
                raise ValueError("Missing focal-class evidence")
            rng = np.random.default_rng(SEED)
            sample = rng.integers(len(pool), size=(10000,len(pool)))
            interval = np.quantile(class_rows.numpy()[sample].mean(axis=1), (.025,.975), axis=0)
            summary = {}
            for i, key in enumerate(CHANNELS):
                summary[key] = {m: {"percent":(100 * class_rows[:,i,j].mean(0)).tolist(),
                                    "interval95":(100 * interval[:,i,j]).tolist(),
                                    "positive_gap_classes":int((class_rows[:,i,j,2] > 0).sum())}
                                for j,m in enumerate(MEASURES)}
            output[panel][foils] = summary
            payload[panel,foils] = class_rows
    record(OUT/f"rivals-summary-{split}.pt", {"query_ids":all_queries.sort().values,"gallery_ids":stores[0]["gallery_ids"],
           "class_summary":payload,"class_ids":class_ids}, revision, {
           "split":split,"queries":len(all_queries),"gallery":len(stores[0]["gallery_ids"]),"classes":meta["classes"],
           "pools":meta["pools"],"channels":list(CHANNELS),"measures":MEASURES,"values":VALUES,"summary":output})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--fold", type=int, choices=(0,1))
    for key in ("revision", "table-revision","visual-revision","name-revision","native-revision"):
        parser.add_argument("--"+key, required=True)
    args = parser.parse_args()
    deterministic()
    if args.fold is None:
        aggregate(args.split, args.revision)
    else:
        run(args)
