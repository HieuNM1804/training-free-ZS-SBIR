"""Estimate channel coordinates on one set of native patches and score disjoint patches."""

import argparse
import json
from pathlib import Path
import time

import torch

from .concept_benchmark import atomic_json, checksum
from .data import SKETCHY_SPLITS
from .encoding import save_tensor
from .native_block_probe import DST, SEED, deterministic


READOUTS = ("aligned", "identity", "permuted")
UPDATES = ("spatially_centered", "raw")


def splits(device):
    rng = torch.Generator().manual_seed(SEED)
    order = torch.randperm(49, generator=rng) + 1
    for fit, hold in ((order[:24], order[24:]), (order[24:], order[:24])):
        cycle = torch.randperm(len(fit), generator=rng) + 1
        perm = torch.arange(len(fit) + 1)
        perm[cycle] = cycle.roll(1)
        yield (torch.cat((torch.tensor([0]), fit)).to(device), hold.to(device), perm.to(device))


def projections(x, delta, fit, hold):
    bases, factors = [], []
    for block in x.split(96):
        q, r = torch.linalg.qr(block[:, fit, :].double().transpose(1, 2), mode="reduced")
        bases.append(q)
        factors.append(r)
    basis = torch.cat(bases)
    held = delta[:, hold, :].double()
    return basis, torch.cat(factors), held, held @ basis


def cosines(a, b):
    def cosine(left, right):
        numerator = (left * right).sum((1, 2))
        denominator = torch.linalg.vector_norm(left, dim=(1, 2)) * torch.linalg.vector_norm(right, dim=(1, 2))
        if (denominator <= 1e-20).any():
            raise ValueError("Degenerate held-out block response")
        return numerator / denominator

    return torch.stack((cosine(a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)),
                        cosine(a, b)), dim=-1)


def align(fit_cross, query, target):
    u, singular, vh = torch.linalg.svd(fit_cross, full_matrices=False, driver="gesvd")
    valid = singular > singular[:, :1] * (768 * torch.finfo(singular.dtype).eps)
    v = vh.transpose(1, 2) * valid[:, None, :]
    image_updates = target @ v
    return cosines((query @ u) * valid[:, None, :], image_updates), v, image_updates, valid.sum(1)


def compare(fields, query_ids, photo_ids, perm):
    basis, factor, held, own = fields
    sq, sp = factor[query_ids], factor[photo_ids]
    qs, ps = own[query_ids], own[photo_ids]
    actual, v, target, rank = align(sq @ sp.transpose(1, 2), qs, ps)
    identity = cosines((held[query_ids] @ basis[photo_ids]) @ v, target)
    shuffled, _, _, perm_rank = align(sq @ sp[:, :, perm].transpose(1, 2), qs, ps)
    return torch.stack((actual, identity, shuffled), dim=1), torch.stack((rank, perm_rank), dim=1)


def known_rotation(device):
    rng = torch.Generator(device=device).manual_seed(SEED)
    x = torch.randn((1, 50, 48), generator=rng, dtype=torch.float64, device=device)
    delta = torch.randn((1, 50, 48), generator=rng, dtype=torch.float64, device=device)
    h, _ = torch.linalg.qr(torch.randn((48, 48), generator=rng, dtype=torch.float64, device=device))
    fit, hold, perm = next(splits(device))
    fields = projections(torch.cat((x, x @ h)), torch.cat((delta, delta @ h)), fit, hold)
    ids = torch.tensor([0], device=device)
    scores, _ = compare(fields, ids, ids + 1, perm)
    error = (scores[:, 0] - 1).abs().max().item()
    if error > 1e-10 or scores[0, 0, 0] <= scores[0, 2, 0]:
        raise ValueError(f"Known-rotation check failed: {scores.tolist()}")
    return error


def comparisons(n_classes, part):
    # Roles within a subject are its photo followed by its two independent drawings.
    rows = []
    for cls in range(part, n_classes, 2):
        for subject in range(32):
            donor = (subject + 16) % 32
            negatives = [(cls + 1 + (subject + k * (n_classes - 1) // 4) % (n_classes - 1)) % n_classes
                         for k in range(4)]
            if len(set(negatives)) != 4 or cls in negatives:
                raise ValueError("Invalid negative class schedule")
            for drawing in (1, 2):
                qid = (32 * cls + subject) * 3 + drawing
                targets = [(0, cls, subject), (1, cls, donor)] + [(2, n, donor) for n in negatives]
                for kind, target_cls, target_subject in targets:
                    pid = (32 * target_cls + target_subject) * 3
                    rows.append((cls, subject, drawing, kind, target_cls, target_subject, qid, pid))
    return torch.tensor(rows, dtype=torch.long)


def load_states(root, name, device):
    tensors = {}
    hashes = []
    for part in (0, 1):
        path = root / f"{name}-{part}-states.pt"
        meta = json.loads(path.with_suffix(".json").read_bytes())
        result = torch.load(path, weights_only=True, mmap=True, map_location="cpu")
        if result["revision"] != meta["revision"] or result["classes"] != meta["classes"]:
            raise ValueError("Inconsistent native feature manifest")
        hashes.append(meta["state_sha256"])
        for i, cls in enumerate(result["classes"]):
            if cls in tensors:
                raise ValueError(f"Duplicate features for {cls}")
            tensors[cls] = (result["x"][i], result["delta"][i])
    classes = sorted(tensors)
    xs, ds = [torch.stack([tensors[c][i] for c in classes]).flatten(0, 2).to(device) for i in (0, 1)]
    return classes, xs, ds, hashes


@torch.inference_mode()
def measure(args):
    start = time.monotonic()
    device = torch.device("cuda")
    error = known_rotation(device)
    classes, x, delta, state_hashes = load_states(args.output, args.split, device)
    rows = comparisons(len(classes), args.part)
    target_ids = rows[:, -2:].to(device)
    all_scores, ranks = [], []
    for fit, hold, perm in splits(device):
        fields = projections(x, delta, fit, hold)
        fold = []
        for ids in target_ids.split(128):
            scores, rank = compare(fields, ids[:, 0], ids[:, 1], perm)
            fold.append(scores.cpu())
            ranks.append(rank.cpu())
        all_scores.append(torch.cat(fold))
        del fields
    scores = torch.stack(all_scores, dim=1)
    if not scores.isfinite().all():
        raise ValueError("Nonfinite held-out scores")
    dest = args.output / f"{args.split}-{args.part}-heldout.pt"
    save_tensor({"classes": classes, "links": rows[:, :6], "scores": scores,
                 "readouts": list(READOUTS), "updates": list(UPDATES)}, dest)
    meta = {"revision": args.revision, "split": args.split, "part": args.part, "seed": SEED,
            "pairs": len(rows), "score_shape": list(scores.shape), "state_hashes": state_hashes,
            "rank_range": [int(torch.cat(ranks).min()), int(torch.cat(ranks).max())],
            "known_rotation_error": error, "device": torch.cuda.get_device_name(),
            "seconds": time.monotonic()-start, "sha256": checksum(dest)}
    atomic_json(dest.with_suffix(".json"), meta)
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--split", choices=SKETCHY_SPLITS, required=True)
    parser.add_argument("--part", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, default=DST)
    deterministic()
    measure(parser.parse_args())
