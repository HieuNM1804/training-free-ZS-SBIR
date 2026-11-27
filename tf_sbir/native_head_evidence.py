"""Read frozen CLIP coordinates on the archived pair of Semantic Dot leaders for every query."""

import argparse
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .concept_benchmark import atomic_json, checksum
from .data import SKETCHY_SPLITS
from .encoding import load_frozen_clip, save_tensor
from .kaggle_unseen_features import DEFAULT_CACHE, index_file
from .native_block_probe import DST, PAIRED, SEED, deterministic, data_loader


READOUTS = ("clip", "before_projection", "retained", "discarded", "random_retained_256")


class Head:
    def __init__(self, model):
        self.model = model
        w = model.visual.proj.detach().double()
        basis, _ = torch.linalg.qr(w, mode="complete")
        g = torch.Generator().manual_seed(SEED)
        random_axes, _ = torch.linalg.qr(torch.randn(512, 256, dtype=torch.float64, generator=g))
        self.kept, self.null = basis[:, :512], basis[:, 512:]
        self.random = self.kept @ random_axes.to(w.device)
        invisible = torch.linalg.vector_norm(self.null.T @ w) / torch.linalg.vector_norm(w)
        self.info = {"matrix_shape": list(w.shape), "matrix_rank": int(torch.linalg.matrix_rank(w)),
                     "null_relative_projection_norm": float(invisible)}
        if self.info["matrix_rank"] != 512 or w.shape != (768, 512) or invisible > 1e-10:
            raise ValueError(f"Unexpected visual projector {self.info}")

    def read(self, pixels):
        saved = []
        def store(module, inputs, output):
            saved.append(output)
        handle = self.model.visual.ln_post.register_forward_hook(store)
        try:
            z = self.model.encode_image(pixels).float()
        finally:
            handle.remove()
        if len(saved) != 1:
            raise RuntimeError("Native final LayerNorm did not run exactly once")
        h = saved[0]
        if h.ndim == 3:
            h = h[:, 0, :]
        h = h.double()
        if "native_head_replay" not in self.info:
            relative = torch.linalg.vector_norm(h @ self.model.visual.proj.double() - z.double(), dim=1)
            relative /= torch.linalg.vector_norm(z.double(), dim=1)
            self.info["native_head_replay"] = float(relative.max())
            if self.info["native_head_replay"] > 0.001:
                raise ValueError(f"Final LayerNorm does not reconstruct CLIP {self.info}")
        arrays = (z, h, h @ self.kept, h @ self.null, h @ self.random)
        return [F.normalize(v.float(), dim=-1) for v in arrays]

    def encode(self, root, names, mirror, batch, workers):
        storage = [[] for _ in READOUTS]
        dtype = self.model.visual.conv1.weight.dtype
        for pixels in data_loader([str(root / n) for n in names], workers, batch):
            pixels = pixels.to(device="cuda", dtype=dtype, non_blocking=True)
            fields = self.read(pixels)
            if mirror:
                reflected = self.read(pixels.flip((-1,)))
                fields = [F.normalize(a + b, dim=-1) for a, b in zip(fields, reflected, strict=True)]
            for arr, v in zip(storage, fields, strict=True):
                arr.append(v.cpu())
        return [torch.cat(x) for x in storage]


@torch.inference_mode()
def audit(args):
    start = time.monotonic()
    manifest = json.loads((DEFAULT_CACHE / "manifest.json").read_bytes())
    spec = manifest["datasets"]["sketchy"]
    source_root = Path(spec["root"])
    manifest_index = torch.load(index_file(DEFAULT_CACHE, "sketchy", spec), weights_only=True)
    old_output = PAIRED / f"real-errors-{args.split}.pt"
    old_fingerprint = checksum(old_output)
    old_meta = json.loads(old_output.with_suffix(".json").read_bytes())
    if old_fingerprint != old_meta["per_query_sha256"]:
        raise ValueError("Original leader archive hash mismatch")
    old = torch.load(old_output, weights_only=True)
    qids, qlabels = manifest_index["sketch"]["views"][args.split]
    pids, plabels = manifest_index["photo"]["views"][args.split]
    if not (torch.equal(qids, old["query_ids"]) and torch.equal(pids, old["gallery_ids"])
            and torch.equal(qlabels, old["labels"])):
        raise ValueError("Dataset order mismatch")
    rows = torch.arange(args.part, len(qids), 2)
    pair = old["measurements"]["semantic"]["endpoints"][rows]
    if not (torch.equal(qlabels[rows], plabels[pair[:, 0]]) and (qlabels[rows] != plabels[pair[:, 1]]).all()
            and (old["source"][rows, None] != pair).all()):
        raise ValueError("Unexpected source photo or class in archived endpoints")
    photos, inverse = pair.unique(return_inverse=True)
    head = Head(load_frozen_clip(manifest["checkpoint"], torch.device("cuda")).eval())
    def names(mode, ids):
        return [manifest_index[mode]["paths"][i] for i in ids.tolist()]

    q = head.encode(source_root, names("sketch", qids[rows]), True, args.batch, args.workers)
    p = head.encode(source_root, names("photo", pids[photos]), False, args.batch, args.workers)
    margins = torch.stack([(query * (gallery[inverse[:, 0]] - gallery[inverse[:, 1]])).sum(-1)
                           for query, gallery in zip(q, p, strict=True)], dim=1)
    original = old["measurements"]["semantic"]
    max_difference = (margins[:, 0] - original["native_margin"][rows]).abs().max().item()
    if max_difference > 0.003 or not margins.isfinite().all():
        raise ValueError(f"Frozen-native cosine replay differs by {max_difference}")
    output = args.output / f"{args.split}-{args.part}-head.pt"
    save_tensor({"rows": rows, "labels": qlabels[rows], "endpoints": pair, "readouts": list(READOUTS),
                 "semantic_margin": original["semantic_margin"][rows], "margins": margins}, output)
    meta = {"revision": args.revision, "split": args.split, "part": args.part, "seed": SEED,
            "checkpoint_sha256": manifest["sha256"], "selected_pairs_sha256": old_fingerprint,
            "queries": len(rows), "distinct_endpoint_photos": len(photos),
            "projector": head.info, "native_replay_max_abs": max_difference,
            "seconds": time.monotonic()-start, "device": torch.cuda.get_device_name(), "sha256": checksum(output)}
    atomic_json(output.with_suffix(".json"), meta)
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--split", choices=SKETCHY_SPLITS, required=True)
    parser.add_argument("--part", type=int, choices=(0, 1), required=True)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=DST)
    deterministic()
    audit(parser.parse_args())
