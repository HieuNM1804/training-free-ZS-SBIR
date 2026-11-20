"""Select between both completed Sketchy studies before either study scores TU-Berlin."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from .concept_benchmark import atomic_json, read
from .crossdomain_benchmark import digest


def freeze(root):
    file = root / "inquiry-joint-freeze.json"
    if file.exists():
        record = read(file)
        signature = record.pop("sha256")
        if signature != digest(record):
            raise ValueError("Invalid joint freeze")
        return record["primary"]
    studies = ("calibrated-concept-retrieval", "reverse-prompt-diffusion")
    splits = ["sketchy_1", "sketchy_2"]
    families = {}
    for study in studies:
        folder = root / study
        lock = read(folder / "freeze.json")
        signature = lock.pop("freeze_sha256")
        if signature != digest(lock) or any((folder / "heldout").glob("tuberlin-*.json")):
            raise ValueError(f"{study} requires a valid Sketchy-only lock")
        scores = []
        for split in splits:
            rows = (read(p) for p in (folder / "sketchy").glob(split + "-*.json"))
            chosen, = (r for r in rows if r["parameters"] == lock["winner"])
            if chosen["revision"] != lock["revision"]:
                raise ValueError("Sketchy score revision mismatch")
            scores.append({field: chosen[field] for field in ("split", "method", "precision", "mAP")})
        families[study] = {"revision": lock["revision"], "freeze_sha256": signature,
                           "tu_rows_prelock": 0, "scores": scores}
    def criterion(study):
        rows = families[study]["scores"]
        return tuple(sum(r[field] for r in rows) / len(rows) for field in ("precision", "mAP"))
    primary = max(studies, key=criterion)
    record = {"utc": datetime.now(timezone.utc).isoformat(),
              "rule": "max equal mean Sketchy precision, then mean mAP", "selection_splits": splits,
              "families": families, "primary": primary}
    atomic_json(file, record | {"sha256": digest(record)})
    return primary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    print(freeze(parser.parse_args().workspace / "results"))
