"""Freeze pairwise candidates using only the two existing Sketchy evaluations."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from .concept_benchmark import atomic_json, checksum, read
from .crossdomain_benchmark import digest
from .data import SKETCHY_SPLITS
from .map_study import score_key
from .public_word_calibration import DECISION


FAMILY = "public-word-calibration"
WINDOW = "winclip-concept-likelihood"
UNIFORM = "shared-concept-likelihood"


def freeze(workspace, include_translation=False):
    root = workspace / "results"
    translation = "public-semantic-translation"
    names = (UNIFORM, WINDOW, FAMILY, translation) if include_translation else (UNIFORM, WINDOW, FAMILY)
    path = root / (f"{translation}/strict-joint.json" if include_translation else DECISION)
    if path.exists():
        record = read(path)
        if record.pop("sha256") != digest(record):
            raise ValueError("Invalid strict selection signature")
        return record["primary"]
    for family in names[2:]:
        if any((root / family / "heldout").glob("tuberlin-*.json")):
            raise ValueError(f"TU results already exist for {family}")
    families = {}
    for name in names:
        folder = root / name
        lock = read(folder / ("protocol.json" if name == UNIFORM else "freeze.json"))
        signature = lock.pop("sha256")
        if signature != digest(lock):
            raise ValueError(f"Bad family lock {name}")
        chosen = "uniform" if name == UNIFORM else lock["winner_id"]
        scores = []
        for split in SKETCHY_SPLITS:
            file = folder / "sketchy" / f"{split}-{chosen}.json"
            row = read(file)
            file_hash = checksum(file)
            if name != UNIFORM and file_hash != lock["sources"][file.name]:
                raise ValueError(f"Sketchy row has changed {file}")
            scores.append({"split": split, "sha256": file_hash, "mAP": row["mAP"],
                           "precision": row["precision"]})
        families[name] = {"name": chosen, "freeze": signature, "scores": scores}
    primary = max(families, key=lambda name: score_key(families[name]["scores"]))
    record = {"utc": datetime.now(timezone.utc).isoformat(), "families": families, "primary": primary,
              "selection_data": list(SKETCHY_SPLITS), "new_tu_rows_at_lock": 0,
              "eligibility": "Scores depend only on one query, one photo, frozen CLIP and public resources",
              "rule": "equal mean Sketchy mAP then equal mean precision"}
    atomic_json(path, record | {"sha256": digest(record)})
    return primary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    parser.add_argument("--include-translation", action="store_true")
    args = parser.parse_args()
    print(freeze(args.workspace, args.include_translation))
