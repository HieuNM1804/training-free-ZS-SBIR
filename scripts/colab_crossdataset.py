"""Tune on both Sketchy splits and evaluate only their frozen choices on TU-Berlin."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from colab_paired import checkout, prepare


DOMAIN = ("domain-prompt-gallery-transport", "f06e76dc5412261706911fa4572dd5130188b5e0", "crossdomain_benchmark")
STRUCTURE = ("universal-self-similarity", "81116309a3d589772cca5458c52ed0c4625c5eb0", "structure_benchmark")
STUDIES = {"domain": DOMAIN, "structure": STRUCTURE}
LOCKS = {"domain": "sketchy-to-tuberlin-lock.json", "structure": "structure-sketchy-to-tuberlin-lock.json"}
TOGETHER = "both-sketchy-locks-before-tu.json"
PLAN = {
    "primary_rule": "Across frame-images, domain-prompt and universal-structure families select the eligible configuration with the highest equal-weight mean precision on Sketchy Ext-1 and Ext-2, then equal-weight mean mAP. Ignore TU scores.",
    "selection_datasets": ["sketchy_1", "sketchy_2"],
    "test_dataset": "tuberlin",
    "subgroup": "Partition TU queries by whether the normalized class name occurs in the two Sketchy splits. Keep the full 27262-photo TU gallery for both groups.",
    "contrasts": [
        "general methods vs native-clip",
        "structure vs matched four-view visual control",
        "primary general method vs historical contour visual and router",
        "paired vs shuffled and domain vs neutral prompts on Sketchy",
    ],
}


def register_plan(folder):
    path = folder / "crossdataset-analysis-plan.json"
    if path.is_file():
        old = json.loads(path.read_text())
        if old["plan"] != PLAN or old["tu_rows_when_declared"] != [0, 0]:
            raise ValueError("The analysis must follow the plan recorded before TU scoring")
        return
    counts = [result_count(folder / (n + "-tuberlin-frozen"), "tuberlin") for n in STUDIES]
    if any(counts):
        raise ValueError("TU data already scored before the analysis plan")
    path.write_text(
        json.dumps(
            {"utc": datetime.now(timezone.utc).isoformat(), "tu_rows_when_declared": counts, "plan": PLAN}, indent=2
        )
    )


def signed_lock(folder, name):
    data = json.loads((folder / LOCKS[name]).read_text())
    unsigned = data.copy()
    expected = unsigned.pop("freeze_sha256")
    raw = json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    splits = data.get("selection_splits", data.get("data_used_for_selection"))
    if (
        hashlib.sha256(raw).hexdigest() != expected
        or data["revision"] != STUDIES[name][1]
        or splits != ["sketchy_1", "sketchy_2"]
    ):
        raise ValueError(f"Invalid Sketchy lock for {name}")
    return data


def result_names(name, lock):
    if name == "structure":
        return set(lock["heldout_controls"]) | {lock["best_including_ablation"], lock["best_using_structure"]}
    return {"native-clip", *(r["name"] for r in lock["winners"].values()), *lock["previous_sketchy_winners"]}


def result_count(folder, split):
    return len(list(folder.glob(split + "-*.json")))


def status(workspace):
    results = workspace / "results"
    state = {}
    for name in STUDIES:
        file = workspace / (name + "-state.json")
        current = json.loads(file.read_text()) if file.is_file() else {}
        state[name] = {
            "sketchy": [result_count(results / (name + "-sketchy"), s) for s in ("sketchy_1", "sketchy_2")],
            "locked": (results / LOCKS[name]).is_file(),
            "tu": result_count(results / (name + "-tuberlin-frozen"), "tuberlin"),
            "worker": current,
        }
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
    return state


def prepare_tu(workspace):
    marker = workspace / "tuberlin-data.json"
    if marker.is_file():
        root = Path(json.loads(marker.read_text())["root"])
    else:
        os.environ["KAGGLEHUB_CACHE"] = str(workspace / "datasets")
        os.environ["DISABLE_COLAB_CACHE"] = "true"
        import kagglehub

        root = Path(kagglehub.dataset_download("b20dccn616nguynhutun/tuberlin/versions/1"))
    if not all(
        (root / mode / category).is_dir() for mode in ("sketch", "photo") for category in ("wrist-watch", "tv")
    ):
        raise ValueError(f"Missing TU-Berlin benchmark class folders under {root}")
    if not marker.is_file():
        marker.write_text(json.dumps({"root": str(root)}))


def bind_selections(folder):
    locks = {name: signed_lock(folder, name) for name in STUDIES}
    path = folder / TOGETHER
    signatures = {LOCKS[n]: v["freeze_sha256"] for n, v in locks.items()}
    if path.is_file():
        saved = json.loads(path.read_text())
        if saved["locks"] != signatures or saved["no_tu_rows_at_release"] != [0, 0]:
            raise ValueError("Stored joint freeze does not match the Sketchy selections")
    else:
        counts = [result_count(folder / (n + "-tuberlin-frozen"), "tuberlin") for n in STUDIES]
        if any(counts):
            raise ValueError("TU measurements already exist without the joint freeze record")
        record = {
            "utc_before_tu": datetime.now(timezone.utc).isoformat(),
            "no_tu_rows_at_release": counts,
            "locks": signatures,
            "revisions": {n: v["revision"] for n, v in locks.items()},
        }
        temp = path.with_suffix(".pending")
        temp.write_text(json.dumps(record, indent=2) + "\n")
        temp.replace(path)
    return locks


def heldout_done(folder, name, lock):
    dest = folder / (name + "-tuberlin-frozen")
    rows = [json.loads(p.read_text()) for p in dest.glob("tuberlin-*.json")]
    allowed = result_names(name, lock)
    for r in rows:
        if (
            r["method"] not in allowed
            or r["freeze_sha256"] != lock["freeze_sha256"]
            or r["revision"] != lock["revision"]
            or r["exploratory_unseen"]
            or (r["queries"], r["gallery"], r["p_k"], r["map_k"]) != (2130, 27262, 100, 0)
        ):
            raise ValueError(f"Invalid frozen TU row in {dest}")
    return {r["method"] for r in rows} == allowed and all(
        (dest / "per-query" / f"tuberlin-{n}.pt").is_file() for n in allowed
    )


def run(workspace):
    import fcntl
    from importlib.util import find_spec
    import torch

    if find_spec("google.colab") is None or not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime in Google Colab")
    workspace.mkdir(parents=True, exist_ok=True)
    guard = (workspace / "crossdataset-run.lock").open("a")
    try:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        status(workspace)
        return
    state = status(workspace)
    for name, item in state.items():
        worker = item["worker"]
        if worker.get("phase") in ("checkout", "sketchy", "freeze", "heldout"):
            cmdline = Path("/proc") / str(worker.get("pid", 0)) / "cmdline"
            if cmdline.is_file() and any(
                x in cmdline.read_bytes() for x in (b"run_crossdomain.py", b"run_universal_structure_study.py")
            ):
                print(f"An existing Colab worker is processing {name}.", flush=True)
                return
    repo = Path(__file__).resolve().parents[1]
    folder = workspace / "results"
    folder.mkdir(exist_ok=True)
    register_plan(folder)
    prepare(workspace, repo)
    prepare_tu(workspace)
    trees = {n: checkout(workspace, repo, b, r) for n, (b, r, _) in STUDIES.items()}
    env = os.environ | {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    }
    workers = 2 if torch.cuda.mem_get_info()[0] >= 19 * 2**30 and len(os.sched_getaffinity(0)) >= 6 else 1

    def warm(dataset):
        code = """from pathlib import Path
import sys, torch
from tf_sbir.crossdomain_inputs import runtime, projection, inputs
w=Path(sys.argv[1]); dataset=sys.argv[2]
cfg=runtime(w,dataset,batch=64,workers=2,crop_batch=192); cfg.crop_workers=6
with torch.inference_mode():
    projection(cfg)
    inputs(cfg,dataset,prepare_sketch_controls=dataset=='tuberlin')"""
        subprocess.run(
            [sys.executable, "-c", code, str(workspace), dataset], cwd=trees["domain"], env=env, check=True
        )

    def stage(name, phase):
        log = workspace / "logs" / f"crossdataset-{name}-{phase}.log"
        log.parent.mkdir(exist_ok=True)
        command = [
            sys.executable,
            "-u",
            "-m",
            "tf_sbir." + STUDIES[name][2],
            "--workspace",
            str(workspace),
            "--stage",
            phase,
        ]
        print(name, phase, log, flush=True)
        with log.open("ab") as output:
            subprocess.run(command, cwd=trees[name], env=env, stdout=output, stderr=subprocess.STDOUT, check=True)

    def run_stage(names, phase):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda n: stage(n, phase), names))

    pending = [n for n in STUDIES if not (folder / LOCKS[n]).is_file()]
    if pending:
        warm("sketchy")
        run_stage(pending, "sketchy")
    run_stage(STUDIES, "freeze")
    locks = bind_selections(folder)
    pending = [n for n, lock in locks.items() if not heldout_done(folder, n, lock)]
    if pending:
        warm("tuberlin")
        run_stage(pending, "heldout")
    if not all(heldout_done(folder, n, lock) for n, lock in locks.items()):
        raise RuntimeError("Incomplete TU-Berlin results")
    status(workspace)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    parser.add_argument("--status", action="store_true")
    options = parser.parse_args()
    if options.status:
        status(options.workspace)
    else:
        run(options.workspace)
