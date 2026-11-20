"""Reproduce the jointly frozen saliency and phase comparison on a Google Colab GPU."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib, json, os
from pathlib import Path
import subprocess, sys

from colab_crossdataset import prepare_tu
from colab_paired import checkout, prepare
from colab_saliency import METHOD, REV, CONTROL_REV, active_worker

PHASE = "phase-agreement-retrieval"
PHASE_REV = "1bd02a9ae257005e79a9069d76021a7184e1b34f"
JOINT = "saliency-phase-both-locks.json"
STUDIES = {"saliency": (METHOD, REV), "phase": (PHASE, PHASE_REV)}
TU_FOLDERS = ("saliency-tuberlin-frozen", "saliency-tuberlin-center", "phase-tuberlin-frozen")


def tu_counts(results):
    return {n: len(list((results / n).glob("tuberlin-*.json"))) for n in TU_FOLDERS}


def combined_lock(results):
    signatures = {}
    for n, (_, rev) in STUDIES.items():
        data = json.loads((results / (n + "-sketchy-to-tuberlin-lock.json")).read_text())
        sig = data.pop("freeze_sha256")
        actual = hashlib.sha256(
            json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if (
            actual != sig
            or data["revision"] != rev
            or data["selection_splits"] != ["sketchy_1", "sketchy_2"]
            or data["tu_classes"] != 30
        ):
            raise ValueError(n)
        signatures[n] = sig
    file = results / JOINT
    if file.is_file():
        result = json.loads(file.read_text())
        if result["locks"] != signatures or result["tu_rows_before_release"] != {n: 0 for n in TU_FOLDERS}:
            raise ValueError("Invalid joint freeze")
        return result
    counts = tu_counts(results)
    if any(counts.values()):
        raise ValueError("TU was scored before both methods were frozen")
    result = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "locks": signatures,
        "tu_rows_before_release": counts,
    }
    tmp = file.with_suffix(".pending")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(file)
    return result


def running(workspace):
    if active_worker(workspace):
        return True
    for n, script in (("phase-study", "phase_study.py"), ("saliency-control", "saliency_diagnostics.py")):
        f = workspace / (n + ".pid")
        if f.exists():
            p = Path("/proc") / f.read_text().strip() / "cmdline"
            if p.is_file() and script.encode() in p.read_bytes():
                return True
    return False


def status(workspace):
    results = workspace / "results"
    print(
        json.dumps(
            {
                n: {
                    "sketchy": [
                        len(list((results / (n + "-sketchy")).glob(s + "-*.json")))
                        for s in ("sketchy_1", "sketchy_2")
                    ],
                    "frozen": (results / (n + "-sketchy-to-tuberlin-lock.json")).is_file(),
                }
                for n in STUDIES
            }
            | {
                "tu": tu_counts(results),
                "joint_freeze": (results / JOINT).is_file(),
                "active_worker": running(workspace),
            },
            indent=2,
        ),
        flush=True,
    )


def run(workspace):
    import fcntl
    from importlib.util import find_spec
    import torch

    if find_spec("google.colab") is None or not torch.cuda.is_available():
        raise RuntimeError("Select a Colab GPU")
    workspace.mkdir(parents=True, exist_ok=True)
    guard = (workspace / "region-studies.lock").open("a")
    try:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        status(workspace)
        return
    if running(workspace):
        status(workspace)
        return
    repo = Path(__file__).resolve().parents[1]
    results = workspace / "results"
    results.mkdir(exist_ok=True)
    plans = {
        "saliency-control-plan.json": {
            "prior_tu_rows_for_this_method": 0,
            "control_revision": CONTROL_REV,
            "plan": "Fix both winners and their crop sizes and replace the saliency center with the geometric image center.",
        },
        "saliency-phase-comparison-plan.json": {
            "tu_rows": {n: 0 for n in TU_FOLDERS},
            "selection_splits": ["sketchy_1", "sketchy_2"],
            "tu_classes": 30,
            "choice": "Choose across the eligible saliency and phase methods by the equal-split mean precision on both Sketchy splits, then mAP.",
        },
    }
    for n, plan in plans.items():
        path = results / n
        if not path.is_file():
            if any(tu_counts(results).values()):
                raise ValueError("A registration record is missing for this round")
            path.write_text(json.dumps(plan | {"utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    prepare(workspace, repo)
    prepare_tu(workspace)
    trees = {n: checkout(workspace, repo, b, rev) for n, (b, rev) in STUDIES.items()}
    trees["control"] = checkout(workspace, repo, METHOD, CONTROL_REV)
    env = os.environ | {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    }
    logs = workspace / "logs"
    logs.mkdir(exist_ok=True)

    def invoke(n, module, *params):
        label = "-".join(v for v in params if not v.startswith("--"))
        with (logs / f"regions-{n}-{module}-{label}.log").open("ab") as f:
            subprocess.run(
                [sys.executable, "-u", "-m", "tf_sbir." + module, "--workspace", str(workspace), *params],
                cwd=trees[n],
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
            )

    def tune(n):
        if (results / (n + "-sketchy-to-tuberlin-lock.json")).is_file():
            return
        invoke(n, n + "_views", "--dataset", "sketchy")
        for stage in ("sketchy", "seed", "refine", "freeze") if n == "saliency" else ("sketchy", "freeze"):
            print(n, stage, flush=True)
            invoke(n, n + "_benchmark", "--stage", stage)

    parallel = 2 if torch.cuda.mem_get_info()[0] >= 18 * 2**30 and len(os.sched_getaffinity(0)) >= 6 else 1
    # Prepare the shared native-image cache once before concurrent methods read it.
    for dataset in ("sketchy", "tuberlin"):
        init = """from pathlib import Path
import sys,torch
from tf_sbir.crossdomain_inputs import runtime
from tf_sbir.benchmark import merge_inputs
from tf_sbir.data import SKETCHY_SPLITS
from tf_sbir.encoding import load_global_inputs
w=Path(sys.argv[1]); ds=sys.argv[2]; cfg=runtime(w,ds,batch=64,workers=2)
splits=SKETCHY_SPLITS if ds=='sketchy' else ('tuberlin',)
q,_=merge_inputs(cfg.root,splits,'sketch'); p,_=merge_inputs(cfg.root,splits,'photo')
with torch.inference_mode():load_global_inputs(cfg,q,p,False)"""
        subprocess.run(
            [sys.executable, "-c", init, str(workspace), dataset], cwd=trees["phase"], env=env, check=True
        )
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        list(pool.map(tune, STUDIES))
    combined_lock(results)

    def held(n):
        invoke(n, n + "_views", "--dataset", "tuberlin")
        invoke(n, n + "_benchmark", "--stage", "heldout")

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        list(pool.map(held, STUDIES))
    for ds in ("sketchy", "tuberlin"):
        invoke("control", "saliency_views", "--dataset", ds, "--at-center")
        invoke("control", "saliency_controls", "--dataset", ds, "--at-center")
    invoke("control", "saliency_controls", "--dataset", "sketchy")
    status(workspace)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    parser.add_argument("--status", action="store_true")
    a = parser.parse_args()
    (status if a.status else run)(a.workspace)
