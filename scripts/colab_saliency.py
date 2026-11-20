"""Run the saliency-coordinate study from a clean Colab GPU runtime or resume a verified run."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

from colab_crossdataset import prepare_tu
from colab_paired import checkout, prepare


METHOD = "saliency-coordinate-transfer"
REV = "5b0f2b4614d6a4bd9b0f251f13d03a95780dc2cb"
CONTROL_REV = "ead74af1c12e6704f8ace05de10dc115afcd5d13"


def active_worker(workspace):
    marker = workspace / "saliency-study.pid"
    if marker.is_file():
        process = Path("/proc") / marker.read_text().strip() / "cmdline"
        return process.is_file() and b"saliency_study.py" in process.read_bytes()
    return False


def status(workspace):
    results = workspace / "results"
    state = workspace / "saliency-stage.json"
    print(
        json.dumps(
            {
                "worker": json.loads(state.read_text()) if state.is_file() else None,
                "sketchy": {
                    s: len(list((results / "saliency-sketchy").glob(s + "-*.json")))
                    for s in ("sketchy_1", "sketchy_2")
                },
                "locked": (results / "saliency-sketchy-to-tuberlin-lock.json").is_file(),
                "tuberlin": len(list((results / "saliency-tuberlin-frozen").glob("tuberlin-*.json"))),
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
        raise RuntimeError("Select a Colab GPU runtime")
    workspace.mkdir(parents=True, exist_ok=True)
    handle = (workspace / "saliency-run.lock").open("a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        status(workspace)
        return
    if active_worker(workspace):
        status(workspace)
        return
    repo = Path(__file__).resolve().parents[1]
    prepare(workspace, repo)
    prepare_tu(workspace)
    tree = checkout(workspace, repo, METHOD, REV)
    control_tree = checkout(workspace, repo, METHOD, CONTROL_REV)
    logs = workspace / "logs"
    logs.mkdir(exist_ok=True)
    env = os.environ | {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    }
    parallel = 2 if torch.cuda.mem_get_info()[0] >= 18 * 2**30 and len(os.sched_getaffinity(0)) >= 6 else 1

    def invoke(module, *params, source=tree):
        label = "-".join(str(k) for k in params if not str(k).startswith("--"))
        with (logs / ("saliency-reproduce-" + label + ".log")).open("ab") as f:
            subprocess.run(
                [sys.executable, "-u", "-m", "tf_sbir." + module, "--workspace", str(workspace), *params],
                cwd=source,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
            )

    results = workspace / "results"
    lock = results / "saliency-sketchy-to-tuberlin-lock.json"
    results.mkdir(exist_ok=True)
    control_plan = results / "saliency-control-plan.json"
    if not control_plan.exists():
        if list((results / "saliency-tuberlin-frozen").glob("tuberlin-*.json")):
            raise ValueError("This study already measured TU before registering the position control")
        from datetime import datetime, timezone

        control_plan.write_text(
            json.dumps(
                {
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "prior_tu_rows_for_this_method": 0,
                    "control_revision": CONTROL_REV,
                    "plan": "Score the two locked Sketchy winners with matched region extent and center both crops at the image center.",
                },
                indent=2,
            )
        )
    if not lock.is_file():
        invoke("saliency_views", "--dataset", "sketchy")
        for stage in ("sketchy", "seed", "refine", "freeze"):
            print(stage, flush=True)
            if stage in ("sketchy", "refine"):
                with ThreadPoolExecutor(max_workers=parallel) as pool:
                    list(
                        pool.map(
                            lambda s: invoke("saliency_benchmark", "--stage", stage, "--split", s),
                            ("sketchy_1", "sketchy_2"),
                        )
                    )
            else:
                invoke("saliency_benchmark", "--stage", stage)
    else:
        invoke("saliency_benchmark", "--stage", "freeze")
    invoke("saliency_views", "--dataset", "tuberlin")
    invoke("saliency_benchmark", "--stage", "heldout")
    for dataset in ("sketchy", "tuberlin"):
        invoke("saliency_views", "--dataset", dataset, "--at-center", source=control_tree)
        invoke("saliency_controls", "--dataset", dataset, "--at-center", source=control_tree)
    invoke("saliency_controls", "--dataset", "sketchy", source=control_tree)
    status(workspace)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    parser.add_argument("--status", action="store_true")
    options = parser.parse_args()
    (status if options.status else run)(options.workspace)
