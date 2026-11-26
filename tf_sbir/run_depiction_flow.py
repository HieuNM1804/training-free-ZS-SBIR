"""Run the predeclared numerical and retrieval stages on an offline four-GPU Kaggle session."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .kaggle_unseen_features import DEFAULT_CACHE

FAMILY = "depiction-state-flow"


def run(revision, start_stage):
    root = DEFAULT_CACHE.parent
    logs = root / ("logs-" + FAMILY)
    logs.mkdir(exist_ok=True)
    command = [sys.executable, "-u", "-m", "tf_sbir.depiction_flow_study"]
    stages = ("protocol", "numeric", "scan", "lock", "official")
    tasks = {
        "protocol": [[]],
        "numeric": [["--depth", str(d)] for d in (1, 3, 6)],
        "scan": [["--split", s, "--part", str(p)] for s in ("sketchy_1", "sketchy_2") for p in (0, 1)],
        "lock": [[]],
        "official": [["--split", s] for s in ("sketchy_1", "sketchy_2", "tuberlin")],
    }
    for stage in stages[stages.index(start_stage):]:
        jobs = []
        for gpu, extra in enumerate(tasks[stage]):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=":4096:8",
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONHASHSEED="42")
            with (logs/f"{stage}-{gpu}.log").open("w") as out:
                proc = subprocess.Popen(command + [stage, "--revision", revision, *extra], env=env,
                                        stdout=out, stderr=subprocess.STDOUT)
            jobs.append(proc)
        print(stage, [p.pid for p in jobs], flush=True)
        codes = [p.wait() for p in jobs]
        if any(codes):
            raise RuntimeError(f"{stage} exit codes {codes}")
        if stage == "numeric":
            work = root/"results"/FAMILY
            records = [json.loads((work/f"numerics-{d}.json").read_bytes()) for d in (1, 3, 6)]
            if any(r["steps"] is None for r in records):
                print("Stopping at the predeclared convergence gate.", flush=True)
                return
    print("Full unseen replay complete.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revision")
    parser.add_argument("--start", default="protocol", choices=("protocol", "numeric", "scan", "lock", "official"))
    args = parser.parse_args()
    run(args.revision, args.start)
