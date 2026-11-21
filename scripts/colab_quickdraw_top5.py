"""Run the three remaining frozen Top 5 methods on the full QuickDraw split."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch


METHODS = (
    ("neighborhood-calibrated-likelihood", "research/neighborhood-calibrated-likelihood",
     "tf_sbir.quickdraw_neighborhood_calibrated"),
    ("concept-translation-retrieval", "research/concept-translation-retrieval",
     "tf_sbir.quickdraw_concept_translation"),
    ("cross-language-relevance", "research/cross-language-relevance",
     "tf_sbir.quickdraw_cross_language_relevance"),
)


def worktree_for(workspace, repository, branch, family):
    subprocess.run(["git", "-C", str(repository), "fetch", "origin", branch], check=True)
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", f"origin/{branch}"], text=True
    ).strip()
    destination = workspace / "worktrees" / f"quickdraw-top5-{family}-{commit[:12]}"
    if not destination.is_dir():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "-C", str(repository), "worktree", "add", "--detach", str(destination), commit],
            check=True,
        )
    actual = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
    if actual != commit:
        raise RuntimeError(f"Unexpected checkout for {family}")
    return destination


def run_method(workspace, repository, family, branch, module, metric_workers):
    driver = worktree_for(workspace, repository, branch, family)
    log_path = workspace / "logs" / "quickdraw-top5" / f"{family}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    command = [sys.executable, "-u", "-m", module, "--workspace", str(workspace),
               "--metric-workers", str(metric_workers)]
    print(f"Starting {family} from {branch} at {driver.name}", flush=True)
    with log_path.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(command, cwd=driver, env=environment, stdout=output, stderr=subprocess.STDOUT)
        cursor = 0
        while process.poll() is None:
            time.sleep(30)
            output.flush()
            with log_path.open(encoding="utf-8") as saved:
                saved.seek(cursor)
                update = saved.read()
                cursor = saved.tell()
            if update:
                print(update, end="" if update.endswith("\n") else "\n", flush=True)
            gpu = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"],
                text=True,
            ).strip()
            print(f"{family} GPU utilization,memory MiB,power W: {gpu}", flush=True)
        output.flush()
    if process.returncode:
        print(log_path.read_text(encoding="utf-8", errors="replace")[-12000:], flush=True)
        raise subprocess.CalledProcessError(process.returncode, command)

    result_path = workspace / "results" / "quickdraw-top5" / family / "quickdraw.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["verified_queries"] != 91491 or result["gallery"] != 56407:
        raise RuntimeError(f"Incomplete QuickDraw result for {family}")
    print(f"Completed {family}: mAP@all={result['mAP']:.6f}, P@200={result['precision']:.6f}", flush=True)
    return result


def run(workspace, repository, metric_workers):
    if not torch.cuda.is_available() or not Path("/content").is_dir():
        raise RuntimeError("Run the Top 5 QuickDraw comparison in the connected Colab GPU runtime")
    workspace = workspace.resolve()
    repository = repository.resolve()
    print(f"Google Colab GPU: {torch.cuda.get_device_name()}", flush=True)
    print(f"Metric worker threads: {metric_workers}", flush=True)
    summaries = {}
    for family, branch, module in METHODS:
        summaries[family] = run_method(workspace, repository, family, branch, module, metric_workers)
    summary_path = workspace / "results" / "quickdraw-top5" / "remaining-three.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Saved summary to {summary_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    parser.add_argument("--repository", type=Path, default=Path("/content/sbir/training-free-ZS-SBIR"))
    parser.add_argument("--metric-workers", type=int, default=min(len(os.sched_getaffinity(0)), 16))
    options = parser.parse_args()
    run(options.workspace, options.repository, options.metric_workers)
