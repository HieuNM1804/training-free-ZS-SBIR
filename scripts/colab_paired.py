"""Run the paper-backed Sketchy comparisons on an attached Google Colab GPU."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request


VISUAL = ("visual-paired-graph", "c268df56c1eda43e0ff288abcb9ecb46e5189072", "visual-graph")
REFLECTION = ("reflection-paired-transfer", "19bc037c54d8953c5b11f3f6247f2e98a47528f7", "reflection-controls")
BALANCE = ("gallery-semantic-balance", "7f411f8f9d56479b45054d105c5ac184438574ae", "gallery-balance")
SCALE = ("paired-neighborhood-scale", "c2036a38218886e17bc6a49ffd69405f017fa695", "paired-scale")
CHECKPOINT = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"


def prepare(workspace, repo):
    ready = workspace / "ready.json"
    if ready.is_file():
        root = Path(json.loads(ready.read_text())["root"])
    else:
        os.environ["KAGGLEHUB_CACHE"] = str(workspace / "datasets")
        os.environ["DISABLE_COLAB_CACHE"] = "true"
        import kagglehub
        dataset = Path(kagglehub.dataset_download("b20dccn616nguynhutun/sketchy/versions/1"))
        root = dataset / "Sketchy" if (dataset / "Sketchy").is_dir() else dataset
    if not all((root / mode / "cup").is_dir() for mode in ("sketch", "photo")):
        raise ValueError(f"Sketchy class folders are missing under {root}")
    assets = workspace / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    checkpoint, words = assets / "ViT-B-32.pt", assets / "open-images-v7-classes.csv"
    for path, url in [(checkpoint, f"https://openaipublic.azureedge.net/clip/models/{CHECKPOINT}/ViT-B-32.pt"),
                      (words, "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv")]:
        if not path.is_file():
            pending = path.with_suffix(".part")
            urllib.request.urlretrieve(url, pending)
            pending.replace(path)
    with checkpoint.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != CHECKPOINT:
            raise ValueError(f"Unexpected OpenAI CLIP checkpoint at {checkpoint}")
    ready.write_text(json.dumps({"root": str(root), "repo": str(repo), "features": str(workspace / "features")}))
    return root, checkpoint, words


def checkout(workspace, repo, branch, revision):
    dest = workspace / "worktrees" / (branch + "-" + revision[:10])
    if not dest.is_dir():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "-C", str(repo), "fetch", "--quiet", "origin", "research/" + branch], check=True)
        subprocess.run(["git", "-C", str(repo), "worktree", "add", "--detach", str(dest), revision], check=True)
    found = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
    if found != revision:
        raise ValueError(f"Unexpected revision in {dest}")
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    parser.add_argument("--phase", choices=("visual", "reflection", "balance", "scale"), required=True)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--crop-batch-size", type=int, default=384)
    parser.add_argument("--score-batch", type=int, default=192)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--metric-workers", type=int, default=4)
    args = parser.parse_args()
    if sys.platform != "linux" or not Path("/content").is_dir():
        raise RuntimeError("Choose a Google Colab GPU kernel")
    import google.colab  # The Colab host must provide its runtime package.
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("The connected Colab runtime needs a GPU")
    if importlib.metadata.version("torchmetrics") != "1.8.2":
        raise RuntimeError("Install the pinned repository requirements on the Colab runtime")
    repo = Path(__file__).resolve().parents[1]
    root, checkpoint, words = prepare(args.workspace, repo)
    branch, revision, stage = {"visual": VISUAL, "reflection": REFLECTION, "balance": BALANCE, "scale": SCALE}[args.phase]
    method = checkout(args.workspace, repo, branch, revision)
    output = args.workspace / "results" / stage
    output.mkdir(parents=True, exist_ok=True)
    cpu = len(os.sched_getaffinity(0))
    command = [sys.executable, "-u", "-m", "tf_sbir.transfer_benchmark", "--stage", stage,
               "--root", str(root), "--cache", str(args.workspace / "features"), "--clip-checkpoint", str(checkpoint),
               "--output", str(output), "--batch-size", str(args.batch_size), "--crop-batch-size", str(args.crop_batch_size),
               "--score-batch", str(args.score_batch), "--workers", str(min(cpu, args.workers)),
               "--metric-workers", str(min(cpu, args.metric_workers)), "--exploratory-unseen"]
    if args.phase != "visual":
        command.extend(["--vocabulary", str(words)])
    print(torch.cuda.get_device_name(), branch, revision, root, flush=True)
    env = os.environ | {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    subprocess.run(command, cwd=method, env=env, check=True)


if __name__ == "__main__":
    main()
