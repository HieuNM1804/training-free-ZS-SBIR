"""Restore the two shared RGB gallery inputs on a fresh Colab runtime."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from .conditional_semantic_index import bare_words
from .cross_diffusion_map import Config, Index
from .encoding import save_tensor
from .foundation import load_words
from .retrieval_diagnosis import global_inputs
from .semantic import read_words
from .shared_concept_likelihood import emission


@torch.inference_mode()
def run(workspace, dataset):
    cfg, _, queries, _, galleries, qraw, praw = global_inputs(workspace, dataset)
    words = read_words(cfg.vocabulary)
    source, target = (v.cuda() for v in load_words(cfg, words))
    _, scale = bare_words(cfg, words)
    settings = Config(neighbors=16, bandwidth=.36, iterations=2, mode="rgb")
    for split, (qi, _) in queries.items():
        index_path = workspace / "cache/pose-conditional-layout" / (split + "-gallery-uniform.pt")
        scores_path = workspace / "cache/pragmatic-gallery-competition" / (split + "-named-rgb16.pt")
        pi, _ = galleries[split]
        if index_path.is_file() and scores_path.is_file():
            continue
        p = praw[pi].cuda()
        graph = Index(SimpleNamespace(p=p), rgb_only=True)
        probability = ((p @ target.T) * scale).softmax(1)
        if not index_path.is_file():
            projected = torch.empty_like(probability)
            for start in range(0, len(words), 384):
                block = slice(start, start+384)
                projected[:,block] = graph.score(settings, probability[:,block].T.contiguous()).T
            projected /= projected.sum(1, keepdim=True)
            save_tensor(projected.cpu(), index_path)
            del projected
        if not scores_path.is_file():
            rows = []
            for q in qraw[qi].cuda().split(256):
                rows.append(graph.score(settings, emission(q, source, scale) @ probability.T).cpu())
            save_tensor(torch.cat(rows), scores_path)
        # The two ways of applying the same image operator must agree on actual query/candidate ranks.
        archive = torch.load(index_path, mmap=True, weights_only=True)
        base = torch.load(scores_path, mmap=True, weights_only=True)
        if archive.shape != (len(pi), len(words)) or base.shape != (len(qi), len(pi)):
            raise ValueError(f"Incomplete RGB inputs for {split}")
        qcheck = torch.arange(0, len(qi), max(1,len(qi)//11))[:11]
        pcheck = torch.arange(0, len(pi), max(1,len(pi)//129))[:129]
        direct = emission(qraw[qi[qcheck]].cuda(), source, scale) @ archive[pcheck].cuda().T
        expected = base[qcheck][:,pcheck].cuda()
        direct /= direct.amax(1,keepdim=True).clamp_min(1e-30)
        expected /= expected.amax(1,keepdim=True).clamp_min(1e-30)
        error = float((direct-expected).abs().max())
        print(split, "ready", len(qi), len(pi), "max normalized composition error", error, flush=True)
        if error > .005:
            raise ValueError("Gallery inputs do not reproduce their shared RGB operator")
        del graph, probability


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/content/sbir"))
    parser.add_argument("--dataset", choices=("sketchy","tuberlin"), required=True)
    args = parser.parse_args()
    run(args.workspace, args.dataset)
