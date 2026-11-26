"""Fit a shared lexical prior from paired prompt readings of unlabelled photos."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import random
import zipfile

import numpy as np
import torch

from .branch_view_study import PrefixReplay
from .concept_benchmark import atomic_json
from .crossdomain_benchmark import digest, revision
from .crossdomain_inputs import CHECKPOINT
from .depiction_path_diagnostic import class_interval
from .encoding import save_tensor
from .geometry_word_flat import COUNTS, SHARD
from .locked_geometry_words import score_rows
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .semantic_disputed_word_risk import inputs, query_posteriors, word_bank
from .shared_concept_likelihood import fit_prior


FAMILY = 'paired-semantic-prior'
SEED = 42
BATCH = 64
ARMS = ('S', 'photo', 'source', 'mean', 'paired', 'permuted')


def prior_bank(p, a):
    estimates, reports = {}, {}

    def record(name, evidence):
        fit, diagnostics = fit_prior(evidence.T)
        estimates[name] = fit['mle']
        reports[name] = diagnostics

    record('photo', p)
    record('source', a)
    buffer = (p + a).mul_(.5)
    record('mean', buffer)
    order = torch.randperm(p.size(1), generator=torch.Generator().manual_seed(SEED)).to(p.device)
    permutation = torch.empty_like(order)
    permutation[order] = order.roll(1)
    assert (permutation != torch.arange(len(order), device=p.device)).all()
    for name in ('paired', 'permuted'):
        for start in range(0, p.size(1), 256):
            sl = slice(start, start + 256)
            paired = a[:, sl] if name == 'paired' else a[:, permutation[sl]]
            # Separate square roots avoid underflow before the geometric mean.
            buffer[:, sl] = p[:, sl].sqrt() * paired.sqrt()
        record(name, buffer)
    return torch.stack([estimates[k] for k in ARMS[1:]]), reports


def baseline_archive(split, qi, pi, labels):
    directory = Path(__file__).resolve().parents[1] / 'docs/results/semantic-disputed-word-risk'
    with zipfile.ZipFile(directory/'paired-T4.zip') as z:
        pack = torch.load(io.BytesIO(z.read(split+'.pt')), weights_only=True)
    for name, expected in (('qi', qi), ('pi', pi), ('labels', labels)):
        if not torch.equal(pack[name], expected):
            raise ValueError(f'Archived baseline changed in {split}/{name}')
    return {'S':pack['rows']['S'], 'photo':pack['rows']['S_gallery_prior']}


def freeze(root):
    rows = [json.loads((root/(s+'.json')).read_bytes()) for s in ('sketchy_1', 'sketchy_2')]
    if any(r['commit'] != revision() for r in rows):
        raise ValueError('Wrong results in the Sketchy lock')
    record = dict(commit=revision(), primary='paired', sketchy_recipes=[r['recipe'] for r in rows],
                  arms=ARMS)
    atomic_json(root/'freeze.json', record | {'sha': digest(record)})


@torch.inference_mode()
def run(dataset):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    w = Path('/content/sbir')
    root = w / 'results' / FAMILY
    lock = None
    if dataset == 'tuberlin':
        lock = json.loads((root/'freeze.json').read_bytes())
        if lock.pop('sha') != digest(lock) or lock['commit'] != revision():
            raise ValueError('Mismatched paired-prior lock')
    qs, gs, q, p, refl, source, target, scale, vocab = inputs(w, dataset)
    kd, _ = load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest() != KD_FUNCTION_SHA:
        raise ValueError('Changed KD-SBIR evaluator')
    for split, (qi, labels) in qs.items():
        pi, gallery_labels = gs[split]
        assert (len(qi), len(pi), len(labels.unique())) == COUNTS[split]
        signature = digest(dict(commit=revision(), device=torch.cuda.get_device_name(), vocab=vocab,
                                seed=SEED, lock=lock, q=qi.tolist(), p=pi.tolist()))
        output = root/(split+'.json')
        if output.exists() and output.with_suffix('.pt').exists():
            if json.loads(output.read_bytes()).get('recipe') == signature:
                continue
        rgb = p[pi].cuda()
        gallery = word_bank(rgb, target, scale)
        priors, stats = prior_bank(gallery, word_bank(rgb, source, scale))
        denominator = priors @ gallery
        if not (denominator > 0).all():
            raise ValueError('Nonpositive posterior normalizer')
        replay = PrefixReplay(kd, gallery_labels, split)
        classes = gallery_labels.cuda()
        shard_dir = root/split/signature
        shard_dir.mkdir(exist_ok=True, parents=True)
        print(split, 'ready', {k: v['effective_number_of_concepts'] for k, v in stats.items()}, flush=True)
        all_rows = {name: [] for name in ARMS}
        with ThreadPoolExecutor(max_workers=min(3, len(os.sched_getaffinity(0)))) as pool:
            for start in range(0, len(qi), SHARD):
                stop = min(start+SHARD, len(qi))
                shard = shard_dir/f'{start:05d}.pt'
                if not shard.exists():
                    blocks = {name: [] for name in ARMS}
                    for block in torch.arange(start, stop).split(BATCH):
                        ids = qi[block]
                        query = query_posteriors(q[ids].cuda(), refl[ids].cuda(), source, scale)
                        weighted = query[:, None] * priors[None]
                        tests = (weighted.flatten(0, 1) @ gallery).reshape(len(block), len(priors), len(pi))
                        tests = tests / denominator
                        scores = (query @ gallery, *tests.unbind(1))
                        if not all(score.isfinite().all() for score in scores):
                            raise ValueError('Nonfinite lexical scores')
                        for name, score in zip(ARMS, scores, strict=True):
                            blocks[name].append(score_rows(score, labels[block], classes, split, replay, pool)[:, :2])
                    save_tensor({'ids':qi[start:stop], 'rows':{k:torch.cat(v) for k,v in blocks.items()}}, shard)
                batch = torch.load(shard, weights_only=True)
                assert torch.equal(qi[start:stop], batch['ids'])
                for name in ARMS:
                    all_rows[name].append(batch['rows'][name])
                print(split, stop, '/', len(qi), flush=True)
        rows = {name: torch.cat(v) for name,v in all_rows.items()}
        replays = {}
        for name, old in baseline_archive(split, qi, pi, labels).items():
            difference = 100*(rows[name].double()-old.double()).abs().mean(0)
            if difference.max() > .04:
                raise ValueError(f'Changed {name} baseline: {difference.tolist()}')
            replays[name] = difference.tolist()
        contrasts = {}
        for name in (n for n in ARMS if n != 'paired'):
            diff = 100*(rows['paired'].double()-rows[name].double())
            contrasts[name] = {'points':diff.mean(0).tolist(),
                               'class_delta':class_interval(diff[:, 0], labels, seed=SEED)}
        report = dict(recipe=signature, commit=revision(), device=torch.cuda.get_device_name(),
                      torch=str(torch.__version__), seed=SEED, evaluator_sha=KD_FUNCTION_SHA,
                      clip_sha=CHECKPOINT, vocab_sha=vocab, sample_rows=len(qi), gallery=len(pi),
                      classes=len(labels.unique()), replay_mae_points=replays, prior_fit=stats,
                      metrics={k:(100*v.double().mean(0)).tolist() for k,v in rows.items()},
                      paired_minus=contrasts)
        save_tensor(dict(recipe=signature, qi=qi, pi=pi, labels=labels, rows=rows), output.with_suffix('.pt'))
        atomic_json(output, report)
        print(split, report['metrics'], flush=True)
        del gallery
    if dataset == 'sketchy':
        freeze(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=('sketchy', 'tuberlin'))
    args = parser.parse_args()
    run(args.dataset)
