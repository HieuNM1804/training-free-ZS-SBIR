"""Match lexical evidence from two fixed sketch views using distinct confirmations."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import os
from pathlib import Path
import random

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
from .semantic_disputed_word_risk import archived_semantic, inputs, word_bank
from .shared_concept_likelihood import fit_prior


FAMILY = 'distinct-lexical-evidence'
SEED = 42
BATCH = 64
METHODS = ('S', 'view_product', 'distinct_words', 'distinct_groups', 'distinct_sham')
ARMS = tuple(name + suffix for suffix in ('', '_prior') for name in METHODS)
NEW_ARMS = ('distinct_words', 'distinct_groups', 'distinct_words_prior', 'distinct_groups_prior')


def nearest_other(text):
    neighbors = []
    for first in range(0, len(text), 512):
        indices = torch.arange(first, min(first + 512, len(text)), device=text.device)
        scores = text[indices] @ text.T
        scores[torch.arange(len(indices), device=text.device), indices] = -torch.inf
        neighbors.append(scores.argmax(1))
    return torch.cat(neighbors)


def reciprocal_pairs(source, target):
    sketch = nearest_other(source)
    photo = nearest_other(target)
    index = torch.arange(len(source), device=source.device)
    good = (sketch == photo) & (sketch[sketch] == index) & (photo[sketch] == index) & (index < sketch)
    left, right = index[good], sketch[good]
    if len(left) < 2 or torch.cat((left, right)).unique().numel() != 2 * len(left):
        raise ValueError('Expected disjoint reciprocal lexical pairs in both prompt banks')
    # A derangement of the existing partners keeps every grouped word and the same group sizes.
    order = torch.randperm(len(right), generator=torch.Generator().manual_seed(SEED)).to(source.device)
    shuffled = torch.empty_like(right)
    shuffled[order] = right[order.roll(1)]
    assert not (right == shuffled).any()
    return left, right, shuffled


class Gallery:
    def __init__(self, p, pairs):
        self.p = p
        self.squared = p.square()
        self.left, self.right, self.shuffled = pairs
        self.alias_products = p[self.left] * p[self.right]
        self.sham_products = p[self.left] * p[self.shuffled]
        self.prior = fit_prior(p.T)[0]['mle']
        self.normalizer = self.prior @ p

    def scores(self, a, b):
        rows = {}
        for suffix, q0, q1, normalizer in (
            ('', a, b, 1.),
            ('_prior', a*self.prior, b*self.prior, self.normalizer),
        ):
            first = q0 @ self.p
            second = q1 @ self.p
            both = first * second
            same_word = (q0*q1) @ self.squared
            distinct = (both - same_word).clamp_min_(0)
            left0, left1 = q0[:, self.left], q1[:, self.left]
            aliases = (left0*q1[:, self.right] + left1*q0[:, self.right]) @ self.alias_products
            sham = (left0*q1[:, self.shuffled] + left1*q0[:, self.shuffled]) @ self.sham_products
            rows.update({
                'S'+suffix: (.5 * (first + second)) / normalizer,
                'view_product'+suffix: both.sqrt() / normalizer,
                'distinct_words'+suffix: distinct.sqrt() / normalizer,
                'distinct_groups'+suffix: (distinct - aliases).clamp_min_(0).sqrt() / normalizer,
                'distinct_sham'+suffix: (distinct - sham).clamp_min_(0).sqrt() / normalizer,
            })
        return rows

    def audit(self, a, b):
        """Verify the factorized score against direct group sums in double precision."""
        a, b = a[:3], b[:3]
        product = self.p[:, :16].double()
        measured = self.scores(a, b)['distinct_groups'][:, :16].double()
        first = a.double()[:, :, None] * product
        second = b.double()[:, :, None] * product
        sums = first.sum(1) * second.sum(1)
        same = (first * second).sum(1)
        same += (first[:, self.left]*second[:, self.right] + first[:, self.right]*second[:, self.left]).sum(1)
        exact = (sums - same).clamp_min_(0).sqrt()
        torch.testing.assert_close(exact, measured, atol=2e-6, rtol=4e-4)
        return float((exact - measured).abs().max())


def read_views(query, mirror, source, scale, ids):
    poses = torch.stack((query[ids].cuda(), mirror[ids].float().cuda()), dim=1)
    return (scale * (poses @ source.T)).softmax(2).unbind(1)


def freeze(root):
    measurements = [json.loads((root/(s+'.json')).read_bytes()) for s in ('sketchy_1', 'sketchy_2')]
    if any(row['commit'] != revision() for row in measurements):
        raise ValueError('Mismatched Sketchy revisions')
    means = {name: [sum(row['metrics'][name][k] for row in measurements) / 2 for k in range(2)]
             for name in NEW_ARMS}
    best = max(means, key=means.get)
    data = {'winner': best, 'means': means, 'commit': revision(),
            'sketchy_recipes': [row['recipe'] for row in measurements]}
    atomic_json(root/'freeze.json', data | {'sha': digest(data)})
    print('Frozen', data, flush=True)


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
            raise ValueError('Wrong Sketchy lock for TU')
    qs, gs, query, photo, mirror, source, target, scale, words_sha = inputs(w, dataset)
    kd, _ = load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest() != KD_FUNCTION_SHA:
        raise ValueError('Changed KD-SBIR evaluator')
    aliases = reciprocal_pairs(source, target)
    for split, (qi, labels) in qs.items():
        pi, photo_labels = gs[split]
        assert (len(qi), len(pi), len(labels.unique())) == COUNTS[split]
        signature = digest(dict(commit=revision(), device=torch.cuda.get_device_name(), seed=SEED,
                                queries=qi.tolist(), photos=pi.tolist(), words=words_sha, frozen=lock))
        output = root / (split + '.json')
        if output.exists() and output.with_suffix('.pt').exists():
            if json.loads(output.read_bytes()).get('recipe') == signature:
                print(split, 'already complete', flush=True)
                continue
        gallery = Gallery(word_bank(photo[pi].cuda(), target, scale), aliases)
        check = gallery.audit(*read_views(query, mirror, source, scale, qi[:3]))
        candidates = photo_labels.cuda()
        replay = PrefixReplay(kd, photo_labels, split)
        shards = root / split / signature
        shards.mkdir(exist_ok=True, parents=True)
        print(split, 'ready', {'groups': len(aliases[0]), 'factorization_error': check}, flush=True)
        files = {name: [] for name in ARMS}
        with ThreadPoolExecutor(max_workers=min(3, len(os.sched_getaffinity(0)))) as pool:
            for start in range(0, len(qi), SHARD):
                stop = min(start + SHARD, len(qi))
                shard = shards / f'{start:05d}.pt'
                if not shard.exists():
                    blocks = {name: [] for name in ARMS}
                    for ix in torch.arange(start, stop).split(BATCH):
                        scores = gallery.scores(*read_views(query, mirror, source, scale, qi[ix]))
                        for name, scores_for_name in scores.items():
                            if not scores_for_name.isfinite().all():
                                raise ValueError(f'Nonfinite scores: {name}')
                            blocks[name].append(score_rows(scores_for_name, labels[ix], candidates, split, replay, pool)[:, :2])
                    save_tensor({'ids': qi[start:stop], 'rows': {k: torch.cat(v) for k, v in blocks.items()}}, shard)
                loaded = torch.load(shard, weights_only=True)
                assert torch.equal(qi[start:stop], loaded['ids'])
                for name in ARMS:
                    files[name].append(loaded['rows'][name])
                print(split, stop, '/', len(qi), flush=True)
        rows = {k: torch.cat(v) for k, v in files.items()}
        mae = 100 * (rows['S'].double() - archived_semantic(split, qi, pi, labels).double()).abs().mean(0)
        if mae.max() > .04:
            raise ValueError(f'S no longer replays: {mae.tolist()}')
        contrasts = {}
        for suffix in ('', '_prior'):
            for left, right in (
                ('view_product', 'S'), ('distinct_words', 'view_product'), ('distinct_words', 'S'),
                ('distinct_groups', 'distinct_words'), ('distinct_groups', 'distinct_sham'),
                ('distinct_groups', 'S'),
            ):
                a, b = left + suffix, right + suffix
                diff = 100 * (rows[a].double() - rows[b].double())
                contrasts[a+' minus '+b] = {
                    'points': diff.mean(0).tolist(),
                    'class_delta': class_interval(diff[:, 0], labels, seed=SEED),
                }
        summary = {
            'recipe': signature, 'commit': revision(), 'device': torch.cuda.get_device_name(),
            'torch': str(torch.__version__), 'seed': SEED, 'evaluator_sha': KD_FUNCTION_SHA,
            'clip_sha': CHECKPOINT, 'vocab_sha': words_sha, 'pairs': len(aliases[0]),
            'sample_rows': len(qi), 'gallery': len(pi), 'classes': len(labels.unique()),
            'factorization_max_absolute_error': check, 'semantic_replay_mae_points': mae.tolist(),
            'metrics': {name: (100*values.double().mean(0)).tolist() for name, values in rows.items()},
            'contrasts': contrasts,
        }
        save_tensor({'recipe': signature, 'rows': rows, 'qi': qi, 'pi': pi, 'labels': labels}, output.with_suffix('.pt'))
        atomic_json(output, summary)
        print(split, summary['metrics'], flush=True)
        del gallery
    if dataset == 'sketchy':
        freeze(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=('sketchy', 'tuberlin'))
    args = parser.parse_args()
    run(args.dataset)
