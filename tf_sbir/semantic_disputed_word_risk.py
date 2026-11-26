"""Audit and constrain unstable word overlap in the global S ranker."""
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
from torch.nn import functional as F

from .benchmark import merge_inputs
from .branch_view_study import PrefixReplay
from .concept_benchmark import atomic_json
from .crossdomain_benchmark import digest, revision
from .crossdomain_inputs import CHECKPOINT, runtime
from .depiction_path_diagnostic import class_interval
from .encoding import load_global_inputs, save_tensor
from .foundation import load_words
from .geometry_word_flat import COUNTS, SHARD
from .locked_geometry_words import score_rows
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .semantic import read_words
from .shared_concept_likelihood import fit_prior
from .standardized_context_audit import audit_inputs


FAMILY = 'semantic-disputed-word-risk'
SEED = 42
BATCH = 32
TRIM_HEAD = 128
LEVELS = (.0625, .25, .5)
BASELINES = ('S', 'S_cosine', 'S_query_top64', 'S_gallery_prior', 'S_prompt_average')


def query_posteriors(query, reflected, source, scale):
    poses = torch.stack((query, reflected.float()), dim=1)
    return (scale * (poses @ source.T)).softmax(2).mean(1)


def word_bank(images, text, scale):
    """Store the transposed posterior matrix, with each word contiguous in memory."""
    bank = images.new_empty((len(text), len(images)))
    for i in range(0, len(images), 256):
        part = slice(i, i + 256)
        bank[:, part] = (scale * (images[part] @ text.T)).softmax(1).T
    return bank


def trim_interval(q_values, q_indices, q_tail, cap, cap_overlap, budgets, nominal):
    """Bound the robust score by computing exactly the removable mass in the head."""
    head = cap[q_indices]
    cumulative = head.cumsum(1)
    before = F.pad(cumulative[:, :-1], (0, 0, 1, 0))
    remaining_overlap = (cap_overlap - (head * q_values[..., None]).sum(1)).clamp_min_(0)
    for budget in budgets:
        removal = torch.minimum(head, (budget[None, None] - before).clamp_min_(0))
        gain = (removal * q_values[..., None]).sum(1)
        tail_mass = (budget[None] - cumulative[:, -1]).clamp_min_(0)
        possible_tail_gain = torch.minimum(tail_mass * q_tail, remaining_overlap)
        denom = 1 - budget[None]
        lower = (nominal - gain - possible_tail_gain).clamp_min_(0) / denom
        yield lower, possible_tail_gain / denom


def audit_trim(queries, p, cap, fractions):
    """Compare the bounded implementation against all 20,931 sorted coordinates."""
    data = []
    torch_rng = torch.Generator().manual_seed(SEED)
    rows = torch.randperm(p.size(1), generator=torch_rng)[:64].to(p.device)
    queries = queries[:4]
    photos, limits = p[:, rows].contiguous(), cap[:, rows].contiguous()
    base, overlap = queries @ photos, queries @ limits
    budgets = [f * limits.sum(0) for f in fractions]
    vals, ids = queries.sort(dim=1, descending=True)
    all_words = trim_interval(vals, ids, vals.new_zeros((len(vals), 1)), limits, overlap, budgets, base)
    head = trim_interval(vals[:, :TRIM_HEAD], ids[:, :TRIM_HEAD], vals[:, TRIM_HEAD, None],
                         limits, overlap, budgets, base)
    for fraction, (exact, _), (lower, gap) in zip(fractions, all_words, head, strict=True):
        if ((exact < lower - 2e-6) | (exact > lower + gap + 2e-6)).any():
            raise ValueError('The truncated robust interval missed the full solution')
        data.append({'fraction': fraction, 'max_interval_width': float(gap.max()),
                     'max_absolute_score_error': float((exact - lower).abs().max())})
    return data


def ranker(q, p, mixed, cap, prior, prior_normalizer, photo_norm, disputed, fractions):
    raw = q @ p
    head64 = q.topk(64, dim=1)
    sparse = q.new_zeros(q.shape).scatter_(1, head64.indices, head64.values)
    base = {
        'S': raw,
        'S_cosine': raw / (q.norm(dim=1, keepdim=True) * photo_norm),
        'S_query_top64': sparse @ p,
        'S_gallery_prior': ((q * prior) @ p) / prior_normalizer,
        'S_prompt_average': q @ mixed,
    }
    disputed_score = q @ cap
    budgets = [fraction * disputed for fraction in fractions]
    head = q.topk(TRIM_HEAD + 1, dim=1)
    arguments = (head.values[:, :-1], head.indices[:, :-1], head.values[:, -1:])
    risk = trim_interval(*arguments, cap, disputed_score, budgets, raw)
    closest = raw.topk(100, dim=1).indices
    widths = {}
    for fraction, budget, (score, width) in zip(fractions, budgets, risk, strict=True):
        base[f'risk-{fraction:g}'] = score
        base[f'static-{fraction:g}'] = (raw - fraction * disputed_score) / (1 - budget)
        widths[f'risk-{fraction:g}'] = (width / raw.clamp_min(1e-25)).gather(1, closest).mean(1)
    uniform = trim_interval(*arguments, p, raw, budgets, raw)
    for fraction, (score, width) in zip(fractions, uniform, strict=True):
        base[f'uniform-{fraction:g}'] = score
        widths[f'uniform-{fraction:g}'] = (width / raw.clamp_min(1e-25)).gather(1, closest).mean(1)
    return base, widths, head64, disputed_score


def diagnose(q, head64, bank, scores, disputed_score, sketch_labels, photo_labels):
    """Inspect baseline errors after ranking, using labels only for the audit."""
    relevant = sketch_labels.cuda()[:, None] == photo_labels[None]
    positive = scores.masked_fill(~relevant, -torch.inf).argmax(1)
    negative = scores.masked_fill(relevant, -torch.inf).argmax(1)
    ids = torch.stack((positive, negative), dim=1)
    products = q[:, None] * bank.T[ids]
    total = products.sum(2)
    head = products.gather(2, head64.indices[:, None].expand(-1, 2, -1)).sum(2)
    tail_share = 1 - head / total.clamp_min(1e-25)
    dispute_share = disputed_score.gather(1, ids) / total.clamp_min(1e-25)
    miss = total[:, 1] > total[:, 0]
    repaired = miss & (head[:, 0] > head[:, 1])
    return torch.column_stack((head64.values.sum(1), q.square().sum(1).reciprocal(),
                               tail_share, products.amax(2) / total.clamp_min(1e-25),
                               miss, repaired, dispute_share)).cpu()


def summarize_diagnosis(rows):
    missed = rows[:, 6].bool()
    bad = rows[missed]
    return {
        'median_query_mass_top64': float(rows[:, 0].median()),
        'median_effective_query_words': float(rows[:, 1].median()),
        'highest_ranked_baseline_photo_is_wrong': int(missed.sum()),
        'same_hard_pair_repaired_by_query_top64': int(rows[:, 7].sum()),
        'median_tail_fraction_on_failed_pairs_positive_then_negative': bad[:, 2:4].median(0).values.tolist(),
        'median_largest_word_fraction_on_failed_pairs_positive_then_negative': bad[:, 4:6].median(0).values.tolist(),
        'median_disputed_fraction_on_failed_pairs_positive_then_negative': bad[:, 8:10].median(0).values.tolist(),
        'failed_pairs_with_higher_disputed_fraction_on_negative': int((bad[:, 9] > bad[:, 8]).sum()),
    }


def inputs(w, dataset):
    cfg = runtime(w, dataset, batch=384, workers=min(4, len(os.sched_getaffinity(0))))
    names = ('sketchy_1', 'sketchy_2') if dataset == 'sketchy' else ('tuberlin',)
    qp, qs = merge_inputs(cfg.root, names, 'sketch')
    pp, gs = merge_inputs(cfg.root, names, 'photo')
    view_file = w / 'features' / f's-only-locked-mirror-{dataset}.pt'
    if not view_file.exists():
        _, _, ids, _, _, _, mirror, _, _, scale = audit_inputs(w, all_sketches=True, dataset=dataset)
        save_tensor({'ids': ids, 'mirror': mirror, 'scale': scale}, view_file)
    views = torch.load(view_file, weights_only=True, mmap=True)
    assert torch.equal(views['ids'], torch.arange(len(qp)))
    q, p, _ = load_global_inputs(cfg, qp, pp, False)
    words = read_words(cfg.vocabulary)
    assert len(words) == 20931
    source, target = (bank.cuda() for bank in load_words(cfg, words))
    return qs, gs, q, p, views['mirror'], source, target, views['scale'], digest(words)


def archived_semantic(split, qi, pi, labels):
    folder = Path(__file__).resolve().parents[1] / 'docs/results/minimal-srg-patches'
    with zipfile.ZipFile(folder / 'paired-T4.zip') as archive:
        old = torch.load(io.BytesIO(archive.read(split + '.pt')), weights_only=True)
    if not all(torch.equal(old[k], value) for k, value in [('qi', qi), ('pi', pi), ('labels', labels)]):
        raise ValueError('Baseline query, gallery or label order changed')
    return old['rows']['S']


def summarize(split, qi, pi, labels, metrics, diag, intervals, trim_audit, signature, vocab):
    replay = 100 * (metrics['S'].double() - archived_semantic(split, qi, pi, labels).double()).abs().mean(0)
    if replay.max() > .04:
        raise ValueError(f'S differs from its archived ranking: {replay.tolist()}')
    paired = {}
    for name, values in metrics.items():
        diff = 100 * (values.double() - metrics['S'].double())
        paired[name] = {'points': diff.mean(0).tolist(), 'class_delta': class_interval(diff[:, 0], labels, seed=SEED)}
    return {
        'recipe': signature, 'commit': revision(), 'device': torch.cuda.get_device_name(),
        'torch': str(torch.__version__), 'seed': SEED, 'evaluator_sha': KD_FUNCTION_SHA,
        'clip_sha': CHECKPOINT, 'vocab_sha': vocab, 'sample_rows': len(qi), 'gallery': len(pi),
        'classes': len(labels.unique()), 'semantic_replay_mae_points': replay.tolist(),
        'metrics': {name: (100 * values.double().mean(0)).tolist() for name, values in metrics.items()},
        'versus_S': paired, 'diagnosis': summarize_diagnosis(diag), 'exact_trim_audit': trim_audit,
        'relative_tail_bound_at_top100': {
            name: {'median': float(values.median()), 'mean': float(values.mean()),
                   'p95': float(values.quantile(.95))}
            for name, values in intervals.items()
        },
    }


def select(w):
    files = [w / FAMILY / (name + '.json') for name in ('sketchy_1', 'sketchy_2')]
    measurements = [json.loads(f.read_bytes()) for f in files]
    if any(r['commit'] != revision() for r in measurements):
        raise ValueError('Selection and full Sketchy scoring must use the same commit')
    candidates = {}
    for fraction in LEVELS:
        name = f'risk-{fraction:g}'
        candidates[name] = [sum(r['metrics'][name][k] for r in measurements) / 2 for k in range(2)]
    key = max(candidates, key=candidates.get)
    spec = {'fraction': float(key.split('-')[1]), 'key': key, 'candidate_means': candidates,
            'commit': revision(), 'sketchy_recipes': [r['recipe'] for r in measurements]}
    atomic_json(w / FAMILY / 'freeze.json', spec | {'sha': digest(spec)})
    print('Frozen', spec, flush=True)


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
    fractions = LEVELS
    frozen = None
    if dataset == 'tuberlin':
        frozen = json.loads((root / 'freeze.json').read_bytes())
        checksum = frozen.pop('sha')
        if digest(frozen) != checksum or frozen['commit'] != revision():
            raise ValueError('Wrong method lock for TU-Berlin')
        fractions = (frozen['fraction'],)

    qs, gs, q, p, reflected, source, target, scale, vocab = inputs(w, dataset)
    kd, _ = load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest() != KD_FUNCTION_SHA:
        raise ValueError('KD-SBIR evaluator has changed')
    arms = (*BASELINES, *(f'{name}-{f:g}' for f in fractions for name in ('risk', 'static', 'uniform')))
    for split, (qi, labels) in qs.items():
        pi, photo_labels = gs[split]
        if (len(qi), len(pi), len(labels.unique())) != COUNTS[split]:
            raise ValueError('Expected the complete unseen benchmark')
        signature = digest(dict(commit=revision(), gpu=torch.cuda.get_device_name(), words=vocab,
                                seed=SEED, queries=qi.tolist(), gallery=pi.tolist(), fractions=fractions,
                                frozen=frozen, kd=KD_FUNCTION_SHA))
        file = root / (split + '.json')
        if file.exists() and file.with_suffix('.pt').exists():
            if json.loads(file.read_bytes()).get('recipe') == signature:
                print(split, 'already complete', flush=True)
                continue

        rgb = p[pi].cuda()
        gallery = word_bank(rgb, target, scale)
        cross_prompt = word_bank(rgb, source, scale)
        cap = (gallery - cross_prompt).clamp_min_(0)
        disputed = cap.sum(0)
        mixed = cross_prompt.add_(gallery).mul_(.5)
        prior, _ = fit_prior(gallery.T)
        mle = prior['mle']
        prior_normalizer = mle @ gallery
        norms = gallery.norm(dim=0)
        labels_cuda = photo_labels.cuda()
        replay = PrefixReplay(kd, photo_labels, split)
        trim_audit = audit_trim(query_posteriors(q[qi[:4]].cuda(), reflected[qi[:4]].cuda(), source, scale),
                               gallery, cap, fractions)
        print(split, 'ready', {'gallery': len(pi), 'queries': len(qi), 'median_disputed_mass': float(disputed.median())}, flush=True)
        chunks = root / split / signature
        chunks.mkdir(exist_ok=True, parents=True)
        rows = {arm: [] for arm in arms}
        diag = []
        widths = {name: [] for name in arms if name.startswith(('risk-', 'uniform-'))}
        with ThreadPoolExecutor(max_workers=min(3, len(os.sched_getaffinity(0)))) as pool:
            for start in range(0, len(qi), SHARD):
                stop = min(start + SHARD, len(qi))
                shard = chunks / f'{start:05d}.pt'
                if not shard.exists():
                    block = {arm: [] for arm in arms}
                    measures = []
                    gaps = {name: [] for name in widths}
                    for batch in torch.arange(start, stop).split(BATCH):
                        ids = qi[batch]
                        reading = query_posteriors(q[ids].cuda(), reflected[ids].cuda(), source, scale)
                        scores, bounds, head64, disputed_score = ranker(
                            reading, gallery, mixed, cap, mle, prior_normalizer, norms, disputed, fractions,
                        )
                        for arm, score in scores.items():
                            if not score.isfinite().all():
                                raise ValueError(f'Nonfinite score in {split}/{arm}')
                            block[arm].append(score_rows(score, labels[batch], labels_cuda, split, replay, pool)[:, :2])
                        measures.append(diagnose(reading, head64, gallery, scores['S'], disputed_score,
                                                 labels[batch], labels_cuda))
                        for name, bound in bounds.items():
                            gaps[name].append(bound.cpu())
                    save_tensor({'ids': qi[start:stop], 'rows': {k: torch.cat(v) for k, v in block.items()},
                                 'diag': torch.cat(measures), 'gap': {k: torch.cat(v) for k, v in gaps.items()}}, shard)
                record = torch.load(shard, weights_only=True)
                assert torch.equal(qi[start:stop], record['ids'])
                for name in arms:
                    rows[name].append(record['rows'][name])
                for name in widths:
                    widths[name].append(record['gap'][name])
                diag.append(record['diag'])
                print(split, stop, '/', len(qi), flush=True)
        metrics = {name: torch.cat(pieces) for name, pieces in rows.items()}
        gaps = {name: torch.cat(pieces) for name, pieces in widths.items()}
        summary = summarize(split, qi, pi, labels, metrics, torch.cat(diag), gaps, trim_audit, signature, vocab)
        summary['median_disputed_photo_mass'] = float(disputed.median())
        save_tensor({'recipe': signature, 'qi': qi, 'pi': pi, 'labels': labels, 'rows': metrics,
                     'diag': torch.cat(diag), 'bounds': gaps}, file.with_suffix('.pt'))
        atomic_json(file, summary)
        print(split, summary['metrics'], flush=True)
        del gallery, mixed, cross_prompt, cap
    if dataset == 'sketchy':
        select(w / 'results')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=('sketchy', 'tuberlin'))
    args = parser.parse_args()
    run(args.dataset)
