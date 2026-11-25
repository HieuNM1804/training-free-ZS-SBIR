"""Score the three locked evidence sources without gallery diffusion or crop selection."""
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

from .branch_view_study import PrefixReplay
from .concept_benchmark import atomic_json
from .crossdomain_benchmark import digest, revision
from .depiction_path_diagnostic import class_interval
from .depiction_pivot_geometry import native_patches
from .encoding import load_frozen_clip, save_tensor
from .gcwe_displacement_consensus import score_update
from .gcwe_displacement_leave_one_out import locked_measure
from .geometry_word_flat import (
    COUNTS, GLOBAL_WEIGHT, REGIONAL_WEIGHT, SHARD, geometry_cache, log_score, make_word_bank,
)
from .locked_geometry_words import score_rows
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .standardized_context_audit import audit_inputs
from .crossdomain_inputs import runtime
from .benchmark import merge_inputs


FAMILY = 'minimal-srg-patches'
SEED = 42
BATCH = 32
PATCH_BANK_BATCH = 64
ARMS = ('S_R_G', 'S_G', 'S_R', 'S')


def patch_words(tokens, text, scale):
    """Every native patch independently votes once over the same public vocabulary."""
    posteriors = (scale * (tokens.to(device=text.device, dtype=text.dtype) @ text.T)).softmax(-1)
    return posteriors.mean(1)


def patch_photo_bank(photo_paths, near, gallery_patches, visual, folder, text, scale):
    bank = text.new_empty((len(photo_paths), len(text)))
    nominees = near.flatten().unique(sorted=True).cpu()
    already_encoded = torch.zeros(len(photo_paths), dtype=torch.bool)
    already_encoded[nominees] = True
    missing = (~already_encoded).nonzero().flatten()
    bank[nominees.to(text.device)] = pooled_batches(gallery_patches, text, scale)
    if len(missing):
        tokens = native_patches([photo_paths[i] for i in missing.tolist()], visual, folder)
        bank[missing.to(text.device)] = pooled_batches(tokens, text, scale)
    return bank


def pooled_batches(tokens, text, scale):
    out = text.new_empty((len(tokens), len(text)))
    for start in range(0, len(tokens), PATCH_BANK_BATCH):
        block = slice(start, start + PATCH_BANK_BATCH)
        out[block] = patch_words(tokens[block], text, scale)
    return out


def reference(folder, split, gallery, queries, labels):
    archive = Path(__file__).resolve().parents[1] / 'docs/results' / folder / 'paired-T4.zip'
    with zipfile.ZipFile(archive) as bundle:
        record = torch.load(io.BytesIO(bundle.read(split + '.pt')), weights_only=True)
    for key, expected in (('pi', gallery), ('qi', queries), ('labels', labels)):
        if not torch.equal(record[key], expected):
            raise ValueError(f'Archived {folder} ordering differs for {split}/{key}')
    return record['rows']


def summarize(rows, prior_ablation, prior_terms, labels):
    comparisons = {
        'S_R_G minus S_G': (rows['S_R_G'], rows['S_G']),
        'S_R_G minus S_R': (rows['S_R_G'], rows['S_R']),
        'S_R minus S': (rows['S_R'], rows['S']),
        'S_G minus S': (rows['S_G'], rows['S']),
        'S_R_G minus prior without diffusion': (rows['S_R_G'], prior_ablation['without_any_rgb_diffusion']),
        'S_R_G minus prior full': (rows['S_R_G'], prior_ablation['same_partition']),
    }
    paired = {}
    for name, (left, right) in comparisons.items():
        diff = 100 * (left.double() - right.double())
        paired[name] = {'points': diff.mean(0).tolist(),
                        'class_delta': class_interval(diff[:, 0], labels, seed=SEED)}
    semantic_mae = 100 * (rows['S'].double() - prior_terms['word_only'].double()).abs().mean(0)
    if semantic_mae.max() > .04:
        raise ValueError(f'The unchanged direct semantic ranker diverged: {semantic_mae.tolist()}')
    return paired, semantic_mae.tolist()


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
    cfg = runtime(w, dataset, batch=384, workers=min(4, len(os.sched_getaffinity(0))))
    qs, gs, encoded, _, query, photo, flip, source, target, scale = audit_inputs(
        w, all_sketches=True, dataset=dataset,
    )
    qp, _ = merge_inputs(cfg.root, tuple(qs), 'sketch')
    pp, _ = merge_inputs(cfg.root, tuple(gs), 'photo')
    assert torch.equal(encoded, torch.arange(len(qp)))
    vision = load_frozen_clip(cfg.clip_checkpoint, torch.device('cuda')).visual
    kd, _ = load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest() == KD_FUNCTION_SHA
    device = torch.cuda.get_device_name()
    commit = revision()
    folder = w / 'cache/depiction-pivot-geometry'
    for split, (qi, labels) in qs.items():
        pi, gallery_labels = gs[split]
        rows_count, gallery_count, classes = COUNTS[split]
        assert (len(qi), len(pi), len(labels.unique())) == (rows_count, gallery_count, classes)
        signature = digest(dict(commit=commit, device=device, seed=SEED, family=FAMILY,
                                evaluator=KD_FUNCTION_SHA, queries=qi.tolist(), gallery=pi.tolist()))
        file = root / (split + '.json')
        if file.exists() and file.with_suffix('.pt').exists():
            stored = json.loads(file.read_bytes())
            if stored.get('recipe') == signature and stored.get('sample_rows') == rows_count:
                print(split, 'already complete', flush=True)
                continue

        prior_ablation = reference('displacement-internal-ablation', split, pi, qi, labels)
        prior_terms = reference('three-component-factorial', split, pi, qi, labels)
        gallery = photo[pi].cuda()
        geometry_gallery = F.normalize(gallery)
        near, compact, patches, sketches, blank = geometry_cache(
            w, split, geometry_gallery, query, flip, qi, pi, qp, pp, vision,
        )
        print(split, 'building uniform patch word bank', flush=True)
        local_bank = patch_photo_bank([pp[i] for i in pi.tolist()], near, patches, vision, folder, target, scale)
        semantic_bank = make_word_bank(gallery, target, scale, 256)
        geometry_bank = make_word_bank(geometry_gallery, target, scale, 160)
        parts = torch.randperm(len(pi), generator=torch.Generator().manual_seed(SEED)).cuda().tensor_split(2)
        folds = torch.empty(len(pi), device='cuda', dtype=torch.bool)
        folds[parts[0]], folds[parts[1]] = False, True
        priors = [geometry_bank[part].mean(0) for part in parts]
        gallery_classes = gallery_labels.cuda()
        replay = PrefixReplay(kd, gallery_labels, split)
        shards = root / split / signature
        shards.mkdir(exist_ok=True, parents=True)
        metrics = {name: [] for name in ARMS}
        with ThreadPoolExecutor(max_workers=min(3, len(os.sched_getaffinity(0)))) as pool:
            for first in range(0, len(qi), SHARD):
                final = min(first + SHARD, len(qi))
                shard = shards / f'{first:05d}.pt'
                if not shard.exists():
                    block = {name: [] for name in ARMS}
                    for batch in torch.arange(first, final).split(BATCH):
                        ids = qi[batch]
                        q_native, q_flip = query[ids].cuda(), flip[ids].float().cuda()
                        poses = torch.stack((q_native, q_flip), 1)
                        words = (scale * (poses @ source.T)).softmax(2).mean(1)
                        s = GLOBAL_WEIGHT * log_score(words @ semantic_bank.T)
                        query_patches = sketches[batch].cuda()
                        local_words = patch_words(query_patches, source, scale)
                        r = REGIONAL_WEIGHT * log_score(local_words @ local_bank.T)
                        geometry_words = ((scale * (q_native @ source.T)).softmax(1)
                                          + (scale * (q_flip @ source.T)).softmax(1)) / 2
                        updated = score_update(
                            locked_measure, query_patches, blank[batch].cuda(),
                            patches[compact[batch].cuda()], None, scale, geometry_words,
                            geometry_bank, near[batch], folds, priors, None, include_original=False,
                        )['same_partition']
                        g = GLOBAL_WEIGHT * (log_score(updated) - log_score(geometry_words @ geometry_bank.T))
                        scores = {'S_R_G': s + r + g, 'S_G': s + g, 'S_R': s + r, 'S': s}
                        for name, value in scores.items():
                            if not value.isfinite().all():
                                raise ValueError(f'Nonfinite score in {split}/{name}')
                            block[name].append(score_rows(value, labels[batch], gallery_classes, split, replay, pool)[:, :2])
                    save_tensor({'ids': qi[first:final], 'rows': {k: torch.cat(v) for k, v in block.items()}}, shard)
                result = torch.load(shard, weights_only=True)
                assert torch.equal(result['ids'], qi[first:final])
                for name in ARMS:
                    metrics[name].append(result['rows'][name])
                print(split, final, '/', len(qi), flush=True)

        rows = {name: torch.cat(chunks) for name, chunks in metrics.items()}
        paired, semantic_mae = summarize(rows, prior_ablation, prior_terms, labels)
        summary = {
            'commit': commit, 'recipe': signature, 'device': device, 'torch': str(torch.__version__),
            'evaluator_sha': KD_FUNCTION_SHA, 'sample_rows': len(qi), 'gallery': len(pi), 'classes': classes,
            'seed': SEED, 'metrics': {k: (100 * v.double().mean(0)).tolist() for k, v in rows.items()},
            'contrasts': paired, 'semantic_replay_mae_points': semantic_mae,
        }
        save_tensor({'recipe': signature, 'qi': qi, 'pi': pi, 'labels': labels, 'rows': rows}, file.with_suffix('.pt'))
        atomic_json(file, summary)
        print(split, summary['metrics'], flush=True)
        del patches, local_bank, semantic_bank, geometry_bank


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('sketchy', 'tuberlin'))
    args = parser.parse_args()
    for dataset in ((args.dataset,) if args.dataset else ('sketchy', 'tuberlin')):
        run(dataset)
