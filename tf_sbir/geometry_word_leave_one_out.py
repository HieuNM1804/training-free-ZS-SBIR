"""Matched one-component removals from the published geometry-conditioned estimator."""
import argparse
import hashlib
import inspect
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .branch_view_inputs import prepare_gallery
from .branch_view_policy import outputs
from .branch_view_study import PrefixReplay
from .concept_benchmark import atomic_json
from .cross_view_regions import CandidateRegions
from .crossdomain_benchmark import digest, revision
from .crossdomain_inputs import CHECKPOINT
from .depiction_path_diagnostic import class_interval
from .depiction_pivot_geometry import native_patches, distance_agreement
from .encoding import load_frozen_clip, save_tensor
from .gallery_panel_resume import semantic_inputs
from .gallery_region_panels import CONFIG, Index
from .geometry_conditioned_words import rank_density, reweight
from .locked_geometry_words import score_rows
from .loo_source_windows import source_windows
from .positive_foil import STRONG
from .query_partition_tree import Hierarchy
from .quickdraw_locked import KD_FUNCTION_SHA
from .regional_intervention_study import index
from .retrieval_evidence import load_kd
from .selective_concept_listener import compile_index
from .shared_concept_likelihood import fit_prior
from .standardized_context_audit import audit_inputs
from .strong_pilot_scores import location


FAMILY = 'geometry-word-leave-one-out'
ROWS = ('full', 'without_diffusion', 'without_regional', 'without_hierarchy', 'without_geometry')
COUNTS = {'sketchy_1': (14179, 17100, 25), 'sketchy_2': (12694, 12552, 21), 'tuberlin': (2130, 27262, 30)}
GLOBAL_WEIGHT = .65
REGIONAL_WEIGHT = .35
SHARD = 256


def log_score(value):
    return value.clamp_min(1e-30).log()


def make_word_bank(gallery, target, scale, block):
    bank = gallery.new_empty(len(gallery), len(target))
    for first in range(0, len(gallery), block):
        sl = slice(first, first + block)
        bank[sl] = (scale * (gallery[sl] @ target.T)).softmax(1)
    return bank


def gallery_hierarchy(w, split, gallery, target, scale, paths):
    packed = prepare_gallery(w, split, gallery, target, scale, paths)
    tree = Hierarchy([v.cuda() for v in packed['ids']], [v.cuda() for v in packed['parents']], packed['sizes'])
    banks = packed['banks'].cuda().flatten(0, 1).T.contiguous()
    prior = torch.bincount(tree.ids[0], minlength=tree.sizes[0]).float() / len(gallery)
    return tree, banks, prior


def hierarchy_inputs(poses, crops, src, scale, banks, prior):
    x = torch.cat((poses[:, None], crops.float()), 1).reshape(-1, poses.size(-1))
    density = []
    for part in x.split(448):
        value = ((scale * (part @ src.T)).softmax(1) @ banks).reshape(-1, 3, len(prior)).clamp_min(1e-32)
        density.append(value / (value * prior).sum(2, keepdim=True))
    return torch.cat(density).reshape(len(poses), 14, 2, 3, len(prior))


def regional_readout(poses, crops, src, scale, words, bank, original):
    reader = CandidateRegions(poses[:, 0], poses[:, 1], crops[:, :, 0], crops[:, :, 1],
                              src, words, bank, src.new_empty((0, len(src))), original)
    reader.prepare(scale)
    return reader.read('self', 1., 'gallery')


def word_update(words, gallery_words, candidates, photos, query, blank, scale, distance, folds, parts, priors):
    def agreement(sketch):
        z = scale * (sketch[:, None] @ photos.transpose(-1, -2)).float()
        joint = (z.softmax(-1) + z.softmax(-2)) / (2 * 49)
        return distance_agreement(joint, distance, distance)
    excess = agreement(query) - agreement(blank)
    word_rows = gallery_words[candidates]
    result = words.new_empty(len(words), len(gallery_words))
    for fold_id, targets in enumerate(parts):
        eligible = folds[candidates] != bool(fold_id)
        if not eligible.any(1).all():
            raise ValueError('Empty opposite gallery fold')
        reference = priors[1 - fold_id]
        normal = eligible.float()[:, None, :].bmm(word_rows).squeeze(1) + reference
        mass = rank_density(excess, eligible, 8)
        observed = mass[:, None, :].bmm(word_rows).squeeze(1) + reference
        update = reweight(words, observed / normal.clamp_min(1e-30))
        result[:, targets] = update @ gallery_words[targets].T
    return result


def paired_summary(scores, labels):
    full = scores['full'].double()
    result = {}
    for name, row in scores.items():
        delta = 100 * (row.double()[:, :2] - full[:, :2])
        result[name] = {
            'metrics_percent': (100 * row.double()[:, :2].mean(0)).tolist(),
            'ablated_minus_full_points': delta.mean(0).tolist(),
            'macro_ap_removal': class_interval(delta[:, 0], labels),
        }
    return result


def geometry_cache(w, split, gallery, query, flipped, qi, pi, qp, pp, vision):
    x = F.normalize(query[qi].cuda() + flipped[qi].float().cuda())
    near = torch.cat([(v @ gallery.T).topk(200).indices for v in x.split(192)])
    used, compact = near.cpu().unique(sorted=True, return_inverse=True)
    folder = w / 'cache/depiction-pivot-geometry'
    gp = native_patches([pp[i] for i in pi[used].tolist()], vision, folder).cuda()
    paths = [qp[i] for i in qi.tolist()]
    # Keep both query caches CPU-mapped so the largest gallery fits on T4.
    sp = native_patches(paths, vision, folder)
    flat = native_patches(paths, vision, folder, uniform=True)
    return near, compact, gp, sp, flat


@torch.inference_mode()
def run(dataset):
    w = Path('/content/sbir')
    root = w / 'results' / FAMILY
    root.mkdir(parents=True, exist_ok=True)
    inputs = semantic_inputs(w, dataset)
    cfg, qp, qs, pp, gs, q, p, src, tgt, scale = inputs
    views = source_windows(w, dataset, inputs)
    aq, ag, encoded, _, gq, photo, flip, gsrc, gtgt, gscale = audit_inputs(w, all_sketches=True, dataset=dataset)
    assert torch.equal(encoded, torch.arange(len(qp))) and scale == gscale
    kd, _ = load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest() == KD_FUNCTION_SHA
    vision = load_frozen_clip(cfg.clip_checkpoint, torch.device('cuda')).visual
    coords = torch.cartesian_prod(torch.arange(7), torch.arange(7)).cuda().float()
    distance = torch.cdist(coords, coords)
    for split, (qi, labels) in qs.items():
        out = root / (split + '.json')
        if out.exists():
            continue
        pi, gl = gs[split]
        assert (len(qi), len(pi), len(gl.unique())) == COUNTS[split]
        assert torch.equal(aq[split][0], qi) and torch.equal(ag[split][0], pi)
        previous = torch.load(w / 'results/geometry-conditioned-word-evidence' / (split + '-official.pt'), mmap=True, weights_only=True)
        assert torch.equal(previous['query_indices'], qi) and torch.equal(previous['labels'], labels)
        base = torch.load(location(w, split, torch.arange(len(qi))), mmap=True, weights_only=True)
        assert torch.equal(base['qi'], qi) and torch.equal(base['pi'], pi)
        paths = [str(pp[i]) for i in pi.tolist()]
        gallery = p[pi].cuda()
        geom_gallery = F.normalize(photo[pi].cuda())
        stamp = digest({'gallery': paths, 'model': CHECKPOINT, 'source': 'regional-64/RGB16'})
        words, compact_bank = index(w, split, gallery, tgt, scale, stamp[:16])
        tree, branches, prior = gallery_hierarchy(w, split, gallery, tgt, scale, paths)
        torch.set_float32_matmul_precision('highest')
        graph = Index(SimpleNamespace(p=gallery), rgb_only=True)
        geom_graph = Index(SimpleNamespace(p=geom_gallery), rgb_only=True)
        bank = make_word_bank(gallery, tgt, scale, 256)
        gallery_prior = fit_prior(bank)[0]['mle']
        raw_words, raw_compact, _ = compile_index(gallery, tgt, scale, gallery_prior, 64, 'visual')
        if not torch.equal(raw_words, words):
            raise ValueError('Raw regional head IDs differ from the locked heads')
        smoothed = graph.score(CONFIG, raw_compact.T.contiguous()).T.contiguous()
        check_index = float((smoothed / smoothed.sum(1, keepdim=True) - compact_bank).abs().max())
        if check_index > 2e-4:
            raise ValueError(f'Regional diffusion cache differs: {check_index}')
        geom_bank = make_word_bank(geom_gallery, gtgt, gscale, 160)
        near, compact, gp, sp, flat = geometry_cache(w, split, geom_gallery, gq, flip, qi, pi, qp, pp, vision)
        parts = torch.randperm(len(pi), generator=torch.Generator().manual_seed(69313)).cuda().tensor_split(2)
        folds = torch.empty(len(pi), dtype=torch.bool, device='cuda')
        folds[parts[0]], folds[parts[1]] = False, True
        priors = [geom_bank[part].mean(0) for part in parts]
        replay = PrefixReplay(kd, gl, split)
        classes = gl.cuda()
        folder = root / split
        folder.mkdir(exist_ok=True)
        all_metrics = {name: [] for name in ROWS}
        max_rebuild = 0.
        with ThreadPoolExecutor(max_workers=min(2, len(os.sched_getaffinity(0)))) as pool:
            for start in range(0, len(qi), SHARD):
                stop = min(len(qi), start + SHARD)
                path = folder / f'{start:05d}.pt'
                if not path.exists():
                    batch = {name: [] for name in ROWS}
                    max_error = 0.
                    for b in range(start, stop, 16):
                        sl = slice(b, min(b + 16, stop))
                        ids = qi[sl]
                        poses = torch.stack((q[ids].cuda(), views['flipped'][ids].float().cuda()), 1)
                        crops = views['regions'][ids].cuda()
                        post = (scale * (poses @ src.T)).softmax(2)
                        first = post[:, 0] @ bank.T
                        raw = post.mean(1) @ bank.T
                        first_d = graph.score(CONFIG, first)
                        lex = log_score(graph.score(CONFIG, raw))
                        regional = regional_readout(poses, crops, src, scale, words, compact_bank, first_d)
                        regional_raw = regional_readout(poses, crops, src, scale, raw_words, raw_compact, first)
                        reg = GLOBAL_WEIGHT * log_score(first_d) + REGIONAL_WEIGHT * regional
                        reg_new = reg + GLOBAL_WEIGHT * (lex - log_score(first_d))
                        density = hierarchy_inputs(poses, crops, src, scale, branches, prior)
                        _, old, _ = next(outputs(tree, density, reg, reg, [STRONG]))
                        h = (old - reg) / reg.std(1, keepdim=True)
                        native_base = reg_new + h * reg_new.std(1, keepdim=True)
                        strong = base['score'][sl].cuda()
                        max_error = max(max_error, float((native_base - strong).abs().max()))
                        gwords = ((gscale * (gq[ids].cuda() @ gsrc.T)).softmax(1)
                                  + (gscale * (flip[ids].float().cuda() @ gsrc.T)).softmax(1)) / 2
                        direct = gwords @ geom_bank.T
                        changed = word_update(gwords, geom_bank, near[sl], gp[compact[sl].cuda()],
                                              sp[sl].cuda(), flat[sl].cuda(), gscale, distance, folds, parts, priors)
                        gd = GLOBAL_WEIGHT * (log_score(geom_graph.score(CONFIG, changed))
                                              - log_score(geom_graph.score(CONFIG, direct)))
                        gr = GLOBAL_WEIGHT * (log_score(changed) - log_score(direct))
                        global_only = GLOBAL_WEIGHT * lex
                        unsmoothed = GLOBAL_WEIGHT * log_score(raw) + REGIONAL_WEIGHT * regional_raw
                        scores = {
                            'full': strong + gd,
                            'without_geometry': strong,
                            'without_hierarchy': reg_new + gd,
                            'without_regional': global_only + h * global_only.std(1, keepdim=True) + gd,
                            'without_diffusion': unsmoothed + h * unsmoothed.std(1, keepdim=True) + gr,
                        }
                        for name, value in scores.items():
                            if not torch.isfinite(value).all():
                                raise ValueError(f'Nonfinite score for {name}')
                            batch[name].append(score_rows(value, labels[sl], classes, split, replay, pool))
                    shard = {'qi': qi[start:stop], 'metrics': {k: torch.cat(v) for k, v in batch.items()},
                             'max_rebuild_score_error': max_error}
                    save_tensor(shard, path)
                shard = torch.load(path, weights_only=True)
                assert torch.equal(shard['qi'], qi[start:stop]) and set(shard['metrics']) == set(ROWS)
                max_rebuild = max(max_rebuild, shard['max_rebuild_score_error'])
                for name in ROWS:
                    all_metrics[name].append(shard['metrics'][name])
                print(split, stop, '/', len(qi), 'max score reconstruction gap', max_rebuild, flush=True)
        raw = {name: torch.cat(chunks) for name, chunks in all_metrics.items()}
        checks = {}
        for name, ref in [('full', 'excess'), ('without_geometry', 'base')]:
            difference = raw[name][:, :2] - previous['per_query'][ref][:, :2]
            checks[name] = {'max_per_query': difference.abs().amax(0).tolist(),
                            'mean_signed_points': (100 * difference.double().mean(0)).tolist(),
                            'mean_absolute_points': (100 * difference.abs().double().mean(0)).tolist()}
            if max(checks[name]['mean_absolute_points']) > .05:
                raise ValueError(f'Official per-query replay diverged: {checks}')
        if max_rebuild > .05:
            raise ValueError(f'Historical strong score reconstruction changed: {max_rebuild}')
        save_tensor({'qi': qi, 'pi': pi, 'labels': labels, 'rows': raw}, root / (split + '.pt'))
        report = {'commit': revision(), 'device': torch.cuda.get_device_name(), 'split': split,
                  'counts': COUNTS[split], 'metrics': ['mAP@200', 'P@200'] if split == 'sketchy_2' else ['mAP@all', 'P@100'],
                  'paired_reference_checks': checks, 'max_strong_score_discrepancy': max_rebuild,
                  'max_regional_index_discrepancy': check_index, 'rows': paired_summary(raw, labels)}
        atomic_json(out, report)
        print(split, json.dumps({k: v['metrics_percent'] for k, v in report['rows'].items()}), flush=True)
        del bank, geom_bank, graph, geom_graph, gp, tree, branches, compact_bank, raw_compact, base


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=('sketchy', 'tuberlin'))
    run(parser.parse_args().dataset)
