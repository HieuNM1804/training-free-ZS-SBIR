"""Probe the semantic evidence unique to geometry after matching the CLIP nomination rank."""
import hashlib
import inspect
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from .geometry_word_flat import (GLOBAL_WEIGHT, REGIONAL_WEIGHT, CONFIG, Index, CHECKPOINT,
    KD_FUNCTION_SHA, PrefixReplay, CandidateRegions, atomic_json, audit_inputs, class_interval,
    digest, distance_agreement, geometry_cache, index, load_frozen_clip, load_kd, log_score,
    make_word_bank, rank_density, revision, reweight, save_tensor, score_rows, semantic_inputs,
    source_windows)
import torch
from torch.nn import functional as F


def adjacent_partner(eligible, order):
    """Exchange consecutive eligible nominees in the specified order; fix any unmatched last one."""
    width = eligible.size(1)
    ordered = order.expand_as(eligible)
    ordered = ordered.gather(1, eligible.gather(1, ordered).int().argsort(dim=1, descending=True, stable=True))
    pos = torch.arange(width, device=eligible.device)[None, :].expand_as(eligible)
    peer = pos ^ 1
    paired = (pos < eligible.sum(1, keepdim=True)) & (peer < eligible.sum(1, keepdim=True))
    partner = torch.empty_like(ordered).scatter_(1, ordered, ordered.gather(1, torch.where(paired, peer, pos)))
    assert torch.equal(partner.gather(1, partner), pos) and torch.equal(eligible.gather(1, partner), eligible)
    return partner


def get_updates(query, blank, patches, dist, scale, words, bank, nominees, fold, priors, random_order):
    def geometry(tokens):
        sim = scale * (tokens[:, None] @ patches.transpose(-1, -2)).float()
        return distance_agreement((sim.softmax(-1) + sim.softmax(-2)) / 98, dist, dist)
    signal = geometry(query) - geometry(blank)
    evidence = bank[nominees]
    natural = torch.arange(nominees.size(1), device=nominees.device)
    updates = {}
    for target_fold in range(2):
        eligible = fold[nominees] != bool(target_fold)
        mass = rank_density(signal, eligible, 8)
        partner = adjacent_partner(eligible, natural)
        sham_partner = adjacent_partner(eligible, random_order)
        baseline = eligible.float()
        paired = .5 * (mass + mass.gather(1, partner))
        sham = .5 * (mass + mass.gather(1, sham_partner))
        weights = torch.stack([baseline, mass, paired, sham], dim=1)
        counts = weights.bmm(evidence) + priors[1-target_fold][None, None]
        uniform, observed, expected, random_expected = counts.unbind(1)
        for name, ratio in [('original', observed / uniform), ('matched', observed / expected),
                            ('between_pairs', expected / uniform), ('random_pairs', observed / random_expected)]:
            updated = reweight(words, ratio.clamp_min(1e-30))
            score = updated @ bank.T
            if name not in updates:
                updates[name] = torch.empty_like(score)
            mask = fold == bool(target_fold)
            updates[name][:, mask] = score[:, mask]
    return updates


@torch.inference_mode()
def run():
    import random
    import numpy as np
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    w = Path('/content/sbir')
    root = w / 'results/gcwe-conditioned-donor-test'
    root.mkdir(parents=True, exist_ok=True)
    inputs = semantic_inputs(w, 'sketchy')
    cfg, qp, qs, pp, gs, q, p, src, tgt, scale = inputs
    views = source_windows(w, 'sketchy', inputs)
    _, _, enc, _, gq, photo, flip, gsrc, gtgt, gscale = audit_inputs(w, all_sketches=True)
    assert torch.equal(enc, torch.arange(len(qp)))
    vision = load_frozen_clip(cfg.clip_checkpoint, torch.device('cuda')).visual
    kd, _ = load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest() == KD_FUNCTION_SHA
    coordinates = torch.cartesian_prod(torch.arange(7), torch.arange(7)).cuda().float()
    dist = torch.cdist(coordinates, coordinates)
    random_order = torch.randperm(200, generator=torch.Generator().manual_seed(42)).cuda()
    for split, (qi, labels) in qs.items():
        if (root / (split+'.json')).exists():
            continue
        pi, gallery_labels = gs[split]
        ix = torch.randperm(len(qi), generator=torch.Generator().manual_seed(42))[:768].sort().values
        ref = torch.load(w/'results/geometry-word-flat-first'/(split+'.pt'), mmap=True, weights_only=True)
        assert torch.equal(qi, ref['qi']) and torch.equal(pi, ref['pi'])
        gallery = p[pi].cuda()
        geom_gallery = F.normalize(photo[pi].cuda())
        graph = Index(SimpleNamespace(p=gallery), rgb_only=True)
        geometry_graph = Index(SimpleNamespace(p=geom_gallery), rgb_only=True)
        paths = [str(pp[i]) for i in pi.tolist()]
        stamp = digest({'gallery': paths, 'model': CHECKPOINT, 'source': 'regional-64/RGB16'})
        heads, compact_bank = index(w, split, gallery, tgt, scale, stamp[:16])
        bank = make_word_bank(gallery, tgt, scale, 256)
        geom_bank = make_word_bank(geom_gallery, gtgt, gscale, 160)
        # Reuse the full-run patch caches with their original path-order hash.
        near, compact, gp, sp, flat = geometry_cache(w, split, geom_gallery, gq, flip, qi, pi, qp, pp, vision)
        parts = torch.randperm(len(pi), generator=torch.Generator().manual_seed(42)).cuda().tensor_split(2)
        folds = torch.empty(len(pi), device='cuda', dtype=torch.bool)
        folds[parts[0]], folds[parts[1]] = False, True
        priors = [geom_bank[part].mean(0) for part in parts]
        freq_order = bank.mean(0).argsort(stable=True)
        partner = torch.arange(len(freq_order), device='cuda') ^ 1
        partner.clamp_max_(len(freq_order)-1)
        perm = torch.empty_like(freq_order).scatter_(0, freq_order, freq_order[partner])
        replay = PrefixReplay(kd, gallery_labels, split)
        classes = gallery_labels.cuda()
        all_rows = {}
        with ThreadPoolExecutor(max_workers=min(3,len(os.sched_getaffinity(0)))) as pool:
            for batch in ix.split(16):
                ids = qi[batch]
                poses = torch.stack((q[ids].cuda(), views['flipped'][ids].float().cuda()), 1)
                crops = views['regions'][ids].cuda()
                post = (scale * (poses @ src.T)).softmax(2).mean(1)
                raw = post @ bank.T
                lex = log_score(graph.score(CONFIG,raw))
                reader = CandidateRegions(poses[:,0], poses[:,1], crops[:,:,0], crops[:,:,1],
                                          src, heads, compact_bank, src.new_empty((0,len(src))), raw)
                reader.prepare(scale)
                regional = reader.read('self',1.,'gallery')
                named = reader.logl - log_score(reader.l.mean(3,keepdim=True))
                alpha = (reader.prior.log()+named).softmax(2)
                shared = log_score((alpha.mean(3,keepdim=True)*reader.l).sum(2).mean(1))
                gwords = ((gscale*(gq[ids].cuda() @ gsrc.T)).softmax(1)
                          +(gscale*(flip[ids].float().cuda() @ gsrc.T)).softmax(1))/2
                direct = log_score(geometry_graph.score(CONFIG, gwords @ geom_bank.T))
                update = get_updates(sp[batch].cuda(),flat[batch].cuda(),gp[compact[batch].cuda()],
                                     dist,gscale,gwords,geom_bank,near[batch],folds,priors,random_order)
                base = GLOBAL_WEIGHT*lex + REGIONAL_WEIGHT*regional
                scores = {'cosine_two_views': F.normalize(poses.mean(1)) @ geom_gallery.T,
                          'semantic_direct': raw, 'semantic_diffused':lex,
                          'semantic_wrong_identity':graph.score(CONFIG,post[:,perm] @ bank.T),
                          'regional_only':regional,'semantic_regional':base}
                for name,val in update.items():
                    delta = GLOBAL_WEIGHT*(log_score(geometry_graph.score(CONFIG,val))-direct)
                    scores[name] = base + delta
                    if name=='original':
                        scores['without_global_term'] = REGIONAL_WEIGHT*regional + delta
                        scores['shared_regional_attention'] = GLOBAL_WEIGHT*lex + REGIONAL_WEIGHT*shared + delta
                for name,val in scores.items():
                    if not val.isfinite().all():raise ValueError(name)
                    all_rows.setdefault(name,[]).append(score_rows(val,labels[batch],classes,split,replay,pool))
        rows = {k:torch.cat(v)[:,:2] for k,v in all_rows.items()}
        gap = (100*(rows['original']-ref['rows']['full'][ix,:2]).double().abs().mean(0))
        if gap.max()>.04:raise ValueError(f'Reconstruction changed by {gap}')
        contrasts = [('original','semantic_regional'),('matched','original'),('matched','semantic_regional'),
                     ('matched','random_pairs'),('between_pairs','semantic_regional'),
                     ('original','shared_regional_attention'),('original','without_global_term'),
                     ('semantic_diffused','semantic_wrong_identity'),('semantic_direct','cosine_two_views')]
        summary = {'commit':revision(),'sample_rows':len(ix),'gallery':len(pi),'seed':42,'replay_mae_points':gap.tolist(),
                   'metrics':{k:(100*v.double().mean(0)).tolist() for k,v in rows.items()},'contrasts':{}}
        for a,b in contrasts:
            diff=100*(rows[a].double()-rows[b].double())
            summary['contrasts'][a+' minus '+b]={'points':diff.mean(0).tolist(),'class_delta':class_interval(diff[:,0],labels[ix],seed=42)}
        save_tensor({'qi':qi[ix],'pi':pi,'labels':labels[ix],'rows':rows},root/(split+'.pt'))
        atomic_json(root/(split+'.json'),summary)
        print(split,summary['metrics'],flush=True)
        del gp,graph,geometry_graph,geom_bank,bank


if __name__=='__main__':
    run()
