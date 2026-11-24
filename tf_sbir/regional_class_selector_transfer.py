"""Match selector photo donors symmetrically around each candidate's frozen semantic rank."""
from pathlib import Path
from types import SimpleNamespace

import torch

from .geometry_word_flat import (CHECKPOINT, CONFIG, Index, atomic_json, class_interval,
    digest, index, make_word_bank, revision, save_tensor, semantic_inputs)
from .regional_category_information import region_observations
from .regional_scale_position_contrast import complete_windows


def donors(scores, candidates, classes):
    order = scores.argsort(dim=1, descending=True, stable=True)
    rank = order.argsort(dim=1).gather(1, candidates)
    steps = torch.arange(1, 65, device=scores.device)
    up = rank[:, :, None] - steps
    down = rank[:, :, None] + steps
    inside = (up >= 0) & (down < scores.size(1))
    rows = torch.arange(len(scores), device=scores.device)[:, None, None]
    above = order[rows, up.clamp_min(0)]
    below = order[rows, down.clamp_max(scores.size(1)-1)]
    foil = candidates.reshape(len(scores), -1, 2).flip(2).reshape_as(candidates)[:, :, None]
    usable = inside & (above != foil) & (below != foil)
    same_above = classes[above] == classes[candidates, None]
    same_below = classes[below] == classes[candidates, None]
    positive, negative = usable & same_above & ~same_below, usable & same_below & ~same_above
    def nearest(array, mask):
        idx = mask.long().argmax(2, keepdim=True)
        return array.gather(2, idx).squeeze(2)
    same = [nearest(above, positive), nearest(below, negative)]
    different = [nearest(below, positive), nearest(above, negative)]
    valid = positive.any(2) & negative.any(2)
    assert all(bool((classes[item][valid] == classes[candidates][valid]).all()) for item in same)
    assert all(bool((classes[item][valid] != classes[candidates][valid]).all()) for item in different)
    at_target = scores.gather(1, candidates)
    tie = torch.stack([scores.gather(1, item) == at_target for item in same + different]).float().mean(0)
    return same, different, valid, tie


def pick(tensor, ids):
    return tensor.gather(3, ids[:, None, None].expand(-1, tensor.size(1), tensor.size(2), -1))


def read_donors(weights, values, selected):
    mixture = (pick(weights, selected[0]) + pick(weights, selected[1])) * .5
    return (mixture * values).sum(2).mean(1).clamp_min(1e-30).log()


def summarize(data, labels):
    a, b = slice(None, None, 2), slice(1, None, 2)
    hits = data['hits']
    mixed = hits[:, a] != hits[:, b]
    eligible = data['valid'][:, a] & data['valid'][:, b] & mixed
    sign = torch.where(hits[:, a], 1., -1.)
    correct, wrong = eligible & hits[:, a], eligible & hits[:, b]
    def average(x, mask):
        return (x * mask).sum(1) / mask.sum(1)
    def stats(x):
        ok = torch.isfinite(x)
        return dict(mean=float(x[ok].mean()), queries=int(ok.sum()),
                    by_class=class_interval(x[ok], labels[ok], seed=42) if ok.any() else None)
    scores = {k:data[k] for k in ('same','different','own')}
    accuracy = {}
    balanced = {}
    for k, v in scores.items():
        margin = sign * (v[:, a] - v[:, b])
        outcome = (margin>0).double() + .5 * (margin==0)
        balanced[k] = 50 * (average(outcome, correct) + average(outcome, wrong))
        accuracy[k] = dict(balanced_percent=stats(balanced[k]),
                           per_query_percent=stats(100 * average(outcome, eligible)))
    change = scores['same'] - scores['different']
    relevant = torch.where(hits[:, a], change[:, a], change[:, b])
    irrelevant = torch.where(hits[:, a], change[:, b], change[:, a])
    ties = (data['tie'][:, a] + data['tie'][:, b]) * .5
    return dict(mixed_pairs=int(mixed.sum()), eligible_pairs=int(eligible.sum()),
                eligible_queries=int(eligible.any(1).sum()),
                original_correct=int(correct.sum()), original_wrong=int(wrong.sum()),
                selector_ties=float(ties[eligible].mean()), accuracy=accuracy,
                same_minus_different=dict(relevant=stats(average(relevant, eligible)),
                    irrelevant=stats(average(irrelevant, eligible)),
                    margin=stats(average(relevant-irrelevant, eligible)),
                    balanced_points=stats(balanced['same']-balanced['different'])))


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    w = Path('/content/sbir')
    root = w/'results/regional-class-selector-transfer'
    root.mkdir(parents=True, exist_ok=True)
    inputs = semantic_inputs(w, 'sketchy')
    _, _, queries, paths, galleries, q, p, src, tgt, scale = inputs
    views = complete_windows(w, 'sketchy', inputs)
    for split, (all_qi, all_labels) in queries.items():
        ref = torch.load(w/'results/regional-category-information'/(split+'.pt'), weights_only=True)
        qi, labels = ref['qi'], ref['labels']
        ix = {v:i for i,v in enumerate(all_qi.tolist())}
        assert torch.equal(labels, all_labels[[ix[v] for v in qi.tolist()]])
        pi, cls = galleries[split]
        assert torch.equal(pi, ref['pi'])
        gallery, cls = p[pi].cuda(), cls.cuda()
        stamp = digest(dict(gallery=[str(paths[i]) for i in pi.tolist()],model=CHECKPOINT,source='regional-64/RGB16'))
        heads, compact = index(w, split, gallery, tgt, scale, stamp[:16])
        bank = make_word_bank(gallery, tgt, scale, 192)
        graph = Index(SimpleNamespace(p=gallery), rgb_only=True)
        records = {}
        for batch in torch.arange(len(qi)).split(24):
            ids = qi[batch]
            poses = torch.stack((q[ids], views['flipped'][ids].float()),1).cuda()
            native = (scale*(poses @ src.T)).softmax(-1).mean(1) @ bank.T
            anchors = graph.score(CONFIG,native).topk(200).indices
            values, weights, full, _, own = region_observations(poses, views['regions'][ids].cuda(),
                src, scale, heads, compact, native)
            same, different, valid, tie = donors(full, anchors, cls)
            vals = pick(values, anchors)
            data = dict(anchors=anchors, hits=cls[anchors] == labels[batch,None].cuda(), valid=valid, tie=tie,
                        same=read_donors(weights,vals,same), different=read_donors(weights,vals,different),
                        own=own.gather(1,anchors), same_ids=torch.stack(same,2), different_ids=torch.stack(different,2))
            for k,v in data.items():
                records.setdefault(k,[]).append(v.cpu())
        data = {k:torch.cat(v) for k,v in records.items()}
        assessment = summarize(data,labels)
        assert assessment['mixed_pairs'] == int(ref['counts'][:,0].sum())
        save_tensor(dict(qi=qi,pi=pi,labels=labels,**data),root/(split+'.pt'))
        atomic_json(root/(split+'.json'),dict(commit=revision(),seed=42,gpu=torch.cuda.get_device_name(),
            queries=len(qi),gallery=len(pi),assessment=assessment))
        print(split, assessment['eligible_pairs'], assessment['same_minus_different']['margin']['mean'],flush=True)


if __name__ == '__main__':
    run()
