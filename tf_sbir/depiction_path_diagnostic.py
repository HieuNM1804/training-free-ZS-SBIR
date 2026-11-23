"""Localize the failing leg of a source-like gallery pivot after ranking."""
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .gallery_context_reversals import coordinates
from .natural_depiction_donors import two_domain_readings
from .standardized_context_audit import audit_inputs
from .strong_pilot_scores import location


def class_interval(values, labels, seed=38451):
    groups = [values[labels == c].mean() for c in labels.unique()]
    if not groups:
        return None
    means = torch.stack(groups).double().cpu()
    draws = torch.randint(len(groups), (10000, len(groups)),
                          generator=torch.Generator().manual_seed(seed))
    return {'classes': len(groups), 'queries': len(values),
            'mean_across_classes': means.mean().item(),
            'positive_classes': int((means > 0).sum()),
            'bootstrap95': means[draws].mean(1).quantile(torch.tensor([.025, .975], dtype=means.dtype)).tolist()}


def endpoint_summary(rank, target_labels, truth):
    hits = target_labels[rank] == truth[:, None]
    return torch.stack((hits[:, 0].float(), hits.any(1).float(), hits.float().mean(1)), dim=1)


@torch.inference_mode()
def run():
    w = Path('/content/sbir')
    folder = w/'results/natural-depiction-donors'
    qs, gs, enc, _, q, p, flip, src, tgt, scale = audit_inputs(w, 768)
    out = {}
    for split, (qi, y) in qs.items():
        rows = torch.randperm(len(qi), generator=torch.Generator().manual_seed(87243))[:768].sort().values
        ix = coordinates(enc, qi[rows])
        labels = y[rows].cuda()
        pi, gl = gs[split]
        gl = gl.cuda()
        baseline = torch.load(location(w, split, rows), weights_only=True, mmap=True)
        assert torch.equal(baseline['qi'], qi[rows]) and torch.equal(baseline['pi'], pi)
        base = baseline['score'].cuda()
        g = F.normalize(p[pi].cuda())
        x = F.normalize((q[ix].cuda() + flip[ix].float().cuda()) / 2)
        style = torch.cat([two_domain_readings(t, src, tgt, scale)[0] for t in g.split(256)])
        upper = style > style.median()
        lower = (~upper).nonzero().flatten()
        gt = gl[lower]
        cosine = x @ g.T
        near = cosine.topk(200).indices
        are_upper = upper[near]
        are_correct = gl[near] == labels[:, None]
        found = (are_correct & are_upper).any(1)
        base_miss = ~(gl[base.topk(200).indices] == labels[:, None]).any(1)
        use = (base_miss & found).nonzero().flatten()
        near = near[use]
        high = are_upper[use]
        good = are_correct[use]
        truth = labels[use]
        # The ordering and endpoint graph are independently computed without truth.
        # Oracle reads a label only when enumerating paths after that ordering.
        first = near.gather(1, high.long().argmax(1)[:, None]).flatten()
        oracle = near.gather(1, (high & good).long().argmax(1)[:, None]).flatten()
        coin = torch.Generator().manual_seed(6635)
        random_same_class = torch.stack([pool[torch.randint(len(pool), (1,), generator=coin).item()]
            for label in truth for pool in [(upper & (gl == label)).nonzero().flatten()]])
        references = {'sketch_cosine': cosine[use][:, lower], 'strong': base[use][:, lower]}
        target_bank = torch.cat([(scale*(t @ tgt.T)).softmax(1) for t in g[lower].split(128)])
        # Evaluate both legs on the same disjoint endpoints, excluding every upper donor.
        for name, anchor in [('first', first), ('earliest_correct_oracle', oracle),
                             ('random_correct_oracle', random_same_class)]:
            references[name+'/cosine'] = g[anchor] @ g[lower].T
            noun = (scale*(g[anchor] @ tgt.T)).softmax(1)
            references[name+'/target_words'] = noun @ target_bank.T
        values = {name: endpoint_summary(v.topk(200).indices, gt, truth)
                  for name, v in references.items()}
        observed = torch.load(folder/(split+'.pt'), weights_only=True)
        obs = observed['diagnostic']
        assert torch.equal(observed['labels'], y[rows])
        recovered = (obs[:, 1] == 0) & (obs[:, 2] > 0)
        assert int((recovered & (obs[:, 3] > 0)).sum()) == len(use)
        out[split] = {
            'pilot_queries': len(ix), 'strong_misses': int(base_miss.sum()),
            'complete_misses_with_correct_upper_donor_in_visual_top200': len(use),
            'first_upper_donor_correct_on_these': int((gl[first] == truth).sum()),
            'lower_endpoint_gallery_size': len(lower),
            'mean_upper_fraction_excess_over_same_class_gallery_macro': class_interval(
                obs[recovered, 5] - obs[recovered, 6], y[rows][recovered]),
            'columns': ['correct_at_1', 'any_correct_in_200', 'fraction_correct_in_200'],
            'endpoints': {name: value.mean(0).tolist() for name, value in values.items()},
            'earliest_correct_oracle_vs_first_cosine_density_class_interval': class_interval(
                values['earliest_correct_oracle/cosine'][:, 2]-values['first/cosine'][:, 2], truth),
            'earliest_correct_oracle_vs_sketch_cosine_density_class_interval': class_interval(
                values['earliest_correct_oracle/cosine'][:, 2]-values['sketch_cosine'][:, 2], truth),
        }
        print(split, json.dumps(out[split]), flush=True)
        del target_bank, base, baseline
    (folder/'path-diagnostic.json').write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
    run()
