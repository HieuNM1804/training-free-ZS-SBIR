"""Diagnostic comparison on fixed positive/negative candidate pairs; labels only identify audit pairs."""
import json
from pathlib import Path

import torch

from .query_fragment_reference_inputs import dataset


def windows(photo, groups, panel, bank):
    # Class-balanced difference in expected group log odds at each window.
    rows = photo[panel.cpu(), :, :, bank].cuda().mean(2)
    k = rows.size(-1)
    counts = torch.bincount(groups[panel], minlength=k)
    assert counts.min() > 0
    means = rows.new_zeros(k, rows.size(1), k)
    means.index_add_(0, groups[panel], rows)
    means /= counts[:, None, None]
    self_scores = means.diagonal(dim1=0, dim2=2).T
    margins = self_scores[:, :, None] - means
    sep = (margins + margins.transpose(0, 2)).permute(0, 2, 1)
    pairs = torch.triu_indices(k, k, offset=1, device=groups.device)
    difficulty = sep[pairs[0], pairs[1], 0]
    ordered = pairs[:, difficulty.argsort(stable=True)]
    other = ordered.roll(1, dims=1)
    pick = sep[:, :, 1:].argmax(-1) + 1
    control = pick.clone()
    control[ordered[0], ordered[1]] = pick[other[0], other[1]]
    control[ordered[1], ordered[0]] = control[ordered[0], ordered[1]]
    return pick, control


@torch.inference_mode()
def run():
    w = Path('/content/sbir')
    _, _, queries, _, gallery, _, _ = dataset(w, 'sketchy')
    report = {}
    for split, (_, labels) in queries.items():
        cache = w/'cache/gallery-calibrated-window-extremes'/split
        def load(path):
            return torch.load(path, mmap=True, weights_only=True)
        layout = load(cache/'structure.pt')
        groups = layout['groups'].cuda()
        parts = [p.cuda() for p in layout['parts']]
        photos = load(cache/'photos-photo.pt')
        sketches = load(cache/'sketches.pt')
        regional = load(w/'cache/gallery-stable-region-evidence'/(split+'-regional-reference.pt'))
        classes = gallery[split][1].cuda()
        counts = torch.zeros(2, 5, dtype=torch.long, device='cuda')
        for i, held in enumerate(parts):
            a, b = (i+1)%3, (i+2)%3
            pick, control = windows(photos, groups, parts[a], b)
            for start in range(0, len(labels), 128):
                sl = slice(start,start+128)
                scores = regional[sl].cuda()[:,held]
                ok = labels[sl].cuda()[:,None] == classes[held]
                pos = groups[held[scores.masked_fill(~ok,-torch.inf).argmax(1)]]
                neg = groups[held[scores.masked_fill(ok,-torch.inf).argmax(1)]]
                valid = ok.any(1) & (~ok).any(1) & (pos != neg)
                idx = torch.arange(len(pos),device='cuda')
                z = sketches[sl].cuda().mean(2)
                delta = z[idx,:,:,pos] - z[idx,:,:,neg]
                hard = valid & ~ok[idx,scores.argmax(1)] & (delta[:,0,a]<0)
                evidence = delta[:,:,b]
                success = torch.stack([torch.ones_like(valid), evidence[:,0]>0,
                    evidence[:,1:].mean(1)>0, evidence[idx,pick[pos,neg]]>0,
                    evidence[idx,control[pos,neg]]>0])
                for row, mask in enumerate([valid, hard]):
                    counts[row] += (success & mask).sum(1)
        report[split] = counts.tolist()
    print(json.dumps(report))


if __name__ == '__main__':
    run()
