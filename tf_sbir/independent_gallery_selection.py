"""Test whether disjoint gallery panels predict which score retrieves a correct held-out top photo."""
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .query_fragment_reference_inputs import dataset
from .query_partition_inputs import place


@torch.inference_mode()
def run():
    w = Path('/content/sbir')
    _, _, sketches, _, photos, _, features = dataset(w, 'sketchy')
    result = {}
    for split, (_, labels) in sketches.items():
        ids, classes = photos[split]
        gallery = F.normalize(features[ids].float().cuda())
        rows = [torch.load(path, weights_only=True, mmap=True) for path in (
            place(w, split) / 'lex.pt',
            w / 'cache/gallery-stable-region-evidence' / (split + '-regional-reference.pt'))]
        folds = torch.randperm(len(ids), generator=torch.Generator().manual_seed(1981)).cuda().tensor_split(3)
        counts = torch.zeros(4, dtype=torch.long, device='cuda')
        decisions = []
        for i, held in enumerate(folds):
            a, b = folds[(i+1)%3], folds[(i+2)%3]
            for start in range(0, len(labels), 128):
                sl = slice(start, start+128)
                agreement, correct = [], []
                for row in rows:
                    scores = row[sl].cuda()
                    means = [F.normalize(gallery[panel[scores[:, panel].topk(5).indices]].mean(1)) for panel in (a,b)]
                    agreement.append((means[0]*means[1]).sum(1))
                    correct.append(classes.cuda()[held[scores[:,held].argmax(1)]] == labels[sl].cuda())
                choose = agreement[0] > agreement[1]
                counts += torch.stack([correct[0].sum(), correct[1].sum(), torch.where(choose,*correct).sum(), choose.sum()])
                decisions.append(torch.stack([choose,*correct]).cpu())
        sel, lex, reg = torch.cat(decisions,1)
        result[split] = dict(observations=sel.numel(), lexical_regional_selected_switches=counts.tolist(),
                             fixed=int((sel & lex & ~reg).sum()), damaged=int((sel & ~lex & reg).sum()))
    print(json.dumps(result), flush=True)
    out = w/'results/independent-gallery-score-selection'
    out.mkdir(exist_ok=True)
    (out/'pilot.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    run()
