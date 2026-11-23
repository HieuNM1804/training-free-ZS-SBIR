"""Paired diagnostic on word witnesses chosen solely from disjoint gallery references."""
from pathlib import Path
import json

import torch
from .gallery_panel_resume import semantic_inputs, region_views


@torch.inference_mode()
def run():
    w = Path('/content/sbir')
    cfg, paths, qs, _, gs, native, _, words, _, scale = semantic_inputs(w, 'sketchy')
    report = {}
    for split, (qi, labels) in qs.items():
        def load(path):
            return torch.load(path, mmap=True, weights_only=True)
        layout = load(w/'cache/gallery-calibrated-window-extremes'/split/'structure.pt')
        groups = layout['groups'].cuda()
        parts = [p.cuda() for p in layout['parts']]
        weights = load(w/'cache/shared-private-name-evidence'/split/'photo-words.pt')
        regional = load(w/'cache/gallery-stable-region-evidence'/(split+'-regional-reference.pt'))
        packed, _ = region_views(w,split,cfg,paths,qi,native)
        classes = gs[split][1].cuda()
        k = int(groups.max())+1
        banks = []
        for panel in parts:
            bank = words.new_zeros(k,len(words))
            for indices in panel.split(256):
                bank.index_add_(0,groups[indices],weights[indices.cpu()].to(bank))
            banks.append(bank/torch.bincount(groups[panel],minlength=k)[:,None].clamp_min(1))
        counts = torch.zeros(5,dtype=torch.long,device='cuda')
        for i, held in enumerate(parts):
            first, second = banks[(i+1)%3], banks[(i+2)%3]
            ids = torch.empty((k,k,8),dtype=torch.long,device='cuda')
            for g in range(k):
                diff = torch.minimum(first[g]-first,second[g]-second)
                ids[g] = diff.topk(8).indices
            for start in range(0,len(qi),128):
                sl = slice(start,start+128)
                scores = regional[sl].cuda()[:,held]
                ok = labels[sl].cuda()[:,None] == classes[held]
                pos = groups[held[scores.masked_fill(~ok,-torch.inf).argmax(1)]]
                neg = groups[held[scores.masked_fill(ok,-torch.inf).argmax(1)]]
                ix = torch.arange(len(pos),device='cuda')
                valid = (pos!=neg)&ok.any(1)&(~ok).any(1)
                vec = torch.cat([native[qi[sl]].cuda()[:,None],packed['regions'][sl].cuda().float().mean(2)],1)
                def evidence(g,h):
                    axis = words[ids[g,h]].mean(1)-words[ids[h,g]].mean(1)
                    return (vec*axis[:,None]).sum(-1)
                d = evidence(pos,neg)
                # Retain the positive group's identity while replacing the opposing group.
                rival = ((neg+1)%k)
                rival = torch.where(rival==pos,(rival+1)%k,rival)
                ctl = evidence(pos,rival)-evidence(neg,rival)
                for j,v in enumerate([valid,d[:,0]>0,d[:,1:].mean(1)>0,
                    d[:,1:].amax(1)+d[:,1:].amin(1)>0,ctl[:,1:].mean(1)>0]):
                    counts[j]+=(v&valid).sum()
        report[split]=counts.tolist()
    print(json.dumps(report))

if __name__ == '__main__':
    run()
