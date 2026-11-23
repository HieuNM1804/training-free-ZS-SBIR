"""Challenge the current top group with the best alternative group in each held-out panel."""
from pathlib import Path
import json
import torch
from .gallery_panel_resume import semantic_inputs


@torch.inference_mode()
def run():
    w = Path('/content/sbir')
    _, _, qs, _, gs, native, _, words, _, _ = semantic_inputs(w, 'sketchy')
    result = {}
    for split,(qi,labels) in qs.items():
        def load(p):
            return torch.load(p, mmap=True, weights_only=True)
        geo = load(w/'cache/gallery-calibrated-window-extremes'/split/'structure.pt')
        group = geo['groups'].cuda()
        k = int(group.max())+1
        weights = load(w/'cache/shared-private-name-evidence'/split/'photo-words.pt')
        regional = load(w/'cache/gallery-stable-region-evidence'/(split+'-regional-reference.pt'))
        banks = []
        for panel in geo['parts']:
            m = words.new_zeros(k,len(words))
            for ids in panel.split(256):
                m.index_add_(0,group[ids.cuda()],weights[ids].to(m))
            banks.append(m/torch.bincount(group[panel.cuda()],minlength=k)[:,None].clamp_min(1))
        counts = torch.zeros(2,4,dtype=torch.long,device='cuda')
        for i,held in enumerate(geo['parts']):
            a,b = banks[(i+1)%3],banks[(i+2)%3]
            frequency = ((a+b)/2).mean(0)
            vectors = words.new_empty(k,k,words.shape[1])
            signature = words.new_empty(k,k,2)
            for g in range(k):
                strength,ids = torch.minimum(a[g]-a,b[g]-b).topk(8)
                vectors[g] = words[ids].mean(1)
                signature[g,:,0] = strength.mean(1)
                signature[g,:,1] = frequency[ids].mean(1)
            # Match alternative opponents within each focal group using stable word
            # separation and mean word frequency, with both coordinates standardized.
            controls = torch.empty_like(vectors)
            for g in range(k):
                s = signature[g]
                dist = torch.cdist(s/s.std(0).clamp_min(1e-12),s/s.std(0).clamp_min(1e-12))
                dist.fill_diagonal_(torch.inf)
                dist[:,g] = torch.inf
                controls[g] = vectors[g,dist.argmin(1)]
            held = held.cuda()
            for start in range(0,len(qi),128):
                sl = slice(start,start+128)
                score = regional[sl].cuda()[:,held]
                top = held[score.argmax(1)]
                g = group[top]
                rival = held[score.masked_fill(group[held][None]==g[:,None],-torch.inf).argmax(1)]
                h = group[rival]
                query = native[qi[sl]].cuda()
                correct = gs[split][1].cuda()[top]==labels[sl].cuda()
                alternative = gs[split][1].cuda()[rival]==labels[sl].cuda()
                for j,bank in enumerate((vectors,controls)):
                    switch = (query*(bank[g,h]-bank[h,g])).sum(1)<0
                    counts[j]+=torch.stack([correct.sum(),switch.sum(),(switch&~correct&alternative).sum(),
                        (switch&correct&~alternative).sum()])
        result[split]=counts.tolist()
    print(json.dumps(result))


if __name__=='__main__':
    run()
