"""Contrast independent and peer-supported semantic evidence on identical retrieved images."""
from pathlib import Path
import json
import torch
from .gallery_panel_resume import semantic_inputs

@torch.inference_mode()
def run():
    w = Path('/content/sbir')
    _,_,qs,_,gs,native,_,words,_,scale = semantic_inputs(w,'sketchy')
    report = {}
    def load(path):
        return torch.load(path,mmap=True,weights_only=True)
    for split,(qi,labels) in qs.items():
        group = load(w/'cache/gallery-calibrated-window-extremes'/split/'structure.pt')['groups'].cuda()
        weights = load(w/'cache/shared-private-name-evidence'/split/'photo-words.pt')
        reg = load(w/'cache/gallery-stable-region-evidence'/(split+'-regional-reference.pt'))
        bank = words.new_zeros(int(group.max())+1,len(words))
        for start in range(0,len(group),256):
            sl = slice(start,start+256)
            bank.index_add_(0,group[sl],weights[sl].to(bank))
        bank /= torch.bincount(group)[:,None]
        perm = torch.randperm(len(words),generator=torch.Generator().manual_seed(3341)).cuda()
        counts = torch.zeros(4,3,dtype=torch.long,device='cuda')
        for start in range(0,len(qi),64):
            sl = slice(start,start+64)
            scores = reg[sl].cuda()
            ranked = scores.argsort(1,descending=True)
            first = torch.full((len(scores),len(bank)),len(group),device='cuda',dtype=torch.long)
            first.scatter_reduce_(1,group[ranked],torch.arange(len(group),device='cuda').expand_as(ranked),reduce='amin')
            candidates = ranked.gather(1,first.sort(1).values[:,:8])
            b = bank[group[candidates]]
            q = (scale*(native[qi[sl]].cuda()@words.T)).softmax(1)[:,None]
            peer = (b.sum(1,keepdim=True)-b)/7
            choices = [(q*b).sum(-1),(q*b*peer).sum(-1),(q*b*peer[:,:,perm]).sum(-1),
                       (q*b*bank.mean(0)).sum(-1)]
            correct = gs[split][1].cuda()[candidates]==labels[sl].cuda()[:,None]
            for i,z in enumerate(choices):
                final = correct.gather(1,z.argmax(1,keepdim=True)).flatten()
                counts[i]+=torch.stack([final.sum(),(final&~correct[:,0]).sum(),(~final&correct[:,0]).sum()])
        report[split] = counts.tolist()
    print(json.dumps(report))

if __name__=='__main__':
    run()
