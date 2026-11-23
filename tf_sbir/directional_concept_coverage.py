"""Test the direction of lexical support with the same public vocabulary and support size."""
import argparse
import json
from pathlib import Path

import torch
from .gallery_panel_resume import semantic_inputs, region_views


def support(posterior, size):
    return torch.zeros_like(posterior).scatter_(1,posterior.topk(size).indices,1.)


@torch.inference_mode()
def run(native_opponents=False):
    w = Path('/content/sbir')
    cfg, paths, queries, photos, galleries, q, p, src, tgt, scale = semantic_inputs(w,'sketchy')
    report = {}
    for s,(qi,y) in queries.items():
        pi,gl = galleries[s]
        cls = gl.cuda()
        views,_ = region_views(w,s,cfg,paths,qi,q)
        keys = {(Path(photos[i]).parent.name,Path(photos[i]).stem):j for j,i in enumerate(pi.tolist())}
        ids = torch.tensor([keys[(Path(paths[i]).parent.name,Path(paths[i]).stem.rsplit('-',1)[0])]
                            for i in qi.tolist()])
        assert torch.equal(gl[ids],y)
        bank = torch.cat([(scale*(part@tgt.T)).softmax(1) for part in p[pi].cuda().split(256)])
        per_k = {}
        for size in [32,128]:
            photo_support = support(bank,size)
            statistics = []
            for start in range(0,len(y),128):
                sl = slice(start,start+128)
                a = (scale*(q[qi[sl]].cuda()@src.T)).softmax(1)
                b = (scale*(views['flipped'][sl].float().cuda()@src.T)).softmax(1)
                post = (a+b)/2
                q_support = support(post,size)
                forward = post@photo_support.T
                reverse = q_support@bank.T
                direct = post@bank.T
                own = ids[sl].cuda()
                ix = torch.arange(len(own),device=bank.device)
                same_class = y[sl,None].cuda()==cls
                # Lexical hardest wrong class is selected only for this diagnostic.
                selector = q[qi[sl]].cuda() @ p[pi].cuda().T if native_opponents else direct
                opponent = selector.masked_fill(same_class,-torch.inf).argmax(1)
                control = selector.masked_fill(~same_class,-torch.inf).scatter(1,own[:,None],-torch.inf).argmax(1)
                row = []
                for z in (direct,forward,reverse,(forward+reverse)/2):
                    own_score = z[ix,own]
                    wrong = z[ix,opponent]
                    paired = own_score-wrong
                    sibling = z[ix,control]-wrong
                    within = (z>own_score[:,None])&same_class
                    ties = (z==own_score[:,None])&same_class
                    rank = 1+within.sum(1)+.5*(ties.sum(1)-1)
                    z = z.scatter(1,own[:,None],-torch.inf)
                    n = 200 if s=='sketchy_2' else len(pi)-1
                    hits = same_class.gather(1,z.topk(n).indices)
                    ap = (hits*hits.cumsum(1)/torch.arange(1,n+1,device=z.device)).sum(1)/hits.sum(1).clamp_min(1)
                    precision = hits[:,:200 if s=='sketchy_2' else 100].float().mean(1)
                    win = lambda x: (x>0).float()+.5*(x==0)
                    row.append(torch.stack([ap,precision,win(paired),win(sibling),rank],1))
                statistics.append(torch.stack(row,1).cpu())
            data = torch.cat(statistics).double()
            per_k[size] = {'queries':len(y),'metrics_percent':(100*data[:,:,:4].mean(0)).tolist(),
                           'median_source_rank':data[:,:,4].median(0).values.tolist()}
            print(s,size,per_k[size],flush=True)
        report[s] = per_k
    directory = w/'results/directional-concept-coverage'
    directory.mkdir(exist_ok=True,parents=True)
    filename = 'native-opponents.json' if native_opponents else 'sketchy.json'
    (directory/filename).write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--native-opponents',action='store_true')
    run(parser.parse_args().native_opponents)
