"""Retrospective reference game. Sample one source photo A, and its visually
nearest source B with disjoint dominant words, or an oracle different-class B.
A query mentioning B without A is a lexical event; it need not be a contradiction.
The evaluation gives equal weight to source pair rather than number of drawings.
"""
from pathlib import Path
import json
import torch
from .gallery_panel_resume import semantic_inputs,region_views


@torch.inference_mode()
def run():
    w=Path('/content/sbir')
    cfg,paths,qs,photos,gs,q,p,src,tgt,scale=semantic_inputs(w,'sketchy')
    report={}
    for split,(qi,y) in qs.items():
        pi,gl=gs[split]
        keys={(Path(photos[i]).parent.name,Path(photos[i]).stem):j for j,i in enumerate(pi.tolist())}
        photo_ids=torch.tensor([keys[(Path(paths[i]).parent.name,Path(paths[i]).stem.rsplit('-',1)[0])] for i in qi.tolist()])
        unique,owner=photo_ids.unique(return_inverse=True)
        assert torch.equal(gl[photo_ids],y)
        g=p[pi[unique]].cuda()
        group=gl[unique].cuda()
        words=(g@tgt.T).topk(8).indices
        views,_=region_views(w,split,cfg,paths,qi,q)
        sketch_terms=[]
        for i in range(0,len(y),128):
            sl=slice(i,i+128)
            a=(scale*(q[qi[sl]].cuda()@src.T)).softmax(1)
            b=(scale*(views['flipped'][sl].cuda().float()@src.T)).softmax(1)
            sketch_terms.append((a+b).topk(32).indices)
        sketch_terms=torch.cat(sketch_terms)
        # Membership table uses no class labels and includes every drawing of each source.
        mentions=torch.zeros(len(qi),len(src),device=g.device,dtype=torch.bool)
        mentions.scatter_(1,sketch_terms,True)
        same_source=owner.cuda()
        per_split={}
        for require_different in (False,True):
            pair_rows=[]
            for i in range(0,len(unique),48):
                batch=torch.arange(i,min(i+48,len(unique)),device=g.device)
                sim=g[batch]@g.T
                eligible=torch.ones_like(sim,dtype=torch.bool)
                for word in words[batch].T:
                    eligible &= (words[None]!=word[:,None,None]).all(2)
                if require_different:
                    eligible &= group[None]!=group[batch,None]
                assert eligible.any(1).all()
                rival=sim.masked_fill(~eligible,-torch.inf).argmax(1)
                # Source C is a different image of B's class, selected with a
                # similar cosine to A. It is never used for ranking.
                opts=eligible & (group[None]==group[rival,None])
                opts.scatter_(1,rival[:,None],False)
                has_control=opts.any(1)
                distances=(sim-sim.gather(1,rival[:,None])).abs()
                foil_control=distances.masked_fill(~opts,torch.inf).argmin(1)
                denotes_a=mentions[:,words[batch]].any(2).T
                denotes_b=mentions[:,words[rival]].any(2).T
                samples=[]
                for sources in (batch,rival,foil_control):
                    absent=(same_source[None]==sources[:,None])&~denotes_a
                    count=absent.sum(1)
                    samples.append(torch.stack([count,(absent&denotes_b).sum(1)],1))
                v=torch.stack(samples,1)
                # Require an eligible query in all three conditions for paired comparison.
                valid=has_control & (v[:,:,0]>0).all(1)
                pair_rows.append(torch.cat([v[:,:,1]/v[:,:,0].clamp_min(1),
                     (group[batch]==group[rival]).float()[:,None],
                     group[batch,None].float()],1)[valid].cpu())
            a=torch.cat(pair_rows)
            differences=a[:,2]-a[:,0]
            by_class=torch.stack([differences[a[:,4]==c].mean() for c in a[:,4].unique()])
            per_split[str(require_different)]={'pairs':len(a),'total_sources':len(unique),
                'per_source_means':a[:,:4].double().mean(0).tolist(),
                'classes_where_other_source_wins':int((by_class>0).sum()),'classes':len(by_class)}
        report[split]=per_split
    print(json.dumps(report),flush=True)
    (w/'results/positive-foil-evidence/reference-game.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    run()

