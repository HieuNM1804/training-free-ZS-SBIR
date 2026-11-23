"""Audit nominee availability and what the existing geometric test does to them."""
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .gallery_context_reversals import coordinates
from .geometry_conditioned_words import rank_density
from .depiction_pivot_geometry import native_patches
from .locked_geometry_words import pair_geometry
from .standardized_context_audit import audit_inputs
from .benchmark import merge_inputs
from .crossdomain_inputs import runtime
from .data import SKETCHY_SPLITS
from .encoding import load_frozen_clip
from .strong_pilot_scores import location


def design_heads(visual,word):
    # Equal-cost alternatives. Saturate a list before uniqueness to ensure
    # every design sends exactly 200 distinct photos to the verifier.
    def interleave(a,b):
        cols=torch.stack([a,b],2).flatten(1,2).cpu().tolist()
        return torch.tensor([list(dict.fromkeys(row))[:200] for row in cols],device=visual.device)
    head=visual.topk(1200).indices
    lex=word.topk(200).indices
    mid=head[:,200:400]
    # One equally spaced representative of each consecutive six-rank band.
    wide=head[:,2::6]
    return {'visual':head[:,:200],'words':lex,'alternating':interleave(head[:,:200],lex),
            'visual_band_201_400':mid,'systematic_visual_1200':wide}


@torch.inference_mode()
def run():
    w=Path('/content/sbir')
    out=w/'results/contrastive-gallery-design'
    out.mkdir(exist_ok=True,parents=True)
    qs,gs,enc,_,q,p,flip,src,tgt,scale=audit_inputs(w,768)
    cfg=runtime(w,'sketchy')
    qp,_=merge_inputs(cfg.root,SKETCHY_SPLITS,'sketch')
    pp,_=merge_inputs(cfg.root,SKETCHY_SPLITS,'photo')
    vision=load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda')).visual
    grid=torch.cartesian_prod(torch.arange(7),torch.arange(7)).cuda().float()
    dist=torch.cdist(grid,grid)
    perm=torch.randperm(49,generator=torch.Generator().manual_seed(33513)).cuda()
    scramble=dist[perm][:,perm]
    folder=w/'cache/depiction-pivot-geometry'
    result={}
    for split,(qi,y) in qs.items():
        rows=torch.randperm(len(qi),generator=torch.Generator().manual_seed(87243))[:768].sort().values
        ix=coordinates(enc,qi[rows])
        pi,gl=gs[split]
        pack=torch.load(location(w,split,rows),mmap=True,weights_only=True)
        assert torch.equal(pack['qi'],qi[rows]) and torch.equal(pack['pi'],pi)
        g=F.normalize(p[pi].cuda())
        bank=torch.cat([(scale*(v@tgt.T)).softmax(1) for v in g.split(160)])
        sketch=F.normalize(q[ix].cuda()+flip[ix].float().cuda())
        visual=sketch@g.T
        words=((scale*(q[ix].cuda()@src.T)).softmax(1)+(scale*(flip[ix].float().cuda()@src.T)).softmax(1))/2
        panels=design_heads(visual,words@bank.T)
        all_ids=torch.cat(list(panels.values()),1).cpu().unique(sorted=True)
        lookup=torch.full((len(g),),-1,dtype=torch.long)
        lookup[all_ids]=torch.arange(len(all_ids))
        gp=native_patches([pp[i] for i in pi[all_ids].tolist()],vision,folder).cuda()
        paths=[qp[i] for i in qi[rows].tolist()]
        sp=native_patches(paths,vision,folder).cuda()
        blank=native_patches(paths,vision,folder,uniform=True).cuda()
        # Construct every list and score every geometric choice before labeling them.
        measurements={}
        for key,ids in panels.items():
            vals=[]
            for start in range(0,len(rows),12):
                sl=slice(start,start+12)
                block=gp[lookup[ids[sl].cpu()].cuda()]
                real,floor,_=pair_geometry(sp[sl],blank[sl],block,scale,dist,scramble)
                valid=torch.ones_like(real,dtype=torch.bool)
                vals.append(rank_density(real-floor,valid,8).cpu()/ids.size(1))
            measurements[key]=torch.cat(vals)
        actual=y[rows]
        top=pack['score'].topk(200).indices
        headlabels=gl[top]
        count=torch.zeros(len(rows),len(gl.unique())).scatter_add_(1,headlabels,torch.ones_like(top,dtype=torch.float))
        wrong=count.clone()
        wrong.scatter_(1,actual[:,None],-1)
        rival=wrong.argmax(1)
        empty=~(headlabels==actual[:,None]).any(1)
        first_miss=headlabels[:,0]!=actual
        summaries={}
        for key,ids in panels.items():
            cl=gl[ids.cpu()]
            pos=cl==actual[:,None]
            neg=cl==rival[:,None]
            mass=measurements[key]
            right=(mass*pos).sum(1)
            wrongmass=(mass*neg).sum(1)
            prior=pos.float().mean(1)
            priorneg=neg.float().mean(1)
            objective=right-prior-(wrongmass-priorneg)
            stats={}
            for group,mask in {'all':torch.ones_like(empty),'base_top1_wrong':first_miss,'base_no_positive_200':empty}.items():
                with_both=(pos.any(1)&neg.any(1)) & mask
                stats[group]={'queries':int(mask.sum()),'with_positive':int((pos.any(1)&mask).sum()),
                        'with_true_and_modal_wrong':int(with_both.sum()),
                        'mean_positive_count':float(pos[mask].float().sum(1).mean()),
                        'mean_rival_count':float(neg[mask].float().sum(1).mean()),
                        'winning_probability_true':float(right[mask].mean()),'winning_probability_rival':float(wrongmass[mask].mean()),
                        'queries_where_geometric_weight_moves_toward_truth_over_rival_if_both':
                            int((with_both&(objective>0)).sum()),
                        'mean_paired_gain_in_true_minus_rival_mass_if_both':
                            float(objective[with_both].mean()) if with_both.any() else None}
            summaries[key]=stats
        result[split]=summaries
        print(split,summaries['visual']['base_no_positive_200'],flush=True)
        del gp,bank
    (out/'nominee-audit.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    run()
