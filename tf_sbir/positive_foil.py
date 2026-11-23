"""Positive evidence for an alternative versus an unmentioned target concept.

Gallery-only construction. For each photo choose an alternative sharing no top-8
public words. Compare the nearest such photo with a similarity-matched alternative.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch
from .branch_view_policy import outputs
from .branch_view_study import PrefixReplay
from .gallery_panel_resume import region_views, semantic_inputs
from .query_partition_tree import Hierarchy
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd

STRONG = "3/0.50000/same/16/regional-cell/0.4"


def gallery_foils(images, terms):
    mapping = []
    rowids = torch.arange(len(images),device=images.device)
    for ix in rowids.split(128):
        similarities = images[ix] @ images.T
        neighbors = similarities.topk(512).indices
        for k in [64,256,512]:
            cand = neighbors[:,:k]
            disjoint = (terms[cand][:,:,:,None] != terms[ix][:,None,None,:]).all(3).all(2)
            if (disjoint.sum(1)>=2).all():
                break
        # Rare cases get an exhaustive gallery search; do not use labels.
        if (disjoint.sum(1)<2).any():
            cand = similarities.argsort(dim=1,descending=True)
            disjoint = torch.ones_like(cand,dtype=torch.bool)
            for col in terms[ix].T:
                disjoint &= (terms[cand]!=col[:,None,None]).all(2)
        assert (disjoint.sum(1)>=2).all()
        valid_scores = similarities.gather(1,cand).masked_fill(~disjoint,-torch.inf)
        mapping.append(cand.gather(1,valid_scores.topk(2).indices))
    return torch.cat(mapping)


def base_parts(w,s):
    folder = w/'cache/branch-conditioned-query-readout'/s
    metadata = torch.load(folder/'gallery.pt',weights_only=True)
    tree = Hierarchy([z.cuda() for z in metadata['ids']],
                     [z.cuda() for z in metadata['parents']],metadata['sizes'])
    reg = torch.load(w/'cache/gallery-stable-region-evidence'/(s+'-regional-reference.pt'),
                     mmap=True,weights_only=True)
    return folder,tree,reg


def raw_ranks(scores, labels, classes, split):
    cutoff = 200 if split=='sketchy_2' else scores.size(1)
    rank = scores.topk(cutoff).indices
    hit = classes[rank]==labels.cuda()[:,None]
    precision = hit[:,:200 if split=='sketchy_2' else 100].float().mean(1)
    ap = (hit*hit.cumsum(1)/torch.arange(1,cutoff+1,device=scores.device)).sum(1)/hit.sum(1).clamp_min(1)
    return rank,torch.stack([ap,precision],1).cpu()


@torch.inference_mode()
def run(w, ds):
    cfg, paths, qs, photos, gs, q, p, src, tgt, scale = semantic_inputs(w,ds)
    kd,_ = load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    result = {}
    for s,(qi,y) in qs.items():
        pi,gl = gs[s]
        classes = gl.cuda()
        images = torch.nn.functional.normalize(p[pi].cuda())
        photo_terms = (images@tgt.T).topk(8).indices
        rivals = gallery_foils(images,photo_terms)
        region, _ = region_views(w,s,cfg,paths,qi,q)
        folder,tree,regional = base_parts(w,s)
        bank = torch.cat([(scale*(part@tgt.T)).softmax(1) for part in images.split(256)])
        # Diagnostics use source identity after constructing gallery foils and scoring.
        if ds=="sketchy":
            keys = {(Path(photos[i]).parent.name,Path(photos[i]).stem):j for j,i in enumerate(pi.tolist())}
            source = torch.tensor([keys[(Path(paths[i]).parent.name,Path(paths[i]).stem.rsplit('-',1)[0])]
                                   for i in qi.tolist()],device=images.device)
            assert torch.equal(classes[source].cpu(),y)
        replayer = PrefixReplay(kd,gl,s)
        rows = {}
        diagnostic = []
        largest_error = 0.
        for start in range(0,len(y),48):
            sl = slice(start,start+48)
            pack_start = start//192*192
            off = start-pack_start
            if off==0:
                pack = torch.load(folder/'full'/f'{pack_start:06d}.pt',mmap=True,weights_only=True)
            sub = slice(off,off+len(y[sl]))
            assert torch.equal(pack['ids'][sub],torch.arange(start,start+len(y[sl])))
            reg = regional[sl].cuda()
            _,strong,_ = next(outputs(tree,pack['densities'][sub].cuda(),reg,reg,[STRONG]))
            a = (scale*(q[qi[sl]].cuda()@src.T)).softmax(1)
            b = (scale*(region['flipped'][sl].float().cuda()@src.T)).softmax(1)
            post = (a+b)/2
            lexical = (post@bank.T).clamp_min(1e-30).log()
            mass = post[:,photo_terms].sum(2)
            def veto(ix):
                return (-mass[:,ix].clamp(max=1-1e-7)).log1p()
            penalty = veto(rivals[:,0])
            placebo = veto(rivals[:,1])
            if ds=='sketchy':
                own = source[sl]
                idx = torch.arange(len(own),device=images.device)
                words = post.topk(32).indices
                contains = lambda ww:(words[:,:,None]==ww[:,None,:]).any(2).any(1)
                best_wrong = lexical.masked_fill(classes[None]==y[sl,None].cuda(),-torch.inf).argmax(1)
                diagnostic.append(torch.stack([contains(photo_terms[own]),contains(photo_terms[rivals[own,0]]),
                    contains(photo_terms[best_wrong]),contains(photo_terms[rivals[best_wrong,0]]),
                    (classes[rivals[own,0]]==classes[own])],1).cpu())
            for name,z in {'strong':strong,'strong_veto':strong+penalty,'strong_control':strong+placebo,
                           'mean':lexical,'mean_veto':lexical+penalty,'mean_control':lexical+placebo}.items():
                rank,metrics = raw_ranks(z,y[sl],classes,s)
                check = replayer.chunk((rank[:1].cpu(),y[sl][:1]))
                error = float((check-metrics[:1]).abs().max())
                largest_error = max(largest_error,error)
                assert error<2e-6
                rows.setdefault(name,[]).append(metrics)
        result[s] = {'queries':len(y),'photos':len(gl),'gpu':torch.cuda.get_device_name(),'code_sha':__import__('subprocess').check_output(
            ['git','rev-parse','HEAD'],text=True).strip(),'kd_sample_max_error':largest_error,
            'metrics':{k:(torch.cat(v).double().mean(0)*100).tolist() for k,v in rows.items()},
            'foil_with_same_label_percent':float((classes[rivals[:,0]]==classes).float().mean()*100),
            'median_difference_in_foil_cosine':float(((images*images[rivals[:,0]]).sum(1)-(images*images[rivals[:,1]]).sum(1)).median())}
        if diagnostic:
            pos,fo,neg,fn,same = torch.cat(diagnostic).T
            result[s]['pairs']={'target_absent':[int((~pos).sum()),int((~pos&fo).sum())],
                                'wrong_absent':[int((~neg).sum()),int((~neg&fn).sum())],
                                'both_absent':[int((~pos&~neg).sum()),int((~pos&~neg&fo).sum()),
                                               int((~pos&~neg&fn).sum())]}
        print(s,json.dumps(result[s]),flush=True)
    out=w/'results/positive-foil-evidence'
    out.mkdir(exist_ok=True,parents=True)
    (out/(ds+'.json')).write_text(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['sketchy','tuberlin'],default='sketchy')
    run(Path('/content/sbir'),parser.parse_args().dataset)

