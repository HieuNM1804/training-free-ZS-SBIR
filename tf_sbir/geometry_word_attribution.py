"""Post-ranking audit of which public words change a held-out class comparison."""
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from .benchmark import merge_inputs
from .branch_view_inputs import literal_word_ids
from .crossdomain_inputs import runtime
from .data import SKETCHY_SPLITS, UNSEEN_CLASSES
from .depiction_path_diagnostic import class_interval
from .depiction_pivot_geometry import native_patches
from .encoding import load_frozen_clip
from .gallery_context_reversals import coordinates
from .geometry_conditioned_words import rank_density, reweight
from .locked_geometry_words import pair_geometry
from .semantic import read_words
from .standardized_context_audit import audit_inputs


def class_emissions(bank, labels, fold):
    n = len(labels.unique())
    means = torch.zeros((n,bank.size(1)), device=bank.device)
    means.index_add_(0,labels[fold],bank[fold])
    return means / labels[fold].bincount(minlength=n).clamp_min(1)[:,None]


@torch.inference_mode()
def run():
    w = Path('/content/sbir')
    family = w/'results/geometry-conditioned-word-evidence'
    qs,gs,enc,_,q,p,flip,src,tgt,scale = audit_inputs(w,768)
    cfg = runtime(w,'sketchy',workers=min(4,len(os.sched_getaffinity(0))))
    paths,_ = merge_inputs(cfg.root,SKETCHY_SPLITS,'sketch')
    photos,_ = merge_inputs(cfg.root,SKETCHY_SPLITS,'photo')
    words = read_words(cfg.vocabulary)
    vision = load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda')).visual
    coords = torch.cartesian_prod(torch.arange(7),torch.arange(7)).cuda().float()
    dist = torch.cdist(coords,coords)
    perm = torch.randperm(49,generator=torch.Generator().manual_seed(33513)).cuda()
    scrambled = dist[perm][:,perm]
    cache = w/'cache/depiction-pivot-geometry'
    results = {}
    for split,(qi,truth) in qs.items():
        rows = torch.randperm(len(qi),generator=torch.Generator().manual_seed(87243))[:768].sort().values
        ix = coordinates(enc,qi[rows])
        pi,gl = gs[split]
        g = F.normalize(p[pi].cuda())
        x = F.normalize(q[ix].cuda()+flip[ix].float().cuda())
        near = (x@g.T).topk(200).indices
        sub,compact = near.cpu().unique(sorted=True,return_inverse=True)
        query_files = [paths[i] for i in qi[rows].tolist()]
        gp = native_patches([photos[i] for i in pi[sub].tolist()],vision,cache).cuda()
        sp = native_patches(query_files,vision,cache).cuda()
        flat = native_patches(query_files,vision,cache,uniform=True).cuda()
        bank = torch.cat([(scale*(v@tgt.T)).softmax(1) for v in g.split(128)])
        parts = torch.randperm(len(g),generator=torch.Generator().manual_seed(69313)).cuda().tensor_split(2)
        fold = torch.empty(len(g),dtype=torch.bool,device=g.device)
        fold[parts[0]],fold[parts[1]] = False,True
        priors = [bank[z].mean(0) for z in parts]
        stored = [[],[]]
        initial = []
        # First finish all word estimates for individual sketches without class labels.
        for start in range(0,len(ix),12):
            sl = slice(start,start+12)
            candidates = near[sl]
            original = ((scale*(q[ix[sl]].cuda()@src.T)).softmax(1)+(scale*(flip[ix[sl]].float().cuda()@src.T)).softmax(1))/2
            real,floor,_ = pair_geometry(sp[sl],flat[sl],gp[compact[sl].cuda()],scale,dist,scrambled)
            donors = bank[candidates]
            for j in range(2):
                valid = fold[candidates] != bool(j)
                ratio = rank_density(real-floor,valid,8)
                reference = valid.float()[:,None,:].bmm(donors).squeeze(1)+priors[1-j]
                changed = ratio[:,None,:].bmm(donors).squeeze(1)+priors[1-j]
                stored[j].append(reweight(original,changed/reference.clamp_min(1e-30)).cpu())
            initial.append(original.cpu())
        # From here on labels only define retrospective comparisons and cannot enter retrieval.
        y = truth[rows].cuda()
        original = torch.cat(initial).cuda()
        updated = [torch.cat(v).cuda() for v in stored]
        emission = [class_emissions(bank,gl.cuda(),v) for v in parts]
        names = UNSEEN_CLASSES[split]
        literal = torch.zeros(len(words),dtype=torch.bool,device=g.device)
        ids = literal_word_ids(words,names)
        for term_ids in ids:literal[term_ids] = True
        # Candidate rival chosen from the original lexical means in the opposite
        # half. Keep it fixed when scoring the target half before and after reweighting.
        tests = {}
        for idx in range(2):
            other = original@emission[1-idx].T
            other.scatter_(1,y[:,None],-torch.inf)
            rival = other.argmax(1)
            contrast = emission[idx][y]-emission[idx][rival]
            scale_margin = (original*(emission[idx][y]+emission[idx][rival])).sum(1).clamp_min(1e-30)
            original_margin = (original*contrast).sum(1)/scale_margin
            update_terms = (updated[idx]-original)*contrast/scale_margin[:,None]
            all_effect = update_terms.sum(1)
            literal_effect = update_terms[:,literal].sum(1)
            positive = updated[idx]@emission[idx].T
            projected = (positive.gather(1,y[:,None])-positive.gather(1,rival[:,None])).flatten()/scale_margin
            if not torch.allclose(original_margin+all_effect,projected,rtol=2e-4,atol=3e-5):
                raise ValueError('Per-noun attribution must reconstruct the observed margin')
            for name,values in {'original':original_margin,'full':original_margin+all_effect,
                    'without_literal_update':original_margin+all_effect-literal_effect,
                    'only_literal_update':original_margin+literal_effect}.items():
                tests.setdefault(name,[]).append(values.cpu())
        test = {k:torch.stack(v,1) for k,v in tests.items()}
        base = test['original']>0
        report = {}
        for name,margins in test.items():
            correct = margins>0
            gains = (~base & correct).sum()
            losses = (base & ~correct).sum()
            report[name] = {'correct_fold_decisions':int(correct.sum()),
                'fixed_lost':[int(gains),int(losses)],'macro_change_points':class_interval(
                    100*(correct.float()-base.float()).mean(1),truth[rows])}
        count = {'queries':len(y),'fold_decisions':2*len(y),'unseen_classes':len(names),
                 'classes_with_literal_word':sum(bool(z) for z in ids),'literal_word_count':int(literal.sum())}
        results[split] = {**count,'outcomes':report}
        print(split,results[split],flush=True)
        del gp,sp,flat,bank,updated
    (family/'word-attribution.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':
    run()
