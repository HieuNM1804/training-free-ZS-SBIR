"""Full replay of the frozen word update with resumable per-query output."""
import argparse
import hashlib
import inspect
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .benchmark import merge_inputs
from .branch_view_study import PrefixReplay
from .crossdomain_inputs import runtime
from .data import SKETCHY_SPLITS
from .depiction_path_diagnostic import class_interval
from .depiction_pivot_geometry import distance_agreement, native_patches
from .encoding import load_frozen_clip, save_tensor
from .gallery_context_reversals import coordinates
from .gallery_region_panels import CONFIG, Index
from .geometry_conditioned_words import rank_density, reweight
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .standardized_context_audit import audit_inputs
from .strong_pilot_scores import location


LOCK = '803720d5c830b715326dea26971482164e2154ee'
ORDERS = ('excess','geometry','shuffled_excess','wrong_canvas','visual','words','base_nominees')
METHODS = (*ORDERS,'expansion_10','expansion_50','expansion_100')
MINIMAL = ('word_reference','word_excess','diffused_word_reference','diffused_word_excess')
SHARD = 384


def pair_geometry(query, blank, photos, scale, dist, scrambled):
    def both(images):
        z = scale*(images[:,None] @ photos.transpose(-1,-2)).float()
        coupling = (z.softmax(-1)+z.softmax(-2))/(2*49)
        return distance_agreement(coupling,dist,dist),distance_agreement(coupling,dist,scrambled)
    real, fake = both(query)
    floor, fake_floor = both(blank)
    return real,floor,fake-fake_floor


def mismatched_canvas(value, eligible):
    # A fixed per-query permutation within the opposite fold preserves the same
    # nominees and marginal distribution of the subtracted positional scores.
    n = value.size(1)
    rng = torch.Generator().manual_seed(24599)
    shuffle = torch.randperm(n,generator=rng).to(value.device)
    ids = shuffle.expand(len(value),-1)
    valid = eligible[:,shuffle]
    order = valid.long().argsort(dim=1,descending=True,stable=True)
    ordered_ids = ids.gather(1,order)
    length = valid.sum(1,keepdim=True)
    step = torch.arange(n,device=value.device)[None,:]
    next_position = torch.where(step<length,(step+1)%length.clamp_min(1),step)
    next_ids = ordered_ids.gather(1,next_position)
    return torch.empty_like(value).scatter_(1,ordered_ids,value.gather(1,next_ids))


def score_rows(scores,truth,classes,split,replay,pool):
    cut = 200 if split=='sketchy_2' else scores.size(1)
    rank = scores.topk(cut).indices
    hits = classes[rank]==truth.cuda()[:,None]
    denom = hits.sum(1).clamp_min(1)
    by_rank = (hits*hits.cumsum(1)/torch.arange(1,cut+1,device=rank.device)).sum(1)/denom
    precision = hits[:,:200 if split=='sketchy_2' else 100].float().mean(1)
    official = replay.score(rank,truth,pool)
    if (official-torch.stack([by_rank,precision],1).cpu()).abs().max()>2e-6:
        raise ValueError('Rank scoring differs from original KD-SBIR')
    return torch.cat((official,hits[:,:200].any(1).cpu()[:,None],hits[:,:1].cpu()),1)


def summarize(raw,labels):
    base = raw['base'].double()
    primary = raw['excess'].double()
    result = {}
    for name,v in raw.items():
        v = v.double()
        delta = v-base
        result[name] = {'metrics':(100*v[:,:2].mean(0)).tolist(),
            'primary_minus_this_macro_ap':class_interval(100*(primary[:,0]-v[:,0]),labels),
            'macro_ap_change':class_interval(100*delta[:,0],labels),
            'gained_lost_top200':[int((delta[:,2]>0).sum()),int((delta[:,2]<0).sum())],
            'fixed_damaged_top1':[int((delta[:,3]>0).sum()),int((delta[:,3]<0).sum())]}
    return result


@torch.inference_mode()
def run(dataset,only):
    w = Path('/content/sbir')
    root = w/'results/geometry-conditioned-word-evidence'
    root.mkdir(exist_ok=True,parents=True)
    splits = SKETCHY_SPLITS if dataset=='sketchy' else ('tuberlin',)
    qs,gs,enc,_,q,p,flip,src,tgt,scale = audit_inputs(w,all_sketches=True,dataset=dataset)
    cfg = runtime(w,dataset,batch=192,workers=min(4,len(os.sched_getaffinity(0))))
    qp,_ = merge_inputs(cfg.root,splits,'sketch')
    pp,_ = merge_inputs(cfg.root,splits,'photo')
    kd,_ = load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    vision = load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda')).visual
    coords = torch.cartesian_prod(torch.arange(7),torch.arange(7)).cuda().float()
    dist = torch.cdist(coords,coords)
    perm = torch.randperm(49,generator=torch.Generator().manual_seed(33513)).cuda()
    scrambled = dist[perm][:,perm]
    cache = w/'cache/depiction-pivot-geometry'
    for split,(qi,labels) in qs.items():
        if only and split!=only: continue
        rows = torch.arange(len(qi))
        pi,gl = gs[split]
        if dataset=='tuberlin' and (len(qi),len(pi),len(gl.unique()))!=(2130,27262,30):
            raise ValueError('Full TU split required')
        pack = torch.load(location(w,split,rows),weights_only=True,mmap=True)
        assert torch.equal(pack['qi'],qi) and torch.equal(pack['pi'],pi)
        root_split = root/('full-'+split)
        root_split.mkdir(exist_ok=True)
        ix = coordinates(enc,qi)
        g = F.normalize(p[pi].cuda())
        x = F.normalize(q[ix].cuda()+flip[ix].float().cuda())
        nominations = [((v@g.T).topk(200)) for v in x.split(192)]
        near = torch.cat([r.indices for r in nominations])
        cosine = torch.cat([r.values for r in nominations])
        subset,compact = near.cpu().unique(sorted=True,return_inverse=True)
        gp = native_patches([pp[i] for i in pi[subset].tolist()],vision,cache).cuda()
        paths = [qp[i] for i in qi.tolist()]
        sp = native_patches(paths,vision,cache).cuda()
        flat = native_patches(paths,vision,cache,uniform=True).cuda()
        bank = torch.cat([(scale*(v@tgt.T)).softmax(1) for v in g.split(160)])
        graph = Index(SimpleNamespace(p=g),rgb_only=True)
        parts = torch.randperm(len(g),generator=torch.Generator().manual_seed(69313)).cuda().tensor_split(2)
        fold = torch.empty(len(g),dtype=torch.bool,device=g.device)
        fold[parts[0]],fold[parts[1]] = False,True
        priors = [bank[part].mean(0) for part in parts]
        replay = PrefixReplay(kd,gl,split)
        cl = gl.cuda()
        pieces = {}
        outcome = {}
        with ThreadPoolExecutor(max_workers=min(2,len(os.sched_getaffinity(0)))) as pool:
            for begin in range(0,len(qi),SHARD):
                end = min(begin+SHARD,len(qi))
                where = root_split/f'{begin:05d}.pt'
                if where.exists():
                    shard = torch.load(where,weights_only=True)
                else:
                    gathered = {}
                    for start in range(begin,end,16):
                        sl = slice(start,min(start+16,end))
                        ix_batch = ix[sl]
                        candidates = near[sl]
                        strong = pack['score'][sl].cuda()
                        words = ((scale*(q[ix_batch].cuda()@src.T)).softmax(1)+(scale*(flip[ix_batch].float().cuda()@src.T)).softmax(1))/2
                        direct = words@bank.T
                        native = graph.score(CONFIG,direct).clamp_min(1e-30).log()
                        appearance,floor,sham = pair_geometry(sp[sl],flat[sl],gp[compact[sl].cuda()],scale,dist,scrambled)
                        fixed = {'excess':appearance-floor,'geometry':appearance,'shuffled_excess':sham,
                                 'visual':cosine[sl],'words':direct.gather(1,candidates),
                                 'base_nominees':strong.gather(1,candidates)}
                        photo_words = bank[candidates]
                        holders = {name:torch.empty_like(direct) for name in METHODS}
                        for fold_id,targets in enumerate(parts):
                            eligible = fold[candidates] != bool(fold_id)
                            if not eligible.any(1).all():raise ValueError('Empty reference fold')
                            reference = priors[1-fold_id]
                            normal = eligible.float()[:,None,:].bmm(photo_words).squeeze(1)+reference
                            evidence = {}
                            for name in ORDERS:
                                signal = appearance-mismatched_canvas(floor,eligible) if name=='wrong_canvas' else fixed[name]
                                weight = rank_density(signal,eligible,8)
                                observed = weight[:,None,:].bmm(photo_words).squeeze(1)+reference
                                evidence[name] = reweight(words,observed/normal.clamp_min(1e-30))
                                if name=='excess':
                                    feedback = F.normalize(observed,p=1,dim=1)
                                    for percent in (10,50,100):
                                        evidence[f'expansion_{percent}'] = torch.lerp(words,feedback,percent/100)
                            projected = torch.cat([evidence[name] for name in METHODS])@bank[targets].T
                            for name,chunk in zip(METHODS,projected.split(len(words)),strict=True):
                                holders[name][:,targets] = chunk
                        updated = graph.score(CONFIG,torch.cat(list(holders.values()))).clamp_min(1e-30).log()
                        batch = {'base':strong,'word_reference':direct,'word_excess':holders['excess'],
                                 'diffused_word_reference':native,'diffused_word_excess':updated[:len(words)]}
                        for name,val in zip(METHODS,updated.split(len(words)),strict=True):
                            batch[name]=strong+.65*(val-native)
                        for name,val in batch.items():
                            exact = score_rows(val,labels[sl],cl,split,replay,pool)
                            gathered.setdefault(name,[]).append(exact)
                    shard = {'lock':LOCK,'qi':qi[begin:end], 'rows':{k:torch.cat(v) for k,v in gathered.items()}}
                    save_tensor(shard,where)
                assert shard['lock']==LOCK and torch.equal(shard['qi'],qi[begin:end])
                assert set(shard['rows'])=={'base',*METHODS,*MINIMAL}
                if (shard['rows']['base'][:,:2]-pack['kd'][begin:end]).abs().max()>2e-6:raise ValueError('Base ranking changed')
                for name,val in shard['rows'].items():pieces.setdefault(name,[]).append(val)
                print(split,end,'/',len(qi),flush=True)
        raw = {k:torch.cat(v) for k,v in pieces.items()}
        outcome = {'lock':LOCK,'queries':len(qi),'gallery':len(g),'classes':len(gl.unique()),'rows':summarize(raw,labels)}
        save_tensor({'lock':LOCK,'query_indices':qi,'labels':labels,'per_query':raw},root/(split+'-official.pt'))
        (root/(split+'-official.json')).write_text(json.dumps(outcome,indent=2))
        print(split,'base',outcome['rows']['base']['metrics'],'primary',outcome['rows']['excess'],flush=True)
        del gp,sp,flat,bank,graph,g,pack


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=('sketchy','tuberlin'),required=True)
    parser.add_argument('--only',choices=(*SKETCHY_SPLITS,'tuberlin'))
    args = parser.parse_args()
    run(args.dataset,args.only)
