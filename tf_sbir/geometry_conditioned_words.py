"""Test whether within-shortlist geometric evidence changes the interpretation of a query."""
import hashlib
import inspect
import json
import os
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
from .encoding import load_frozen_clip
from .gallery_context_reversals import coordinates
from .gallery_region_panels import CONFIG, Index
from .positive_foil import raw_ranks
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .standardized_context_audit import audit_inputs
from .strong_pilot_scores import location


def rank_density(scores, eligible, draws):
    """Max of iid draws from a finite uniform proposal, averaged over tied ranks."""
    sorted_scores = scores.masked_fill(~eligible,torch.inf).sort(1).values
    count = eligible.sum(1,keepdim=True).clamp_min(1)
    lower = torch.searchsorted(sorted_scores,scores.contiguous())/count
    upper = torch.searchsorted(sorted_scores,scores.contiguous(),right=True)/count
    density = (upper.pow(draws)-lower.pow(draws))/(upper-lower).clamp_min(1e-9)
    density = density.masked_fill(~eligible,0.)
    if (density.sum(1)-eligible.sum(1)).abs().max()>.003: raise ValueError('Order statistic lost mass')
    return density


def spatial_readouts(sketch, blank, photos, scale, dist, shuffled):
    def read(query, check_shuffle=False):
        logits = scale*(query[:,None] @ photos.transpose(-1,-2)).float()
        joint = (logits.softmax(-1)+logits.softmax(-2))/(2*49)
        real = distance_agreement(joint,dist,dist)
        return (real,distance_agreement(joint,dist,shuffled)) if check_shuffle else real
    real,scrambled = read(sketch,True)
    canvas = read(blank)
    return {'geometry':real,'excess':real-canvas,'canvas':canvas,'scrambled':scrambled}


def reweight(query, likelihood):
    return F.normalize(query*likelihood,p=1,dim=-1)


@torch.inference_mode()
def run():
    w = Path('/content/sbir')
    root = w/'results/geometry-conditioned-word-evidence'
    root.mkdir(exist_ok=True,parents=True)
    qs,gs,enc,_,q,p,flip,src,tgt,scale = audit_inputs(w,768)
    cfg = runtime(w,'sketchy',batch=192,workers=min(4,len(os.sched_getaffinity(0))))
    qp,_ = merge_inputs(cfg.root,SKETCHY_SPLITS,'sketch')
    pp,_ = merge_inputs(cfg.root,SKETCHY_SPLITS,'photo')
    kd,_ = load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    vision = load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda')).visual
    coords = torch.cartesian_prod(torch.arange(7),torch.arange(7)).cuda().float()
    dist = torch.cdist(coords,coords)
    perm = torch.randperm(49,generator=torch.Generator().manual_seed(33513)).cuda()
    shuffled = dist[perm][:,perm]
    cache = w/'cache/depiction-pivot-geometry'
    out = {}
    for split,(qi,y) in qs.items():
        rows = torch.randperm(len(qi),generator=torch.Generator().manual_seed(87243))[:768].sort().values
        ix = coordinates(enc,qi[rows])
        pi,gl = gs[split]
        pack = torch.load(location(w,split,rows),weights_only=True,mmap=True)
        assert torch.equal(pack['qi'],qi[rows]) and torch.equal(pack['pi'],pi)
        base = pack['score'].cuda()
        g = F.normalize(p[pi].cuda())
        x = F.normalize(q[ix].cuda()+flip[ix].float().cuda())
        cosine,near = (x@g.T).topk(200)
        subset,compact = near.cpu().unique(sorted=True,return_inverse=True)
        gp = native_patches([pp[i] for i in pi[subset].tolist()],vision,cache).cuda()
        query_paths = [qp[i] for i in qi[rows].tolist()]
        sp = native_patches(query_paths,vision,cache).cuda()
        flat = native_patches(query_paths,vision,cache,uniform=True).cuda()
        bank = torch.cat([(scale*(v@tgt.T)).softmax(1) for v in g.split(160)])
        graph = Index(SimpleNamespace(p=g),rgb_only=True)
        fold = torch.empty(len(g),dtype=torch.bool,device=g.device)
        parts = torch.randperm(len(g),generator=torch.Generator().manual_seed(69313)).cuda().tensor_split(2)
        fold[parts[0]],fold[parts[1]] = False,True
        replay = PrefixReplay(kd,gl,split)
        ranking = {}
        audits = []
        max_replay = 0.
        for start in range(0,len(rows),12):
            sl = slice(start,start+12)
            candidates = near[sl]
            words = ((scale*(q[ix[sl]].cuda()@src.T)).softmax(1)+(scale*(flip[ix[sl]].float().cuda()@src.T)).softmax(1))/2
            direct = words@bank.T
            native = graph.score(CONFIG,direct).clamp_min(1e-30).log()
            strong = base[sl]
            holders = {}
            photo_nouns = bank[candidates]
            signals = spatial_readouts(sp[sl],flat[sl],gp[compact[sl].cuda()],scale,dist,shuffled)
            signals.update({'visual':cosine[sl],'words':direct.gather(1,candidates),'base':strong.gather(1,candidates)})
            sampling = {key:([2,8,32] if key in ('geometry','excess') else [2,8]) for key in signals}
            fold_details = []
            for part,targets in enumerate(parts):
                eligible = fold[candidates] != bool(part)
                if not eligible.any(1).all():raise ValueError('No opposite-fold donor')
                # One unlabeled prior photo worth of shrinkage for scarcely represented terms.
                reference = bank[parts[1-part]].mean(0)
                normal = eligible.float()[:,None,:].bmm(photo_nouns).squeeze(1)+reference
                estimates = {}
                for key,score in signals.items():
                    for m in sampling[key]:
                        mass = rank_density(score,eligible,m)
                        observed = mass[:,None,:].bmm(photo_nouns).squeeze(1)+reference
                        ratio = observed/normal.clamp_min(1e-30)
                        estimates[f'{key}/max{m}'] = reweight(words,ratio)
                        if key in ('geometry','excess','visual') and m==8:
                            estimates[f'{key}/uncorrected8'] = reweight(words,observed)
                            estimates[f'{key}/gallery8'] = F.normalize(reference*ratio,p=1,dim=1)
                        if key=='geometry' and m==8:
                            fold_details.append(ratio)
                names = list(estimates)
                # Fix donor self-nomination before RGB propagation by scoring each photo from the opposite fold.
                scores = torch.cat([estimates[name] for name in names])@bank[targets].T
                for name,chunk in zip(names,scores.split(len(words)),strict=True):
                    holders.setdefault(name,torch.empty_like(direct))[:,targets] = chunk
            combined = torch.cat(list(holders.values()))
            translated = graph.score(CONFIG,combined).clamp_min(1e-30).log()
            def record(name,score):
                rank,fast = raw_ranks(score,y[rows[sl]],gl.cuda(),split)
                kdrow = replay.chunk((rank.cpu(),y[rows[sl]]))
                if float((fast-kdrow).abs().max())>2e-6:raise ValueError('Original evaluator mismatch')
                hit = gl.cuda()[rank[:,:200]]==y[rows[sl]].cuda()[:,None]
                ranking.setdefault(name,[]).append(torch.cat([kdrow,hit.any(1).cpu()[:,None],hit[:,:1].cpu()],1))
            record('base',strong)
            record('native',native)
            for name,score in zip(holders,translated.split(len(words)),strict=True):
                # Replace only the original .65 times log native-word term of the strong reference.
                change = score-native
                record(name,strong+.65*change)
                if name in ('geometry/max2','geometry/max8','excess/max8','base/max8','words/max8'):
                    record('native/'+name,score)
            max_replay=max(max_replay,float((native-pack['native'][sl].cuda()).abs().max()))
            first,second = [p.clamp_min(1e-25).log() for p in fold_details]
            def centered(a):return a-(a*words).sum(1,keepdim=True)
            a,b = centered(first),centered(second)
            agreement = (words*a*b).sum(1)/((words*a.square()).sum(1)*(words*b.square()).sum(1)).clamp_min(1e-20).sqrt()
            audits.append(agreement.cpu())
        measures = {name:torch.cat(rowscores).double() for name,rowscores in ranking.items()}
        strong = measures['base']
        assert (strong[:,:2]-pack['kd']).abs().max()<2e-6
        table = {}
        for name,values in measures.items():
            d = values-strong
            table[name] = {'metrics':(100*values[:,:2].mean(0)).tolist(),
                'gained_lost_top200':[int((d[:,2]>0).sum()),int((d[:,2]<0).sum())],
                'macro_map_delta':class_interval(100*d[:,0],y[rows])}
        out[split] = {'queries':len(rows),'gallery':len(gl),'rows':table,'native_reconstruction_max_log_difference':max_replay,
            'mean_cross_fold_correlation_of_log_word_evidence':float(torch.cat(audits).mean())}
        torch.save({'metrics':measures,'labels':y[rows],'cross_fold_word_agreement':torch.cat(audits)},root/(split+'.pt'))
        print(split,out[split]['mean_cross_fold_correlation_of_log_word_evidence'],table['geometry/max2'],table['geometry/max8'],flush=True)
        del gp,sp,flat,base,pack,bank,graph
    (root/'pilot.json').write_text(json.dumps(out,indent=2))


if __name__=='__main__':
    run()
