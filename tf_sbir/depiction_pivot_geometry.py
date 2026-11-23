"""Can spatial agreement identify a content-preserving gallery intermediary?"""
import hashlib
import inspect
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .benchmark import merge_inputs
from .branch_view_study import PrefixReplay
from .crossdomain_inputs import runtime
from .data import Images, SKETCHY_SPLITS, configure_image_worker, normal_transform
from .encoding import load_frozen_clip, save_tensor
from .gallery_context_reversals import coordinates
from .natural_depiction_donors import two_domain_readings
from .positive_foil import raw_ranks
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .standardized_context_audit import audit_inputs
from .strong_pilot_scores import location


def native_patches(paths, vision, folder):
    stamp = hashlib.sha256(('ViT-B32-openai/full-grid/v1\n'+'\n'.join(map(str, paths))).encode()).hexdigest()
    cache = folder/(stamp+'.pt')
    if cache.is_file():
        return torch.load(cache, weights_only=True, mmap=True)
    batches = DataLoader(Images(paths, normal_transform(), 'rgb'), batch_size=192,
                         num_workers=min(4,len(os.sched_getaffinity(0))), pin_memory=True,
                         worker_init_fn=configure_image_worker)
    features = torch.empty((len(paths),49,512), dtype=torch.float16)
    start = 0
    for batch in batches:
        tokens = vision.transformer(vision._embeds(batch.cuda(non_blocking=True).to(vision.conv1.weight.dtype)))
        patches = F.normalize((vision.ln_post(tokens[:,1:]) @ vision.proj).float(), dim=-1)
        features[start:start+len(batch)] = patches.half().cpu()
        start += len(batch)
    assert start == len(paths)
    save_tensor(features, cache)
    return features


def distance_agreement(prob, dist, target_dist):
    # Two independent correspondences are drawn from the same fixed joint measure.
    # The independent-marginal null preserves each image's location weights.
    qs, gs = prob.sum(-1), prob.sum(-2)
    def moments(marginal, d):
        mean = ((marginal @ d)*marginal).sum(-1)
        return mean, (((marginal @ d.square())*marginal).sum(-1)-mean.square()).clamp_min(0)
    mu, var = moments(qs, dist)
    nu, other_var = moments(gs, target_dist)
    joint = (prob*(dist @ prob @ target_dist.T)).sum((-1,-2))
    return (joint-mu*nu)/(var*other_var).sqrt().clamp_min(1e-7)


def zscore(values, mask):
    n = mask.sum(1,keepdim=True).clamp_min(1)
    center = values-(values*mask).sum(1,keepdim=True)/n
    return center/((center.square()*mask).sum(1,keepdim=True)/n).sqrt().clamp_min(1e-7)


@torch.inference_mode()
def run():
    w = Path('/content/sbir')
    family = 'depiction-pivot-geometry'
    folder = w/'results'/family
    folder.mkdir(exist_ok=True,parents=True)
    qs, gs, enc, _, q, p, flip, src, tgt, scale = audit_inputs(w,768)
    cfg = runtime(w,'sketchy',batch=192,workers=min(4,len(os.sched_getaffinity(0))))
    qp,_ = merge_inputs(cfg.root,SKETCHY_SPLITS,'sketch')
    pp,_ = merge_inputs(cfg.root,SKETCHY_SPLITS,'photo')
    model = load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda'))
    kd,_ = load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest() == KD_FUNCTION_SHA
    coord = torch.cartesian_prod(torch.arange(7),torch.arange(7)).cuda().float()
    dist = torch.cdist(coord,coord)
    perm = torch.randperm(49,generator=torch.Generator().manual_seed(33513)).cuda()
    scrambled = dist[perm][:,perm]
    results = {}
    for split,(qi,y) in qs.items():
        rows = torch.randperm(len(qi),generator=torch.Generator().manual_seed(87243))[:768].sort().values
        ix = coordinates(enc,qi[rows])
        pi,gl = gs[split]
        cl = gl.cuda()
        basepack = torch.load(location(w,split,rows),weights_only=True,mmap=True)
        assert torch.equal(basepack['qi'],qi[rows]) and torch.equal(basepack['pi'],pi)
        base = basepack['score'].cuda()
        x = F.normalize((q[ix].cuda()+flip[ix].float().cuda())/2)
        g = F.normalize(p[pi].cuda())
        visual,near = (x@g.T).topk(200)
        style = torch.cat([two_domain_readings(part,src,tgt,scale)[0] for part in g.split(160)])
        eligible = style[near]>style.median()
        fallback = ~eligible.any(1)
        eligible[fallback,0] = True
        subset,compact = near.cpu().unique(sorted=True,return_inverse=True)
        gp = native_patches([pp[i] for i in pi[subset].tolist()],model.visual,w/'cache'/family).cuda()
        sp = native_patches([qp[i] for i in qi[rows].tolist()],model.visual,w/'cache'/family).cuda()
        bank = torch.cat([(scale*(part@tgt.T)).softmax(1) for part in g.split(160)])
        # Each sketch ranks its own nominated photos. Subset batching only caches identical image encodes.
        metrics, donor_stats, endpoint_stats = {}, {}, {}
        chosen = {}
        replay = PrefixReplay(kd,gl,split)
        for start in range(0,len(rows),6):
            block = slice(start,start+6)
            idx, mask = near[block], eligible[block]
            appearance = (sp[block,None] @ gp[compact[block].cuda()].transpose(-1,-2)).float()
            energy = scale*appearance
            # Equal prior on which view initiates a match. Allows nonuniform salience at the other endpoint.
            prob = (energy.softmax(-1)+energy.softmax(-2))/(2*49)
            consistency = distance_agreement(prob,dist,dist)
            sham = distance_agreement(prob,dist,scrambled)
            bag = (prob*appearance).sum((-1,-2))
            direct_word = (((scale*(q[ix[block]].cuda()@src.T)).softmax(1) +
                            (scale*(flip[ix[block]].float().cuda()@src.T)).softmax(1))/2) @ bank.T
            origin = zscore(visual[block],mask)
            scores = {'first':origin,'spatial':consistency,'scrambled':sham,'bag':bag,
                      'visual_spatial':origin+zscore(consistency,mask),
                      'visual_scrambled':origin+zscore(sham,mask),'visual_bag':origin+zscore(bag,mask),
                      'words':direct_word.gather(1,idx),'strong':base[block].gather(1,idx)}
            for name,score in scores.items():
                order = score.masked_fill(~mask,-torch.inf).argsort(dim=1,descending=True,stable=True)
                donor = idx.gather(1,order[:,:1]).flatten()
                chosen.setdefault(name,[]).append(donor.cpu())
                hits = (cl[idx.gather(1,order)]==y[rows[block]].cuda()[:,None]) & mask.gather(1,order)
                rr = torch.where(hits.any(1),1/(1+hits.float().argmax(1).float()),0.)
                donor_stats.setdefault(name,[]).append(torch.stack((hits[:,0].float(),rr),1).cpu())
        # All donor selections above precede endpoint scoring. Truth is only read for summary stats.
        lower = (style<=style.median()).nonzero().flatten()
        truth = y[rows].cuda()
        available = ((cl[near]==truth[:,None]) & eligible).any(1)
        strong_miss = ~(cl[base.topk(200).indices]==truth[:,None]).any(1)
        hard = strong_miss & available & ~fallback
        def evaluate(name,score):
            exact = []
            for start in range(0,len(rows),64):
                sl = slice(start,start+64)
                rank,quick = raw_ranks(score[sl],y[rows[sl]],cl,split)
                raw = replay.chunk((rank.cpu(),y[rows[sl]]))
                if (raw-quick).abs().max()>2e-6: raise ValueError('KD replay differs')
                exact.append(raw)
            metrics[name] = torch.cat(exact)
        evaluate('base',base)
        assert (metrics['base']-basepack['kd']).abs().max()<2e-6
        for name,chunks in chosen.items():
            donor = torch.cat(chunks).cuda()
            density = []
            gains = []
            for i in range(0,len(donor),64):
                sl = slice(i,i+64)
                sim = (bank[donor[sl]] @ bank.T).clamp_min(1e-30).log()
                top = sim[:,lower].topk(200).indices
                density.append((cl[lower[top]]==truth[sl,None]).float().mean(1))
                # The named intermediary receives its exact native strong score.
                gain = (sim-sim.mean(1,keepdim=True))/sim.std(1,keepdim=True).clamp_min(1e-7)
                gain.scatter_(1,donor[sl,None],0.)
                gains.append(gain)
            endpoint_stats[name] = torch.cat(density).cpu()
            inc = torch.cat(gains)*base.std(1,keepdim=True)
            evaluate(name+'/0.03125',base+.03125*inc)
        hard = hard.cpu()
        report = {'queries':len(rows),'hard_cases':int(hard.sum()),'fallback_cases':int(fallback.sum()),
                  'mean_marginal_donor_accuracy_mrr':{},'mean_lower_target_p200':{},
                  'official_full_gallery':{name:(100*v.double().mean(0)).tolist() for name,v in metrics.items()}}
        for name,pieces in donor_stats.items():
            stat = torch.cat(pieces)
            report['mean_marginal_donor_accuracy_mrr'][name] = {
                'all':stat.mean(0).tolist(),'hard':stat[hard].mean(0).tolist(),'hard_correct':int(stat[hard,0].sum())}
            report['mean_lower_target_p200'][name] = {
                'all':float(endpoint_stats[name].mean()),'hard':float(endpoint_stats[name][hard].mean())}
        results[split] = report
        torch.save({'per_query_kd':metrics,'chosen':{k:torch.cat(v) for k,v in chosen.items()},'hard':hard,
                    'donors':{k:torch.cat(v) for k,v in donor_stats.items()},'density':endpoint_stats,
                    'labels':y[rows]},folder/(split+'.pt'))
        print(split,'hard',report['hard_cases'],report['mean_marginal_donor_accuracy_mrr'],flush=True)
        del gp,sp,base,basepack,bank
    (folder/'pilot.json').write_text(json.dumps(results,indent=2))


if __name__ == '__main__':
    run()
