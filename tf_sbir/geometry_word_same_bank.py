"""Frozen domain-name ablation of the independent, two-stage retrieval method."""
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
from .conditional_semantic_index import bare_words
from .crossdomain_inputs import runtime
from .data import SKETCHY_SPLITS
from .depiction_path_diagnostic import class_interval
from .depiction_pivot_geometry import native_patches
from .encoding import load_frozen_clip, save_tensor
from .gallery_context_reversals import coordinates
from .gallery_region_panels import CONFIG, Index
from .geometry_conditioned_words import rank_density, reweight
from .locked_geometry_words import LOCK, pair_geometry, score_rows
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .semantic import read_words
from .standardized_context_audit import audit_inputs


@torch.inference_mode()
def run(dataset):
    w = Path('/content/sbir')
    root = w/'results/geometry-conditioned-word-evidence'
    folder = root/'domain-name-ablation'
    folder.mkdir(exist_ok=True,parents=True)
    transfer = dataset=='tuberlin'
    qs,gs,enc,_,q,p,flip,_,_,scale = audit_inputs(w,768,all_sketches=transfer,dataset=dataset)
    cfg = runtime(w,dataset,workers=min(4,len(os.sched_getaffinity(0))))
    splits = ('tuberlin',) if transfer else SKETCHY_SPLITS
    qpaths,_ = merge_inputs(cfg.root,splits,'sketch')
    ppaths,_ = merge_inputs(cfg.root,splits,'photo')
    names = read_words(cfg.vocabulary)
    terms,own_scale = bare_words(cfg,names)
    assert len(names)==20931 and abs(own_scale-scale)<.01
    vision = load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda')).visual
    kd,_=load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    coords = torch.cartesian_prod(torch.arange(7),torch.arange(7)).cuda().float()
    dist = torch.cdist(coords,coords)
    perm = torch.randperm(49,generator=torch.Generator().manual_seed(33513)).cuda()
    shuffled = dist[perm][:,perm]
    cache = w/'cache/depiction-pivot-geometry'
    results = {}
    for split,(qi,labels) in qs.items():
        rows = torch.arange(len(qi)) if transfer else torch.randperm(len(qi),generator=torch.Generator().manual_seed(87243))[:768].sort().values
        ix = coordinates(enc,qi[rows])
        pi,gl = gs[split]
        if transfer and (len(qi),len(pi),len(gl.unique()))!=(2130,27262,30):raise ValueError('Incomplete TU transfer')
        g = F.normalize(p[pi].cuda())
        x = F.normalize(q[ix].cuda()+flip[ix].float().cuda())
        near = torch.cat([(z@g.T).topk(200).indices for z in x.split(128)])
        subset,compact = near.cpu().unique(sorted=True,return_inverse=True)
        gp = native_patches([ppaths[i] for i in pi[subset].tolist()],vision,cache).cuda()
        paths = [qpaths[i] for i in qi[rows].tolist()]
        sp = native_patches(paths,vision,cache).cuda()
        flat = native_patches(paths,vision,cache,uniform=True).cuda()
        # Both sides encode exactly the same bare noun. No domain description or class metadata is used.
        bank = torch.cat([(scale*(z@terms.T)).softmax(1) for z in g.split(160)])
        parts = torch.randperm(len(g),generator=torch.Generator().manual_seed(69313)).cuda().tensor_split(2)
        fold = torch.empty(len(g),device=g.device,dtype=torch.bool)
        fold[parts[0]],fold[parts[1]] = False,True
        priors = [bank[ids].mean(0) for ids in parts]
        graph = Index(SimpleNamespace(p=g),rgb_only=True)
        cl = gl.cuda()
        replay = PrefixReplay(kd,gl,split)
        results_per_query = {}
        with ThreadPoolExecutor(max_workers=min(2,len(os.sched_getaffinity(0)))) as pool:
            for start in range(0,len(rows),16):
                sl = slice(start,start+16)
                candidates = near[sl]
                ix_batch = ix[sl]
                words = ((scale*(q[ix_batch].cuda()@terms.T)).softmax(1)+(scale*(flip[ix_batch].float().cuda()@terms.T)).softmax(1))/2
                direct = words@bank.T
                real,blank,sham = pair_geometry(sp[sl],flat[sl],gp[compact[sl].cuda()],scale,dist,shuffled)
                selections = {'excess':real-blank,'scrambled':sham,'visual':(x[sl]@g.T).gather(1,candidates)}
                delta = {k:torch.empty_like(direct) for k in selections}
                donors = bank[candidates]
                for j,targets in enumerate(parts):
                    valid = fold[candidates] != bool(j)
                    reference = valid.float()[:,None,:].bmm(donors).squeeze(1)+priors[1-j]
                    for k,signal in selections.items():
                        ratios = rank_density(signal,valid,8)
                        changed = ratios[:,None,:].bmm(donors).squeeze(1)+priors[1-j]
                        posterior = reweight(words,changed/reference.clamp_min(1e-30))
                        delta[k][:,targets] = posterior@bank[targets].T
                inputs = {'original':direct,**delta}
                smoothed = graph.score(CONFIG,torch.cat(list(inputs.values())))
                for name,part in zip(list(inputs),smoothed.split(len(words)),strict=True):
                    for tag,score in [(name,inputs[name]),('diffused/'+name,part)]:
                        metric = score_rows(score,labels[rows[sl]],cl,split,replay,pool)
                        results_per_query.setdefault(tag,[]).append(metric)
        raw = {k:torch.cat(v) for k,v in results_per_query.items()}
        data = {}
        for kind,val in raw.items():
            reference = raw['diffused/original' if kind.startswith('diffused/') else 'original']
            d = 100*(val[:,0]-reference[:,0])
            data[kind] = {'metrics':(100*val[:,:2].double().mean(0)).tolist(),
                          'class_macro_delta_vs_own_neutral_base':class_interval(d,labels[rows])}
        results[split] = {'lock':LOCK,'queries':len(rows),'gallery':len(gl),'rows':data}
        save_tensor({'scores':raw,'labels':labels[rows],'qi':qi[rows]},folder/(split+'.pt'))
        print(split,data['original']['metrics'],data['excess']['metrics'],flush=True)
        del bank,gp,sp,flat,graph
    (folder/(dataset+'.json')).write_text(json.dumps(results,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=('sketchy','tuberlin'),required=True)
    run(parser.parse_args().dataset)
