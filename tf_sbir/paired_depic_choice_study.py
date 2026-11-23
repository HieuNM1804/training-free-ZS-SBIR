"""Do changes in the winner of a matched visual choice identify retrieval-relevant concepts?"""
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
from .depiction_pivot_geometry import native_patches
from .encoding import load_frozen_clip,save_tensor
from .gallery_context_reversals import coordinates
from .gallery_region_panels import CONFIG,Index
from .geometry_conditioned_words import rank_density,reweight
from .locked_geometry_words import pair_geometry,score_rows
from .paired_depic_choice_law import check_choice_law,held_logscore,infer
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .standardized_context_audit import audit_inputs
from .strong_pilot_scores import location


@torch.inference_mode()
def run(full_words=False):
    w = Path('/content/sbir')
    folder = w/'results/paired-depiction-choices'
    if full_words:
        from .paired_choice_full_vocabulary import infer_full
        from .paired_choice_penalty import check_penalty_kernel
        folder = folder/'full-vocabulary'
    folder.mkdir(parents=True,exist_ok=True)
    law_error = check_choice_law()
    penalty_error = check_penalty_kernel() if full_words else None
    qs,gs,enc,_,q,p,flip,src,tgt,scale = audit_inputs(w,768)
    cfg = runtime(w,'sketchy',workers=min(4,len(os.sched_getaffinity(0))))
    qp,_ = merge_inputs(cfg.root,SKETCHY_SPLITS,'sketch')
    pp,_ = merge_inputs(cfg.root,SKETCHY_SPLITS,'photo')
    kd,_ = load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    vision = load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda')).visual
    grid = torch.cartesian_prod(torch.arange(7),torch.arange(7)).cuda().float()
    dist = torch.cdist(grid,grid)
    perm = torch.randperm(49,generator=torch.Generator().manual_seed(33513)).cuda()
    scrambled = dist[perm][:,perm]
    cache = w/'cache/depiction-pivot-geometry'
    results = {}
    for split,(qi,y) in qs.items():
        rows = torch.randperm(len(qi),generator=torch.Generator().manual_seed(87243))[:768].sort().values
        ix = coordinates(enc,qi[rows])
        pi,gl = gs[split]
        pack = torch.load(location(w,split,rows),mmap=True,weights_only=True)
        assert torch.equal(pack['qi'],qi[rows]) and torch.equal(pack['pi'],pi)
        g = F.normalize(p[pi].cuda())
        x = F.normalize(q[ix].cuda()+flip[ix].float().cuda())
        near = (x@g.T).topk(200).indices
        sub,compact = near.cpu().unique(sorted=True,return_inverse=True)
        gp = native_patches([pp[i] for i in pi[sub].tolist()],vision,cache).cuda()
        paths = [qp[i] for i in qi[rows].tolist()]
        sp = native_patches(paths,vision,cache).cuda()
        flat = native_patches(paths,vision,cache,uniform=True).cuda()
        bank = torch.cat([(scale*(v@tgt.T)).softmax(1) for v in g.split(160)])
        parts = torch.randperm(len(g),generator=torch.Generator().manual_seed(69313)).cuda().tensor_split(2)
        fold = torch.empty(len(g),device=g.device,dtype=torch.bool)
        fold[parts[0]],fold[parts[1]]=False,True
        priors = [bank[ids].mean(0) for ids in parts]
        graph = Index(SimpleNamespace(p=g),rgb_only=True)
        replay = PrefixReplay(kd,gl,split)
        cl = gl.cuda()
        measures,predictive,diagnostics={},{},[]
        with ThreadPoolExecutor(max_workers=min(2,len(os.sched_getaffinity(0)))) as pool:
            for start in range(0,len(rows),8):
                sl=slice(start,start+8)
                ii=ix[sl]
                words=((scale*(q[ii].cuda()@src.T)).softmax(1)+(scale*(flip[ii].float().cuda()@src.T)).softmax(1))/2
                # Keep the remaining public vocabulary in a single explicit other category.
                ids=words.topk(512).indices
                cand=near[sl]
                donors=bank[cand]
                real,blank,_=pair_geometry(sp[sl],flat[sl],gp[compact[sl].cuda()],scale,dist,scrambled)
                direct=words@bank.T
                native=graph.score(CONFIG,direct).clamp_min(1e-30).log()
                strong=pack['score'][sl].cuda()
                holders={}
                posts=[]
                metas=[]
                for j,targets in enumerate(parts):
                    valid=fold[cand]!=bool(j)
                    reference=priors[1-j]
                    if full_words:
                        post,meta=infer_full(words,donors,real,blank,valid,reference)
                    else:
                        post,meta=infer(words,donors,real,blank,valid,reference,ids)
                    weights=rank_density(real-blank,valid,8)
                    normal=(valid.float()[:,None]@donors).squeeze(1)+reference
                    excess=(weights[:,None]@donors).squeeze(1)+reference
                    post['previous']=reweight(words,excess/normal.clamp_min(1e-30))
                    projections=torch.cat(list(post.values()))@bank[targets].T
                    for name,chunk in zip(post,projections.split(len(words)),strict=True):
                        holders.setdefault(name,torch.empty_like(direct))[:,targets]=chunk
                    posts.append({**post,'initial':words})
                    metas.append(meta)
                for j in range(2):
                    if not full_words:
                        hold=held_logscore(metas[j],posts[1-j],ids)
                        for k,values in hold.items():predictive.setdefault(k,[]).append(values.cpu())
                    diagnostics.append(torch.stack([metas[j][k] for k in ('retained','switch_mass','separated_switch_mass')],1).cpu())
                improved=graph.score(CONFIG,torch.cat(list(holders.values()))).clamp_min(1e-30).log()
                scores={'base':strong,'word_original':direct}
                for name,part in zip(holders,improved.split(len(words)),strict=True):
                    scores[name]=strong+.65*(part-native)
                for key in (('direction_only' if full_words else 'matched_one'),'matched_log','independent_log','previous'):
                    scores['word/'+key]=holders[key]
                for name,score in scores.items():
                    val=score_rows(score,y[rows[sl]],cl,split,replay,pool)
                    measures.setdefault(name,[]).append(val)
                if start%192==0:print(split,start,flush=True)
        raw={k:torch.cat(v).double() for k,v in measures.items()}
        assert (pack['kd']-raw['base'][:,:2]).abs().max()<2e-6
        earlier=torch.load(w/'results/geometry-conditioned-word-evidence'/(split+'.pt'),weights_only=True)
        paired_mean_difference=float((100*(raw['previous'][:,:2]-earlier['metrics']['excess/max8'][:,:2]).mean(0)).abs().max())
        if paired_mean_difference>.025:raise ValueError('Unexplained change in paired reference')
        table={}
        for name,val in raw.items():
            delta=val-raw['previous']
            table[name]={'metrics':(100*val[:,:2].mean(0)).tolist(),
                         'macro_map_change_vs_previous':class_interval(100*delta[:,0],y[rows])}
        diag=torch.cat(diagnostics)
        pred={k:torch.cat(v) for k,v in predictive.items()}
        experiment={'sample':len(rows),'gallery':len(g),'law_check':law_error,'penalty_kernel_check':penalty_error,'max_mean_replay_difference_pp':paired_mean_difference,
              'mean_retained_prior_and_switch_probabilities':diag.mean(0).tolist(),'rows':table,
              'opposite_fold_predictive_log_gain_over_chance':{k:float(v.mean()) for k,v in pred.items()}}
        results[split]=experiment
        save_tensor({'rows':raw,'labels':y[rows],'heldout':pred,'diagnostics':diag},folder/(split+'.pt'))
        print(split,'result',table['matched_log'],table['direction_only' if full_words else 'matched_one'],flush=True)
        del bank,gp,sp,flat,graph
    (folder/'pilot.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--full-words',action='store_true')
    run(parser.parse_args().full_words)
