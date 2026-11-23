"""Test content localization versus shared absolute-coordinate preferences between modalities."""
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
from .depiction_pivot_geometry import distance_agreement,native_patches
from .encoding import load_frozen_clip,save_tensor
from .gallery_context_reversals import coordinates
from .gallery_region_panels import CONFIG,Index
from .geometry_conditioned_words import rank_density,reweight
from .independent_patch_frames import affinities,frame_indices,mutual,read_frames
from .locked_geometry_words import score_rows
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .standardized_context_audit import audit_inputs
from .strong_pilot_scores import location


@torch.inference_mode()
def run():
    w=Path('/content/sbir')
    root=w/'results/independent-patch-frames'
    root.mkdir(exist_ok=True,parents=True)
    qs,gs,enc,_,q,p,flip,src,tgt,scale=audit_inputs(w,768)
    cfg=runtime(w,'sketchy',workers=min(4,len(os.sched_getaffinity(0))))
    qp,_=merge_inputs(cfg.root,SKETCHY_SPLITS,'sketch')
    pp,_=merge_inputs(cfg.root,SKETCHY_SPLITS,'photo')
    kd,_=load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    vision=load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda')).visual
    frames=frame_indices('cuda')
    coords=torch.cartesian_prod(torch.arange(7),torch.arange(7)).cuda().float()
    dist=torch.cdist(coords,coords)
    summaries={}
    for split,(qi,y) in qs.items():
        rows=torch.randperm(len(qi),generator=torch.Generator().manual_seed(87243))[:768].sort().values
        ix=coordinates(enc,qi[rows])
        pi,gl=gs[split]
        pack=torch.load(location(w,split,rows),mmap=True,weights_only=True)
        assert torch.equal(pack['qi'],qi[rows]) and torch.equal(pack['pi'],pi)
        g=F.normalize(p[pi].cuda())
        x=F.normalize(q[ix].cuda()+flip[ix].float().cuda())
        near=(x@g.T).topk(200).indices
        sub,compact=near.cpu().unique(sorted=True,return_inverse=True)
        gp=read_frames(w,[pp[i] for i in pi[sub].tolist()],vision,frames)
        paths=[qp[i] for i in qi[rows].tolist()]
        sp=read_frames(w,paths,vision,frames)
        blank=native_patches(paths,vision,w/'cache/depiction-pivot-geometry',uniform=True).cuda()
        bank=torch.cat([(scale*(z@tgt.T)).softmax(1) for z in g.split(160)])
        parts=torch.randperm(len(g),generator=torch.Generator().manual_seed(69313)).cuda().tensor_split(2)
        fold=torch.empty(len(g),dtype=torch.bool,device=g.device)
        fold[parts[0]],fold[parts[1]]=False,True
        priors=[bank[z].mean(0) for z in parts]
        graph=Index(SimpleNamespace(p=g),rgb_only=True)
        replay=PrefixReplay(kd,gl,split)
        cl=gl.cuda()
        collected,probe,anchors={},{},{}
        max_error=0.
        with ThreadPoolExecutor(max_workers=min(2,len(os.sched_getaffinity(0)))) as pool:
            for start in range(0,len(rows),6):
                sl=slice(start,start+6)
                ii=ix[sl]
                cand=near[sl]
                qviews=[z[sl] for z in sp]
                frame_b=[blank[sl][:,r] for r in frames]
                gviews=[z[compact[sl].cuda()] for z in gp]
                logits=affinities(qviews,gviews,scale)
                flat_logits=affinities(frame_b,gviews,scale)
                if start==0:
                    subset=[z[:1,:2].float() for z in gviews]
                    expected=sum(qz[:1,None].float()@gz.transpose(-1,-2) for qz in qviews for gz in subset)*(scale/16)
                    offdiag=sum(qz[:1,None].float()@gz.transpose(-1,-2) for i,qz in enumerate(qviews)
                                for j,gz in enumerate(subset) if i!=j)*(scale/12)
                    max_error=max(float((expected-logits['independent'][:1,:2]).abs().max()),
                                  float((offdiag-logits['mismatch'][:1,:2]).abs().max()))
                    if max_error>.002:raise ValueError('Frame moment identity failed')
                geom={}
                for name,z in logits.items():
                    j=mutual(z); b=mutual(flat_logits[name])
                    full=distance_agreement(j,dist,dist)
                    floor=distance_agreement(b,dist,dist)
                    geom[name]=full-floor
                    if name in ('native','same','independent'):
                        geom['plain/'+name]=full
                    def diagonal(m):return m.diagonal(dim1=-2,dim2=-1).sum(-1).mean(1)
                    trace=torch.stack([diagonal(j),diagonal(b),full.mean(1),floor.mean(1)],1)
                    probe.setdefault(name,[]).append(trace.cpu())
                words=((scale*(q[ii].cuda()@src.T)).softmax(1)+(scale*(flip[ii].float().cuda()@src.T)).softmax(1))/2
                direct=words@bank.T
                native=graph.score(CONFIG,direct).clamp_min(1e-30).log()
                strong=pack['score'][sl].cuda()
                donors=bank[cand]
                holders={k:torch.empty_like(direct) for k in geom}
                for j,targets in enumerate(parts):
                    valid=fold[cand]!=bool(j)
                    prior=priors[1-j]
                    normal=(valid.float()[:,None]@donors).squeeze(1)+prior
                    post=[]
                    for k,z in geom.items():
                        weights=rank_density(z,valid,8)
                        observed=(weights[:,None]@donors).squeeze(1)+prior
                        post.append(reweight(words,observed/normal.clamp_min(1e-30)))
                    scored=torch.cat(post)@bank[targets].T
                    for name,chunk in zip(geom,scored.split(len(words)),strict=True):holders[name][:,targets]=chunk
                diff=graph.score(CONFIG,torch.cat(list(holders.values()))).clamp_min(1e-30).log()
                scores={'base':strong}
                for name,z in zip(holders,diff.split(len(words)),strict=True):scores[name]=strong+.65*(z-native)
                for name in ('native','same','independent','mismatch','shared_component'):
                    scores['word/'+name]=holders[name]
                for name,z in scores.items():
                    v=score_rows(z,y[rows[sl]],cl,split,replay,pool)
                    collected.setdefault(name,[]).append(v)
                # Relevance only characterizes the automatic within-nomination ranking.
                relevant=cl[cand]==y[rows[sl]].cuda()[:,None]
                for name,z in geom.items():
                    rank=z.argsort(1,descending=True)
                    hits=relevant.gather(1,rank)
                    anchors.setdefault(name,[]).append(torch.stack([hits[:,0].float(),hits[:,:16].float().mean(1)],1).cpu())
                if start%192==0:print(split,start,flush=True)
        raw={k:torch.cat(v).double() for k,v in collected.items()}
        anchor={k:torch.cat(v) for k,v in anchors.items()}
        assert (raw['base'][:,:2]-pack['kd']).abs().max()<2e-6
        previous=torch.load(w/'results/geometry-conditioned-word-evidence'/(split+'.pt'),weights_only=True)['metrics']['excess/max8']
        replay_error=float((100*(raw['native'][:,:2]-previous[:,:2]).mean(0)).abs().max())
        if replay_error>.025:raise ValueError(f'Changed native readout: {replay_error}')
        table={}
        for name,z in raw.items():
            reference=raw['word/native'] if name.startswith('word/') else raw['native']
            table[name]={'metrics':(100*z[:,:2].mean(0)).tolist(),
                    'macro_delta_vs_native':class_interval(100*(z[:,0]-reference[:,0]),y[rows])}
        measures={k:torch.cat(v) for k,v in probe.items()}
        summaries[split]={'queries':len(rows),'gallery':len(g),'max_kernel_identity_error':max_error,
            'max_mean_reference_error_pp':replay_error,'rows':table,
            'nominee_metrics_top1_top16':{k:v.mean(0).tolist() for k,v in anchor.items()},
            'mean_real_blank_diagonal_then_real_blank_geom':{k:v.mean(0).tolist() for k,v in measures.items()}}
        save_tensor({'scores':raw,'labels':y[rows],'anchors':anchor,'probe':measures},root/(split+'.pt'))
        print(split,table['independent'],summaries[split]['mean_real_blank_diagonal_then_real_blank_geom'],flush=True)
        del bank,gp,sp,blank,graph
    (root/'pilot.json').write_text(json.dumps(summaries,indent=2))


if __name__=='__main__':
    run()
