"""Assign a word to the positions which actually support the pair's spatial verification."""
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
from .locked_geometry_words import score_rows
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .standardized_context_audit import audit_inputs
from .strong_pilot_scores import location


def local_support(logits,dist):
    joint=(logits.softmax(-1)+logits.softmax(-2))/(2*logits.size(-1))
    a,b=joint.sum(-1),joint.sum(-2)
    def moments(m):
        mu=(m*(m@dist)).sum(-1)
        var=(m*(m@dist.square())).sum(-1)-mu.square()
        return mu,var.clamp_min(0)
    mx,vx=moments(a)
    my,vy=moments(b)
    dx=dist-mx[...,None,None]
    dy=dist-my[...,None,None]
    conditional=(dx@joint@dy.transpose(-1,-2))/(vx*vy).sqrt().clamp_min(1e-7)[...,None,None]
    attribution=(joint*conditional).sum(-2)
    rho=attribution.sum(-1)
    centered=attribution-rho[...,None]*b
    return rho,centered,joint


def vector_weight(mass,features):
    return (mass.unsqueeze(-2)@features.float()).squeeze(-2)


@torch.inference_mode()
def run():
    w=Path('/content/sbir')
    folder=w/'results/geometry-grounded-words'
    folder.mkdir(parents=True,exist_ok=True)
    qs,gs,enc,_,q,p,flip,src,tgt,scale=audit_inputs(w,768)
    cfg=runtime(w,'sketchy',workers=min(4,len(os.sched_getaffinity(0))))
    qp,_=merge_inputs(cfg.root,SKETCHY_SPLITS,'sketch')
    pp,_=merge_inputs(cfg.root,SKETCHY_SPLITS,'photo')
    kd,_=load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    vision=load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda')).visual
    grid=torch.cartesian_prod(torch.arange(7),torch.arange(7)).cuda().float()
    dist=torch.cdist(grid,grid)
    shuffled=torch.randperm(49,generator=torch.Generator().manual_seed(33513)).cuda()
    mirrored=torch.arange(49,device='cuda').view(7,7).flip(1).flatten()
    source=w/'cache/depiction-pivot-geometry'
    result={}
    for s,(qi,y) in qs.items():
        rows=torch.randperm(len(qi),generator=torch.Generator().manual_seed(87243))[:768].sort().values
        pi,gl=gs[s]
        ix=coordinates(enc,qi[rows])
        pack=torch.load(location(w,s,rows),weights_only=True,mmap=True)
        assert torch.equal(pack['qi'],qi[rows]) and torch.equal(pack['pi'],pi)
        g=F.normalize(p[pi].cuda())
        x=F.normalize(q[ix].cuda()+flip[ix].float().cuda())
        near=(x@g.T).topk(200).indices
        subset,local=near.cpu().unique(sorted=True,return_inverse=True)
        gp=native_patches([pp[i] for i in pi[subset].tolist()],vision,source).cuda()
        paths=[qp[i] for i in qi[rows].tolist()]
        sp=native_patches(paths,vision,source).cuda()
        canvas=native_patches(paths,vision,source,uniform=True).cuda()
        bank=torch.cat([(scale*(z@tgt.T)).softmax(1) for z in g.split(160)])
        parts=torch.randperm(len(g),generator=torch.Generator().manual_seed(69313)).cuda().tensor_split(2)
        fold=torch.empty(len(g),dtype=torch.bool,device='cuda')
        fold[parts[0]],fold[parts[1]]=False,True
        priors=[bank[v].mean(0) for v in parts]
        graph=Index(SimpleNamespace(p=g),rgb_only=True)
        replay=PrefixReplay(kd,gl,s)
        cl=gl.cuda()
        collected,candidate_tests,probe={},{},[]
        max_error=0.
        with ThreadPoolExecutor(max_workers=min(2,len(os.sched_getaffinity(0)))) as pool:
            for start in range(0,len(ix),8):
                sl=slice(start,start+8)
                cand=near[sl]
                feat=gp[local[sl].cuda()]
                logits=scale*(sp[sl,None]@feat.transpose(-1,-2)).float()
                blank_logits=scale*(canvas[sl,None]@feat.transpose(-1,-2)).float()
                score,real_patch,joint=local_support(logits,dist)
                reference,blank_patch,blank_joint=local_support(blank_logits,dist)
                real=distance_agreement(joint,dist,dist)
                blank=distance_agreement(blank_joint,dist,dist)
                max_error=max(max_error,float((score-real).abs().max()),float((reference-blank).abs().max()))
                if max_error>1e-5:raise ValueError('Local decomposition no longer represents the same geometric score')
                mass=real_patch-blank_patch
                if mass.sum(-1).abs().max()>1e-5:raise ValueError('Target-location residual must sum to zero')
                correct=vector_weight(mass,feat)
                variants={'native':None,'aligned':correct,'real_only':vector_weight(real_patch,feat),
                    'reverse':-correct,'shuffled':vector_weight(mass,feat[:,:,shuffled]),
                    'mirror':vector_weight(mass,feat[:,:,mirrored]),'aligned_4':4*correct,
                    'shuffled_4':4*vector_weight(mass,feat[:,:,shuffled])}
                words=((scale*(q[ix[sl]].cuda()@src.T)).softmax(1)+(scale*(flip[ix[sl]].float().cuda()@src.T)).softmax(1))/2
                direct=words@bank.T
                native=graph.score(CONFIG,direct).clamp_min(1e-30).log()
                strong=pack['score'][sl].cuda()
                initial=bank[cand]
                logp=initial.clamp_min(1e-30).log()
                holders={}
                previous_mean=0.
                wd={}
                for k,vec in variants.items():
                    noun=initial if vec is None else (logp+scale*(vec@tgt.T)).softmax(-1)
                    wd[k]=(noun@words[:,:,None]).squeeze(-1)
                    if k=='aligned':
                        previous_mean=float((noun*(noun.clamp_min(1e-30).log()-logp)).sum(-1).mean())
                    for j,target in enumerate(parts):
                        valid=fold[cand]!=bool(j)
                        weight=rank_density(real-blank,valid,8)
                        base_mean=(valid.float()[:,None]@initial).squeeze(1)+priors[1-j]
                        enriched=(weight[:,None]@noun).squeeze(1)+priors[1-j]
                        post=reweight(words,enriched/base_mean.clamp_min(1e-30))
                        holders.setdefault(k,torch.empty_like(direct))[:,target]=post@bank[target].T
                        if k in ('aligned','shuffled'):
                            # Condition numerator and denominator on the same local reading.
                            denom=(valid.float()[:,None]@noun).squeeze(1)+priors[1-j]
                            post=reweight(words,enriched/denom.clamp_min(1e-30))
                            holders.setdefault(k+'_both',torch.empty_like(direct))[:,target]=post@bank[target].T
                diffuse=graph.score(CONFIG,torch.cat(list(holders.values()))).clamp_min(1e-30).log()
                outputs={'base':strong}
                for k,dv in zip(holders,diffuse.split(len(words)),strict=True):
                    outputs[k]=strong+.65*(dv-native)
                for k in ('aligned','native','shuffled','aligned_both'):outputs['word/'+k]=holders[k]
                for k,z in outputs.items():
                    row=score_rows(z,y[rows[sl]],cl,s,replay,pool)
                    collected.setdefault(k,[]).append(row)
                # Evaluate within the same shortlist after computing each candidate's revised nouns.
                truth=cl[cand]==y[rows[sl]].cuda()[:,None]
                for k,z in wd.items():
                    hit=truth.gather(1,z.argsort(1,descending=True))
                    candidate_tests.setdefault(k,[]).append(torch.stack([hit[:,0].float(),hit[:,:16].float().mean(1)],1).cpu())
                probe.append([float(correct.norm(dim=-1).mean()),float(mass.abs().sum(-1).mean()),previous_mean])
                if start%192==0:print(s,start,flush=True)
        raw={k:torch.cat(v).double() for k,v in collected.items()}
        assert (raw['base'][:,:2]-pack['kd']).abs().max()<2e-6
        old=torch.load(w/'results/geometry-conditioned-word-evidence'/(s+'.pt'),weights_only=True)['metrics']['excess/max8']
        discrepancy=float((100*(old[:,:2]-raw['native'][:,:2]).mean(0)).abs().max())
        if discrepancy>.025:raise ValueError('Paired native readout changed')
        table={}
        for k,v in raw.items():
            ref=raw['word/native'] if k.startswith('word/') else raw['native']
            table[k]={'metrics':(100*v[:,:2].mean(0)).tolist(),'class_macro_map_gain':class_interval(100*(v[:,0]-ref[:,0]),y[rows])}
        nom={k:torch.cat(v) for k,v in candidate_tests.items()}
        result[s]={'queries':len(ix),'gallery':len(g),'max_geometric_decomposition_error':max_error,
            'previous_mean_discrepancy_pp':discrepancy,'mean_vector_norm_patch_mass_KL':torch.tensor(probe).mean(0).tolist(),
            'candidate_top1_top16':{k:v.mean(0).tolist() for k,v in nom.items()},'rows':table}
        print(s,table['aligned'],table['native'],flush=True)
        save_tensor({'labels':y[rows],'rows':raw,'candidate':nom},folder/(s+'.pt'))
        del gp,sp,canvas,bank,graph
    (folder/'pilot.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    run()
