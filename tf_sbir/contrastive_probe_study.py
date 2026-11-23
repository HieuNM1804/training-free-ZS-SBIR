"""Select comparisons that can discriminate noun hypotheses, then observe their geometry."""
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
from .contrastive_probe_design import BLOCKS,SIZE,design,orientation_posterior,smooth_utilities,test_likelihood
from .crossdomain_inputs import runtime
from .data import SKETCHY_SPLITS
from .depiction_path_diagnostic import class_interval
from .depiction_pivot_geometry import native_patches
from .encoding import load_frozen_clip,save_tensor
from .gallery_context_reversals import coordinates
from .gallery_region_panels import CONFIG,Index
from .geometry_conditioned_words import rank_density,reweight
from .locked_geometry_words import pair_geometry,score_rows
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .standardized_context_audit import audit_inputs
from .strong_pilot_scores import location


def select_pair(ids,a,b,k):
    return torch.stack([ids.gather(2,a[k][:,:,None]).squeeze(2),ids.gather(2,b[k][:,:,None]).squeeze(2)],2)


def pull(values,idx):
    return values.gather(1,idx.flatten(1)).view(idx.shape)


@torch.inference_mode()
def run():
    w=Path('/content/sbir')
    folder=w/'results/contrastive-gallery-design'
    folder.mkdir(exist_ok=True,parents=True)
    integration_error=test_likelihood()
    qs,gs,enc,_,q,p,flip,src,tgt,scale=audit_inputs(w,768)
    cfg=runtime(w,'sketchy',workers=min(4,len(os.sched_getaffinity(0))))
    qp,_=merge_inputs(cfg.root,SKETCHY_SPLITS,'sketch')
    pp,_=merge_inputs(cfg.root,SKETCHY_SPLITS,'photo')
    kd,_=load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    vision=load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda')).visual
    coords=torch.cartesian_prod(torch.arange(7),torch.arange(7)).cuda().float()
    dist=torch.cdist(coords,coords)
    perm=torch.randperm(49,generator=torch.Generator().manual_seed(33513)).cuda()
    scrambled=dist[perm][:,perm]
    cache=w/'cache/depiction-pivot-geometry'
    outputs={}
    for s,(qi,y) in qs.items():
        rows=torch.randperm(len(qi),generator=torch.Generator().manual_seed(87243))[:768].sort().values
        ix=coordinates(enc,qi[rows])
        pi,gl=gs[s]
        pack=torch.load(location(w,s,rows),mmap=True,weights_only=True)
        assert torch.equal(pack['qi'],qi[rows]) and torch.equal(pack['pi'],pi)
        g=F.normalize(p[pi].cuda())
        x=F.normalize(q[ix].cuda()+flip[ix].float().cuda())
        sim=x@g.T
        all_near=sim.topk(200).indices
        targets=torch.randperm(len(g),generator=torch.Generator().manual_seed(69313)).cuda().tensor_split(2)
        fold=torch.empty(len(g),dtype=torch.bool,device='cuda')
        fold[targets[0]],fold[targets[1]]=False,True
        banks=torch.cat([(scale*(u@tgt.T)).softmax(1) for u in g.split(160)])
        priors=[banks[t].mean(0) for t in targets]
        pools=[t[sim[:,t].topk(BLOCKS*SIZE).indices] for t in targets]
        used=torch.cat(pools,1).cpu().unique(sorted=True)
        lookup=torch.full((len(g),),-1,dtype=torch.long);lookup[used]=torch.arange(len(used))
        gp=native_patches([pp[i] for i in pi[used].tolist()],vision,cache).cuda()
        paths=[qp[i] for i in qi[rows].tolist()]
        sp=native_patches(paths,vision,cache).cuda()
        blank=native_patches(paths,vision,cache,uniform=True).cuda()
        replay=PrefixReplay(kd,gl,s)
        cl=gl.cuda()
        graph=Index(SimpleNamespace(p=g),rgb_only=True)
        grouping=[torch.randperm(BLOCKS*SIZE,generator=torch.Generator().manual_seed(41903+k)).cuda().view(BLOCKS,SIZE) for k in range(2)]
        coins=[torch.randint(SIZE*(SIZE-1)//2,(1,BLOCKS),generator=torch.Generator().manual_seed(70201+k)).cuda() for k in range(2)]
        metrics,design_probe,held,head_obs={},{},{},{}
        progress=[]
        with ThreadPoolExecutor(max_workers=min(2,len(os.sched_getaffinity(0)))) as threads:
            for start in range(0,len(rows),8):
                sl=slice(start,start+8)
                ii=ix[sl]
                words=((scale*(q[ii].cuda()@src.T)).softmax(1)+(scale*(flip[ii].float().cuda()@src.T)).softmax(1))/2
                direct=words@banks.T
                native=graph.score(CONFIG,direct).clamp_min(1e-30).log()
                strong=pack['score'][sl].cuda()
                holders={}
                post_folds=[]
                data_folds=[]
                traces={}
                for source_id in range(2):
                    target=targets[1-source_id]
                    prior=priors[source_id]
                    ids=pools[source_id][sl]
                    geom=[]
                    # Offline comparison evaluates the union of possible probes.
                    # Each deployable design reads only its selected two photos per block.
                    for subset in ids.split(100,dim=1):
                        patches=gp[lookup[subset.cpu()].cuda()]
                        real,canvas,_=pair_geometry(sp[sl],blank[sl],patches,scale,dist,scrambled)
                        geom.append(real-canvas)
                    geom=torch.cat(geom,1)
                    grouped=ids[:,grouping[source_id]]
                    choices,info,ai,bi,agreement,regret=design(banks,grouped,words,prior,coins[source_id],check=(start%96==0))
                    closest_positions=grouping[source_id].argsort(1)[:,:2].sort(1).values
                    cidx=((ai[:,None]==closest_positions[:,0])&(bi[:,None]==closest_positions[:,1])).int().argmax(0)
                    choices['closest']=cidx.expand(len(ii),-1)
                    if agreement is not None:progress.append(torch.stack([agreement,regret],1).cpu())
                    posts={}
                    orientation_data={}
                    for name,choice in choices.items():
                        pairs=select_pair(grouped,ai,bi,choice)
                        opts=select_pair(grouping[source_id].expand(len(ii),-1,-1),ai,bi,choice)
                        actual=pull(geom,opts)
                        sign=(actual[:,:,0]-actual[:,:,1]).sign()
                        prob=smooth_utilities(banks,pairs,prior)
                        estimate,words_rel,evidence=orientation_posterior(words,prob,sign)
                        posts[name+'/bayes']=estimate
                        one=(1+(2*prob-1)*sign[:,:,None]).mean(1)
                        posts[name+'/one']=reweight(words,one)
                        if name in ('information','random'):
                            flipped,_,_=orientation_posterior(words,prob,-sign)
                            posts[name+'/reverse']=flipped
                        selected=pairs.flatten(1)
                        weight=rank_density(actual.flatten(1),torch.ones_like(selected,dtype=torch.bool),8)
                        pool_words=banks[selected]
                        enriched=(weight[:,None]@pool_words).squeeze(1)+prior
                        normal=pool_words.sum(1)+prior
                        posts[name+'/frequency']=reweight(words,enriched/normal.clamp_min(1e-30))
                        orientation_data[name]=(pairs,sign)
                        # Dot with future probabilities to include word/reliability dependence.
                        posts[name+'/weighted_internal']=words_rel
                        chosen_info=info.gather(2,choice[:,:,None]).squeeze(2).mean(1)
                        meas=torch.stack([chosen_info,words_rel.sum(1),evidence],1).cpu()
                        design_probe.setdefault(name,[]).append(meas)
                        traces.setdefault(name,[]).append(selected)
                    # Actual top-200 reference, only the opposite-fold donors.
                    mask=fold[all_near[sl]]==bool(source_id)
                    threshold=sim[sl].gather(1,all_near[sl][:,-1:])
                    relevant_for_base=sim[sl].gather(1,ids)>=threshold
                    # Resolve near ties by exact head membership.
                    if not torch.equal(relevant_for_base.sum(1),mask.sum(1)):
                        relevant_for_base=(ids[:,:,None]==all_near[sl,None,:]).any(2)
                    mass=rank_density(geom,relevant_for_base,8)
                    donor_words=banks[ids]
                    numerator=(mass[:,None]@donor_words).squeeze(1)+prior
                    denominator=(relevant_for_base.float()[:,None]@donor_words).squeeze(1)+prior
                    posts['previous']=reweight(words,numerator/denominator.clamp_min(1e-30))
                    keys=[k for k in posts if not k.endswith('_internal')]
                    projected=torch.cat([posts[k] for k in keys])@banks[target].T
                    for name,part in zip(keys,projected.split(len(ii)),strict=True):
                        holders.setdefault(name,torch.empty_like(direct))[:,target]=part
                    post_folds.append(posts)
                    data_folds.append(orientation_data)
                for j in range(2):
                    for name in data_folds[j]:
                        pair,observed=data_folds[j][name]
                        prob=smooth_utilities(banks,pair,priors[j])
                        signed=(2*prob-1)*observed[:,:,None]
                        opp=post_folds[1-j]
                        def logpred(mix):
                            return torch.log1p((signed@mix[:,:,None]).squeeze(2).clamp(-.9999999,.9999999)).mean(1)
                        # A uniform reliability prior has mean 1/2 before observing a comparison.
                        held.setdefault(name,[]).append(torch.stack([logpred(.5*words),
                            logpred(opp[name+'/weighted_internal']),logpred(.5*opp[name+'/frequency']),
                            logpred(.5*opp['previous'])],1).cpu())
                changed=graph.score(CONFIG,torch.cat(list(holders.values()))).clamp_min(1e-30).log()
                scores={'base':strong}
                for name,part in zip(holders,changed.split(len(ii)),strict=True):
                    scores[name]=strong+.65*(part-native)
                for name in ('information/bayes','information/frequency','random/bayes','previous'):
                    scores['word/'+name]=holders[name]
                for name,score in scores.items():
                    vals=score_rows(score,y[rows[sl]],cl,s,replay,threads)
                    metrics.setdefault(name,[]).append(vals)
                # Labels only assess available correct images after final automatic rankings.
                truth=y[rows[sl]].cuda()[:,None]
                miss=~(cl[strong.topk(200).indices]==truth).any(1)
                for name,lists in traces.items():
                    panel=torch.cat(lists,1)
                    if not (panel.sort(1).values.diff(1)!=0).all():raise ValueError('Duplicate probe photo')
                    hits=cl[panel]==truth
                    head_obs.setdefault(name,[]).append(torch.stack([miss,hits.any(1),hits.sum(1)],1).cpu())
                if start%192==0:print(s,start,flush=True)
        raw={k:torch.cat(v).double() for k,v in metrics.items()}
        assert (raw['base'][:,:2]-pack['kd']).abs().max()<2e-6
        prev=torch.load(w/'results/geometry-conditioned-word-evidence'/(s+'.pt'),weights_only=True)['metrics']['excess/max8']
        discrepancy=float((100*(raw['previous'][:,:2]-prev[:,:2]).mean(0)).abs().max())
        if discrepancy>.025:raise ValueError('Existing estimator failed paired replay')
        table={}
        for name,vals in raw.items():
            reference=raw['word/previous'] if name.startswith('word/') else raw['previous']
            table[name]={'metrics':(100*vals[:,:2].mean(0)).tolist(),
                         'macro_map_delta':class_interval(100*(vals[:,0]-reference[:,0]),y[rows])}
        probe={k:torch.cat(v) for k,v in design_probe.items()}
        predict={k:torch.cat(v) for k,v in held.items()}
        heads={k:torch.cat(v) for k,v in head_obs.items()}
        outcomes={}
        for name,data in heads.items():
            mask=data[:,0].bool()
            outcomes[name]={'base_misses':int(mask.sum()),'contains_correct_on_base_miss':int(data[mask,1].sum()),
                'mean_positives_on_base_miss':float(data[mask,2].float().mean()),
                'mean_design_MI_reliability_log_evidence':probe[name].mean(0).tolist(),
                'held_logscore_prior_bayes_frequency_previous':predict[name].mean(0).tolist()}
        outputs[s]={'queries':len(rows),'gallery':len(g),'integration_error':integration_error,
            'previous_mean_replay_error_pp':discrepancy,'rows':table,'design':outcomes,
            'agreement_512_vs_2048_then_MI_regret':torch.cat(progress).mean(0).tolist()}
        save_tensor({'rows':raw,'labels':y[rows],'panels':heads,'design':probe,'held':predict},folder/(s+'.pt'))
        print(s,table['information/bayes'],outcomes['information'],flush=True)
        del gp,banks,graph
    (folder/'pilot.json').write_text(json.dumps(outputs,indent=2))


if __name__=='__main__':
    run()
