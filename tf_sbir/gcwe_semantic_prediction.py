"""Predict unobserved sketch content under a scene word, a literal photo or independent meanings."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import os
from pathlib import Path
from types import SimpleNamespace

from .geometry_word_flat import (CONFIG, COUNTS, Index, KD_FUNCTION_SHA, PrefixReplay, atomic_json, class_interval,
    load_frozen_clip, load_kd, make_word_bank, revision, save_tensor, score_rows, semantic_inputs)
from .regional_predictive_observation import observation_tasks, predictions, scene_bank
from .regional_scale_position_contrast import complete_windows
import torch
from torch.nn import functional as F


def condition(prior,emissions,observed):
    joint=prior[:,:,None]*emissions
    marginal=joint.sum(1)
    posterior=(joint*(observed/marginal.clamp_min(1e-30))[:,None]).sum(2)
    if not torch.allclose(posterior.sum(1),torch.ones_like(posterior[:,0]),atol=2e-5):
        raise ValueError('Soft observation changed total posterior mass')
    return posterior


def log_score(forecast,readout):
    return (readout*forecast.clamp_min(1e-30).log()).sum(-1)


def mean_emission(prior,emissions):
    return (prior[:,None] @ emissions).squeeze(1)


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    w=Path('/content/sbir')
    out=w/'results/gcwe-semantic-prediction'
    out.mkdir(parents=True,exist_ok=True)
    inputs=semantic_inputs(w,'sketchy')
    cfg,_,qs,pp,gs,q,p,src,tgt,scale=inputs
    views=complete_windows(w,'sketchy',inputs)
    vision=load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda')).visual
    kd,_=load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    tasks=list(observation_tasks())
    for split,(qi,labels) in qs.items():
        pi,gl=gs[split]
        assert (len(qi),len(pi),len(gl.unique()))==COUNTS[split]
        ix=torch.randperm(len(qi),generator=torch.Generator().manual_seed(42))[4608:5376].sort().values
        parts=torch.randperm(len(pi),generator=torch.Generator().manual_seed(42)).cuda().tensor_split(2)
        heads,photo_regions=scene_bank(w,split,pp,pi,p,tgt,scale,vision)
        gallery=p[pi].cuda()
        bank=make_word_bank(gallery,tgt,scale,128)
        endpoint_banks=[bank[part].T.contiguous() for part in parts]
        donor_keys=[gallery[part].T.contiguous() for part in parts]
        graph=Index(SimpleNamespace(p=gallery),rgb_only=True)
        replay=PrefixReplay(kd,gl,split)
        classes=gl.cuda()
        calc_names=('word','independent','literal_donor','wrong_layout','scene_prior','scene_updated','scene_direct','scene_coverage')
        retrieval_names=('word','prior','direct','wrong_layout')
        all_local,all_metrics=[],{}
        with ThreadPoolExecutor(max_workers=min(2,len(os.sched_getaffinity(0)))) as pool:
            for batch in ix.split(12):
                ids=qi[batch]
                n=len(ids)
                crops=views['regions'][ids,:9].float().cuda()
                logits=scale*(crops @ src.T)
                complete=logits.softmax(-1).mean(2)
                local=logits[:,:,:,heads].softmax(-1).mean(2)
                # Both views encode only A when the proposal and transport are constructed.
                crop_visual=F.normalize(crops.mean(2),dim=-1)
                scores={k:gallery.new_empty(n,len(pi)) for k in retrieval_names}
                local_metrics=gallery.new_zeros(n,len(calc_names))
                for fold in (0,1):
                    eligible=parts[1-fold]
                    accumulated={k:gallery.new_zeros(n,len(src)) for k in retrieval_names}
                    for first in range(0,len(tasks),2):
                        a=tasks[first][0]
                        mass,scene=complete[:,a].topk(64,dim=-1)
                        prior=mass/mass.sum(1,keepdim=True)
                        near=eligible[(crop_visual[:,a] @ donor_keys[1-fold]).topk(100).indices]
                        emission=bank[near[:,None,:],scene[:,:,None]]
                        donors=emission/emission.sum(2,keepdim=True).clamp_min(1e-30)
                        donor_prior=mean_emission(prior,donors)
                        observations=[]
                        for anchor,b,s,d in tasks[first:first+2]:
                            assert anchor==a
                            outputs=predictions(local[:,a],photo_regions[near],s,d)
                            observations.append((b,outputs['conditional'],outputs['shuffle1']))
                        for j in (0,1):
                            b,hb,shb=observations[j]
                            c,hc,shc=observations[1-j]
                            mb,mc=donors @ hb,donors @ hc
                            unconditional=mean_emission(prior,mc)
                            posterior=condition(prior,mb,local[:,b])
                            literal=condition(donor_prior,hb,local[:,b])
                            altered=condition(prior,donors @ shb,local[:,b])
                            actual=mean_emission(posterior,mc)
                            sham=mean_emission(altered,donors @ shc)
                            if not torch.allclose(unconditional,mean_emission(donor_prior,hc),atol=1e-6):
                                raise ValueError('Different priors in competing generative models')
                            if not torch.allclose(prior,condition(prior,mb,mean_emission(prior,mb)),atol=2e-6):
                                raise ValueError('Neutral observation changed the semantic prior')
                            b_scene=complete[:,b].gather(1,scene)
                            pooled=.5*(prior+b_scene/b_scene.sum(1,keepdim=True).clamp_min(1e-30))
                            c_scene=complete[:,c].gather(1,scene)
                            coverage=c_scene.sum(1)
                            truth=c_scene/coverage[:,None].clamp_min(1e-30)
                            metrics=[log_score(f,local[:,c]) for f in
                                     (actual,unconditional,mean_emission(literal,hc),sham)]
                            metrics += [log_score(f,truth) for f in (prior,posterior,pooled)]+[coverage]
                            local_metrics+=torch.stack(metrics,1)/16
                            for k,v in zip(retrieval_names,(posterior,prior,pooled,altered)):
                                accumulated[k].scatter_add_(1,scene,v/8)
                    targets=parts[fold]
                    for k,v in accumulated.items():
                        scores[k][:,targets]=v @ endpoint_banks[fold]
                all_local.append(local_metrics.cpu())
                poses=torch.stack((q[ids].cuda(),views['flipped'][ids].float().cuda()),1)
                scores['native_semantic']=(scale*(poses @ src.T)).softmax(2).mean(1) @ bank.T
                for k,v in scores.items():
                    val=score_rows(graph.score(CONFIG,v),labels[batch],classes,split,replay,pool)
                    all_metrics.setdefault(k,[]).append(val[:,:2])
        local_rows=torch.cat(all_local).double()
        retrieval={k:torch.cat(v).double() for k,v in all_metrics.items()}
        effects={}
        for k in ('independent','literal_donor','wrong_layout'):
            delta=local_rows[:,0]-local_rows[:,calc_names.index(k)]
            effects[k]=dict(mean=float(delta.mean()),class_delta=class_interval(delta,labels[ix],seed=42))
        save_tensor(dict(qi=qi[ix],pi=pi,labels=labels[ix],rows=retrieval,predictive=local_rows),out/(split+'.pt'))
        atomic_json(out/(split+'.json'),dict(commit=revision(),seed=42,device=torch.cuda.get_device_name(),split=split,
            queries=len(ix),gallery=len(pi),evaluator=KD_FUNCTION_SHA,predictive_columns=list(calc_names),
            predictive_means=dict(zip(calc_names,local_rows.mean(0).tolist())),word_minus=effects,
            retrieval_percent={k:(100*v.mean(0)).tolist() for k,v in retrieval.items()}))
        print(split,effects,flush=True)


if __name__=='__main__':
    run()
