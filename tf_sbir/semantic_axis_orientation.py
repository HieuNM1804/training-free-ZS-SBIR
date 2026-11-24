"""Measure what matched second-order visual geometry explains in CLIP word retrieval."""
import hashlib
import inspect
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from .geometry_word_flat import (CONFIG, COUNTS, Index, KD_FUNCTION_SHA, PrefixReplay,
    atomic_json, class_interval, load_kd, revision, save_tensor, score_rows, semantic_inputs)
from .regional_scale_position_contrast import complete_windows
import torch
import torch.nn.functional as F


def entropy(prob):
    return -(prob * prob.clamp_min(1e-30).log()).sum(-1)


def entropy_match(logits, target):
    logits = logits - logits.amax(-1, keepdim=True)
    upper = torch.ones_like(target)
    for _ in range(16):
        above = entropy((upper[:, None] * logits).softmax(-1)) > target
        if not above.any():
            break
        upper = torch.where(above, 2 * upper, upper)
    lower = torch.zeros_like(upper)
    for _ in range(19):
        mid = (lower + upper) * .5
        above = entropy((mid[:, None] * logits).softmax(-1)) > target
        lower = torch.where(above, mid, lower)
        upper = torch.where(above, upper, mid)
    result = (((lower + upper) * .5)[:, None] * logits).softmax(-1)
    if (entropy(result) - target).abs().max() > 2e-4:
        raise ValueError('Entropy matching failed at the specified tolerance')
    return result


def posterior_bank(features, prompts, scale, mean, transform, target_entropy=None):
    bank = features.new_empty((len(features),len(prompts)))
    entropies = []
    for first in range(0,len(features),192):
        sl = slice(first,first+192)
        x = features[sl]
        if transform is not None:
            x = (x - mean) @ transform + mean
        logits = scale * (x @ prompts.T)
        prob = logits.softmax(-1) if target_entropy is None else entropy_match(logits,target_entropy[sl])
        bank[sl] = prob
        entropies.append(entropy(prob))
    return bank,torch.cat(entropies)


def reflection_bank(gallery):
    x = gallery.double()
    mean = x.mean(0)
    centered = x - mean
    cov = centered.T @ centered / len(x)
    _, vec = torch.linalg.eigh(cov.cpu())
    vec = vec.to(gallery.device)
    generator = torch.Generator().manual_seed(42)
    matrices,checks = [],[]
    for _ in range(3):
        signs = (2*torch.randint(2,(len(mean),),generator=generator)-1).to(vec)
        r = ((vec*signs) @ vec.T).float()
        rd = r.double()
        error = float((rd.T @ cov @ rd - cov).norm() / cov.norm())
        orth = float((rd.T @ rd - torch.eye(len(mean),device=rd.device)).abs().max())
        if error > 2e-6 or orth > 2e-6:
            raise ValueError('The transform failed its moment or geometry identity')
        checks.append(dict(relative_covariance_error=error,orthogonality_error=orth))
        matrices.append(r)
    return mean.float(),matrices,checks


@torch.inference_mode()
def run():
    import random
    import numpy as np
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    w = Path('/content/sbir')
    root = w/'results/semantic-axis-orientation'
    root.mkdir(parents=True,exist_ok=True)
    inputs = semantic_inputs(w,'sketchy')
    _,_,qs,_,gs,q,p,src,tgt,scale = inputs
    views = complete_windows(w,'sketchy',inputs)
    kd,_ = load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest() == KD_FUNCTION_SHA
    for split,(qi,labels) in qs.items():
        pi,gl = gs[split]
        assert (len(qi),len(pi),len(gl.unique())) == COUNTS[split]
        ix = torch.randperm(len(qi),generator=torch.Generator().manual_seed(42))[:1536].sort().values
        ids = qi[ix]
        poses = torch.stack((q[ids].float(),views['flipped'][ids].float()),1).cuda()
        gallery = p[pi].cuda()
        graph = Index(SimpleNamespace(p=gallery),rgb_only=True)
        mean,rotations,checks = reflection_bank(gallery)
        query = poses.flatten(0,1)
        rows,stats = {},{}
        replay = PrefixReplay(kd,gl,split)
        classes = gl.cuda()
        settings = [('native',None,None,False)] + [
            (f'shared_{i}',r,r,False) for i,r in enumerate(rotations)]
        settings += [('separate',rotations[1],rotations[0],False),
                     ('entropy_matched',rotations[0],rotations[0],True)]
        photo_ent = query_ent = None
        with ThreadPoolExecutor(max_workers=min(2,len(os.sched_getaffinity(0)))) as pool:
            for name,rq,rp,match in settings:
                bank,pe = posterior_bank(gallery,tgt,scale,mean,rp,photo_ent if match else None)
                post,qe = posterior_bank(query,src,scale,mean,rq,query_ent if match else None)
                if name == 'native':
                    photo_ent,query_ent = pe,qe
                stats[name] = dict(gallery_entropy=float(pe.mean()),view_entropy=float(qe.mean()),
                    max_gallery_entropy_change=float((pe-photo_ent).abs().max()),
                    max_view_entropy_change=float((qe-query_ent).abs().max()))
                post = post.reshape(len(ix),2,-1).mean(1)
                for first in range(0,len(ix),48):
                    sl = slice(first,first+48)
                    raw = post[sl] @ bank.T
                    scores = {name+'_direct':raw,name+'_diffused':graph.score(CONFIG,raw)}
                    if name=='native':
                        scores['visual_cosine'] = F.normalize(poses[sl].mean(1)) @ F.normalize(gallery).T
                    for key,value in scores.items():
                        rows.setdefault(key,[]).append(score_rows(value,labels[ix[sl]],classes,split,replay,pool)[:,:2])
                del bank,post
                print(split,name,flush=True)
        rows = {k:torch.cat(v) for k,v in rows.items()}
        contrasts = {}
        for key,val in rows.items():
            if key=='visual_cosine' or key.startswith('native'):
                continue
            diff = 100*(rows['native_'+key.rsplit('_',1)[1]].double()-val.double())
            contrasts[key]=dict(points=diff.mean(0).tolist(),class_delta=class_interval(diff[:,0],labels[ix],seed=42))
        metrics={k:(100*v.double().mean(0)).tolist() for k,v in rows.items()}
        save_tensor(dict(qi=ids,pi=pi,labels=labels[ix],rows=rows),root/(split+'.pt'))
        atomic_json(root/(split+'.json'),dict(commit=revision(),split=split,seed=42,queries=len(ix),gallery=len(pi),
            checkpoint=str(inputs[0].clip_checkpoint),device=torch.cuda.get_device_name(),torch=str(torch.__version__),
            estimator_sha=KD_FUNCTION_SHA,rotations=checks,entropy=stats,metrics=metrics,native_minus_control=contrasts))


if __name__=='__main__':
    run()
