"""Adjust CLIP query-word logits by an analytic conditional Gaussian log moment."""
import hashlib
import inspect
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from .geometry_word_flat import (CONFIG, COUNTS, Index, KD_FUNCTION_SHA, PrefixReplay, atomic_json,
    class_interval, load_kd, make_word_bank, revision, save_tensor, score_rows, semantic_inputs)
from .regional_scale_position_contrast import complete_windows
from .semantic_axis_orientation import entropy,entropy_match
import torch


def matched_word_permutation(means):
    order=means.argsort(stable=True).cpu()
    perm=torch.arange(len(order))
    rng=torch.Generator().manual_seed(42)
    for stratum in order.split(64):
        perm[stratum]=stratum[torch.randperm(len(stratum),generator=rng)]
    return perm.to(means.device)


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    w=Path('/content/sbir')
    root=w/'results/semantic-null-evidence'
    root.mkdir(parents=True,exist_ok=True)
    inputs=semantic_inputs(w,'sketchy')
    _,_,qs,_,gs,q,p,src,tgt,scale=inputs
    views=complete_windows(w,'sketchy',inputs)
    kd,_=load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    for split,(qi,labels) in qs.items():
        pi,gl=gs[split]
        assert (len(qi),len(pi),len(gl.unique()))==COUNTS[split]
        ix=torch.randperm(len(qi),generator=torch.Generator().manual_seed(42))[1536:2304].sort().values
        gallery=p[pi].cuda()
        center=gallery.double().mean(0)
        deviation=gallery.double()-center
        cov=deviation.T @ deviation / len(gallery)
        eigenvalues,eigenvectors=torch.linalg.eigh(cov.cpu())
        eig=eigenvectors.float().cuda()
        text_energy=(src@eig).square().T.contiguous()
        pooled=(.5*scale**2)*(eigenvalues.clamp_min(0).float().cuda() @ text_energy)
        mean=center.float()
        perm=matched_word_permutation(src @ mean)
        bank=make_word_bank(gallery,tgt,scale,192)
        graph=Index(SimpleNamespace(p=gallery),rgb_only=True)
        classes=gl.cuda()
        replay=PrefixReplay(kd,gl,split)
        rows,entropies={},{}
        with ThreadPoolExecutor(max_workers=min(2,len(os.sched_getaffinity(0)))) as pool:
            for batch in ix.split(32):
                ids=qi[batch]
                poses=torch.stack((q[ids].float(),views['flipped'][ids].float()),1).flatten(0,1).cuda()
                logits=scale*(poses @ src.T)
                a2=((poses-mean) @ eig).square()
                penalty=.5*scale**2*(a2 @ text_energy)
                if not torch.isfinite(penalty).all() or penalty.min()<0:
                    raise ValueError('Invalid Gaussian log moment')
                primary=(logits-penalty).softmax(-1)
                alts={'native':logits.softmax(-1),'conditional':primary,'pooled':(logits-pooled).softmax(-1),
                      'permuted':(logits-penalty[:,perm]).softmax(-1),'excess':(logits-penalty+pooled).softmax(-1),
                      'entropy_only':entropy_match(logits,entropy(primary))}
                for name,prob in alts.items():
                    entropies.setdefault(name,[]).append(entropy(prob).cpu())
                    raw=prob.reshape(len(batch),2,-1).mean(1) @ bank.T
                    for mode,scores in [('direct',raw),('diffused',graph.score(CONFIG,raw))]:
                        values=score_rows(scores,labels[batch],classes,split,replay,pool)[:,:2]
                        rows.setdefault(name+'_'+mode,[]).append(values)
        rows={k:torch.cat(v) for k,v in rows.items()}
        contrasts={}
        for k,v in rows.items():
            primary=rows['conditional_'+k.rsplit('_',1)[1]]
            delta=100*(primary.double()-v.double())
            contrasts[k]=dict(points=delta.mean(0).tolist(),class_delta=class_interval(delta[:,0],labels[ix],seed=42))
        metrics={k:(100*v.double().mean(0)).tolist() for k,v in rows.items()}
        save_tensor(dict(qi=qi[ix],pi=pi,labels=labels[ix],rows=rows),root/(split+'.pt'))
        atomic_json(root/(split+'.json'),dict(commit=revision(),seed=42,device=torch.cuda.get_device_name(),
            queries=len(ix),gallery=len(pi),evaluator=KD_FUNCTION_SHA,metrics=metrics,conditional_minus=contrasts,
            mean_entropy={k:float(torch.cat(v).mean()) for k,v in entropies.items()}))
        print(split,metrics,flush=True)


if __name__=='__main__':
    run()
