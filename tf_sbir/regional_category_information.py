"""Attribute class-boundary resolutions to crop observations and to a specific candidate's use of them."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import os
from pathlib import Path
from types import SimpleNamespace

from .cross_view_regions import CandidateRegions
from .geometry_word_flat import (CHECKPOINT,CONFIG,Index,KD_FUNCTION_SHA,PrefixReplay,atomic_json,
    class_interval,digest,index,load_kd,make_word_bank,revision,save_tensor,score_rows,semantic_inputs)
from .regional_scale_position_contrast import complete_windows
import torch


def assessment(ids,cls,truth,scores):
    a,b=ids[:,:200:2],ids[:,1:200:2]
    a_hit=cls[a]==truth[:,None]
    b_hit=cls[b]==truth[:,None]
    correct=a_hit&~b_hit
    wrong=~a_hit&b_hit
    mixed=correct|wrong
    def avg(v,m):
        return (v*m).sum(1)/m.sum(1).float()
    results={}
    for k,s in scores.items():
        margin=(s.gather(1,a)-s.gather(1,b))*torch.where(a_hit,1.,-1.)
        success=(margin>0).float()+.5*(margin==0).float()
        resc=avg(success,wrong)
        kept=avg(success,correct)
        results[k]=torch.stack((avg(success,mixed),resc,kept,(resc+kept)/2,(success*wrong).sum(1),
                               ((1-success)*correct).sum(1)),1).cpu()
    return results,torch.stack((mixed.sum(1),correct.sum(1),wrong.sum(1)),1).cpu()


def score_arms(poses,crops,src,scale,heads,bank,native,query_ids):
    empty=src.new_empty((0,len(src)))
    reader=CandidateRegions(poses[:,0],poses[:,1],crops[:,:,0],crops[:,:,1],src,heads,bank,empty,native)
    reader.prepare(scale)
    actual=reader.read('self',1.,'gallery')
    uniform=reader.read('uniform',1.,'gallery')
    repeated=poses[:,None].expand(-1,crops.size(1),-1,-1)
    full=CandidateRegions(poses[:,0],poses[:,1],repeated[:,:,0],repeated[:,:,1],src,heads,bank,empty,native)
    full.prepare(scale)
    global_read=full.read('self',1.,'gallery')
    torch.testing.assert_close(global_read,full.read('uniform',1.,'gallery'),atol=1e-5,rtol=1e-6)
    weights=(reader.prior.log()+reader.logl-reader.l.mean(3,keepdim=True).clamp_min(1e-30).log()).softmax(2)
    # The window selector sees a neighboring candidate under the whole-query readout; the value always uses the real photo.
    order=global_read.argsort(1,descending=True,stable=True)
    n=order.size(1)
    permutation=[]
    for i,qid in enumerate(query_ids.tolist()):
        offset=int(torch.randint(2,(1,),generator=torch.Generator().manual_seed(42+qid)))
        perm=torch.arange(n,device=order.device)
        end=offset+2*((n-offset)//2)
        segment=perm[offset:end].reshape(-1,2).flip(1).flatten()
        perm[offset:end]=segment
        pairs=torch.empty_like(perm)
        pairs[order[i]]=order[i,perm]
        permutation.append(pairs)
    perm=torch.stack(permutation)
    other=weights.gather(3,perm[:,None,None].expand_as(weights))
    matched=(other*reader.l).sum(2).mean(1).clamp_min(1e-30).log()
    delta_obs=uniform-global_read
    delta_cond=actual-uniform
    torch.testing.assert_close(actual-global_read,delta_obs+delta_cond,atol=3e-6,rtol=1e-6)
    return {'global_heads':global_read,'window_marginal':uniform,'candidate':actual,
        'adjacent_selector':matched,'observation_increment':delta_obs,'conditional_increment':delta_cond,
        'identity_increment':actual-matched}


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    w=Path('/content/sbir')
    root=w/'results/regional-category-information'
    root.mkdir(parents=True,exist_ok=True)
    inputs=semantic_inputs(w,'sketchy')
    _,_,qs,pp,gs,q,p,src,tgt,scale=inputs
    views=complete_windows(w,'sketchy',inputs)
    kd,_=load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    for split,(qi,labels) in qs.items():
        ref=torch.load(w/'results/semantic-category-boundaries'/(split+'.pt'),weights_only=True)
        pi,gl=gs[split]
        assert torch.equal(pi,ref['pi'])
        pos={int(v):j for j,v in enumerate(qi)}
        ix=torch.tensor([pos[int(v)] for v in ref['qi']])
        assert torch.equal(labels[ix],ref['labels'])
        gallery=p[pi].cuda()
        stamp=digest(dict(gallery=[str(pp[i]) for i in pi.tolist()],model=CHECKPOINT,source='regional-64/RGB16'))
        heads,compact=index(w,split,gallery,tgt,scale,stamp[:16])
        bank=make_word_bank(gallery,tgt,scale,192)
        graph=Index(SimpleNamespace(p=gallery),rgb_only=True)
        replay=PrefixReplay(kd,gl,split)
        cls=gl.cuda()
        metrics,paired,counts={},{},[]
        with ThreadPoolExecutor(max_workers=min(2,len(os.sched_getaffinity(0)))) as pool:
            for batch in ix.split(24):
                ids=qi[batch]
                poses=torch.stack((q[ids],views['flipped'][ids].float()),1).cuda()
                windows=views['regions'][ids].cuda()
                native=(scale*(poses @ src.T)).softmax(-1).mean(1) @ bank.T
                order=graph.score(CONFIG,native).topk(200).indices
                scored=score_arms(poses,windows,src,scale,heads,compact,native,ids)
                assess,count=assessment(order,cls,labels[batch].cuda(),scored)
                for k,s in scored.items():
                    if not k.endswith('increment'):
                        metrics.setdefault(k,[]).append(score_rows(s,labels[batch],cls,split,replay,pool)[:,:2])
                    paired.setdefault(k,[]).append(assess[k])
                counts.append(count)
        metrics={k:torch.cat(v).double() for k,v in metrics.items()}
        paired={k:torch.cat(v).double() for k,v in paired.items()}
        count=torch.cat(counts)
        assert torch.equal(count,ref['counts'])
        torch.testing.assert_close(metrics['candidate'],ref['rows']['regional_original'],atol=2e-5,rtol=1e-5)
        valid=torch.isfinite(paired['candidate'][:,3])
        effects={}
        for k in ('global_heads','window_marginal','adjacent_selector'):
            delta=100*(paired['candidate'][valid,3]-paired[k][valid,3])
            official=100*(metrics['candidate']-metrics[k])
            effects[k]=dict(balanced_points=float(delta.mean()),class_balanced=class_interval(delta,labels[ix][valid],seed=42),
                official_points=official.mean(0).tolist(),class_ap=class_interval(official[:,0],labels[ix],seed=42))
        save_tensor(dict(qi=qi[ix],pi=pi,labels=labels[ix],rows=metrics,conditional=paired,counts=count),root/(split+'.pt'))
        atomic_json(root/(split+'.json'),dict(commit=revision(),gpu=torch.cuda.get_device_name(),seed=42,evaluator=KD_FUNCTION_SHA,
            queries=len(ix),gallery=len(pi),both_directions=int(valid.sum()),mixed=int((count[:,0]>0).sum()),
            measures=['all_mixed','global_wrong','global_correct','within_query_balanced','rescued_total','damaged_total'],
            conditional={k:torch.cat((v[:,:4].nanmean(0),v[:,4:].sum(0))).tolist() for k,v in paired.items()},
            metrics={k:(100*v.mean(0)).tolist() for k,v in metrics.items()},candidate_minus=effects))
        print(split,effects['window_marginal']['official_points'],flush=True)


if __name__=='__main__':
    run()
