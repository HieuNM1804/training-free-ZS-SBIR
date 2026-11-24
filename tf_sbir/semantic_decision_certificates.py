"""Identify binary semantic decisions a candidate could not transmit through any common regional channel."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from .gallery_corner_observations import read_corners
from .gcwe_relational_completion import CORNERS
from .geometry_word_flat import (CHECKPOINT,CONFIG,Index,KD_FUNCTION_SHA,PrefixReplay,atomic_json,
    class_interval,digest,index,load_kd,make_word_bank,revision,save_tensor,score_rows,semantic_inputs)
from .regional_scale_position_contrast import complete_windows
from .semantic_category_boundaries import boundaries


ROUNDING=2e-6


def profiles(crops,text,scale):
    """Distribution over location given word, marginalized over the two image renderings."""
    prob=(scale*(crops.cuda().float()@text.T)).softmax(dim=1).mean(2).transpose(1,2)
    return prob/prob.sum(-1,keepdim=True)


def decisions(prob,pair):
    return .5*(prob[:,pair[0]]-prob[:,pair[1]]).abs().sum(-1)


def maximum_deficit(query,photos):
    return (.5*(query[:,None]-photos[None]).amax(-1)).clamp_min(0)


def ordered(value,native_order):
    # The integer ranks reproduce the exact lexicographic order in the official KD evaluator.
    order=native_order.gather(1,value.gather(1,native_order).argsort(dim=1,stable=True))
    rank=torch.arange(order.size(1),device=order.device).expand_as(order)
    return torch.empty_like(order).scatter_(1,order,-rank).float()


def validate():
    rng=torch.Generator().manual_seed(42)
    p=torch.rand(5,8,4,generator=rng,dtype=torch.float64)
    p/=p.sum(-1,keepdim=True)
    k=torch.rand(5,4,4,generator=rng,dtype=torch.float64)
    k/=k.sum(-1,keepdim=True)
    pair=torch.triu_indices(8,8,offset=1)
    observable=p@k
    u,v=decisions(observable,pair),decisions(p,pair)
    assert maximum_deficit(u,v).diag().max()<1e-12
    other=observable.roll(1,dims=1)
    radius=.5*(other-observable).abs().sum(-1).amax(-1)
    assert (maximum_deficit(decisions(other,pair),v).diag()<=radius+1e-12).all()
    torch.testing.assert_close(v,decisions(p.flip([-1]),pair))
    assert (maximum_deficit(u.sort().values,v.sort().values)<=maximum_deficit(u,v)+1e-12).all()
    torch.testing.assert_close(ordered(torch.tensor([[0.,0.,1.]]),torch.tensor([[1,2,0]])),torch.tensor([[-1.,0.,-2.]]))


def assess_masks(named,cls,labels,order):
    # Only here, after candidate scores are fixed, look up evaluation labels.
    ok=named<=ROUNDING
    top=order[:,:200]
    bound=named.gather(1,top).reshape(-1,100,2)
    hits=(cls[top]==labels[:,None]).reshape_as(bound)
    mixed=hits[:,:,0]^hits[:,:,1]
    good=torch.where(hits[:,:,0],bound[:,:,0],bound[:,:,1])
    bad=torch.where(hits[:,:,0],bound[:,:,1],bound[:,:,0])
    def avg(x):
        return (x*mixed).sum(1)/mixed.sum(1)
    return torch.stack((ok.float().mean(1),ok.gather(1,top).float().mean(1),avg(good),avg(bad),
                        avg((good<=ROUNDING).float()),avg((bad<=ROUNDING).float()),mixed.sum(1)),1).cpu()


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    validate()
    w=Path('/content/sbir')
    root=w/'results/semantic-decision-certificates'
    root.mkdir(parents=True,exist_ok=True)
    inputs=semantic_inputs(w,'sketchy')
    _,_,qs,paths,gs,q,p,src,tgt,scale=inputs
    views=complete_windows(w,'sketchy',inputs)
    photos=read_corners(w,paths)
    kd,_=load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    pair=torch.triu_indices(64,64,offset=1,device='cuda')
    swap=torch.arange(64,device='cuda')^1
    for split,(all_qi,all_labels) in qs.items():
        pi,gl=gs[split]
        ref=torch.load(w/'results/semantic-category-boundaries'/(split+'.pt'),weights_only=True)
        assert torch.equal(pi,ref['pi'])
        qi,labels=ref['qi'],ref['labels']
        id_to_pos={v:i for i,v in enumerate(all_qi.tolist())}
        assert torch.equal(labels,all_labels[[id_to_pos[v] for v in qi.tolist()]])
        gallery=p[pi].cuda()
        stamp=digest(dict(gallery=[str(paths[i]) for i in pi.tolist()],model=CHECKPOINT,source='regional-64/RGB16'))
        heads,_=index(w,split,gallery,tgt,scale,stamp[:16])
        local=torch.cat([profiles(photos[sl],tgt[heads],scale) for sl in pi.split(192)])
        named=torch.cat([decisions(row,pair) for row in local.split(192)])
        swapped=torch.cat([decisions(row[:,swap],pair) for row in local.split(192)])
        capacities=[named,swapped,named.sort(dim=-1).values]
        graph=Index(SimpleNamespace(p=gallery),rgb_only=True)
        bank=make_word_bank(gallery,tgt,scale,192)
        replay=PrefixReplay(kd,gl,split)
        cls=gl.cuda()
        metrics,accuracy,diagnostics={},{},[]
        with ThreadPoolExecutor(max_workers=min(2,len(os.sched_getaffinity(0)))) as pool:
            for batch in torch.arange(len(qi)).split(24):
                ids=qi[batch]
                sk=profiles(views['regions'][ids][:,CORNERS],src[heads],scale)
                decision=decisions(sk,pair)
                free=decision.sort(dim=-1).values
                poses=torch.stack((q[ids],views['flipped'][ids].float()),1).cuda()
                lexical=(scale*(poses@src.T)).softmax(-1).mean(1)@bank.T
                native=graph.score(CONFIG,lexical)
                order=native.argsort(dim=1,descending=True,stable=True)
                scores={'native':native}
                bounds={}
                for k,query,cap in zip(('named','swapped','free'),(decision,decision,free),capacities):
                    bounds[k]=torch.cat([maximum_deficit(query,chunk) for chunk in cap.split(256)],1)
                assert (bounds['named']+ROUNDING>=bounds['free']).all()
                bounds['identity_excess']=(bounds['named']-bounds['free']).clamp_min(0)
                bounds['reverse']=torch.cat([maximum_deficit(chunk,decision).T for chunk in named.split(256)],1)
                for k,v in bounds.items():
                    scores[k]=ordered(v,order)
                scores['filter']=ordered((bounds['named']>ROUNDING).float(),order)
                diagnostics.append(assess_masks(bounds['named'],cls,labels[batch].cuda(),order))
                # Match the previous native top-200 procedure even when an exact score tie exists.
                ranks=native.topk(200).indices
                official,counts=boundaries(ranks,scores,cls,labels[batch])
                assert torch.equal(counts,ref['counts'][batch])
                for k,v in scores.items():
                    metrics.setdefault(k,[]).append(score_rows(v,labels[batch],cls,split,replay,pool)[:,:2])
                    accuracy.setdefault(k,[]).append(official[k])
        rows={k:torch.cat(v).double() for k,v in metrics.items()}
        acc={k:torch.cat(v).double() for k,v in accuracy.items()}
        diagnostics=torch.cat(diagnostics).double()
        torch.testing.assert_close(rows['native'],ref['rows']['semantic/diffused'],atol=2e-5,rtol=1e-5)
        effects={}
        for arm,ctrl in [('filter','native'),('named','swapped'),('named','free'),('named','reverse'),('identity_excess','free')]:
            gap=100*(rows[arm]-rows[ctrl])
            effects[arm+' minus '+ctrl]=dict(points=gap.mean(0).tolist(),class_ap=class_interval(gap[:,0],labels,seed=42))
        save_tensor(dict(qi=qi,pi=pi,labels=labels,heads=heads.cpu(),rows=rows,accuracy=acc,diagnostics=diagnostics),root/(split+'.pt'))
        atomic_json(root/(split+'.json'),dict(commit=revision(),seed=42,gpu=torch.cuda.get_device_name(),evaluator=KD_FUNCTION_SHA,
            queries=len(qi),gallery=len(pi),pairs=pair.size(1),effects=effects,metrics={k:(100*v.mean(0)).tolist() for k,v in rows.items()},
            balanced_percent={k:float(100*v[:,3].nanmean()) for k,v in acc.items()},
            support_columns=['fraction_passing_all','fraction_passing_native_top200','mixed_true_bound','mixed_foil_bound',
                             'mixed_true_pass_rate','mixed_foil_pass_rate','mixed_pairs_per_query'],
            support=diagnostics.nanmean(0).tolist()))
        print(split,effects['filter minus native']['points'],flush=True)
        del local,named,swapped,capacities,graph,bank


if __name__=='__main__':
    run()
