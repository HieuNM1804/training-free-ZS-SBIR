"""Test semantic evidence deletion and relocation through an absorbing unobserved outcome."""
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
from .semantic_decision_certificates import ROUNDING,assess_masks,maximum_deficit,ordered


def events(crops,text,heads,scale):
    logits=scale*(crops.cuda().float()@text.T)
    joint=logits.softmax(-1).index_select(-1,heads).mean(2).transpose(1,2)/4
    assert (joint>=0).all() and (joint.sum(1)<=.250001).all()
    return joint


def tests(table,pairs):
    difference=table[:,pairs[0]]-table[:,pairs[1]]
    return torch.cat((difference.clamp_min(0).sum(-1),(-difference).clamp_min(0).sum(-1)),1)


def signatures(table,pairs,permutation):
    unary=table.sum(-1)
    shuffled=table.gather(2,permutation.expand_as(table))
    return [unary,tests(table,pairs),tests(unary[:,:,None],pairs),tests(shuffled,pairs)]


def validate():
    gen=torch.Generator().manual_seed(42)
    p=torch.rand(4,8,4,generator=gen,dtype=torch.double)/8
    k=torch.rand(4,4,5,generator=gen,dtype=torch.double)
    k=(k/k.sum(-1,keepdim=True))[:,:,:4]
    q=p@k
    pair=torch.triu_indices(8,8,offset=1)
    assert (q.sum(-1)<=p.sum(-1)+1e-12).all()
    assert (tests(q,pair)<=tests(p,pair)+1e-12).all()
    torch.testing.assert_close(tests(q,pair),tests(q[:,:,[1,3,2,0]],pair))
    richer=torch.tensor([[[.125,0],[0,.125]]])
    pooled=richer.mean(1,keepdim=True).expand_as(richer)
    two=torch.tensor([[0],[1]])
    torch.testing.assert_close(pooled.sum(-1),richer.sum(-1))
    assert (tests(richer,two)>tests(pooled,two)).all()
    assert (richer.sum(-1)>torch.zeros_like(richer).sum(-1)).all()


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    validate()
    w=Path('/content/sbir')
    root=w/'results/semantic-word-erasure'
    root.mkdir(parents=True,exist_ok=True)
    inputs=semantic_inputs(w,'sketchy')
    _,_,qs,paths,gs,q,p,src,tgt,scale=inputs
    views=complete_windows(w,'sketchy',inputs)
    photos=read_corners(w,paths)
    kd,_=load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    pair=torch.triu_indices(64,64,offset=1,device='cuda')
    gen=torch.Generator().manual_seed(42)
    permutation=torch.stack([torch.randperm(4,generator=gen) for _ in range(64)]).cuda()
    for split,(all_qi,all_labels) in qs.items():
        pi,gl=gs[split]
        ref=torch.load(w/'results/semantic-category-boundaries'/(split+'.pt'),weights_only=True)
        qi,labels=ref['qi'],ref['labels']
        pos={k:i for i,k in enumerate(all_qi.tolist())}
        assert torch.equal(pi,ref['pi']) and torch.equal(labels,all_labels[[pos[k] for k in qi.tolist()]])
        gallery=p[pi].cuda()
        stamp=digest(dict(gallery=[str(paths[i]) for i in pi.tolist()],model=CHECKPOINT,source='regional-64/RGB16'))
        heads,_=index(w,split,gallery,tgt,scale,stamp[:16])
        components=[signatures(events(photos[batch],tgt,heads,scale),pair,permutation) for batch in pi.split(128)]
        pools=[torch.cat(items) for items in zip(*components)]
        del components
        bank=make_word_bank(gallery,tgt,scale,192)
        graph=Index(SimpleNamespace(p=gallery),rgb_only=True)
        replay=PrefixReplay(kd,gl,split)
        cls=gl.cuda()
        metrics,conditional,support,strength={},{},[],[]
        with ThreadPoolExecutor(max_workers=min(2,len(os.sched_getaffinity(0)))) as pool:
            for batch in torch.arange(len(qi)).split(24):
                ids=qi[batch]
                x=events(views['regions'][ids][:,CORNERS],src,heads,scale)
                mass=x.sum(-1)
                spatial=tests(x,pair)
                query=[mass,spatial,tests(mass[:,:,None],pair),spatial]
                bounds={}
                for name,u,all_p in zip(('unary','located','pooled','sham'),query,pools):
                    bound=torch.cat([maximum_deficit(u,subset) for subset in all_p.split(192)],1)
                    # maximum_deficit carries the 1/2 factor for a two-row certificate.
                    bounds[name]=2*bound if name=='unary' else torch.maximum(bounds['unary'],bound)
                poses=torch.stack((q[ids],views['flipped'][ids].float()),1).cuda()
                native=graph.score(CONFIG,(scale*(poses@src.T)).softmax(-1).mean(1)@bank.T)
                order=native.argsort(dim=1,descending=True,stable=True)
                scores={'native':native,**{k:ordered(v,order) for k,v in bounds.items()}}
                scores['filter']=ordered((bounds['located']>ROUNDING).float(),order)
                support.append(assess_masks(bounds['located'],cls,labels[batch].cuda(),order))
                excess=bounds['located']-bounds['unary']
                strength.append(torch.stack(((excess>0).float().mean(1),(excess>ROUNDING).float().mean(1),
                    (excess.amax(1)==0).float(),excess.mean(1),mass.amax(1)),1).cpu())
                paired,counts=boundaries(native.topk(200).indices,scores,cls,labels[batch])
                assert torch.equal(counts,ref['counts'][batch])
                for name,v in scores.items():
                    metrics.setdefault(name,[]).append(score_rows(v,labels[batch],cls,split,replay,pool)[:,:2])
                    conditional.setdefault(name,[]).append(paired[name])
        rows={k:torch.cat(v).double() for k,v in metrics.items()}
        acc={k:torch.cat(v).double() for k,v in conditional.items()}
        torch.testing.assert_close(rows['native'],ref['rows']['semantic/diffused'],atol=2e-5,rtol=1e-5)
        sup,st=torch.cat(support).double(),torch.cat(strength).double()
        effects={}
        for name,ctrl in [('located','pooled'),('located','sham'),('located','unary'),('filter','native')]:
            d=100*(rows[name]-rows[ctrl])
            effects[name+' minus '+ctrl]=dict(points=d.mean(0).tolist(),class_ap=class_interval(d[:,0],labels,seed=42))
        save_tensor(dict(qi=qi,pi=pi,labels=labels,heads=heads.cpu(),permutation=permutation.cpu(),rows=rows,
                         accuracy=acc,support=sup,strength=st),root/(split+'.pt'))
        atomic_json(root/(split+'.json'),dict(commit=revision(),gpu=torch.cuda.get_device_name(),seed=42,evaluator=KD_FUNCTION_SHA,
            queries=len(qi),gallery=len(pi),metrics={k:(100*v.mean(0)).tolist() for k,v in rows.items()},effects=effects,
            balanced_percent={k:float(100*v[:,3].nanmean()) for k,v in acc.items()},
            support_columns=['fraction_passing_all','fraction_passing_native_top200','mixed_true_bound','mixed_foil_bound',
                             'mixed_true_pass_rate','mixed_foil_pass_rate','mixed_pairs_per_query'],support=sup.nanmean(0).tolist(),
            strength_columns=['fraction_stronger_exact','fraction_stronger_over_2e-6','all_bounds_equal_for_query',
                              'mean_extra_bound','maximum_word_detection'],strength=st.mean(0).tolist()))
        print(split,effects['located minus pooled']['points'],flush=True)
        del pools,graph,bank,gallery


if __name__=='__main__':
    run()
