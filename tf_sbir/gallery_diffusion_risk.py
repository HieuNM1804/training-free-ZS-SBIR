"""Does cross-domain diffusion overvalue rare, strongly agreeing gallery endpoints?"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .branch_view_policy import outputs
from .branch_view_study import PrefixReplay
from .gallery_panel_resume import region_views, semantic_inputs
from .gallery_region_panels import CONFIG, Index
from .positive_foil import base_parts, raw_ranks, STRONG
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd


def endpoint_scores(graph, mass):
    # K is the normalized diffusion kernel applied by the existing RGB ranker.
    # All expectations use exactly the same operator and sketch evidence.
    log_mass=mass.clamp_min(1e-30).log()
    direct=graph.score(CONFIG,mass).clamp_min(1e-30).log()
    geometric=graph.score(CONFIG,log_mass)
    half=2*graph.score(CONFIG,mass.sqrt()).clamp_min(1e-30).log()
    double=.5*graph.score(CONFIG,mass.square()).clamp_min(1e-30).log()
    return {'original':direct,'geometric':geometric,'half':half,'double':double}


def distribution_control(reference, modified):
    # Preserve exactly the new marginal score distribution for each sketch,
    # while assigning these scores in the original diffusion order.
    reference_order=reference.argsort(dim=1,stable=True)
    return torch.empty_like(modified).scatter_(1,reference_order,modified.sort(dim=1).values)


def adjacent_errors(base_ranks, difference, labels, gallery_labels):
    # Restrict to adjacent members of the original top 200 with opposite relevance.
    head=base_ranks[:,:200]
    hit=gallery_labels[head]==labels.cuda()[:,None]
    sign=hit[:,:-1].float()-hit[:,1:].float()
    gap=difference.gather(1,head)
    # Positive when the wrong photo depends more on concentrated endpoint evidence.
    wrong_minus_right=(gap[:,1:]-gap[:,:-1])*sign
    def collect(mask):
        return torch.stack([mask.sum(1),(wrong_minus_right*mask).sum(1),
                            ((wrong_minus_right>0)&mask).sum(1)],1)
    return torch.stack([collect(sign==-1),collect(sign==1)],1).cpu()


@torch.inference_mode()
def run(workspace,ds,official):
    cfg,paths,queries,_,galleries,q,p,source,target,scale=semantic_inputs(workspace,ds)
    kd,_=load_kd(workspace)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    if ds=='tuberlin' and not official:raise ValueError('Transfer requires full KD replay')
    result={}
    for s,(qi,labels) in queries.items():
        pi,gold=galleries[s]
        if ds=='tuberlin' and (len(gold.unique())!=30 or len(labels.unique())!=30):raise ValueError('Incomplete TU transfer')
        gc=gold.cuda()
        g=F.normalize(p[pi].cuda())
        bank=torch.cat([(scale*(part@target.T)).softmax(1) for part in g.split(256)])
        graph=Index(SimpleNamespace(p=g),rgb_only=True)
        views,_=region_views(workspace,s,cfg,paths,qi,q)
        folder,tree,regional=base_parts(workspace,s)
        replay=PrefixReplay(kd,gold,s)
        per_query={}
        diagnostics=[]
        error=0.
        worst_jensen=0.
        for start in range(0,len(qi),48):
            sl=slice(start,start+48)
            offset=start%192
            if offset==0:pack=torch.load(folder/'full'/f'{start:06d}.pt',mmap=True,weights_only=True)
            sub=slice(offset,offset+len(qi[sl]))
            assert torch.equal(pack['ids'][sub],torch.arange(start,start+len(qi[sl])))
            reg=regional[sl].cuda()
            _,old_combined,_=next(outputs(tree,pack['densities'][sub].cuda(),reg,reg,[STRONG]))
            residual=(old_combined-reg)/reg.std(1,keepdim=True)
            a=(scale*(q[qi[sl]].cuda()@source.T)).softmax(1)
            b=(scale*(views['flipped'][sl].cuda().float()@source.T)).softmax(1)
            original_single=graph.score(CONFIG,a@bank.T).clamp_min(1e-30).log()
            rows=endpoint_scores(graph,((a+b)/2)@bank.T)
            ordinary=rows['original']
            gap=ordinary-rows['geometric']
            worst_jensen=min(worst_jensen,float(gap.min()))
            for name in ('half','double','geometric'):
                rows[name+'_marginal_control']=distribution_control(ordinary,rows[name])
            for name,vector in rows.items():
                modes=('standalone','strong') if 'control' not in name else ('strong',)
                for mode in modes:
                    if mode=='strong':
                        updated=reg+.65*(vector-original_single)
                        score=updated+updated.std(1,keepdim=True)*residual
                    else:
                        score=vector
                    rank,quick=raw_ranks(score,labels[sl],gc,s)
                    n=len(quick) if official else 1
                    checked=replay.chunk((rank[:n].cpu(),labels[sl][:n]))
                    error=max(error,float((checked-quick[:n]).abs().max()))
                    assert error<2e-6
                    per_query.setdefault(mode+'/'+name,[]).append(checked if official else quick)
                    if mode=='strong' and name=='original':
                        diagnostics.append(adjacent_errors(rank,gap,labels[sl],gc))
        values={key:torch.cat(rows).double() for key,rows in per_query.items()}
        diagnostics=torch.cat(diagnostics).double()
        summary={}
        for j,k in enumerate(['wrong_immediately_above_right','right_immediately_above_wrong']):
            total,gap_sum,positive=diagnostics[:,j].sum(0).tolist()
            summary[k]={'n':int(total),'mean_wrong_minus_right_jensen_gap':gap_sum/total,
                        'fraction_wrong_larger_gap':positive/total}
        ci={}
        for k in ('geometric','half','double'):
            for compare in ('original',k+'_marginal_control'):
                delta=(values['strong/'+k]-values['strong/'+compare])*100
                macro=torch.stack([delta[labels==c].mean(0) for c in labels.unique()])
                boot=torch.randint(len(macro),(10000,len(macro)),generator=torch.Generator().manual_seed(173))
                ci[k+' minus '+compare]={'class_mean':macro.mean(0).tolist(),
                                        'ci95':macro[boot].mean(1).quantile(torch.tensor([.025,.975],dtype=macro.dtype),dim=0).tolist()}
        result[s]={'queries':len(qi),'gallery':len(g),'official_all':official,'maximum_kd_error':error,
                   'minimum_jensen_gap_or_zero':worst_jensen,'head_diagnostics':summary,
                   'metrics':{k:(100*v.mean(0)).tolist() for k,v in values.items()},'paired':ci}
        print(s,json.dumps(result[s]),flush=True)
    directory=workspace/'results/gallery-diffusion-risk'
    directory.mkdir(parents=True,exist_ok=True)
    (directory/(ds+('-official' if official else '')+'.json')).write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['sketchy','tuberlin'],default='sketchy')
    parser.add_argument('--official',action='store_true')
    options=parser.parse_args()
    run(Path('/content/sbir'),options.dataset,options.official)
