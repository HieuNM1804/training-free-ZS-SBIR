"""Standardize the total donor exposure to public concepts for each candidate photo."""
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
from .gallery_panel_resume import region_views,semantic_inputs
from .gallery_region_panels import CONFIG,Index
from .positive_foil import STRONG,base_parts,raw_ranks
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd


def endpoint_matrix(graph):
    """Materialize the normalized operator once using the sparse local transitions."""
    graph.prepare(CONFIG)
    kernel=graph.kernels[0]
    dense=kernel.status
    for _ in range(CONFIG.iterations):
        dense=torch.sparse.mm(kernel.local,dense)
    dense=dense.T.contiguous()
    for _ in range(CONFIG.iterations):
        dense=torch.sparse.mm(kernel.local,dense)
    dense=dense.T.contiguous()
    return dense.div_(dense.sum(1,keepdim=True))


def grouped_permutation(groups):
    rng=torch.Generator().manual_seed(5728)
    cpu=groups.cpu()
    perm=torch.arange(len(groups))
    for group in cpu.unique():
        ids=(cpu==group).nonzero().flatten()
        ids=ids[torch.randperm(len(ids),generator=rng)]
        perm[ids]=ids.roll(1)
    return perm.to(groups.device)


class StandardizedDonors:
    def __init__(self,w,bank):
        self.w=w
        words,self.group=bank.argmax(1).unique(sorted=True,return_inverse=True)
        self.target=bank[:,words].contiguous()
        self.permutation=grouped_permutation(self.group)
        self.count=self.group.bincount().float()
        self.vocabulary_count=len(words)

    def score(self,evidence):
        outputs={k:torch.empty_like(evidence) for k in ('matrix_native','standard','permuted','same_backoff','without_rgb')}
        for start in range(0,len(self.group),160):
            sl=slice(start,start+160)
            w=self.w[sl]
            n=len(w)
            codes=self.group.expand(n,-1)
            observed=w.new_zeros(n,len(self.count)).scatter_add_(1,codes,w)
            support=observed>0
            conditional=w/observed.gather(1,codes).clamp_min(torch.finfo(w.dtype).tiny)
            original=evidence@w.T
            outputs['matrix_native'][:,sl]=original
            for name,preferred in [('standard',self.target[sl]),('permuted',self.target[self.permutation[sl]])]:
                target=preferred*support
                fallback=(1-target.sum(1)).clamp(0,1)
                scaled=conditional*target.gather(1,codes)
                outputs[name][:,sl]=evidence@scaled.T+evidence[:,sl]*fallback
                if name=='standard':
                    outputs['same_backoff'][:,sl]=original*(1-fallback)+evidence[:,sl]*fallback
                    flat=(target/self.count).gather(1,codes)
                    outputs['without_rgb'][:,sl]=evidence@flat.T+evidence[:,sl]*fallback
        return outputs


@torch.inference_mode()
def run(workspace,ds,full):
    cfg,paths,qs,_,gs,q,p,src,tgt,scale=semantic_inputs(workspace,ds)
    kd,_=load_kd(workspace)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    if ds=='tuberlin' and not full:raise ValueError('Transfer requires every query')
    result={}
    for s,(qi,labels) in qs.items():
        pi,gold=gs[s]
        if ds=='tuberlin' and (len(gold.unique())!=30 or len(labels.unique())!=30):raise ValueError('Incomplete transfer')
        rows=torch.arange(len(qi)) if full else torch.randperm(len(qi),generator=torch.Generator().manual_seed(87243))[:768].sort().values
        g=F.normalize(p[pi].cuda())
        bank=torch.cat([(scale*(chunk@tgt.T)).softmax(1) for chunk in g.split(256)])
        graph=Index(SimpleNamespace(p=g),rgb_only=True)
        estimator=StandardizedDonors(endpoint_matrix(graph),bank)
        views,_=region_views(workspace,s,cfg,paths,qi,q)
        folder,tree,regional=base_parts(workspace,s)
        replay=PrefixReplay(kd,gold,s)
        scores={}
        top1={}
        max_error=0.
        max_log_error=0.
        mapped=[]
        for start in range(0,len(qi),192):
            selected=rows[(rows>=start)&(rows<start+192)]
            if not len(selected):continue
            pack=torch.load(folder/'full'/f'{start:06d}.pt',mmap=True,weights_only=True)
            for ix in selected.split(96):
                offset=ix-start
                assert torch.equal(pack['ids'][offset],ix)
                reg=regional[ix].cuda()
                _,old,_=next(outputs(tree,pack['densities'][offset].cuda(),reg,reg,[STRONG]))
                cell=(old-reg)/reg.std(1,keepdim=True)
                a=(scale*(q[qi[ix]].cuda()@src.T)).softmax(1)
                b=(scale*(views['flipped'][ix].cuda().float()@src.T)).softmax(1)
                evidence=((a+b)/2)@bank.T
                first=graph.score(CONFIG,a@bank.T).clamp_min(1e-30).log()
                native=graph.score(CONFIG,evidence)
                alternatives=estimator.score(evidence)
                materialized=alternatives.pop('matrix_native')
                max_error=max(max_error,float((materialized-native).abs().max()))
                max_log_error=max(max_log_error,float((materialized.clamp_min(1e-30).log()-native.clamp_min(1e-30).log()).abs().max()))
                alternatives={'baseline':native,'matrix_native':materialized,**alternatives}
                for name,mass in alternatives.items():
                    log=mass.clamp_min(1e-30).log()
                    if not log.isfinite().all():raise ValueError('Nonfinite standardized score')
                    new=reg+.65*(log-first)
                    combined=new+new.std(1,keepdim=True)*cell
                    rank,quick=raw_ranks(combined,labels[ix],gold.cuda(),s)
                    exact=replay.chunk((rank.cpu(),labels[ix]))
                    assert (exact-quick).abs().max()<2e-6
                    scores.setdefault(name,[]).append(exact)
                    ok=gold[rank[:,0].cpu()]==labels[ix]
                    if name=='baseline':prior=ok
                    top1.setdefault(name,[]).append(torch.stack([~prior&ok,prior&~ok],1))
                mapped.append(ix)
        assert torch.equal(rows,torch.cat(mapped))
        z={k:torch.cat(v).double() for k,v in scores.items()}
        y=labels[rows]
        ci={}
        for k in ('baseline','permuted','same_backoff','without_rgb'):
            delta=(z['standard']-z[k])*100
            macro=torch.stack([delta[y==c].mean(0) for c in y.unique()])
            boot=torch.randint(len(macro),(10000,len(macro)),generator=torch.Generator().manual_seed(5728))
            ci[k]={'class_mean':macro.mean(0).tolist(),'ci95':macro[boot].mean(1).quantile(torch.tensor([.025,.975],dtype=macro.dtype),dim=0).tolist()}
        result[s]={'queries':len(rows),'gallery':len(gold),'active_names':estimator.vocabulary_count,
            'mean_candidate_posterior_mass_on_active_names':float(estimator.target.sum(1).mean()),
            'fraction_permutation_moved':float((estimator.permutation!=torch.arange(len(g),device=g.device)).float().mean()),
            'native_operator_max_absolute_error':max_error,'native_operator_max_log_error':max_log_error,
            'metrics':{k:(100*v.mean(0)).tolist() for k,v in z.items()},'top1_fixed_damaged':{k:torch.cat(v).sum(0).tolist() for k,v in top1.items()},
            'standard_minus_control':ci}
        print(s,json.dumps(result[s]),flush=True)
        del estimator,graph,bank
    out=workspace/'results/public-context-standardization'
    out.mkdir(parents=True,exist_ok=True)
    (out/(ds+('-full' if full else '-pilot')+'.json')).write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',default='sketchy',choices=['sketchy','tuberlin'])
    parser.add_argument('--full',action='store_true')
    options=parser.parse_args()
    run(Path('/content/sbir'),options.dataset,options.full)
