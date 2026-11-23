"""Predict candidate visual evidence from the observed rank tail of its other donors."""
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.special import betaincc,betaln
import torch
import torch.nn.functional as F

from .branch_view_study import PrefixReplay
from .gallery_context_reversals import coordinates
from .gallery_diffusion import nearest_gallery
from .gallery_region_panels import CONFIG,Index
from .positive_foil import raw_ranks
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .standardized_context_audit import audit_inputs


def visual_ranks(score,baseline=None):
    n=score.size(1)
    if baseline is None:
        order=score.argsort(dim=1,descending=True,stable=True)
        ranks=torch.arange(n,device=score.device).expand_as(order)
        return torch.empty_like(score).scatter_(1,order,(ranks+.5)/n),n
    # Equal-rank strata approximate conditioning on the current native score.
    # Keep a full-sized last bin by splitting as evenly as possible.
    bins=max(1,round(math.sqrt(n)))
    br=baseline.argsort(dim=1,descending=True,stable=True)
    p=torch.empty_like(score)
    groups=br.tensor_split(bins,1)
    for width in sorted({ix.size(1) for ix in groups}):
        ids=torch.stack([ix for ix in groups if ix.size(1)==width],1)
        x=score.gather(1,ids.flatten(1)).reshape_as(ids)
        order=x.argsort(dim=2,descending=True,stable=True)
        ranks=((torch.arange(width,device=x.device)+.5)/width).expand_as(x)
        values=torch.empty_like(x).scatter_(2,order,ranks)
        p.scatter_(1,ids.flatten(1),values.flatten(1))
    return p,min(ix.size(1) for ix in groups)


class TailPrediction:
    def __init__(self,neighbors,nref):
        self.ids=neighbors
        n=neighbors.size(1)
        k=np.arange(n+1)
        self.tables=[]
        grid=2.**(-np.arange(1,math.floor(math.log2(nref))+1))
        for t in grid:
            # Model donor counts with u ~ Uniform(t,1). Integrate out u
            # rather than selecting its fitted value on the candidate itself.
            q=betaincc(k+1,n-k+1,t)
            log_bf=betaln(k+1,n-k+1)+np.log(q)-(n-k+1)*np.log1p(-t)-k*np.log(t)
            mean_u=(k+1)/(n+2)*betaincc(k+2,n-k+1,t)/q
            vals=np.stack([log_bf-math.log(len(grid)),np.log(mean_u/t),
                           np.log(np.maximum(1-mean_u,1e-14)/(1-t))])
            self.tables.append((t,torch.from_numpy(vals).float().to(neighbors.device)))

    def predict(self,p):
        # Zero log terms correspond to a 1:1 prior between the no-enrichment
        # model and the threshold mixture. Each density integrates to one.
        numerator=torch.zeros_like(p)
        denominator=torch.zeros_like(p)
        donors=p[:,self.ids]
        for t,tab in self.tables:
            inside=p<=t
            count=(donors<=t).sum(2)
            weight=tab[0,count]
            density=torch.where(inside,tab[1,count],tab[2,count])
            numerator=torch.logaddexp(numerator,weight+density)
            denominator=torch.logaddexp(denominator,weight)
        value=numerator-denominator
        if not value.isfinite().all():raise ValueError('Nonfinite predictive density')
        return value


def rank_matched(x,source):
    order=source.argsort(dim=1,stable=True)
    return torch.empty_like(x).scatter_(1,order,x.sort(dim=1).values)


@torch.inference_mode()
def run(w,ds,full):
    if ds=='tuberlin' and not full:raise ValueError('Only locked full TU transfer')
    qs,gs,enc,_,q,p,fl,src,tgt,scale=audit_inputs(w,768,all_sketches=full,dataset=ds)
    kd,_=load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    result={}
    for s,(qi,y) in qs.items():
        sample=torch.arange(len(y)) if full else torch.randperm(len(y),generator=torch.Generator().manual_seed(87243))[:768].sort().values
        where=coordinates(enc,qi[sample])
        pi,gl=gs[s]
        g=F.normalize(p[pi].cuda())
        bank=torch.cat([(scale*(v@tgt.T)).softmax(1) for v in g.split(256)])
        graph=Index(SimpleNamespace(p=g),rgb_only=True)
        neighbors=nearest_gallery(F.normalize(bank@tgt),32)[0]
        if (neighbors==torch.arange(len(g),device=g.device)[:,None]).any():raise ValueError('Donor contains its target')
        rgb=graph.neighbors[0][0][:,:32]
        perm=torch.randperm(len(g),generator=torch.Generator().manual_seed(87017)).cuda()
        sham=perm[neighbors[perm.argsort()]]
        if (sham==torch.arange(len(g),device=g.device)[:,None]).any():raise ValueError('Shuffled donor contains its target')
        sizes={'raw':len(g),'conditional':len(g)//round(math.sqrt(len(g)))}
        readers={k:TailPrediction(ids,sizes[mode]) for k,ids,mode in [
            ('semantic_raw',neighbors,'raw'),('semantic_conditional',neighbors,'conditional'),
            ('rgb_conditional',rgb,'conditional'),('shuffled_conditional',sham,'conditional')]}
        replay=PrefixReplay(kd,gl,s)
        records={}
        cats=gl.cuda()
        # Evaluate the full grid in the pilot; the primary mechanism uses conditional
        # ranks and semantic donors. The endpoint itself is held out of its donor list.
        for start in range(0,len(where),24):
            ix=where[start:start+24]
            actual=y[sample[start:start+24]]
            a=(scale*(q[ix].cuda()@src.T)).softmax(1)
            b=(scale*(fl[ix].float().cuda()@src.T)).softmax(1)
            lex=((a+b)/2)@bank.T
            nu=graph.score(CONFIG,lex).clamp_min(1e-30).log()
            visual=((q[ix].cuda()+fl[ix].float().cuda())/2)@g.T
            raw,_=visual_ranks(visual)
            cond,nref=visual_ranks(visual,nu)
            assert nref==sizes['conditional']
            signals={k:r.predict(raw if k=='semantic_raw' else cond) for k,r in readers.items()}
            smooth=graph.score(CONFIG,(scale*(visual-visual.amax(1,keepdim=True))).exp()).clamp_min(1e-30)
            local=signals['semantic_conditional']
            signals['matched_visual']=rank_matched(local,visual)
            signals['matched_conditional_rank']=rank_matched(local,-cond)
            signals['matched_smoothed_visual']=rank_matched(local,smooth)
            def evaluate(name,scores):
                rank,quick=raw_ranks(scores,actual,cats,s)
                exact=replay.chunk((rank.cpu(),actual))
                if float((exact-quick).abs().max())>2e-6:raise ValueError('KD mismatch')
                hit=cats[rank[:,:200]]==actual.cuda()[:,None]
                records.setdefault(name,[]).append(torch.cat([exact,hit.any(1).cpu()[:,None],hit[:,:1].cpu()],1))
            evaluate('native',nu)
            for name,increment in signals.items():
                for value in (.125,.25,.5,1.):
                    evaluate(f'{name}/{value:g}',nu+value*increment)
        vals={k:torch.cat(v).double() for k,v in records.items()}
        base=vals['native']
        means={}
        for k,v in vals.items():
            means[k]={'metrics':(100*v[:,:2].mean(0)).tolist(),
                      'recovered_and_lost_top200':[((v[:,2]-base[:,2])==1).sum().item(),((v[:,2]-base[:,2])==-1).sum().item()],
                      'fixed_and_damaged_top1':[((v[:,3]-base[:,3])==1).sum().item(),((v[:,3]-base[:,3])==-1).sum().item()]}
        result[s]={'queries':len(sample),'gallery':len(g),'rows':means,
                   'native_miss_200':int((base[:,2]==0).sum()),'class_ids':y[sample].unique().tolist()}
        root=w/'results/predictive-visual-tail'
        root.mkdir(exist_ok=True,parents=True)
        torch.save({'labels':y[sample],'per_query':vals},root/(s+('-full' if full else '-pilot')+'.pt'))
        print(s,'unit predictive',means['semantic_conditional/1'],flush=True)
        del graph,bank
    (root/(ds+('-full' if full else '-pilot')+'.json')).write_text(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['sketchy','tuberlin'],default='sketchy')
    parser.add_argument('--full',action='store_true')
    options=parser.parse_args()
    run(Path('/content/sbir'),options.dataset,options.full)
