"""Does the signed response to adding unlabeled photos transfer to another gallery?"""
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .branch_view_study import PrefixReplay
from .gallery_context_reversals import coordinates
from .gallery_region_panels import CONFIG,Index
from .positive_foil import raw_ranks
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .standardized_context_audit import audit_inputs


def diffuse(g,evidence):
    index=Index(SimpleNamespace(p=g),rgb_only=True)
    return torch.cat([index.score(CONFIG,x).clamp_min(1e-30).log() for x in evidence.split(192)])


def partition_scores(g,evidence,partition):
    values=torch.empty_like(evidence)
    membership=torch.empty(len(g),device=g.device,dtype=torch.long)
    for k,ids in enumerate(partition):
        membership[ids]=k
        values[:,ids]=diffuse(g[ids],evidence[:,ids])
    return values,membership


def directional_report(predicted,actual,start,end):
    valid=(actual!=0)&(predicted!=0)
    scale=actual.square().mean().sqrt().clamp_min(1e-15)
    r={
        'pairs':len(start),'nonzero_comparisons':int(valid.sum()),
        'fraction_direction_correct':float((predicted[valid]*actual[valid]>0).float().mean()),
        'normalized_mse_neutral':1.,
        'normalized_mse_forecast':float(((actual-predicted)/scale).square().mean()),
        'normalized_mse_reversed_direction':float(((actual+predicted)/scale).square().mean()),
        'signed_projection':float((actual*predicted).sum()/predicted.square().sum().clamp_min(1e-30)),
        'target_sign_reversals':int((start*end<0).sum())}
    for name,adjust in [('baseline',start),('forecast',start+predicted),('reverse',start-predicted)]:
        switched=adjust*start<0
        r[name]={'reproduces_correct_target_sign':int((adjust*end>0).sum()),
                 'switches':int(switched.sum()),'beneficial_vs_category':int((switched&(adjust>0)).sum()),
                 'harmful_vs_category':int((switched&(adjust<0)).sum())}
    return r


def retrieval(scores,labels,classes,split,replay):
    results=[]
    for v,y in zip(scores.split(48),labels.split(48),strict=True):
        rank,check=raw_ranks(v,y,classes.cuda(),split)
        official=replay.chunk((rank.cpu(),y))
        if float((check-official).abs().max())>2e-6:raise ValueError('KD replay mismatch')
        results.append(official)
    return torch.cat(results).double()


@torch.inference_mode()
def run():
    w=Path('/content/sbir')
    qs,gs,encoded,cp,q,p,flip,src,tgt,scale=audit_inputs(w,768)
    fixture=torch.load(w/'results/gallery-context-reversals/fixed-boundaries.pt',weights_only=True)
    assert torch.equal(cp,fixture['photos'])
    shared=fixture['queries']
    iq,jl,jr,sign=[fixture[k].cuda() for k in ('rows','left','right','orientation')]
    kd,_=load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    result={}
    across={}
    for s,(qi,labels) in qs.items():
        sample=torch.randperm(len(qi),generator=torch.Generator().manual_seed(87243))[:768].sort().values
        requested=torch.cat([shared,qi[sample]]).unique(sorted=True)
        where=coordinates(encoded,requested)
        eval_rows=coordinates(requested,qi[sample])
        shared_rows=coordinates(requested,shared).cuda()
        pi,classes=gs[s]
        cols=coordinates(pi,cp).cuda()
        g=F.normalize(p[pi].cuda())
        bank=torch.cat([(scale*(x@tgt.T)).softmax(1) for x in g.split(256)])
        e=[]
        for idx in where.split(128):
            a=(scale*(q[idx].cuda()@src.T)).softmax(1)
            b=(scale*(flip[idx].float().cuda()@src.T)).softmax(1)
            e.append(((a+b)/2)@bank.T)
        e=torch.cat(e)
        del bank
        full=diffuse(g,e)
        l,r=cols[jl],cols[jr]
        query=shared_rows[iq]
        def margin(score):
            return (score[query,l]-score[query,r])*sign
        half_sum=torch.zeros_like(full)
        half_reads=[]
        quarter_reads=[]
        for draw in range(8):
            perm=torch.randperm(len(pi),generator=torch.Generator().manual_seed(97031+draw)).cuda()
            quarters=perm.tensor_split(4)
            halves=[torch.cat(quarters[:2]),torch.cat(quarters[2:])]
            half,hm=partition_scores(g,e,halves)
            quarter,qm=partition_scores(g,e,quarters)
            half_sum+=half/8
            half_reads.append((margin(half),hm[l]==hm[r]))
            quarter_reads.append((margin(quarter),qm[l]==qm[r]))
            del half,quarter
            print(s,'partitions',draw+1,flush=True)
        base=margin(full)
        across[s]={'base':base.cpu(),'half':margin(half_sum).cpu()}
        # Independent random partition draws for estimating the direction and
        # measuring the subsequent move. Both photos must coexist in the subgraph.
        def mean_of(values,masks):
            count=torch.stack(masks).sum(0)
            return (torch.stack(values)*torch.stack(masks)).sum(0)/count.clamp_min(1),count
        forecasts=[]
        for train,test in [(range(4),range(4,8)),(range(4,8),range(4))]:
            move,n1=mean_of([half_reads[j][0]-quarter_reads[j][0] for j in train],
                            [quarter_reads[j][1] for j in train])
            start,n2=mean_of([half_reads[j][0] for j in test],[half_reads[j][1] for j in test])
            valid=(n1>0)&(n2>0)
            forecasts.append(directional_report(.5*move[valid],base[valid]-start[valid],start[valid],base[valid]))
        replay=PrefixReplay(kd,classes,s)
        native=full[eval_rows]
        change=native-half_sum[eval_rows]
        # Only the alignment of the change to a particular candidate is broken.
        perm=torch.randperm(len(pi),generator=torch.Generator().manual_seed(33511)).cuda()
        tests={'native':native,'extrapolated':native+change,'half_average':native-change,
               'placebo':native+change[:,perm]}
        values={}
        for k,sc in tests.items():
            values[k]=retrieval(sc,labels[sample],classes,s,replay)
        delta=100*(values['extrapolated']-values['native'])
        y=labels[sample]
        macro=torch.stack([delta[y==c].mean(0) for c in y.unique()])
        boot=torch.randint(len(macro),(10000,len(macro)),generator=torch.Generator().manual_seed(419))
        result[s]={'queries_evaluated':len(sample),'gallery':len(pi),'crossfit_half_to_full':forecasts,
                   'metrics':{k:(100*v.mean(0)).tolist() for k,v in values.items()},
                   'class_mean_delta':macro.mean(0).tolist(),
                   'class_ci95':macro[boot].mean(1).quantile(torch.tensor([.025,.975],dtype=torch.double),dim=0).tolist()}
        print(s,'retrieval',result[s]['metrics'],flush=True)
        del full,half_sum,e
    for a,b in [('sketchy_1','sketchy_2'),('sketchy_2','sketchy_1')]:
        v=across[a]['base']
        u=across[b]['base']
        result[a]['native_gallery_forecast']=directional_report(v-across[a]['half'],u-v,v,u)
    for variant in ['native','extrapolated','half_average']:
        def projected(s):
            v=across[s]['base']
            d=v-across[s]['half']
            return v+{'native':0,'half_average':-1,'extrapolated':1}[variant]*d
        a,b=projected('sketchy_1'),projected('sketchy_2')
        result.setdefault('fixed_pairs',{})[variant]={'pairs':len(a),'flips':int((a*b<0).sum()),
                                                    'pair_correct':[int((a>0).sum()),int((b>0).sum())]}
    out=w/'results/finite-gallery-response'
    out.mkdir(exist_ok=True,parents=True)
    (out/'sketchy.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':
    run()
