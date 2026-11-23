"""Do naturally source-like photos in an unlabeled gallery mediate retrieval across depiction?"""
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .branch_view_study import PrefixReplay
from .gallery_context_reversals import coordinates
from .gallery_region_panels import Index
from .positive_foil import raw_ranks
from .public_context_standardization import endpoint_matrix,grouped_permutation
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .standardized_context_audit import audit_inputs
from .strong_pilot_scores import location


def two_domain_readings(x,src,tgt,scale):
    # Same nouns with equal prior in the existing source and target prompt sets.
    # The log-odds are an operational CLIP proxy, not human style labels.
    source=scale*(x@src.T)
    target=scale*(x@tgt.T)
    return source.logsumexp(1)-target.logsumexp(1),target.argmax(1)


def pool(visual,kernel,weight,expected):
    signal=visual*weight
    prior=signal.sum(1,keepdim=True)/weight.sum()
    value=(signal@kernel.T)/expected.clamp_min(1e-30)
    return value/prior.clamp_min(1e-30)


@torch.inference_mode()
def run():
    w=Path('/content/sbir')
    qs,gs,encoded,_,q,p,flip,src,tgt,scale=audit_inputs(w,768)
    kd,_=load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    root=w/'results/natural-depiction-donors'
    root.mkdir(parents=True,exist_ok=True)
    result={}
    for s,(qi,y) in qs.items():
        rows=torch.randperm(len(qi),generator=torch.Generator().manual_seed(87243))[:768].sort().values
        where=coordinates(encoded,qi[rows])
        pi,labels=gs[s]
        pack=torch.load(location(w,s,rows),mmap=True,weights_only=True)
        assert torch.equal(pack['qi'],qi[rows]) and torch.equal(pack['pi'],pi)
        g=F.normalize(p[pi].cuda())
        readings=[two_domain_readings(x,src,tgt,scale) for x in g.split(256)]
        logodds,words=(torch.cat(v) for v in zip(*readings,strict=True))
        wsource=logodds.sigmoid()
        wtarget=(-logodds).sigmoid()
        perm=grouped_permutation(words)
        weights={'all':torch.ones_like(wsource),'source':wsource,'target':wtarget,
                 'shuffled_source':wsource[perm],'shuffled_target':wtarget[perm]}
        graph=Index(SimpleNamespace(p=g),rgb_only=True)
        kernel=endpoint_matrix(graph)
        del graph
        diag=kernel.diagonal().clone()
        kernel.fill_diagonal_(0)
        rowmass=kernel.sum(1,keepdim=True)
        if (rowmass<1e-8).any():raise ValueError('No independent donor')
        kernel/=rowmass
        denominators={k:(kernel@v).clamp_min(1e-30) for k,v in weights.items()}
        high=logodds>logodds.median()
        record={}
        probe=[]
        replay=PrefixReplay(kd,labels,s)
        cl=labels.cuda()
        for start in range(0,len(where),48):
            ix=where[start:start+48]
            sl=slice(start,start+len(ix))
            actual=y[rows[sl]]
            base=pack['score'][sl].cuda()
            x=F.normalize((q[ix].cuda()+flip[ix].float().cuda())/2)
            v=scale*(x@g.T)
            signal=(v-v.amax(1,keepdim=True)).exp()
            q_source=two_domain_readings(x,src,tgt,scale)[0].sigmoid()[:,None]
            donors={k:pool(signal,kernel,weights[k],ref) for k,ref in denominators.items()}
            mixture=q_source*donors['source']+(1-q_source)*donors['target']
            shuffled=q_source*donors['shuffled_source']+(1-q_source)*donors['shuffled_target']
            reverse=(1-q_source)*donors['source']+q_source*donors['target']
            def log(t):return t.clamp_min(1e-30).log()
            mods={'all':log(donors['all']),'source':log(donors['source']),'target':log(donors['target']),
                  'query':log(mixture),'reverse':log(reverse),'shuffled':log(shuffled),
                  'depiction_delta':log(mixture)-log(donors['all']),
                  'reverse_delta':log(reverse)-log(donors['all']),'shuffled_delta':log(shuffled)-log(donors['all']),
                  'direct_cosine':v}
            def evaluate(name,score):
                rank,fast=raw_ranks(score,actual,cl,s)
                exact=replay.chunk((rank.cpu(),actual))
                if float((exact-fast).abs().max())>2e-6:raise ValueError('KD mismatch')
                hits=cl[rank[:,:200]]==actual.cuda()[:,None]
                record.setdefault(name,[]).append(torch.cat([exact,hits.any(1).cpu()[:,None],hits[:,:1].cpu()],1))
            evaluate('strong',base)
            for name,feature in mods.items():
                # Compare strength-matched after removing per-query location/scale.
                z=(feature-feature.mean(1,keepdim=True))/feature.std(1,keepdim=True).clamp_min(1e-8)
                if name in ('all','source','target','query','direct_cosine'):
                    evaluate('only/'+name,feature)
                for alpha in (.03125,.125,.5):
                    evaluate(f'{name}/{alpha:g}',base+alpha*base.std(1,keepdim=True)*z)
            vh=v.topk(200).indices
            good=cl[vh]==actual.cuda()[:,None]
            any_strong=(cl[base.topk(200).indices]==actual.cuda()[:,None]).any(1)
            matched_prior=torch.stack([high[cl==c].float().mean() for c in actual.cuda()])
            fraction=(good&high[vh]).sum(1).float()/good.sum(1).clamp_min(1)
            probe.append(torch.stack([q_source[:,0],any_strong,good.any(1),(good&high[vh]).any(1),
               (good&~high[vh]).any(1),fraction,matched_prior],1).cpu())
        raw={k:torch.cat(v).double() for k,v in record.items()}
        baseline=raw['strong']
        if (pack['kd']-baseline[:,:2]).abs().max()>2e-6:raise ValueError('Strong reference differs')
        diagnostics=torch.cat(probe)
        eligible=(diagnostics[:,1]==0)&(diagnostics[:,2]>0)
        trace=diagnostics[eligible]
        stats={'strong_misses_top200':int((diagnostics[:,1]==0).sum()),'recoverable_by_visual_top200':len(trace),
               'recoveries_only_upper_style_half':int(((trace[:,3]>0)&(trace[:,4]==0)).sum()),
               'recoveries_only_lower_style_half':int(((trace[:,3]==0)&(trace[:,4]>0)).sum()),
               'recoveries_both_halves':int(((trace[:,3]>0)&(trace[:,4]>0)).sum()),
               'mean_fraction_visual_positives_in_upper_half':float(trace[:,5].mean()),
               'mean_gallery_fraction_in_upper_half_same_true_classes':float(trace[:,6].mean()),
               'mean_source_prompt_probability_sketch':float(diagnostics[:,0].mean()),
               'mean_source_prompt_probability_gallery':float(wsource.mean()),
               'gallery_source_probability_quantiles':wsource.quantile(wsource.new_tensor([.1,.5,.9])).tolist(),
               'mean_removed_self_weight':float(diag.mean())}
        table={}
        for k,v in raw.items():
            diff=v-baseline
            table[k]={'metrics':(100*v[:,:2].mean(0)).tolist(),
                      'rescued_lost_top200':[int((diff[:,2]>0).sum()),int((diff[:,2]<0).sum())],
                      'fixed_damaged_top1':[int((diff[:,3]>0).sum()),int((diff[:,3]<0).sum())]}
        result[s]={'queries':len(rows),'gallery':len(labels),'diagnostic':stats,'rows':table}
        torch.save({'per_query':raw,'diagnostic':diagnostics,'labels':y[rows]},root/(s+'.pt'))
        print(s,stats,table['query/0.125'],flush=True)
        del kernel
    (root/'sketchy-pilot.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':
    run()
