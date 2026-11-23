"""Cross-fitted gallery pivots with a query partition function for each donor."""
import hashlib
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.stats import spearmanr
import torch
import torch.nn.functional as F

from .branch_view_study import PrefixReplay
from .gallery_context_reversals import coordinates
from .gallery_region_panels import CONFIG,Index
from .positive_foil import raw_ranks
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .standardized_context_audit import audit_inputs
from .strong_pilot_scores import location


def style_bank(g,src,tgt,scale):
    src_z=[]
    tgt_z=[]
    emission=[]
    target=[]
    for x in g.split(160):
        a=scale*(x@src.T)
        b=scale*(x@tgt.T)
        u,v=a.logsumexp(1),b.logsumexp(1)
        # Same public noun index under both already approved description prompts.
        mix=torch.logaddexp(a,b)-torch.logaddexp(u,v)[:,None]
        emission.append(mix.exp())
        target.append(b.softmax(1))
        src_z.append(u-math.log(len(src)))
        tgt_z.append(v-math.log(len(tgt)))
    return torch.cat(src_z),torch.cat(tgt_z),torch.cat(emission),torch.cat(target)


def image_partition(g,scale,logh):
    # Hold out the donor's own photograph from the proposed query measure.
    # A common shift of every log normalizer cancels in a posterior over donors.
    n=len(g)
    logmass=(logh.exp().sum()-logh.exp()).log()
    zs=[]
    zu=[]
    for i in range(0,n,128):
        block=g[i:i+128]
        scores=scale*(block@g.T)
        ar=torch.arange(len(block),device=g.device)
        scores[ar,i+ar]=-torch.inf
        zs.append((scores+logh[None,:]).logsumexp(1)-logmass[i:i+len(block)])
        zu.append(scores.logsumexp(1)-math.log(n-1))
    return torch.cat(zs),torch.cat(zu)


def ranked_increment(x,ref):
    idx=ref.argsort(dim=1,stable=True)
    return torch.empty_like(x).scatter_(1,idx,x.sort(dim=1).values)


def equal_style_shuffle(h):
    ids=h.argsort()
    perm=torch.arange(len(h),device=h.device)
    gen=torch.Generator().manual_seed(60219)
    for g in ids.tensor_split(32):
        perm[g]=g[torch.randperm(len(g),generator=gen).cuda()]
    return perm


@torch.inference_mode()
def run():
    w=Path('/content/sbir')
    qs,gs,enc,_,q,p,flip,src,tgt,scale=audit_inputs(w,768)
    kd,_=load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    folder=w/'results/normalized-gallery-pivot'
    folder.mkdir(parents=True,exist_ok=True)
    out={}
    for s,(qi,y) in qs.items():
        rows=torch.randperm(len(qi),generator=torch.Generator().manual_seed(87243))[:768].sort().values
        where=coordinates(enc,qi[rows])
        pi,gl=gs[s]
        pack=torch.load(location(w,s,rows),mmap=True,weights_only=True)
        assert torch.equal(pack['pi'],pi) and torch.equal(pack['qi'],qi[rows])
        g=F.normalize(p[pi].cuda())
        zs,zt,mix,target=style_bank(g,src,tgt,scale)
        logh=F.logsigmoid(zs-zt)
        zimg,zall=image_partition(g,scale,logh)
        logjoint=torch.logaddexp(zs,zt)
        identity_error=float((logh-zs+logjoint).abs().max())
        assert identity_error<2e-5
        exposures={'raw_pivot':torch.zeros_like(zs),'style_only':logh,
                   'source_partition_only':-zs,'joint_text_partition':-logjoint,
                   'image_style_partition':logh-zimg,'image_uniform_partition':-zall,
                   'target_text_partition':-zt}
        split=torch.randperm(len(g),generator=torch.Generator().manual_seed(69313)).cuda().tensor_split(2)
        replayer=PrefixReplay(kd,gl,s)
        graph=Index(SimpleNamespace(p=g),rgb_only=True)
        prior=target.mean(0).clamp_min(1e-30)
        gathered={}
        visual_logits=[]
        donor_style=[]
        for start in range(0,len(where),48):
            ix=where[start:start+48]
            sl=slice(start,start+len(ix))
            actual=y[rows[sl]]
            base=pack['score'][sl].cuda()
            x=(q[ix].cuda()+flip[ix].float().cuda())/2
            v=scale*(x@g.T)
            visual_logits.append(v.cpu())
            inferred={}
            for train,test in [(split[0],split[1]),(split[1],split[0])]:
                shuf=equal_style_shuffle(logh[train])
                for key,bias in exposures.items():
                    weights=(v[:,train]+bias[None,train]).softmax(1)
                    if key=='raw_pivot':
                        donor_style.append((weights@logh[train].exp()).cpu())
                    codes=weights@mix[train]
                    val=(codes/prior)@target[test].T
                    inferred.setdefault(key,torch.empty_like(v))[:,test]=val
                    if key=='joint_text_partition':
                        fake=weights@mix[train[shuf]]
                        inferred.setdefault('shuffled_word_identity',torch.empty_like(v))[:,test]=(fake/prior)@target[test].T
            # The query-word-only control has the same target prior correction.
            qwords=((scale*(q[ix].cuda()@src.T)).softmax(1)+(scale*(flip[ix].float().cuda()@src.T)).softmax(1))/2
            inferred['direct_words']=(qwords/prior)@target.T
            true_log=inferred['joint_text_partition'].clamp_min(1e-30).log()
            increments={k:value.clamp_min(1e-30).log() for k,value in inferred.items()}
            increments['matched_visual']=ranked_increment(true_log,v)
            smooth_visual=graph.score(CONFIG,(v-v.amax(1,keepdim=True)).exp())
            increments['matched_smoothed_visual']=ranked_increment(true_log,smooth_visual)
            def score(name,value):
                rank,quick=raw_ranks(value,actual,gl.cuda(),s)
                precise=replayer.chunk((rank.cpu(),actual))
                if (precise-quick).abs().max()>2e-6:raise ValueError('Original KD replay mismatch')
                good=gl.cuda()[rank[:,:200]]==actual.cuda()[:,None]
                gathered.setdefault(name,[]).append(torch.cat([precise,good.any(1).cpu()[:,None],good[:,:1].cpu()],1))
            score('strong',base)
            for name,val in increments.items():
                if name in ('joint_text_partition','raw_pivot','image_style_partition','direct_words'):
                    score('only/'+name,val)
                inc=(val-val.mean(1,keepdim=True))/val.std(1,keepdim=True).clamp_min(1e-8)
                for alpha in (.03125,.125,.5):
                    score(f'{name}/{alpha:g}',base+alpha*base.std(1,keepdim=True)*inc)
        z={k:torch.cat(v).double() for k,v in gathered.items()}
        base=z['strong']
        assert (base[:,:2]-pack['kd']).abs().max()<2e-6
        report={}
        for k,v in z.items():
            delta=v-base
            report[k]={'metrics':(100*v[:,:2].mean(0)).tolist(),
                       'top200_recovered_lost':[int((delta[:,2]>0).sum()),int((delta[:,2]<0).sum())],
                       'top1_fixed_damaged':[int((delta[:,3]>0).sum()),int((delta[:,3]<0).sum())]}
        # Only after completing every score, measure donor popularity across
        # independent sketches. These held-sketch measurements never enter scoring.
        visual=torch.cat(visual_logits)
        y_eval=y[rows]
        labels=y_eval.unique()
        actual_partition=torch.stack([visual[y_eval==c].logsumexp(0)-math.log(int((y_eval==c).sum())) for c in labels]).logsumexp(0)-math.log(len(labels))
        proxy={}
        for k,data in {'source_text':zs,'target_text':zt,'text_joint':logjoint,
                       'gallery_source_like':zimg,'gallery_uniform':zall,'source_vs_target':zs-zt}.items():
            x=data.cpu().numpy()
            proxy[k]={'all_gallery_spearman':float(spearmanr(x,actual_partition.numpy()).statistic),
                'within_class_median_spearman':float(np.median([spearmanr(x[gl==c],actual_partition[gl==c].numpy()).statistic for c in gl.unique()]))}
        out[s]={'queries':len(where),'gallery':len(g),'identity_error':identity_error,
                'mean_source_style_weight_of_visual_donors':float(torch.cat(donor_style).mean()),
                'mean_gallery_source_style_weight':float(logh.exp().mean()),'observed_query_partition':proxy,'rows':report}
        torch.save({'per_query':z,'labels':y_eval,'observed_query_partition':actual_partition},folder/(s+'.pt'))
        print(s,'evidence',proxy,'pilot',report['joint_text_partition/0.125'],flush=True)
        del graph,target,mix
    (folder/'sketchy-pilot.json').write_text(json.dumps(out,indent=2))

if __name__=='__main__':
    run()
