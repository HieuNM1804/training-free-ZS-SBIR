"""Single-view, full-gallery analysis of visual and inferred semantic evidence."""
import argparse
import hashlib
import inspect
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F

from .branch_view_study import PrefixReplay
from .depiction_path_diagnostic import class_interval
from .encoding import save_tensor
from .gallery_panel_resume import semantic_inputs
from .locked_geometry_words import score_rows
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd


def rescale(v):
    lo=v.amin(1,keepdim=True)
    return (v-lo)/(v.amax(1,keepdim=True)-lo).clamp_min(1e-15)


def harris(a,b):
    return a*b-.1*(a+b).square()


def uniform_ranks(v):
    # Midranks keep ties neutral.
    ordered=v.sort(1).values.contiguous()
    v=v.contiguous()
    lower=torch.searchsorted(ordered,v)
    upper=torch.searchsorted(ordered,v,right=True)
    return (upper+lower-1).float()/(2*(v.size(1)-1))


def assess_pair(a,b,truth,cl):
    order_a=a.topk(1000).indices
    order_b=b.topk(1000).indices
    target=truth[:,None]
    ha=cl[order_a]==target
    hb=cl[order_b]==target
    both=torch.zeros_like(a,dtype=torch.bool).scatter_(1,order_a[:,:200],True)
    overlap=both.gather(1,order_b[:,:200]).sum(1)
    def matched(order,hit,other):
        # Adjacent disjoint pairs are matched by the first ranking, inside its top 1,000.
        left,right=order[:,::2],order[:,1::2]
        l,r=hit[:,::2],hit[:,1::2]
        use=l^r
        margin=(other.gather(1,left)-other.gather(1,right))*(l.float()-r.float())
        win=((margin>0).float()+.5*(margin==0))*use
        return torch.stack([use.sum(1),win.sum(1)],1)
    return torch.cat([ha[:,:1],hb[:,:1],overlap[:,None],matched(order_a,ha,b),matched(order_b,hb,a)],1).double().cpu()


def comparison(raw,evidence,first,second,labels):
    a,b=raw[first][:,0].double(),raw[second][:,0].double()
    u,v,overlap,n1,w1,n2,w2=evidence.T
    def discr(n,w):
        valid=n>0
        return {'pairs':int(n.sum()),'fraction_correct':float(w.sum()/n.sum()),
                'queries_with_pairs':int(valid.sum()),'mean_query_accuracy':float((w[valid]/n[valid]).mean())}
    return {'visual_only_correct_at_one':int(((u==1)&(v==0)).sum()),'semantic_only_correct_at_one':int(((u==0)&(v==1)).sum()),
        'both_correct':int(((u==1)&(v==1)).sum()),'neither_correct':int(((u==0)&(v==0)).sum()),
        'AP_wins_visual_semantic_ties':[int((a>b).sum()),int((a<b).sum()),int((a==b).sum())],
        'mean_top200_intersection':float(overlap.mean()),'retrospective_max_of_two_mean_AP':float(100*torch.maximum(a,b).mean()),
        'oracle_uplift_over_semantic':float(100*(a-b).clamp_min(0).mean()),
        'semantic_orders_visual_matched_pairs':discr(n1,w1),'visual_orders_semantic_matched_pairs':discr(n2,w2)}


def summarize(rows,labels,anchor):
    return {k:{'metrics':(100*v[:,:2].double().mean(0)).tolist(),
               'class_macro_map_delta':class_interval(100*(v[:,0].double()-rows[anchor][:,0].double()),labels)}
            for k,v in rows.items()}


@torch.inference_mode()
def run(ds,pilot):
    w=Path('/content/sbir')
    root=w/'results/basic-isolated-channels'
    root.mkdir(parents=True,exist_ok=True)
    cfg,_,qs,_,gs,q,p,src,tgt,scale=semantic_inputs(w,ds)
    kd,_=load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    if ds=='tuberlin' and pilot:raise ValueError('Full 30-class TU required')
    mid=(src+tgt)/2
    pos=mid-mid.mean(0)
    factor=pos.T@pos/len(pos)
    delta=(tgt-src)/2
    noise=delta.T@delta/len(delta)
    def projection(mat):
        e,u=torch.linalg.eigh(mat)
        return u[:,-min(250,int((e>0).sum())):]
    cp=projection(.8*factor-.2*noise)
    pp=projection(factor)
    out={}
    tag='pilot' if pilot else 'full'
    for split,(qi,y) in qs.items():
        keep=torch.randperm(len(qi),generator=torch.Generator().manual_seed(87243))[:768].sort().values if pilot else torch.arange(len(qi))
        pi,gl=gs[split];cl=gl.cuda()
        g=F.normalize(p[pi].cuda())
        mean=g.mean(0);center_g=g-mean
        bank=torch.cat([(scale*(z@tgt.T)).softmax(1) for z in g.split(192)])
        replay=PrefixReplay(kd,gl,split)
        results,diagnostic={},{}
        shardroot=root/(tag+'-'+split+'-shards');shardroot.mkdir(exist_ok=True)
        with ThreadPoolExecutor(max_workers=min(2,len(os.sched_getaffinity(0)))) as pool:
            for begin in range(0,len(keep),512):
                end=min(begin+512,len(keep))
                path=shardroot/(str(begin)+'.pt')
                if path.exists():
                    part=torch.load(path,weights_only=True)
                    assert torch.equal(part['ids'],qi[keep[begin:end]])
                else:
                    vals,diags={},{}
                    for start in range(begin,end,64):
                        rows=keep[start:min(end,start+64)]
                        x=F.normalize(q[qi[rows]].cuda())
                        pword=(scale*(x@src.T)).softmax(1)
                        t=F.normalize(pword@tgt)
                        v=x@g.T
                        st=t@g.T
                        sp=pword@bank.T
                        vn,sn=rescale(v),rescale(st)
                        vr,sr=uniform_ranks(v),uniform_ranks(sp)
                        vs,ss=uniform_ranks(v),uniform_ranks(st)
                        scores={'visual':v,'inferred_text':st,'word_overlap':sp,
                            'mean_text':(vn+sn)/2,'product_text':vn*sn,'harris_text':harris(vn,sn),
                            'rank_mean_words':(vr+sr)/2,'rank_harris_words':harris(vr,sr),
                            'rank_mean_text':(vs+ss)/2,'rank_harris_text':harris(vs,ss)}
                        if pilot:
                            centered=(x-mean)@center_g.T
                            proj=((x-mean)@cp)@(center_g@cp).T
                            pca=((x-mean)@pp)@(center_g@pp).T
                            top=v.topk(5).indices
                            memory=torch.cat([x[:,None],g[top]],1)
                            qe=F.normalize(((.1*(memory@x[:,:,None])).softmax(1)*memory).sum(1))@g.T
                            scores.update({'visual_center':centered,'visual_pca250':pca,'visual_cpca250':proj,'visual_qe5':qe,
                                'text_center':(t-tgt.mean(0))@center_g.T,'text75_linear':.25*vn+.75*sn,
                                'text90_linear':.1*vn+.9*sn,'word75_rank':.25*vr+.75*sr,'word90_rank':.1*vr+.9*sr})
                        for name,score in scores.items():
                            vals.setdefault(name,[]).append(score_rows(score,y[rows],cl,split,replay,pool))
                        for name,sem in [('inferred_text',st),('word_overlap',sp)]:
                            stats=assess_pair(v,sem,y[rows].cuda(),cl)
                            diags.setdefault(name,[]).append(stats)
                    part={'ids':qi[keep[begin:end]],'rows':{k:torch.cat(v) for k,v in vals.items()},
                          'diagnostic':{k:torch.cat(v) for k,v in diags.items()}}
                    save_tensor(part,path)
                for k,v in part['rows'].items():results.setdefault(k,[]).append(v)
                for k,v in part['diagnostic'].items():diagnostic.setdefault(k,[]).append(v)
                print(split,end,len(keep),flush=True)
        raw={k:torch.cat(v) for k,v in results.items()}
        diag={k:torch.cat(v) for k,v in diagnostic.items()}
        output={'queries':len(keep),'photos':len(g),'classes':len(cl.unique()),'fit':'gallery only; fixed protocol',
            'rows':summarize(raw,y[keep],'inferred_text'),'complementarity':{k:comparison(raw,v,'visual',k,y[keep]) for k,v in diag.items()}}
        out[split]=output
        save_tensor({'ids':qi[keep],'labels':y[keep],'rows':raw,'diagnostics':diag},root/(tag+'-'+split+'.pt'))
        (root/(tag+'-'+split+'.json')).write_text(json.dumps(output,indent=2))
        del bank
    (root/(tag+'-'+ds+'.json')).write_text(json.dumps(out,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',choices=['sketchy','tuberlin'],required=True)
    ap.add_argument('--pilot',action='store_true')
    a=ap.parse_args()
    run(a.dataset,a.pilot)
