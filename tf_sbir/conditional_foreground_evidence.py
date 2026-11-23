"""What extra evidence does foreground assignment provide once noun and order are known?"""
import hashlib
import inspect
import json
from collections import defaultdict
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F

from .branch_view_study import PrefixReplay
from .encoding import load_frozen_clip
from .finite_gallery_response import retrieval
from .foreground_word_role import axes,make_tasks,prepare
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .semantic import read_words
from .standardized_context_audit import audit_inputs


def increment(xa,xb,obs,feature,predictors):
    """Exact Gaussian conditional log likelihood ratio at equal conditional variance."""
    a,b=xa.mean(0),xb.mean(0)
    centered=torch.cat([xa-a,xb-b])
    if predictors:
        beta=torch.linalg.lstsq(centered[:,predictors],centered[:,feature:feature+1],driver='gelsd').solution[:,0]
        def residual(x):return x[...,feature]-x[...,predictors]@beta
    else:
        def residual(x):return x[...,feature]
    res=residual(centered)
    df=len(res)-2-len(predictors)
    if df<=0 or float(res.square().sum())<=1e-12:return obs.new_zeros(len(obs))
    va=res.square().sum()/df
    mu_a,mu_b=residual(a),residual(b)
    return (mu_a-mu_b)/va*(residual(obs)-(mu_a+mu_b)/2)


def gaussian_readings(data,task,lookup,features):
    q,_,_,ua,ub=task
    pool=defaultdict(list)
    for i,(a,b) in enumerate(zip(ua.tolist(),ub.tolist(),strict=True)):
        pool[min(a,b),max(a,b)].append(i)
    groups=defaultdict(list)
    for i,t in enumerate(data['terms'].tolist()):groups[t].append(i)
    readers={'name':(0,[]),'role':(2,[]),'role_given_name_order':(2,[0,1]),
             'order_given_name_role':(1,[0,2])}
    updates={key:torch.zeros(len(data['yl'])) for key in (*readers,'permuted_role_given_name_order')}
    group_accuracy=[]
    holdout_loss=[]
    high_confidence=[]
    all_used=torch.zeros(len(data['yl']),dtype=torch.bool)
    for pair,jj in pool.items():
        ia,ib=(torch.tensor(groups[k]) for k in pair)
        if min(len(ia),len(ib))<2 or len(ia)+len(ib)<=4:continue
        j=torch.tensor(jj)
        ids=q[j]
        k=lookup[pair]
        axis=F.normalize(torch.stack([features[n][k] for n in ('isolated','position','role')]))
        xa=(data['photos'][ia]@axis.T).double()
        xb=(data['photos'][ib]@axis.T).double()
        obs=(data['query'][ids]@axis.T).double()
        sign=torch.where(ua[j]<ub[j],1.,-1.)
        for name,(col,pred) in readers.items():
            updates[name][ids]=(sign*increment(xa,xb,obs,col,pred)).float()
        x=torch.cat([xa,xb])
        rng=torch.Generator().manual_seed(5317+pair[0]*20931+pair[1])
        shuffle=x[torch.randperm(len(x),generator=rng)]
        updates['permuted_role_given_name_order'][ids]=(sign*increment(shuffle[:len(xa)],shuffle[len(xa):],obs,2,[0,1])).float()
        all_used[ids]=True
        # Held-photo controls use donor top-word identities only. These are
        # automatic lexical groundings, not class annotations.
        if min(len(ia),len(ib))>=6:
            aa=xa[torch.randperm(len(xa),generator=rng)]
            bb=xb[torch.randperm(len(xb),generator=rng)]
            train_a,test_a=aa.tensor_split(2)
            train_b,test_b=bb.tensor_split(2)
            vals=[]
            loss=[]
            for t,orientation in [(test_a,1),(test_b,-1)]:
                noun=increment(train_a,train_b,t,0,[])
                position=increment(train_a,train_b,t,1,[0])
                extra=increment(train_a,train_b,t,2,[0,1])
                base=orientation*(noun+position)
                augmented=base+orientation*extra
                vals.append(torch.stack([(base>0).double().mean(),(augmented>0).double().mean()]))
                loss.append(torch.stack([F.softplus(-base).mean(),F.softplus(-augmented).mean()]))
            group_accuracy.append(torch.stack(vals).mean(0))
            holdout_loss.append(torch.stack(loss).mean(0))
        high_confidence.append(updates['role_given_name_order'][ids].abs())
    return updates,{'identifiable_queries':int(all_used.sum()),'tested_donor_word_pairs':len(group_accuracy),
        'heldout_balanced_word_accuracy_before_after_role':torch.stack(group_accuracy).mean(0).tolist(),
        'heldout_equal_pair_log_loss_before_after_role':torch.stack(holdout_loss).mean(0).tolist(),
        'absolute_role_log_odds_quantiles':torch.cat(high_confidence).quantile(torch.tensor([.5,.9,.99])).tolist()}


def shift_scores(score,terms,a,b,log_odds):
    # Preserve the joint positive retrieval mass of the two nominated groups.
    # Each group keeps its native relative order and all outside scores stay fixed.
    ma=terms[None,:]==a[:,None]
    mb=terms[None,:]==b[:,None]
    va=score.masked_fill(~ma,-torch.inf).logsumexp(1)
    vb=score.masked_fill(~mb,-torch.inf).logsumexp(1)
    odds=va-vb+log_odds
    total=torch.logaddexp(va,vb)
    new_a=total-F.softplus(-odds)
    new_b=total-F.softplus(odds)
    delta=torch.where(ma,(new_a-va)[:,None],torch.where(mb,(new_b-vb)[:,None],0.))
    return score+torch.where((a!=b)[:,None],delta,0.)


@torch.inference_mode()
def run():
    w=Path('/content/sbir')
    qs,gs,encoded,_,q,p,fl,src,tgt,scale=audit_inputs(w,768)
    data={s:prepare(s,qs,gs,encoded,q,p,fl,src,tgt,scale,False,retain_scores=True) for s in qs}
    tasks={s:make_tasks(d,only_incumbent=True)['first_different_name'] for s,d in data.items()}
    pairs=sorted({(min(a,b),max(a,b)) for task in tasks.values() for a,b in zip(task[3].tolist(),task[4].tolist(),strict=True)})
    model=load_frozen_clip(str(w/'assets/ViT-B-32.pt'),torch.device('cuda'))
    feats,stats=axes(model,open_clip.get_tokenizer('ViT-B-32'),pairs,read_words(w/'assets/open-images-v7-classes.csv'))
    kd,_=load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    result={'text':stats}
    for s,d in data.items():
        corrections,study=gaussian_readings(d,tasks[s],{pair:i for i,pair in enumerate(pairs)},feats)
        a=d['terms'][d['order'][:,0].long()].cuda()
        b=d['terms'][d['alt']].cuda()
        terms=d['terms'].cuda()
        replay=PrefixReplay(kd,d['gl'],s)
        measures={}
        zero=torch.zeros(len(a))
        for name,delta in {'baseline':zero,**corrections}.items():
            values=[]
            top_hits=[]
            for start in range(0,len(a),48):
                sl=slice(start,start+48)
                native=d['log_scores'][sl].cuda()
                score=shift_scores(native,terms,a[sl],b[sl],delta[sl].cuda()) if name!='baseline' else native
                values.append(retrieval(score,d['yl'][sl],d['gl'],s,replay))
                top_hits.append(d['gl'][score.argmax(1).cpu()]==d['yl'][sl])
            v=torch.cat(values)
            hit=torch.cat(top_hits)
            if name=='baseline':base=v;old=hit
            diff=(v-base)*100
            labels=d['yl']
            by_class=torch.stack([diff[labels==c].mean(0) for c in labels.unique()])
            boots=torch.randint(len(by_class),(10000,len(by_class)),generator=torch.Generator().manual_seed(518))
            measures[name]={'metrics':(100*v.mean(0)).tolist(),'class_delta':by_class.mean(0).tolist(),
                'ci95':by_class[boots].mean(1).quantile(torch.tensor([.025,.975],dtype=torch.double),dim=0).tolist(),
                'top1_fixed_damaged':[int((~old&hit).sum()),int((old&~hit).sum())]}
        result[s]={'queries':len(a),'gallery':len(d['gl']),'study':study,'official':measures}
        print(s,study,measures['role_given_name_order'],flush=True)
    out=w/'results/conditional-foreground-evidence'
    out.mkdir(parents=True,exist_ok=True)
    (out/'sketchy.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':
    run()
