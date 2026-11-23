"""Counterbalance a noun's semantic role and mention position on real retrieval rivals."""
import hashlib
import argparse
import inspect
import json
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F

from .branch_view_study import PrefixReplay
from .encoding import load_frozen_clip
from .gallery_region_panels import CONFIG,Index
from types import SimpleNamespace
from .gallery_context_reversals import coordinates
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .semantic import read_words
from .standardized_context_audit import audit_inputs


def encode(model,tokenize,texts):
    return F.normalize(model.encode_text(tokenize(texts).cuda()).float())


def axes(model,tokenize,pairs,words):
    names=('isolated','foreground_first','foreground_last','role','position','role_residual')
    accum={k:[] for k in names}
    cosines=[]
    lengths=[]
    for ix in range(0,len(pairs),256):
        ab=[(words[a],words[b]) for a,b in pairs[ix:ix+256]]
        def caption(a,b,where1,where2):
            return f'an image showing {a} in the {where1} and {b} in the {where2}.'
        def contrast(first,last):
            pos=encode(model,tokenize,[caption(a,b,first,last) for a,b in ab])
            neg=encode(model,tokenize,[caption(b,a,first,last) for a,b in ab])
            return pos-neg
        early=contrast('foreground','background')
        late=-contrast('background','foreground')
        role=(early+late)/2
        order=(early-late)/2
        literal=encode(model,tokenize,[f'an image showing {a}.' for a,_ in ab])
        literal-=encode(model,tokenize,[f'an image showing {b}.' for _,b in ab])
        unit=F.normalize(literal)
        unit_order=F.normalize(order-(order*unit).sum(1,keepdim=True)*unit)
        rem=role-(role*unit).sum(1,keepdim=True)*unit
        rem-=((rem*unit_order).sum(1,keepdim=True))*unit_order
        for key,delta in zip(names,[literal,early,late,role,order,rem],strict=True):
            accum[key].append(delta.cpu())
        cosines.append(F.cosine_similarity(early,late).cpu())
        lengths.append((rem.norm(dim=1)/role.norm(dim=1).clamp_min(1e-12)).cpu())
    return {k:torch.cat(v) for k,v in accum.items()},{
        'unique_public_name_pairs':len(pairs),
        'mean_cosine_of_role_when_mentioned_first_vs_last':float(torch.cat(cosines).mean()),
        'mean_fraction_role_norm_outside_name_and_order_span':float(torch.cat(lengths).mean())}


def prepare(s,qs,gs,encoded,q,p,flip,src,tgt,scale,all_queries):
    qi,yl=qs[s]
    sample=(torch.arange(len(qi)) if all_queries else
            torch.randperm(len(qi),generator=torch.Generator().manual_seed(87243))[:768].sort().values)
    pi,gl=gs[s]
    where=coordinates(encoded,qi[sample])
    g=F.normalize(p[pi].cuda())
    bank=torch.cat([(scale*(part@tgt.T)).softmax(1) for part in g.split(256)])
    terms=bank.argmax(1).cpu()
    graph=Index(SimpleNamespace(p=g),rgb_only=True)
    if len(g)>torch.iinfo(torch.int16).max:raise ValueError('Gallery rank storage would overflow')
    order=torch.empty((len(where),len(g)),dtype=torch.int16)
    for start in range(0,len(where),128):
        ids=where[start:start+128]
        a=(scale*(q[ids].cuda()@src.T)).softmax(1)
        b=(scale*(flip[ids].float().cuda()@src.T)).softmax(1)
        scores=graph.score(CONFIG,((a+b)/2)@bank.T).clamp_min(1e-30).log()
        order[start:start+len(ids)]=scores.topk(scores.size(1)).indices.to(torch.int16).cpu()
    top=order[:,:200].long()
    named=terms[top]!=terms[top[:,:1]]
    valid=named.any(1)
    pos=named.int().argmax(1)
    row=torch.arange(len(sample))
    alt=top[row,pos]
    return {'query':((q[where]+flip[where].float())/2),'photos':g.cpu(),'terms':terms,
            'order':order,'valid':valid,'idx':pos,'alt':alt,'yl':yl[sample],'gl':gl}


def make_tasks(data,only_incumbent=False):
    top=data['order'][:,:64].long()
    row=torch.arange(len(top))
    candidates={
        'first_different_name':(row,top[:,0],data['alt'],data['valid']),
        'adjacent_pairs':(row[:,None].expand(-1,32).reshape(-1),top[:,::2].reshape(-1),
                          top[:,1::2].reshape(-1),torch.ones(len(top)*32,dtype=torch.bool))}
    out={}
    for key,(q,a,b,ok) in candidates.items():
        if only_incumbent and key!='first_different_name':continue
        u,v=data['terms'][a],data['terms'][b]
        keep=ok&(u!=v)
        out[key]=(q[keep],a[keep],b[keep],u[keep],v[keep])
    return out


def evaluate(data,task,pairs_to_id,features):
    q,a,b,u,v=task
    ids=torch.tensor([pairs_to_id[min(x,y),max(x,y)] for x,y in zip(u.tolist(),v.tolist(),strict=True)])
    sign=torch.where(u<v,1.,-1.)
    decisions={}
    margins={}
    photo_checks={}
    separates={}
    for k,bank in features.items():
        qmargin=[]
        photo_margin=[]
        individual=[]
        for j in range(0,len(q),1024):
            sl=slice(j,j+1024)
            axis=bank[ids[sl]].cuda()*sign[sl,None].cuda()
            qmargin.append((data['query'][q[sl]].cuda()*axis).sum(1).cpu())
            pa=data['photos'][a[sl]].cuda()
            pb=data['photos'][b[sl]].cuda()
            da=pa-pb
            photo_margin.append((da*axis).sum(1).cpu())
            individual.append((((pa*axis).sum(1)>0)&((pb*axis).sum(1)<0)).cpu())
        margin=torch.cat(qmargin)
        margins[k]=margin
        decisions[k]=margin<0
        photo_checks[k]=torch.cat(photo_margin)
        separates[k]=torch.cat(individual)
    # Can each candidate photo itself answer the very same two questions?
    # A fallback to the incumbent is well-defined even if either abstains.
    identifiable=separates['foreground_first']&separates['foreground_last']
    pose_agreement=margins['foreground_first']*margins['foreground_last']>0
    readable=identifiable&pose_agreement
    policy={
        'both_orders_only':pose_agreement&decisions['role'],
        'counterbalanced_verified':readable&decisions['role'],
        'isolated_on_same_opportunities':readable&decisions['isolated'],
        'position_on_same_opportunities':readable&decisions['position'],
        'literal_self_verified':separates['isolated']&decisions['isolated']}
    # Evaluation categories enter only after pair nomination and all text readings.
    ya=data['gl'][a]==data['yl'][q]
    yb=data['gl'][b]==data['yl'][q]
    contested=ya^yb
    result={'eligible':len(q),'mixed_relevance':int(contested.sum()),
            'incumbent_correct_on_mixed':int((ya&contested).sum()),'readings':{}}
    for k,switch in decisions.items():
        picked=torch.where(switch,yb,ya)
        result['readings'][k]={
            'mixed_correct':int((picked&contested).sum()),'fixed':int((switch&~ya&yb).sum()),
            'damaged':int((switch&ya&~yb).sum()),
            'gallery_noun_pair_orientation_positive':float((photo_checks[k]>0).float().mean()),
            'balanced_accuracy':.5*(float((~switch)[ya&~yb].float().mean())+float(switch[yb&~ya].float().mean()))}
    literal=margins['isolated']
    role=margins['role']
    for k,agreement in [('role_and_isolated_agree',role*literal>0),('role_and_isolated_disagree',role*literal<0)]:
        m=contested&agreement
        result[k]={'n':int(m.sum()),'role_correct':int((m&torch.where(role<0,yb,ya)).sum())}
    result['verification']={'two_photos_ground_both_orders':int(identifiable.sum()),
        'joint_opportunities':int(readable.sum()),'mixed_joint_opportunities':int((readable&contested).sum()),
        'policies':{k:{'switches':int(take.sum()),'fixed':int((take&~ya&yb).sum()),
                      'damaged':int((take&ya&~yb).sum())} for k,take in policy.items()}}
    decisions.update(policy)
    measures={}
    for name,switch in decisions.items():
        improved=(switch&yb&~ya).int()-(switch&ya&~yb).int()
        literal=decisions['isolated']
        cmp=improved-((literal&yb&~ya).int()-(literal&ya&~yb).int())
        if len(q):
            classes=data['yl'][q]
            category=torch.stack([cmp[classes==c].double().mean()*100 for c in classes.unique()])
            ids=torch.randint(len(category),(10000,len(category)),generator=torch.Generator().manual_seed(632))
            measures[name]={'opportunities':len(q),'net_fixed_minus_isolated':int(cmp.sum()),
                'class_balanced_correctness_delta_pp':float(category.mean()),
                'paired_class_ci95':category[ids].mean(1).quantile(torch.tensor([.025,.975],dtype=torch.double)).tolist()}
    result['comparison_with_isolated']=measures
    return result,decisions


def opportunities(d,base,swapped,qid,decisions,split,root):
    # Compare the original KD outputs of the two predetermined rankings. This
    # bound and category counts are computed only after scoring and ranking.
    heads=d['order'][:,:200].long()
    y=d['yl']
    good=d['gl'][heads]==y[:,None]
    originally_wrong=~good[:,0]
    names=d['terms'][heads].numpy()
    budgets=(1,2,4,8,16,32)
    reps=torch.full((len(y),max(budgets)),-1,dtype=torch.long)
    for i,words in enumerate(names):
        uniq={}
        for j,t in enumerate(words):
            uniq.setdefault(int(t),j)
            if len(uniq)==max(budgets):break
        pos=list(uniq.values())
        reps[i,:len(pos)]=torch.tensor(pos)
    proposed=good.gather(1,reps.clamp_min(0))&(reps>=0)
    delta=(swapped-base)*100
    compared={}
    masks={}
    for name in ['isolated','role','role_residual','position','counterbalanced_verified']:
        mask=torch.zeros(len(y),dtype=torch.bool)
        mask[qid]=decisions[name]
        masks[name]=mask
        taken=delta*mask[:,None]
        compared[name]={'per_query_points_gained':taken.clamp_min(0).mean(0).tolist(),
                        'per_query_points_lost':(-taken).clamp_min(0).mean(0).tolist(),
                        'number_increases':(taken>1e-9).sum(0).tolist(),
                        'number_decreases':(taken< -1e-9).sum(0).tolist()}
    file=root/(split+'-choices.pt')
    torch.save({'classes':y,'baseline':base,'swapped':swapped,'switched':masks,
                'rival_position_zero_based':d['idx'],'correct_in_first_200':good.sum(1),
                'named_representatives_correct':proposed},file)
    return {'native_top1_errors':int(originally_wrong.sum()),
            'native_errors_with_correct_photo_in_top200':int((originally_wrong&good.any(1)).sum()),
            'errors_with_correct_photo_among_first_named_representatives':{
                k:int((originally_wrong&proposed[:,:k].any(1)).sum()) for k in budgets},
            'max_mean_gain_given_two_precomputed_rankings_pp':delta.clamp_min(0).mean(0).tolist(),
            'possible_mean_loss_given_two_precomputed_rankings_pp':(-delta).clamp_min(0).mean(0).tolist(),
            'observed':compared}


@torch.inference_mode()
def run(full=False,dataset='sketchy'):
    w=Path('/content/sbir')
    if dataset=='tuberlin' and not full:raise ValueError('Only a frozen complete TU transfer is supported')
    qs,gs,encoded,_,q,p,flip,src,tgt,scale=audit_inputs(w,768,all_sketches=full,dataset=dataset)
    words=read_words(w/'assets/open-images-v7-classes.csv')
    kd,_=load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    data={s:prepare(s,qs,gs,encoded,q,p,flip,src,tgt,scale,full) for s in qs}
    tasks={s:make_tasks(d,only_incumbent=full) for s,d in data.items()}
    names=set()
    for subset in tasks.values():
        for _,_,_,u,v in subset.values():
            names.update((min(x,y),max(x,y)) for x,y in zip(u.tolist(),v.tolist(),strict=True))
    pairs=sorted(names)
    print('name pairs',len(pairs),flush=True)
    model=load_frozen_clip(str(w/'assets/ViT-B-32.pt'),torch.device('cuda'))
    features,stat=axes(model,open_clip.get_tokenizer('ViT-B-32'),pairs,words)
    lookup={p:i for i,p in enumerate(pairs)}
    result={'text':stat}
    root=w/'results/foreground-word-role'
    root.mkdir(exist_ok=True,parents=True)
    for s,d in data.items():
        entry={'total_queries':len(d['yl']),'gallery':len(d['gl'])}
        for key,task in tasks[s].items():
            entry[key],decisions=evaluate(d,task,lookup,features)
            if key!='first_different_name':continue
            # Only the nominated rival and incumbent exchange positions. Preserve
            # every other rank and consume the original KD evaluator for every query.
            qid=task[0]
            replay=PrefixReplay(kd,d['gl'],s)
            metrics={}
            native=[]
            swapped=[]
            for start in range(0,len(d['yl']),48):
                sl=slice(start,start+48)
                order=d['order'][sl,:200] if s=='sketchy_2' else d['order'][sl]
                order=order.long()
                native.append(replay.chunk((order,d['yl'][sl])))
                ii=torch.arange(len(order))
                pos=d['idx'][sl]
                order[ii,0],order[ii,pos]=order[ii,pos].clone(),order[ii,0].clone()
                swapped.append(replay.chunk((order,d['yl'][sl])))
            native,swapped=torch.cat(native).double(),torch.cat(swapped).double()
            metrics['native']=(100*native.mean(0)).tolist()
            for name,sw in decisions.items():
                switch=torch.zeros(len(native),dtype=torch.bool)
                switch[qid]=sw
                metrics[name]=(100*torch.where(switch[:,None],swapped,native).mean(0)).tolist()
            entry['full_gallery_metrics']=metrics
            if full:
                entry['attainable_from_nomination']=opportunities(d,native,swapped,qid,decisions,s,root)
        result[s]=entry
        print(s,entry['first_different_name']['mixed_relevance'],entry['full_gallery_metrics'],flush=True)
    (root/(dataset+('-full' if full else '')+'.json')).write_text(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--full',action='store_true')
    parser.add_argument('--dataset',choices=['sketchy','tuberlin'],default='sketchy')
    options=parser.parse_args()
    run(options.full,options.dataset)
