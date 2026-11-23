"""Minimum-displacement reranking from counterbalanced, pairwise foreground readings."""
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import open_clip
import torch

from .branch_view_study import PrefixReplay
from .encoding import load_frozen_clip
from .finite_gallery_response import retrieval
from .foreground_word_role import axes,prepare
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_evidence import load_kd
from .semantic import read_words
from .standardized_context_audit import audit_inputs


def nominate(data):
    head=data['order'][:,:200].long().numpy()
    terms=data['terms'].numpy()
    groups=[]
    for photos in head:
        seen={}
        for photo in photos:
            seen.setdefault(int(terms[photo]),int(photo))
            if len(seen)==8:break
        groups.append((list(seen),list(seen.values())))
    return groups


def decreasing_projection(peaks,weights):
    # Pool adjacent violators for weighted least squares in the prescribed order.
    blocks=[]
    for idx,(peak,weight) in enumerate(zip(peaks,weights,strict=True)):
        blocks.append((idx,idx+1,peak*weight,weight))
        while len(blocks)>1:
            a,b=blocks[-2:]
            if a[2]/a[3]>=b[2]/b[3]:break
            blocks[-2:]=[(a[0],b[1],a[2]+b[2],a[3]+b[3])]
    fit=np.empty_like(peaks)
    for start,end,mass,total in blocks:fit[start:end]=mass/total
    return fit


def pair_readings(data,groups,pairs_to_id,text):
    readings={key:[] for key in ('isolated','role','position','role_residual')}
    pairs=[]
    for idx,(words,_) in enumerate(groups):
        for i,u in enumerate(words):
            for j,v in enumerate(words[i+1:],i+1):
                pairs.append((idx,i,j,pairs_to_id[min(u,v),max(u,v)],1 if u<v else -1))
        for rows in readings.values():rows.append(np.zeros((len(words),len(words))))
    for start in range(0,len(pairs),2048):
        batch=pairs[start:start+2048]
        q,i,j,w,orient=np.array(batch).T
        visual=data['query'][q].cuda()
        for key,rows in readings.items():
            margin=((visual*text[key][w].cuda()).sum(1).cpu().numpy()*orient)
            for qid,a,b,value in zip(q,i,j,np.sign(margin),strict=True):
                rows[qid][a,b]=value
                rows[qid][b,a]=-value
    return readings


def updated(data,groups,readings,capacity,spread):
    native=data['log_scores'].numpy()
    term=data['terms'].numpy()
    members={int(t):np.flatnonzero(term==t) for t in np.unique(term)}
    final=native.astype('float64')
    changes=0
    altered=[]
    for idx,(words,photos) in enumerate(groups):
        words=words[:capacity]
        photos=photos[:capacity]
        m=len(photos)
        votes=readings[idx][:m,:m].sum(1)
        order=np.argsort(-votes,kind='stable')
        if np.all(order==np.arange(m)):
            altered.append(0)
            continue
        changes+=1
        row=final[idx]
        peaks=row[photos].copy()
        support=[members[t] for t in words] if spread else [np.array([p]) for p in photos]
        masses=np.array([np.exp(row[ids]).sum() for ids in support])
        fit=np.empty_like(peaks)
        fit[order]=decreasing_projection(peaks[order],masses[order])
        # Resolve projected ties in favor of the nominated ordinal relation.
        for high,low in zip(order[:-1],order[1:],strict=True):
            if fit[low]>=fit[high]:
                fit[low]=np.nextafter(fit[high],-np.inf)
        delta=fit-peaks
        common=np.log(masses.sum()/np.dot(masses,np.exp(delta)))
        for j,ids in enumerate(support):row[ids]+=delta[j]+common
        row[photos]=fit+common
        # Verify mass conservation; check the projected order up to machine precision.
        if not np.isclose(sum(np.exp(row[a]).sum() for a in support),masses.sum(),rtol=2e-12):
            raise ValueError('Ordinal projection changed the selected mass')
        if (np.diff(row[np.array(photos)[order]])>1e-12).any():raise ValueError('Projection order violated')
        altered.append(sum(len(ids) for ids in support))
    return torch.from_numpy(final),{'changed_hypothesis_order':changes,'mean_selected_photos_for_changed_orders':sum(altered)/max(1,changes)}


@torch.inference_mode()
def run():
    w=Path('/content/sbir')
    qs,gs,encoded,_,q,p,fl,src,tgt,scale=audit_inputs(w,768)
    data={s:prepare(s,qs,gs,encoded,q,p,fl,src,tgt,scale,False,retain_scores=True) for s in qs}
    groups={s:nominate(d) for s,d in data.items()}
    pairs=sorted({(min(a,b),max(a,b)) for batch in groups.values() for words,_ in batch
                  for j,a in enumerate(words) for b in words[j+1:]})
    model=load_frozen_clip(str(w/'assets/ViT-B-32.pt'),torch.device('cuda'))
    text,stats=axes(model,open_clip.get_tokenizer('ViT-B-32'),pairs,read_words(w/'assets/open-images-v7-classes.csv'))
    print('ordinal pairs',stats,flush=True)
    kd,_=load_kd(w)
    assert hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()==KD_FUNCTION_SHA
    result={'text':stats}
    for s,d in data.items():
        y=d['yl']
        reads=pair_readings(d,groups[s],{pair:i for i,pair in enumerate(pairs)},text)
        replay=PrefixReplay(kd,d['gl'],s)
        tests=[('baseline',None,0,False)]
        tests.extend((f'{key}/{cap}/groups',key,cap,True) for cap in (2,8) for key in reads)
        tests.extend((f'role/{cap}/leaders','role',cap,False) for cap in (2,8))
        raw={}
        for name,key,cap,spread in tests:
            if key is None:
                score=d['log_scores']
                audit={}
            else:
                score,audit=updated(d,groups[s],reads[key],cap,spread)
            values=[]
            heads=[]
            for start in range(0,len(y),64):
                sl=slice(start,start+64)
                subset=score[sl].cuda()
                values.append(retrieval(subset,y[sl],d['gl'],s,replay))
                heads.append(d['gl'][subset.argmax(1).cpu()]==y[sl])
            v=torch.cat(values)
            hits=torch.cat(heads)
            raw[name]=v
            if key is None:old=hits
            audit['metrics']=(100*v.mean(0)).tolist()
            audit['fixed_damaged_top1']=[int((hits&~old).sum()),int((~hits&old).sum())]
            result.setdefault(s,{'queries':len(y),'gallery':len(d['gl']),'methods':{}})['methods'][name]=audit
        for cap in (2,8):
            v=raw[f'role/{cap}/groups']
            for other in ('baseline',f'isolated/{cap}/groups',f'position/{cap}/groups',f'role/{cap}/leaders'):
                dist=(v-raw[other])*100
                macro=torch.stack([dist[y==c].mean(0) for c in y.unique()])
                boot=torch.randint(len(macro),(10000,len(macro)),generator=torch.Generator().manual_seed(2131))
                result[s].setdefault('paired',{})[f'role/{cap}/groups minus {other}']={
                    'class_mean':macro.mean(0).tolist(),'ci95':macro[boot].mean(1).quantile(torch.tensor([.025,.975],dtype=torch.double),dim=0).tolist()}
        print(s,result[s]['methods']['role/8/groups'],flush=True)
    out=w/'results/ordinal-foreground-ranking'
    out.mkdir(exist_ok=True,parents=True)
    (out/'sketchy.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':
    run()
