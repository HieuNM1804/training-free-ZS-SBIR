"""Partition correct semantic decisions into photo-pair preference and source-dependent excess."""
from pathlib import Path

import numpy as np
import torch
from .geometry_word_flat import atomic_json,revision


def measure(chosen,sign,own,pairs):
    records=[]
    single_choice=two_choices=0
    for k in np.unique(pairs):
        ran=pairs==k
        options=np.unique(own[ran])
        pred=np.where(sign[ran]>0,own[ran],np.where(own[ran]==options[0],options[1],options[0]))
        keep=chosen[ran]
        observed=int((pred[keep]==own[ran][keep]).sum())
        expected=sum(int((pred[keep]==p).sum())*(own[ran]==p).mean() for p in options)
        distinct=np.unique(pred[keep])
        if len(distinct)==1:
            single_choice+=1
        if len(distinct)==2:
            two_choices+=1
        records.append([int(keep.sum()),observed,expected])
    summaries=np.array(records)
    n,obs,null=summaries.sum(0)
    if n==0:
        return {'accepted':0}
    rng=np.random.default_rng(42)
    picks=rng.integers(0,len(records),(5000,len(records)))
    counts,correct,chance=summaries[picks].sum(1).T
    delta=(correct-chance)/np.maximum(counts,1)
    return dict(accepted=int(n),accuracy=float(obs/n),conditional_null=float(null/n),excess=float((obs-null)/n),
        excess_pair_bootstrap95=np.quantile(delta,[.025,.975]).tolist(),one_preferred_photo=single_choice,
        both_photos_predicted=two_choices,total_pairs=len(records))


def run():
    out=Path('/content/sbir/results/semantic-identification')
    result={}
    for split in ('sketchy_1','sketchy_2'):
        rec=torch.load(out/(split+'.pt'),weights_only=True)
        for mode,k in enumerate(('same','different')):
            select=rec['trials'][:,3]==mode
            val=rec['values'][select].numpy()
            tr=rec['trials'][select].numpy()
            sign=np.where(val[:,3]>1e-11,1,np.where(val[:,4]< -1e-11,-1,0))
            matched=np.zeros(len(tr),bool)
            matched[np.argsort(-val[:,2],kind='stable')[:np.count_nonzero(sign)]]=True
            native=np.sign(val[:,:2].mean(1))
            result[split+'/'+k]={name:measure(t,s,tr[:,1],tr[:,4]) for name,t,s in
                [('identified',sign!=0,sign),('confidence_matched',matched,native),
                 ('native',np.ones(len(tr),bool),native)]}
    atomic_json(out/'pair-preference.json',dict(commit=revision(),assessment=result))
    print({k:v['identified']['excess'] for k,v in result.items()})


if __name__=='__main__':
    run()
