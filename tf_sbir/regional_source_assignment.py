"""Measure source-specific information after eliminating a photo pair's fixed scoring bias."""
from pathlib import Path

import numpy as np
import torch
from .geometry_word_flat import atomic_json,revision


def half_correct(margin):
    return (margin>0).astype(float)+.5*(margin==0)


def summarize_pairs(rows,names):
    matrix=np.asarray(rows)
    result={}
    for idx,name in enumerate(names):
        keep=np.isfinite(matrix[:,idx])
        scores=matrix[keep,idx]
        if not len(scores):
            result[name]={'pairs':0,'mean':None,'pair_bootstrap95':None}
            continue
        draws=np.random.default_rng(42).integers(0,len(scores),(5000,len(scores)))
        result[name]={'pairs':len(scores),'mean':float(scores.mean()),
                      'pair_bootstrap95':np.quantile(scores[draws].mean(1),[.025,.975]).tolist()}
    return result


def run():
    w=Path('/content/sbir/results')
    output=w/'regional-predictive-observation'
    report={}
    for split in ('sketchy_1','sketchy_2'):
        draw=torch.load(w/'semantic-identification'/(split+'.pt'),weights_only=True)
        reg=torch.load(output/(split+'.pt'),weights_only=True)
        assert all(torch.equal(draw[k],reg[k]) for k in ('trials','qi','pi'))
        trials=draw['trials'].numpy()
        global_score=draw['values'][:,:2].numpy().mean(1)
        regional_scores=(reg['scores'][:,:,0]-reg['scores'][:,:,1]).numpy()
        named=['global','conditional','marginal','shift','increment','shift_increment']
        all_scores=np.column_stack((global_score,regional_scores,
            regional_scores[:,0]-regional_scores[:,1],regional_scores[:,2]-regional_scores[:,1]))
        for mode,context in enumerate(('same','different')):
            selections={k:[] for k in ('all','global_wrong','global_correct')}
            comparison_counts={k:0 for k in selections}
            class_record=[]
            for pid in np.unique(trials[trials[:,3]==mode,4]):
                mask=(trials[:,3]==mode)&(trials[:,4]==pid)
                tr=trials[mask]
                sides=np.unique(tr[:,1])
                assert len(sides)==2
                a,b=sides
                assert np.all(tr[tr[:,1]==a,2]==b) and np.all(tr[tr[:,1]==b,2]==a)
                # Stored margins are own-minus-foil. The second source therefore already has the opposite sign.
                s=all_scores[mask]
                advantage=(s[tr[:,1]==a,None,:]+s[tr[:,1]==b][None,:,:]).reshape(-1,len(named))
                accuracy=half_correct(advantage)
                extras=np.stack([accuracy[:,4]-accuracy[:,5],accuracy[:,1]-accuracy[:,2]],axis=1)
                outcomes=np.column_stack((accuracy,extras))
                for name,keep in [('all',np.ones(len(advantage),dtype=bool)),('global_wrong',advantage[:,0]<0),
                                  ('global_correct',advantage[:,0]>0)]:
                    selections[name].append(outcomes[keep].mean(0) if keep.any() else np.full(outcomes.shape[1],np.nan))
                    comparison_counts[name]+=int(keep.sum())
                class_record.append([int(pid),int(draw['labels'][tr[0,0]]),len(advantage),
                                     *selections['all'][-1].tolist(),*[float(v) if np.isfinite(v) else None
                                     for v in selections['global_wrong'][-1]]])
            names=named+['increment_minus_shift','conditional_minus_marginal']
            report[split+'/'+context]={'summaries':{k:summarize_pairs(v,names) for k,v in selections.items()},
                'drawing_comparisons':comparison_counts,'pair_columns':names,'by_pair':class_record}
    atomic_json(output/'source-assignment.json',dict(commit=revision(),results=report))
    print({k:v['summaries']['global_wrong']['increment']['mean'] for k,v in report.items()})


if __name__=='__main__':
    run()
