"""Locate source discrimination before and after transporting the geometric action."""
import json
from pathlib import Path

import numpy as np
import torch
from .concept_benchmark import atomic_json
from .crossdomain_benchmark import revision
from .regional_predictive_observation import cluster_effect


def hits(margin):
    return (margin>1e-7).astype(float)+.5*(np.abs(margin)<=1e-7)


def run():
    folder=Path('/content/sbir/results/gcwe-transported-intervention')
    summary={}
    for split in ('sketchy_1','sketchy_2'):
        record=torch.load(folder/(split+'.pt'),weights_only=True)
        arms=json.loads((folder/(split+'.json')).read_text())['arms']
        meta=record['meta'].numpy()
        summary[split]={}
        for frame,row in record['rows'].items():
            geometry=row['geometry'].numpy()
            risk=row['scores'][:,:,:,0].numpy()
            for m,group in enumerate(('same','different')):
                selected=meta[:,4]==m
                idx=meta[selected,5]
                ids=np.unique(idx)
                assert all((idx==pid).sum()==2 for pid in ids)
                margins={}
                for j,k in enumerate(('known_frame_geometry','peak_geometry','isometry_mass')):
                    x=geometry[selected,j]
                    margins[k]=x[:,0]-x[:,1]
                for j,k in enumerate(arms):
                    x=risk[selected,j]
                    margins[k]=x[:,1]-x[:,0]
                results={}
                for k,delta in margins.items():
                    pair=np.asarray([delta[idx==pid].mean() for pid in ids])
                    results[k]=dict(single=cluster_effect(hits(delta),idx),
                        paired=cluster_effect(hits(pair),ids))
                first=hits(margins['peak_geometry'])
                after=hits(margins['square'])
                joint=first+after
                results['square_minus_peak_single_accuracy']=cluster_effect(after-first,idx)
                results['both_correct']=int((joint==2).sum())
                results['geometry_only_correct']=int(((first==1)&(after==0)).sum())
                results['semantic_only_correct']=int(((first==0)&(after==1)).sum())
                results['both_wrong']=int((joint==0).sum())
                summary[split][frame+'/'+group]=results
    atomic_json(folder/'stage-localization.json',dict(commit=revision(),seed=42,
        interpretation='post-hoc matched-source mechanism analysis, larger rank confidence alone is a diagnostic',
        assessment=summary))
    for s in summary:print(s,summary[s]['native/same']['peak_geometry']['single'],flush=True)


if __name__=='__main__':
    run()
