"""Check paired gains of additive position/scale evidence and report untouched-query estimates."""
import json
from pathlib import Path

import torch

from .concept_benchmark import atomic_json
from .depiction_path_diagnostic import class_interval


def summarize(delta, labels, keep):
    selected=100*delta[keep].double()
    selected_labels=labels[keep]
    result={'queries':len(selected), 'mean_points':selected.mean(0).tolist(),
            'class_ap':class_interval(selected[:,0],selected_labels,seed=42),
            'class_precision':class_interval(selected[:,1],selected_labels,seed=42)}
    return result


def run():
    root=Path('/content/sbir/results')
    output=root/'regional-additive-full'
    inputs={split:torch.load(output/(split+'.pt'),mmap=True,weights_only=True)
            for split in ('sketchy_1','sketchy_2','tuberlin')}
    discovery_ids={}
    for split in ('sketchy_1','sketchy_2'):
        groups=[]
        for study in ('regional-scale-position-contrast','regional-additive-confirmation'):
            pack=torch.load(root/study/(split+'.pt'),mmap=True,weights_only=True)
            assert torch.equal(pack['pi'],inputs[split]['pi'])
            groups.append(pack['qi'])
        ids=torch.cat(groups)
        assert len(ids)==4608 and len(ids.unique())==len(ids)
        discovery_ids[split]=ids
    any_development=torch.cat(list(discovery_ids.values())).unique()
    report={}
    for split,record in inputs.items():
        rows=record['rows'];qi=record['qi'];labels=record['labels']
        complete=rows['full']
        delta=rows['additive'].double()-complete.double()
        no_geometry=rows['additive_without_geometry'].double()-rows['without_geometry'].double()
        subsets={'all':torch.ones(len(qi),dtype=torch.bool)}
        if split in discovery_ids:
            subsets['outside_native_development']=~torch.isin(qi,discovery_ids[split])
            subsets['outside_either_development']=~torch.isin(qi,any_development)
        by_subset={}
        for name,keep in subsets.items():
            native=rows['full'][keep,:2].double().mean(0)
            proposed=rows['additive'][keep,:2].double().mean(0)
            by_subset[name]={'native_metrics':(100*native).tolist(),'additive_metrics':(100*proposed).tolist(),
                 'with_gcwe':summarize(delta[:,:2],labels,keep),
                 'without_gcwe':summarize(no_geometry[:,:2],labels,keep),
                 'interaction':summarize((delta-no_geometry)[:,:2],labels,keep),
                 'top200_gained_lost':[int((delta[keep,2]>0).sum()),int((delta[keep,2]<0).sum())],
                 'top1_gained_lost':[int((delta[keep,3]>0).sum()),int((delta[keep,3]<0).sum())]}
        report[split]=by_subset
    atomic_json(output/'paired-effects.json',report)
    print(json.dumps({s:r['all']['with_gcwe']['mean_points'] for s,r in report.items()}))


if __name__=='__main__':
    run()
