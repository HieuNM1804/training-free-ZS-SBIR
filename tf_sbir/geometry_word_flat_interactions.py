"""Read the finished seed-42 runs and summarize paired module effects on ranking failures."""
import json
from pathlib import Path

import torch

from .concept_benchmark import atomic_json
from .depiction_path_diagnostic import class_interval


def compare(a,b,labels):
    d=a.double()-b.double()
    return {'change_points':(100*d[:,:2].mean(0)).tolist(),
            'class_ap':class_interval(100*d[:,0],labels,seed=42),
            'top1_gained_lost':[int((d[:,3]>0).sum()),int((d[:,3]<0).sum())],
            'top200_gained_lost':[int((d[:,2]>0).sum()),int((d[:,2]<0).sum())]}


def run():
    root=Path('/content/sbir/results')
    output=root/'geometry-word-flat-terms'
    report={}
    for split in ('sketchy_1','sketchy_2','tuberlin'):
        terms=torch.load(output/(split+'.pt'),mmap=True,weights_only=True)
        first=torch.load(root/'geometry-word-flat-first'/(split+'.pt'),mmap=True,weights_only=True)
        assert all(torch.equal(terms[k],first[k]) for k in ('labels','pi','qi'))
        rows,labels=terms['rows'],terms['labels']
        contrasts={('full',k) for k in ('without_global','without_regional','without_geometry')}
        # Each pair toggles G or R at fixed semantic base, permitting a paired interaction in utility.
        contrasts.update({('gcwe_word_retrieval','semantic_only'),('without_geometry','semantic_only'),
                          ('without_regional','semantic_only'),('full','visual_cosine')})
        summary={a+' minus '+b:compare(rows[a],rows[b],labels) for a,b in sorted(contrasts)}
        g_with_r=rows['full'].double()-rows['without_geometry'].double()
        g_without_r=rows['without_regional'].double()-rows['semantic_only'].double()
        d=g_with_r-g_without_r
        summary['G_by_R_interaction_on_S']={'change_points':(100*d[:,:2].mean(0)).tolist(),
                                            'class_ap':class_interval(100*d[:,0],labels,seed=42)}
        summary['diffusion_inside_full']=compare(first['rows']['full'],first['rows']['without_diffusion'],labels)
        report[split]=summary
    atomic_json(output/'interactions.json',report)
    print(json.dumps({s:r['G_by_R_interaction_on_S']['change_points'] for s,r in report.items()}))


if __name__=='__main__':
    run()
