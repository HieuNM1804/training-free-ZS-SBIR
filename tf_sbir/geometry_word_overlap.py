"""Compare official outcomes for the exact same unseen Sketchy sketches in two galleries."""
import json
from pathlib import Path

import torch

from .benchmark import merge_inputs
from .crossdomain_inputs import runtime
from .data import SKETCHY_SPLITS,UNSEEN_CLASSES
from .gallery_context_reversals import coordinates


def main():
    w=Path('/content/sbir')
    folder=w/'results/geometry-conditioned-word-evidence'
    cfg=runtime(w,'sketchy')
    _,queries=merge_inputs(cfg.root,SKETCHY_SPLITS,'sketch')
    _,photos=merge_inputs(cfg.root,SKETCHY_SPLITS,'photo')
    s1,s2=SKETCHY_SPLITS
    qshared=torch.tensor(sorted(set(queries[s1][0].tolist()) & set(queries[s2][0].tolist())))
    pshared=torch.tensor(sorted(set(photos[s1][0].tolist()) & set(photos[s2][0].tolist())))
    assert (len(qshared),len(pshared))==(2231,2205)
    sub={}
    base={}
    method={}
    by={}
    names_per={}
    metrics={}
    for s in SKETCHY_SPLITS:
        pack=torch.load(folder/(s+'-official.pt'),weights_only=True)
        q,y=queries[s]
        assert torch.equal(pack['query_indices'],q) and torch.equal(pack['labels'],y)
        ids=coordinates(q,qshared)
        sub[s]=pack['per_query']['excess'][ids].double()-pack['per_query']['base'][ids].double()
        base[s]=pack['per_query']['base'][ids]
        method[s]=pack['per_query']['excess'][ids]
        names_per[s]=[UNSEEN_CLASSES[s][c] for c in y[ids].tolist()]
        def fixed_lost(values,col):
            v=values[:,col]
            return [int((v>0).sum()),int((v<0).sum())]
        metrics[s]={'all_shared_queries':len(ids),
            'base_misses_top1':int((base[s][:,3]==0).sum()),'base_no_correct_photo_top200':int((base[s][:,2]==0).sum()),
            'top1_fixed_lost':fixed_lost(sub[s],3),'top200_fixed_lost':fixed_lost(sub[s],2),
            'native_protocol_mAP_change_on_shared':float(sub[s][:,0].mean()*100)}
        by[s]={}
        for c in sorted(set(names_per[s])):
            mask=torch.tensor([v==c for v in names_per[s]])
            c_id=UNSEEN_CLASSES[s].index(c)
            pi,gl=photos[s]
            correct_ids=pi[gl==c_id]
            by[s][c]={'queries':int(mask.sum()),'gallery_relevant':len(correct_ids),
                'relevant_gallery_shared':int(torch.isin(correct_ids,pshared).sum()),
                'base_no_correct_photo_top200':int((base[s][mask,2]==0).sum()),
                'top1_fixed_lost':fixed_lost(sub[s][mask],3),'top200_fixed_lost':fixed_lost(sub[s][mask],2)}
    assert names_per[s1]==names_per[s2]
    shared_changes={}
    for col,name in [(3,'top1'),(2,'top200')]:
        a,b=sub[s1][:,col],sub[s2][:,col]
        shared_changes[name]={'baseline_hit_disagrees_between_galleries':int((base[s1][:,col]!=base[s2][:,col]).sum()),
            'updated_hit_disagrees_between_galleries':int((method[s1][:,col]!=method[s2][:,col]).sum()),
            'intervention_fixed_in_both':int(((a>0)&(b>0)).sum()),'intervention_lost_in_both':int(((a<0)&(b<0)).sum()),
            'opposite_signs_in_two_galleries':int((a*b<0).sum()),
            'one_gallery_changed_other_did_not':int(((a!=0)^(b!=0)).sum())}
    result={'overlap':{'queries':len(qshared),'photos':len(pshared),'classes':sorted(set(names_per[s1]))},
            'official_rows':metrics,'matched_query_outcomes':shared_changes,'by_class':by}
    (folder/'overlap-diagnostic.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(shared_changes),flush=True)


if __name__=='__main__':
    main()
