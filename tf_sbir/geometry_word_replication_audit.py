"""Paired replication on queries outside the union of both prior pilot samples."""
import json
from pathlib import Path

import torch

from .benchmark import merge_inputs
from .crossdomain_inputs import runtime
from .data import SKETCHY_SPLITS
from .depiction_path_diagnostic import class_interval
from .locked_geometry_words import LOCK


def mask_stats(record, mask):
    raw = record['per_query']
    labels = record['labels'][mask]
    primary = raw['excess'][mask,:2].double()
    view = {}
    for name, full in raw.items():
        val = full[mask,:2].double()
        diff = 100*(primary-val)
        view[name] = {'metrics':(100*val.mean(0)).tolist(),
                      'primary_minus_control_points':diff.mean(0).tolist(),
                      'paired_macro_map_interval':class_interval(diff[:,0],labels)}
    return {'queries':len(labels),'rows':view}


def main():
    w = Path('/content/sbir')
    root = w/'results/geometry-conditioned-word-evidence'
    cfg = runtime(w,'sketchy')
    _,groups = merge_inputs(cfg.root,SKETCHY_SPLITS,'sketch')
    test_ids = {}
    for s,(ids,_) in groups.items():
        sel = torch.randperm(len(ids),generator=torch.Generator().manual_seed(87243))[:768].sort().values
        test_ids[s] = ids[sel]
    dev = torch.cat(list(test_ids.values())).unique()
    out = {}
    for s in (*SKETCHY_SPLITS,'tuberlin'):
        result = torch.load(root/(s+'-official.pt'),weights_only=True)
        assert result['lock']==LOCK
        ids = result['query_indices']
        if s=='tuberlin':
            out[s] = {'full':mask_stats(result,torch.ones(len(ids),dtype=torch.bool))}
            continue
        assert torch.equal(groups[s][0],ids)
        pilot = torch.isin(ids,test_ids[s])
        novel = ~torch.isin(ids,dev)
        observed = torch.load(root/(s+'.pt'),weights_only=True)
        delta = result['per_query']['excess'][pilot,:2].double()-observed['metrics']['excess/max8'][:,:2]
        out[s] = {'pilot_ids':int(pilot.sum()),'any_split_pilot_ids':int((~novel).sum()),
                  'pilot_replay_mean_difference_points':(100*delta.mean(0)).tolist(),
                  'pilot_replay_mean_absolute_difference_points':(100*delta.abs().mean(0)).tolist(),
                  'outside_both_pilots':mask_stats(result,novel),'all':mask_stats(result,torch.ones_like(novel))}
    (root/'replication-audit.json').write_text(json.dumps(out,indent=2))
    print(json.dumps({k:{section:value['queries'] for section,value in v.items() if isinstance(value,dict) and 'queries' in value} for k,v in out.items()}),flush=True)


if __name__=='__main__':
    main()
