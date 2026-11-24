"""Decompose each paired diffusion effect by whether it retains a relevant top-200 result."""
import argparse
import io
import json
import zipfile
from pathlib import Path

import torch


def report(raw):
    rows=raw['rows']
    comparisons={
        'global_given_regional':('full','regional_diffusion_only'),
        'regional_given_global':('full','global_diffusion_only'),
        'global_alone':('global_diffusion_only','without_diffusion'),
        'regional_alone':('regional_diffusion_only','without_diffusion'),
    }
    out={}
    for name,(on,off) in comparisons.items():
        before,after=rows[off].double(),rows[on].double()
        delta=(after[:,:2]-before[:,:2])*100
        prev,now=before[:,2].bool(),after[:,2].bool()
        groups={'retained':prev&now,'lost':prev&~now,'rescued':~prev&now,'missed_twice':~prev&~now}
        assert torch.stack(list(groups.values())).sum(0).eq(1).all()
        data={}
        for group,valid in groups.items():
            data[group]={'n':int(valid.sum()),'contribution_points':(delta[valid].sum(0)/len(delta)).tolist(),
                         'within_group_change_points':delta[valid].mean(0).tolist() if valid.any() else None}
        check=torch.tensor([g['contribution_points'] for g in data.values()],dtype=torch.double).sum(0)
        torch.testing.assert_close(check,delta.mean(0))
        out[name]={'all_change_points':delta.mean(0).tolist(),'groups':data}
    return out


def run(bundle,destination):
    summary={}
    with zipfile.ZipFile(bundle) as z:
        for s in ('sketchy_1','sketchy_2','tuberlin'):
            raw=torch.load(io.BytesIO(z.read(s+'.pt')),weights_only=True)
            summary[s]=report(raw)
    destination.write_text(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('bundle',type=Path)
    parser.add_argument('destination',type=Path)
    args=parser.parse_args()
    run(args.bundle,args.destination)
