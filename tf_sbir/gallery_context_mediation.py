"""Attribute native rank reversals to changed weights on shared or other endpoints."""
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .gallery_context_reversals import coordinates
from .gallery_panel_resume import region_views, semantic_inputs
from .gallery_region_panels import CONFIG, Index


def native_scores(workspace):
    cfg,paths,queries,_,galleries,q,p,src,tgt,scale=semantic_inputs(workspace,'sketchy')
    names=('sketchy_1','sketchy_2')
    def common(groups):
        return torch.tensor(sorted(set(groups[names[0]][0].tolist())&set(groups[names[1]][0].tolist())))
    shared_q,shared_p=common(queries),common(galleries)
    outputs={}
    checks={}
    for name in names:
        qi,_=queries[name]
        pi,_=galleries[name]
        rows,columns=coordinates(qi,shared_q),coordinates(pi,shared_p)
        g=F.normalize(p[pi].cuda())
        graph=Index(SimpleNamespace(p=g),rgb_only=True)
        bank=torch.cat([(scale*(v@tgt.T)).softmax(1) for v in g.split(256)])
        views,_=region_views(workspace,name,cfg,paths,qi,q)
        indicator=g.new_zeros(1,len(g))
        indicator[:,columns]=1
        context_mass=graph.score(CONFIG,indicator)[:,columns].cpu().double()
        parts={k:[] for k in ('native','shared','private','direct')}
        maximum_residual=0.
        for ix in rows.split(48):
            original=(scale*(q[qi[ix]].cuda()@src.T)).softmax(1)
            flipped=(scale*(views['flipped'][ix].float().cuda()@src.T)).softmax(1)
            evidence=((original+flipped)/2)@bank.T
            # Linear decomposition of a completed native score. Every component
            # uses the exact same full native graph, including all intermediate nodes.
            entire=graph.score(CONFIG,evidence)[:,columns]
            shared=graph.score(CONFIG,evidence*indicator)[:,columns]
            private=graph.score(CONFIG,evidence*(1-indicator))[:,columns]
            maximum_residual=max(maximum_residual,float((entire-shared-private).abs().max()))
            for k,v in [('native',entire),('shared',shared),('private',private),('direct',evidence[:,columns])]:
                parts[k].append(v.cpu().double())
        outputs[name]={k:torch.cat(v) for k,v in parts.items()}
        outputs[name]['mass']=context_mass
        checks[name]={'max_abs_linear_residual':maximum_residual,
                      'mean_native_mass_on_shared_endpoints':float(context_mass.mean())}
        del graph,bank
    qloc,ploc=coordinates(queries[names[0]][0],shared_q),coordinates(galleries[names[0]][0],shared_p)
    labels=queries[names[0]][1][qloc]
    truth=labels[:,None]==galleries[names[0]][1][ploc][None,:]
    return [outputs[n] for n in names],truth,checks


def mechanism(row,native,flips):
    return {'correct_on_fixed_pairs':int((row>0).sum()),
            'changes_from_native':int((row*native<0).sum()),
            'reaches_other_order_on_native_flips':int(((row*native<0)&flips).sum())}


@torch.inference_mode()
def run():
    w=Path('/content/sbir')
    (a,b),truth,checks=native_scores(w)
    top=a['direct'].topk(200).indices
    hits=truth.gather(1,top)
    i,j=(hits[:,:-1]^hits[:,1:]).nonzero(as_tuple=True)
    left,right=top[i,j],top[i,j+1]
    polarity=hits[i,j].double()*2-1
    def margin(values):
        return (values[i,left]-values[i,right])*polarity
    d1,d2=margin(a['native']),margin(b['native'])
    flips=d1*d2<0
    # As mixed-precision graph multiplies are approximate, distinguish swaps
    # whose margin changes are too small to resolve at reconstruction precision.
    rebuild1=margin(a['shared']+a['private'])
    rebuild2=margin(b['shared']+b['private'])
    resolved=(d1.abs()>(rebuild1-d1).abs())&(d2.abs()>(rebuild2-d2).abs())
    result={'checks':checks,'shared_queries':len(truth),'shared_photos':truth.size(1),'boundary_pairs':len(i),
            'native_flips':int(flips.sum()),'resolved_flips':int((flips&resolved).sum())}
    for name,first,second,orig in [('ext1_to_ext2',a,b,d1),('ext2_to_ext1',b,a,d2)]:
        common=first['shared']
        unique=first['private']
        p=first['mass'].clamp_min(1e-10)
        o=second['mass'].clamp_min(1e-10)
        destination=second['shared']
        swaps={'swap_only_shared_endpoints':destination+unique,
               'swap_only_other_endpoints':common+second['private'],
               'swap_only_total_exposure_to_shared':common/p*o+unique,
               'swap_only_mix_within_shared':destination/o*p+unique}
        common_changes=margin(swaps['swap_only_shared_endpoints'])*orig<0
        unique_changes=margin(swaps['swap_only_other_endpoints'])*orig<0
        stable=flips&resolved
        result[name]={'swaps':{k:mechanism(margin(v),orig,flips) for k,v in swaps.items()},
                      'resolved_flips_one_part_suffices':{
                         'only_shared':int((stable&common_changes&~unique_changes).sum()),
                         'only_other':int((stable&~common_changes&unique_changes).sum()),
                         'either':int((stable&common_changes&unique_changes).sum()),
                         'neither':int((stable&~common_changes&~unique_changes).sum())}}
    out=w/'results/gallery-context-reversals'
    out.mkdir(exist_ok=True,parents=True)
    (out/'endpoint-attribution.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    run()
