"""Post-score audit of the actually reachable changes and source-response enrichment."""
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .depiction_path_diagnostic import class_interval
from .natural_depiction_donors import two_domain_readings
from .standardized_context_audit import audit_inputs
from .strong_pilot_scores import location


@torch.inference_mode()
def tu_fractions(w):
    qs,gs,enc,_,q,p,flip,src,tgt,scale = audit_inputs(w,all_sketches=True,dataset='tuberlin')
    qi,y = qs['tuberlin']
    pi,classes = gs['tuberlin']
    assert torch.equal(enc,qi) and len(classes.unique())==30
    x = F.normalize((q[qi].cuda()+flip[qi].float().cuda())/2)
    g = F.normalize(p[pi].cuda())
    style = torch.cat([two_domain_readings(z,src,tgt,scale)[0] for z in g.split(192)])
    upper = style>style.median()
    hi = (x@g.T).topk(200).indices
    truth = y.cuda()
    cl = classes.cuda()
    good = cl[hi]==truth[:,None]
    base = torch.load(location(w,'tuberlin',torch.arange(len(y))),weights_only=True,mmap=True)
    assert torch.equal(base['pi'],pi) and torch.equal(base['qi'],qi)
    strong = base['score'].cuda().topk(200).indices
    missed = ~(cl[strong]==truth[:,None]).any(1)
    with_recovery = missed & good.any(1)
    observed = ((good & upper[hi]).sum(1)/good.sum(1).clamp_min(1))[with_recovery]
    frequency = torch.zeros(len(cl.unique()),device=g.device)
    for c in cl.unique():
        frequency[c] = upper[cl==c].float().mean()
    matched = frequency[truth[with_recovery]]
    return {'strong_misses':int(missed.sum()),'recovered_by_visual':int(with_recovery.sum()),
            'recovered_by_upper_visual':int((missed & (good & upper[hi]).any(1)).sum()),
            'mean_fraction_among_relevant_visual_photos':float(observed.mean()),
            'matched_gallery_fraction':float(matched.mean()),
            'class_macro_fraction_difference':class_interval(observed-matched,truth[with_recovery])}


def main():
    w = Path('/content/sbir')
    root = w/'results/depiction-pivot-geometry'
    result = {'tu_visual_recovery':tu_fractions(w),'paired':{}}
    for split in ('sketchy_1','sketchy_2','tuberlin'):
        raw = torch.load(root/(split+'.pt'),weights_only=True)
        m = raw['per_query_kd']
        d = (m['spatial/0.03125'][:,0]-m['base'][:,0]).double()
        if 'flat_canvas' not in raw['chosen']:
            raise ValueError('Expected the locked rerun with the blank-canvas control')
        subset = raw['hard']
        reference = raw['donors']['strong'][:,0]>0
        geometry = raw['donors']['spatial'][:,0]>0
        words = raw['donors']['words'][:,0]>0
        blank = raw['donors']['flat_canvas'][:,0]>0
        def paired(a,b,mask):
            return [int((a & ~b & mask).sum()),int((~a & b & mask).sum())]
        all_rows = torch.ones_like(subset)
        result['paired'][split] = {'queries':len(d),'hard':int(subset.sum()),
            'full_gallery_mean_delta_ap_points':float(100*d.mean()),
            'max_gain_if_oracle_turns_spatial_correction_on_or_off':float(100*d.clamp_min(0).mean()),
            'class_macro_full_gallery_delta_ap_points':class_interval(100*d,raw['labels']),
            'exclusive_correct_donors_geometry_then_control':{
                name:{'all':paired(geometry,alt,all_rows),'hard':paired(geometry,alt,subset)}
                for name,alt in [('strong',reference),('direct_words',words),('content_free',blank)]}}
    (root/'failure-localization.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
