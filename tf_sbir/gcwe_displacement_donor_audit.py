"""Follow a changed verifier from donor selection to word evidence using labels only after weighting."""
from pathlib import Path

from .geometry_word_flat import (atomic_json, audit_inputs, class_interval, distance_agreement,
    geometry_cache, load_frozen_clip, make_word_bank, rank_density, reweight, revision, save_tensor)
from .gcwe_displacement_consensus import concentration_controls, offset_geometry
from .benchmark import merge_inputs
from .crossdomain_inputs import runtime
from .data import SKETCHY_SPLITS
import torch
from torch.nn import functional as F


def donor_measurements(tokens, patches, scale, dist, geometry):
    logits = scale*(tokens[:,None] @ patches.transpose(-1,-2)).float()
    coupling = (logits.softmax(-1)+logits.softmax(-2))/98
    options = concentration_controls(coupling,geometry)
    options['original'] = distance_agreement(coupling,dist,dist)
    return {k:options[k] for k in ('original','same_partition','wrong_coordinates','individual_matches','uncentered_energy')}


def target_word_contrasts(bank, classes, mask):
    # Applied only to already emitted word distributions, as a label-based diagnostic.
    subset = bank[mask]
    local_classes = classes[mask]
    distinct = classes.unique(sorted=True)
    total = subset.sum(0)
    contrasts = []
    for label in distinct:
        hit = local_classes==label
        if not hit.any() or hit.all():
            raise ValueError('Missing positive or negative held-out gallery group')
        positive = subset[hit].mean(0)
        negative = (total - positive*hit.sum()) / (~hit).sum()
        contrasts.append(positive-negative)
    return distinct,torch.stack(contrasts)


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    w = Path('/content/sbir')
    output = w/'results/gcwe-displacement-donor-audit'
    output.mkdir(exist_ok=True,parents=True)
    cfg = runtime(w,'sketchy',batch=256,workers=2)
    qp,_ = merge_inputs(cfg.root,SKETCHY_SPLITS,'sketch')
    pp,_ = merge_inputs(cfg.root,SKETCHY_SPLITS,'photo')
    qs,gs,ids,_,q,p,flipped,src,tgt,scale = audit_inputs(w,all_sketches=True)
    assert torch.equal(ids,torch.arange(len(qp)))
    vision = load_frozen_clip(cfg.clip_checkpoint,torch.device('cuda')).visual
    coord = torch.cartesian_prod(torch.arange(7),torch.arange(7)).float().cuda()
    dist = torch.cdist(coord,coord)
    geometry = offset_geometry(coord.device)
    for split,(qi,labels) in qs.items():
        pi,gl = gs[split]
        ix = torch.randperm(len(qi),generator=torch.Generator().manual_seed(42))[9984:11520].sort().values
        matched = torch.load(w/'results/gcwe-displacement-concentration'/(split+'.pt'),weights_only=True,mmap=True)
        assert torch.equal(qi[ix],matched['qi']) and torch.equal(pi,matched['pi']) and torch.equal(labels[ix],matched['labels'])
        gallery = F.normalize(p[pi].cuda())
        bank = make_word_bank(gallery,tgt,scale,160)
        near,compact,gp,sp,blank = geometry_cache(w,split,gallery,q,flipped,qi,pi,qp,pp,vision)
        parts = torch.randperm(len(pi),generator=torch.Generator().manual_seed(42)).cuda().tensor_split(2)
        fold = torch.zeros(len(pi),device='cuda',dtype=torch.bool)
        fold[parts[1]]=True
        priors=[bank[part].mean(0) for part in parts]
        classes=gl.cuda()
        targets=[target_word_contrasts(bank,classes,fold==bool(t)) for t in range(2)]
        records={}
        for batch in ix.split(8):
            nominees=near[batch]
            patches=gp[compact[batch].cuda()]
            real=donor_measurements(sp[batch].cuda(),patches,scale,dist,geometry)
            null=donor_measurements(blank[batch].cuda(),patches,scale,dist,geometry)
            words=((scale*(q[qi[batch]].cuda()@src.T)).softmax(-1)
                    +(scale*(flipped[qi[batch]].float().cuda()@src.T)).softmax(-1))/2
            evidence=bank[nominees]
            truth=labels[batch].cuda()
            hit=classes[nominees]==truth[:,None]
            batch_results={}
            for target in range(2):
                eligible=fold[nominees]!=bool(target)
                n=eligible.sum(1)
                den=eligible.float()[:,None].bmm(evidence).squeeze(1)+priors[1-target]
                ids,margin=targets[target]
                query_margin=margin[torch.searchsorted(ids,truth)]
                for name in ('uniform',*real):
                    mass=eligible.float() if name=='uniform' else rank_density(real[name]-null[name],eligible,8)
                    observed=mass[:,None].bmm(evidence).squeeze(1)+priors[1-target]
                    updated=reweight(words,observed/den.clamp_min(1e-30))
                    purity=(mass*hit).sum(1)/n
                    utility=((updated-words)*query_margin).sum(1)
                    values=torch.stack([purity,utility],1) * (len(parts[target])/len(pi))
                    batch_results[name]=values if target==0 else batch_results[name]+values
            for k,v in batch_results.items():
                records.setdefault(k,[]).append(v.cpu())
        rows={k:torch.cat(v).double() for k,v in records.items()}
        assert rows['uniform'][:,1].abs().max()<2e-7
        ap_gain=100*(matched['rows']['same_partition'][:,0]-matched['rows']['original'][:,0]).double()
        donor_gain=100*(rows['same_partition'][:,0]-rows['original'][:,0])
        margin_gain=10000*(rows['same_partition'][:,1]-rows['original'][:,1])
        grouped={}
        for name,v in [('donor_purity_points',donor_gain),('heldout_margin_x10000',margin_gain)]:
            grouped[name]=class_interval(v,labels[ix],seed=42)
            # Center within retrieval class before computing descriptive correlations.
            x,y=v.clone(),ap_gain.clone()
            for c in labels[ix].unique():
                mask=labels[ix]==c
                x[mask]-=x[mask].mean()
                y[mask]-=y[mask].mean()
            grouped[name]['within_class_correlation_with_ap_change']=float(F.cosine_similarity(x,y,dim=0))
        save_tensor(dict(qi=qi[ix],pi=pi,labels=labels[ix],rows=rows,ap_gain=ap_gain),output/(split+'.pt'))
        atomic_json(output/(split+'.json'),dict(commit=revision(),seed=42,device=torch.cuda.get_device_name(),queries=len(ix),
            columns=['probability_selected_donor_has_query_class','change_in_heldout_positive_minus_negative_word_overlap'],
            means={k:v.mean(0).tolist() for k,v in rows.items()},primary_minus_original=grouped))
        print(split,grouped,flush=True)
        del bank,gp


if __name__=='__main__':
    run()
