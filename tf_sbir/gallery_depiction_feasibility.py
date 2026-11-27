"""Does natural within-photo variation predict subject-dependent drawing displacements?

Labels identify classes exclusively in this diagnostic. No candidate scores are modified.
"""

import argparse
import json
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F

from .concept_benchmark import atomic_json
from .data import SKETCHY_SPLITS, UNSEEN_CLASSES
from .encoding import load_frozen_clip
from .kaggle_unseen_features import DEFAULT_CACHE, load_visual_split
from .paired_patch_interaction_probe import DST, SEED, read
from .profiled_study import true_sources


PROMPTS = ("a drawing of an object.", "a photo of an object.")
RANDOM_DIRECTIONS = 128


def remove_axis(x, axis):
    return x - (x @ axis)[...,None] * axis


def incremental_domain_check(centroid, photo_axis, domain_axis, y):
    errors = [[],[]]
    for i in range(len(y)):
        keep = torch.arange(len(y),device=y.device)!=i
        ymean = y[keep].mean(0)
        for j,items in enumerate(((centroid,photo_axis),(centroid,photo_axis,domain_axis))):
            means = [t[keep].mean(0) for t in items]
            x = torch.stack([(t[keep]-m).flatten() for t,m in zip(items,means,strict=True)],dim=-1)
            beta = torch.linalg.solve(x.T@x,x.T@(y[keep]-ymean).flatten())
            prediction = ymean + sum(b*(t[i]-m) for t,m,b in zip(items,means,beta,strict=True))
            errors[j].append((y[i]-prediction).square().sum())
    base,combined=(torch.stack(e).sum() for e in errors)
    return (1-combined/base).item()


def leave_class_out(centroid, slopes, observed):
    """Vector intercept and scalar feature coefficients, refit excluding each test class."""
    c = len(centroid)
    errors, base_errors, weights = [],[],[]
    for i in range(c):
        train = torch.arange(c, device=observed.device)!=i
        x = centroid[train]
        v = slopes[:,train]
        y = observed[train]
        xbar,vbar,ybar = x.mean(0),v.mean(1),y.mean(0)
        x,v,y = x-xbar, v-vbar[:,None], y-ybar
        g11 = x.square().sum()
        g12 = (v*x).sum(dim=(1,2))
        c1 = (x*y).sum()
        # Orthogonalize the new predictor against the centroid on training classes only.
        wbase = c1 / g11
        orth = v - (g12/g11)[:,None,None] * x
        beta = (orth*y).sum(dim=(1,2)) / orth.square().sum(dim=(1,2)).clamp_min(1e-18)
        dx = centroid[i]-xbar
        residual = observed[i]-ybar-wbase*dx
        new = slopes[:,i]-vbar-(g12/g11)[:,None]*dx
        errors.append((residual-beta[:,None]*new).square().sum(-1))
        base_errors.append(residual.square().sum())
        weights.append(beta)
    errors = torch.stack(errors, dim=1)
    base_errors = torch.stack(base_errors)
    return 1.-errors.sum(1)/base_errors.sum(), torch.stack(weights,dim=1), base_errors, errors


def source_means(q,p,source,labels):
    ids,group = source.unique(return_inverse=True)
    mean = q.new_zeros(len(ids),q.shape[1]).index_add_(0,group,q)
    difference = p[ids]-F.normalize(mean,dim=1)
    classes = labels[ids]
    return torch.stack([difference[classes==c].mean(0) for c in classes.unique()]), ids


def probes(q,axis,mean):
    rng=torch.Generator().manual_seed(SEED)
    random = torch.randn(RANDOM_DIRECTIONS,q.shape[1],generator=rng,dtype=torch.float64).to(q)
    random = F.normalize(remove_axis(random,axis),dim=-1)
    return torch.cat([axis[None],F.normalize(mean,dim=0)[None],random])


def fit_populations(p, labels, valid, directions):
    nclasses = len(labels.unique())
    banks,centroids,sizes = [],[],[]
    rng = torch.Generator().manual_seed(SEED)
    for c in range(nclasses):
        inds = torch.where(valid & (labels==c))[0]
        if len(inds)<10:
            raise ValueError("Insufficient extension photos for disjoint replication")
        inds = inds[torch.randperm(len(inds),generator=rng)]
        sizes.append(len(inds))
        rep = []
        for part in (inds,inds[::2],inds[1::2]):
            ext = p[part]
            centroid = ext.mean(0)
            residual=ext-centroid
            z=residual@directions.T
            z=z/z.square().mean(0).sqrt().clamp_min(1e-10)
            rep.append((z.T @ residual)/len(ext))
        banks.append(torch.stack(rep))
        centroids.append(p[inds].mean(0))
    return torch.stack(centroids),torch.stack(banks,dim=2),sizes


def summarize(centroids, banks, truth, axis, selected):
    b=remove_axis(centroids[selected],axis)
    y=remove_axis(truth[selected],axis)
    v=remove_axis(banks[:,:,selected],axis)
    gain,weights,base_errors,errors=leave_class_out(b,v[0],y)
    random=gain[2:]
    parts=[]
    for k in (1,2):
        part,_,_,_=leave_class_out(b,v[k],y)
        parts.append(float(part[0]))
    ca=y-y.mean(0)
    cy=[v[h,0]-v[h,0].mean(0) for h in (0,1,2)]
    return {"classes":int(selected.sum()),"photo_minus_drawing_gain_fraction":float(gain[0]),
            "photo_mean_axis_gain":float(gain[1]),"random_axis_gain_median":float(random.median()),
            "additional_fraction_over_centroid_and_photo_mean_axis":incremental_domain_check(b,v[0,1],v[0,0],y),
            "random_axis_gain95":float(random.quantile(.95)),
            "random_at_least_as_large":int((random>=gain[0]).sum()),"random_axes":len(random),
            "gain_in_extension_halves":parts,"mean_direction_cos_to_actual_shift":float(F.cosine_similarity(cy[0],ca).mean()),
            "mean_extension_split_half_cos":float(F.cosine_similarity(cy[1],cy[2]).mean()),
            "positive_coefficients":int((weights[0]>0).sum()),"median_coefficient":float(weights[0].median()),
            "centroid_only_sse":base_errors.tolist(),"centroid_plus_depiction_sse":errors[0].tolist(),
            "classes_with_lower_prediction_error":int((errors[0]<base_errors).sum())}


@torch.inference_mode()
def run(args):
    split=load_visual_split(args.cache,args.split)
    im=F.normalize(F.normalize(split.sketch.float(),dim=1)+F.normalize(split.mirror.float(),dim=1),dim=1)
    p=F.normalize(split.photo.float(),dim=1).double()
    q=im.double()
    gl=split.photo_labels.cuda()
    src=true_sources(split,args.cache)
    if (src<0).any():
        raise ValueError("Missing source identities")
    truth,ids=source_means(q,p,src,gl)
    is_extension=torch.ones(len(p),dtype=torch.bool,device=p.device)
    is_extension[ids]=False
    model=load_frozen_clip(read(args.cache/"manifest.json")["checkpoint"],torch.device("cuda")).float()
    tokens=open_clip.get_tokenizer("ViT-B-32")(list(PROMPTS)).cuda()
    captions=F.normalize(model.encode_text(tokens).double(),dim=1)
    axis=F.normalize(captions[1]-captions[0],dim=0)
    directions=probes(q,axis,p[is_extension].mean(0))
    means,banks,sizes=fit_populations(p,gl,is_extension,directions)
    names=UNSEEN_CLASSES[args.split]
    shared=set(UNSEEN_CLASSES["sketchy_1"])&set(UNSEEN_CLASSES["sketchy_2"])
    all_classes=torch.ones(len(names),device=p.device,dtype=torch.bool)
    exclusive=torch.tensor([n not in shared for n in names],device=p.device)
    full=summarize(means,banks,truth,axis,all_classes)
    disjoint=summarize(means,banks,truth,axis,exclusive)
    ql=split.sketch_labels.cuda()
    pred=q @ F.normalize(means,dim=1).T
    correct=(pred.argmax(1)==ql)
    photo_mean_accuracy=100*correct.double().mean().item()
    tq,tp=q@axis,p@axis
    limits=torch.stack([tp[is_extension & (gl==c)].quantile(torch.tensor([.01,.99],device=p.device,dtype=p.dtype))
                        for c in gl.unique()])
    absolute=torch.stack([torch.stack([tp[is_extension & (gl==c)].min(),tp[is_extension & (gl==c)].max()])
                          for c in gl.unique()])
    support={"query_below_gallery_class_p01_fraction":(tq<limits[ql,0]).double().mean().item(),
             "query_above_gallery_class_p99_fraction":(tq>limits[ql,1]).double().mean().item(),
             "query_outside_observed_gallery_class_range_fraction":
                 ((tq<absolute[ql,0]) | (tq>absolute[ql,1])).double().mean().item(),
             "all_sketch_median":tq.median().item(),"all_extension_median":tp[is_extension].median().item()}
    result={"revision":args.revision,"seed":SEED,"split":args.split,"prompt_axis":PROMPTS,"labels_only_in_diagnosis":True,
            "source_photos":len(ids),"extension_photos":int(is_extension.sum()),"extension_per_class":dict(zip(names,sizes)),
            "class_names":names,"shared_classes":sorted(shared),"all":full,"excluding_shared_classes":disjoint,
            "native_top1_against_labeled_photo_centroids_percent":photo_mean_accuracy,"empirical_support":support,
            "passes":full["photo_minus_drawing_gain_fraction"]>max(0.,full["random_axis_gain95"]) and
                     min(full["gain_in_extension_halves"])>0 and disjoint["photo_minus_drawing_gain_fraction"]>0}
    atomic_json(args.output/f"gallery-response-{args.split}.json",result)
    print(json.dumps({k:full[k] for k in ("photo_minus_drawing_gain_fraction","random_axis_gain95","gain_in_extension_halves")}),
          result["passes"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,default=DEFAULT_CACHE)
    parser.add_argument("--output",type=Path,default=DST)
    parser.add_argument("--split",choices=SKETCHY_SPLITS,required=True)
    parser.add_argument("--revision",required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("highest")
    run(args)
