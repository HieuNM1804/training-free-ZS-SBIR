"""Test conditional rank transport against known monotone shifts and real photometric changes on held-out gallery images."""

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from .concept_benchmark import atomic_json, checksum, read
from .conditional_patch_rank_transport import FAMILY,GROUPS,K,LAYERS,METHODS,N,cdf,measure,prepare,quantile,rank_among_patches
from .crossdomain_benchmark import revision
from .data import CLIP_MEAN, CLIP_STD
from .encoding import load_frozen_clip, save_tensor
from .gallery_panel_resume import semantic_inputs
from .semantic_neuron_counterfactuals import embeddings
from .semantic_neuron_gallery import FAMILY as MEMORY, batches, native, to_gpu


def algebra(device):
    gen=torch.Generator().manual_seed(16219)
    loc=torch.randn(2,49,N,1,generator=gen).to(device)
    scale=torch.rand(2,49,N,1,generator=gen).to(device)+.5
    sample=loc+scale*torch.linspace(-1.,1.,K,device=device)[None,None,None]
    u=torch.rand(2,49,N,generator=gen).to(device)*.58+.25
    transformed=u.square()
    v1,v2=quantile(sample,u),quantile(sample,transformed)
    a,b=cdf(sample,v1),cdf(sample,v2)
    invfloat=(a-u).abs().amax()
    rank_error=(rank_among_patches(a)-rank_among_patches(b)).abs().amax()
    tie=torch.full((2,49,N),.6,device=device)
    tie_error=(rank_among_patches(tie)-.5).abs().max()
    if invfloat>1e-4 or rank_error>0 or tie_error!=0:
        raise ValueError("Conditional probability or order invariance failed")
    return {"calibration_inverse_abs_error":float(invfloat),"monotone_rank_error":float(rank_error),
            "all_tied_rank_error":float(tie_error),"ties_share_their_midrank":True}


def real_camera(x,mode):
    if mode=="clean":return x
    mu=x.new_tensor(CLIP_MEAN)[None,:,None,None]
    sd=x.new_tensor(CLIP_STD)[None,:,None,None]
    rgb=(x*sd+mu).clamp(0,1)
    if mode=="gamma":rgb=rgb.pow(1.8)
    elif mode=="contrast":rgb=.5+.55*(rgb-.5)
    elif mode=="mirror":return x.flip(-1)
    else:raise ValueError(mode)
    return (rgb-mu)/sd


def summaries(table):
    return {k:{"images":len(v),"median_normalized_mse":float(v[:,0].median()),"mean_normalized_mse":float(v[:,0].mean()),
               "median_original_embedding_distance":float(v[:,1].median()),"mean_original_embedding_distance":float(v[:,1].mean()),
               "improves_activations_fraction":float(v[:,2].mean()),"improves_embedding_fraction":float(v[:,3].mean()),
               "median_fraction_PIT_tails":float(v[:,4].median())} for k,v in table.items()}


def synthetic(context,truth,sd):
    obs=context["observations"];u=context["PIT"]
    # The clean scalar, including its possible finite-bank tails, is retained as the independent measurement target.
    warped=quantile(obs,u.clamp(0,1).pow(1.8))
    r=rank_among_patches(cdf(obs,warped))
    h=context["template"]
    methods={"observed":warped,"conditional_uniform":quantile(obs,r),"conditional_scene":quantile(obs,quantile(h,r)),
             "rotate_ranks":quantile(obs,quantile(h,r.roll(13,1))),"conditional_mean":obs.mean(-1)}
    return {k:((v-truth)/sd).square().mean((1,2)).cpu() for k,v in methods.items()}


@torch.inference_mode()
def run(w):
    inputs=semantic_inputs(w,"sketchy")
    cfg,_,qs,pp,*_=inputs
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    vision=load_frozen_clip(cfg.clip_checkpoint,torch.device("cuda")).visual
    root=w/"results"/FAMILY
    baseline_checks=algebra("cuda")
    for s in qs:
        path=root/("diagnosis-"+s+".pt")
        if path.exists() and path.with_suffix(".json").exists():
            if read(path.with_suffix(".json"))["per_query_sha256"]!=checksum(path):raise ValueError("Stale trace")
            continue
        saved=torch.load(w/"cache"/MEMORY/(s+"-memory-v2.pt"),weights_only=True,mmap=True)
        ids=saved["holdout"]
        bank=to_gpu(saved["table"])
        units={l:{g:x.cuda() for g,x in groups.items()} for l,groups in saved["units"].items()}
        cache=prepare(bank)
        std={tag:v[...,:N].flatten(0,1).float().std(0).clamp_min(1e-4) for tag,v in bank["values"].items()}
        arms=[(l,g,N,m) for l in LAYERS for g in GROUPS for m in METHODS]
        records,warped=defaultdict(list),defaultdict(list)
        for no,px in enumerate(batches(pp,ids,cfg.workers,12)):
            clean=px.cuda(non_blocking=True)
            ref=native(vision,clean,units)
            for mode in ("clean","contrast","gamma","mirror"):
                src=ref if mode=="clean" else native(vision,real_camera(clean,mode),units)
                options,contexts=measure(src,bank,cache)
                changed=embeddings(vision,src,units,options,arms)
                original=ref["scene"]
                before=(src["scene"]-original).norm(dim=1)
                for j,(l,g,n,m) in enumerate(arms):
                    tag=l,g
                    correct=ref["values"][tag][...,:N].float()
                    if mode=="mirror":
                        correct=correct.reshape(-1,7,7,N).flip(2).reshape(-1,49,N)
                    diff=((options[tag][m]-correct)/std[tag]).square().mean((1,2))
                    native_err=((contexts[tag]["value"]-correct)/std[tag]).square().mean((1,2))
                    distance=(src["scene"]+changed[:,j]-original).norm(dim=1)
                    row=torch.stack((diff,distance,(diff<native_err).float(),(distance<before-1e-5).float(),
                                     contexts[tag]["near_saturation"]),1)
                    records[f"{mode}/{l}/{g}/{m}"].append(row.cpu())
                if mode=="clean":
                    for tag in contexts:
                        l,g=tag
                        syn=synthetic(contexts[tag],contexts[tag]["value"],std[tag])
                        for m,val in syn.items():warped[f"{l}/{g}/{m}"].append(val)
            if no%4==0:print("held-out photo control",s,no*12,len(ids),flush=True)
        data={k:torch.cat(v) for k,v in records.items()}
        synth={k:torch.cat(v) for k,v in warped.items()}
        save_tensor({"photo_ids":ids,"camera_interventions":data,"known_warp_normalized_mse":synth},path)
        atomic_json(path.with_suffix(".json"),{"revision":revision(),"gallery_held_out":len(ids),"device":torch.cuda.get_device_name(),
                    "memory":MEMORY,"neighbors":K,"checked_algebra":baseline_checks,"results":summaries(data),
                    "known_warp_median_normalized_mse":{k:float(v.median()) for k,v in synth.items()},
                    "per_query_sha256":checksum(path),
                    "scope":"camera perturbations of independent photos with their unclipped original values as reference; this is a mechanism check and does not estimate SBIR mAP"})
        print("completed held-out photo test",s,flush=True)
        del cache,bank


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(cli.parse_args().workspace)
