"""Fixed per-image patch instance normalization before the frozen CLIP transformer."""

import argparse
from datetime import datetime,timezone
from pathlib import Path

import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from .concept_benchmark import atomic_json,checksum,read
from .crossdomain_benchmark import SPLIT_HASHES,digest,revision
from .crossdomain_inputs import CHECKPOINT
from .data import Images,UNSEEN_CLASSES,normal_transform,threaded_batches
from .encoding import global_cache_name,load_frozen_clip,save_tensor
from .metrics import cosine_batches
from .retrieval_diagnosis import global_inputs,stratified_sample
from .retrieval_evidence import load_kd,paired_delta,tensor
from .transfer_benchmark import write_result


PRIMARY="instance"
VARIANTS=("native","center","scale","instance")


def finish(v,tokens):
    pooled,_=v._pool(v.transformer(tokens))
    return F.normalize((pooled@v.proj).float(),dim=-1)


def prepare(v,pixels):
    before=[]
    handle=v.ln_pre.register_forward_pre_hook(lambda _,args:before.append(args[0]))
    try:
        original=v._embeds(pixels)
    finally:
        handle.remove()
    width=original.size(-1)
    patches=F.layer_norm(before[0][:,1:].float(),(width,),eps=v.ln_pre.eps)
    return original,patches


def normalized(patches,eps,variant):
    variance,mean=torch.var_mean(patches,dim=1,correction=0,keepdim=True)
    centered=patches-mean
    if variant=="center":
        return centered
    standardized=centered*torch.rsqrt(variance+eps)
    return standardized+mean if variant=="scale" else standardized


def affine(v,patches,original):
    scaled=patches*v.ln_pre.weight.float()+v.ln_pre.bias.float()
    return torch.cat((original[:,:1],scaled.to(original.dtype)),1)


def perturbation(v,native,patches):
    gen=torch.Generator().manual_seed(1229); width=patches.size(-1)
    gain=(torch.randn(width,generator=gen)*.4).exp().to(patches.device)
    bias=(torch.randn(width,generator=gen)*.4).to(patches.device)
    shifted=patches*gain+bias
    def rep(x,variant):
        h=x if variant=="native" else normalized(x,v.ln_pre.eps,variant)
        return finish(v,affine(v,h,native))
    out={}
    for mode in VARIANTS:
        u,vv=rep(patches,mode),rep(shifted,mode)
        out[mode]=float((1-(u*vv).sum(1)).mean())
    return out


@torch.inference_mode()
def features(args,paths,modality):
    filename=global_cache_name(args.root,paths,modality,"patch-style-fixed-v1","fp16",args.clip_checkpoint)
    file=args.cache/filename
    if not file.exists():
        model=load_frozen_clip(args.clip_checkpoint,torch.device("cuda"))
        v=model.visual
        batch=192
        stream=threaded_batches(Images(paths,normal_transform()),batch,8,True)
        store={k:[] for k in (*VARIANTS[1:],"moments")}
        checks={}
        for pixels in tqdm(stream,total=(len(paths)+batch-1)//batch,desc=f"Patch moments {modality}",mininterval=2):
            pixels=pixels.to("cuda",dtype=v.conv1.weight.dtype,non_blocking=True)
            orig,patch=prepare(v,pixels)
            if not checks:
                n=16
                ref=F.normalize(model.encode_image(pixels[:n]).float())
                checks["native_path_max_abs"]=float((finish(v,orig[:n])-ref).abs().max())
                checks["perturbation_mean_cosine_distance"]=perturbation(v,orig[:n],patch[:n])
            variance,mean=torch.var_mean(patch,dim=1,correction=0)
            store["moments"].append(torch.cat((mean,variance.sqrt()),1).half().cpu())
            encoded=finish(v,torch.cat([affine(v,normalized(patch,v.ln_pre.eps,key),orig) for key in VARIANTS[1:]]))
            for key,block in zip(VARIANTS[1:],encoded.split(len(pixels)),strict=True):
                store[key].append(block.cpu())
        final={k:torch.cat(xs) for k,xs in store.items()}
        save_tensor(final|{"checks":checks},file)
    return torch.load(file,map_location="cpu",weights_only=True,mmap=True)


def retrieval_analysis(q,p,ql,pl,qstyle,pstyle,k):
    labels=ql.cuda(); gallery_labels=pl.cuda()
    moments_q=qstyle.cuda().float(); moments_p=pstyle.cuda().float()
    diff=[]; ranks=[]; wrongids=[]
    for start in range(0,len(q),128):
        block=q[start:start+128]
        sim=block@p.T
        target=labels[start:start+len(block),None]==gallery_labels
        best=sim.argmax(1)
        good=sim.masked_fill(~target,-torch.inf).argmax(1)
        wrong=gallery_labels[best]!=labels[start:start+len(block)]
        qs=moments_q[start:start+len(block)]
        # Squared diagonal Gaussian Wasserstein distance up to the common scale.
        ds_wrong=(qs-moments_p[best]).square().mean(1)
        ds_good=(qs-moments_p[good]).square().mean(1)
        diff.append((ds_good-ds_wrong)[wrong].cpu())
        ranks.append((1+(sim>sim.gather(1,good[:,None])).sum(1)).cpu())
        wrongids.append(wrong.cpu())
    delta=torch.cat(diff)
    ids=stratified_sample(pl,20,6901)
    score=p[ids]@p.T
    score[torch.arange(len(ids),device=p.device),ids]=-torch.inf
    correct=gallery_labels[score.topk(k).indices]==gallery_labels[ids,None]
    return {"native_space_wrong_top1":int(torch.cat(wrongids).sum()),"first_positive_median":int(torch.cat(ranks).median()),
            "mean_Dpositive_minus_Dwrong":float(delta.mean()) if len(delta) else None,
            "fraction_nearer_wrong_in_moment_space":float((delta>0).float().mean()) if len(delta) else None,
            "photo_to_photo_excluding_self_sample_P":float(correct.float().mean()),"photo_queries":len(ids)}


def lock_protocol(root):
    path=root/"protocol.json"; protocol={"revision":revision(),"checkpoint_sha256":CHECKPOINT,
        "primary":PRIMARY,"controls":list(VARIANTS),"layer":"visual.ln_pre","reference":"frozen ln_pre gain and bias",
        "hyperparameter_search":False,"per_query_independent":True,"gallery_calibration":False,
        "requirement":"higher P and mAP than native on each split for a cross-dataset benefit claim"}
    if path.exists():
        saved=read(path)
        if any(saved[k]!=v for k,v in protocol.items()) or digest({k:v for k,v in saved.items() if k!="sha256"})!=saved["sha256"]:
            raise ValueError("Conflict in locked protocol")
        return saved
    if any(root.glob("*/sketchy_*-*.json")) or any(root.glob("*/tuberlin-*.json")):
        raise ValueError("Scores predate the protocol")
    protocol["utc"]=datetime.now(timezone.utc).isoformat()
    atomic_json(path,protocol|{"sha256":digest(protocol)})
    return read(path)


@torch.inference_mode()
def run(w,dataset):
    root=w/"results/patch-style-invariance"
    protocol=lock_protocol(root)
    held=dataset=="tuberlin"
    if held:
        src=read(root/"sketchy-complete.json")
        if src["protocol"]!=protocol["sha256"]:
            raise ValueError("Inconsistent method for transfer")
    args,qp,qs,pp,gs,raw,photos=global_inputs(w,dataset)
    qbank=features(args,qp,"sketch"); pbank=features(args,pp,"photo")
    out=root/("heldout" if held else "sketchy"); out.mkdir(parents=True,exist_ok=True)
    kd,kd_sha=load_kd(w)
    report={"protocol":protocol["sha256"],"kd_function_file":kd_sha,"sketch_checks":qbank["checks"],
            "photo_checks":pbank["checks"],"splits":{}}
    for s,(qi,ql) in qs.items():
        pi,pl=gs[s]; k=200 if s=="sketchy_2" else 100
        if digest(UNSEEN_CLASSES[s])!=SPLIT_HASHES[s] or len(ql.unique())!=len(UNSEEN_CLASSES[s]):
            raise ValueError("Wrong split")
        qscores={}; metrics={}; diagnosis={}
        for mode in VARIANTS:
            q=(raw if mode=="native" else qbank[mode])[qi].cuda()
            p=(photos if mode=="native" else pbank[mode])[pi].cuda()
            meta={"revision":protocol["revision"],"backbone":"OpenAI CLIP ViT-B/32","gpu":torch.cuda.get_device_name(),
                  "checkpoint_sha256":CHECKPOINT,"split_sha256":SPLIT_HASHES[s],"classes":len(UNSEEN_CLASSES[s]),
                  "freeze_sha256":protocol["sha256"],"exploratory_unseen":False,"external_vocabulary":0}
            args.output=out; args.metric_workers=6 if s!="sketchy_2" else 4
            def rows():
                for block in cosine_batches(q,p,4):
                    yield from block.cpu()
            write_result(args,s,mode,rows(),ql,pl,meta,{"patch_norm":mode,"score":"native cosine"})
            queries=tensor(out/"per-query"/f"{s}-{mode}.pt")
            recorded=read(out/f"{s}-{mode}.json")
            if (recorded["queries"],recorded["gallery"])!=(len(ql),len(pl)) or not torch.isfinite(queries).all():
                raise ValueError("Incomplete evaluation")
            qscores[mode]=queries
            metrics[mode]=dict(mAP=recorded["mAP"],precision=recorded["precision"])
            if mode in ("native",PRIMARY):
                diagnosis[mode]=retrieval_analysis(q,p,ql,pl,qbank["moments"][qi],pbank["moments"][pi],k)
            if mode==PRIMARY:
                ap,pr,mk,pk=kd(q,p,ql,pl,s)
                metrics[mode]["original_KD"]={"mAP":float(ap),"precision":float(pr),"map_k":mk,"p_k":pk,
                    "max_abs_difference":max(abs(float(ap)-recorded["mAP"]),abs(float(pr)-recorded["precision"]))}
        deltas={v:paired_delta(qscores[v],qscores["native"],ql) for v in VARIANTS[1:]}
        by_class={n:{v:qscores[v][ql==i].mean(0).tolist() for v in VARIANTS} for i,n in enumerate(UNSEEN_CLASSES[s])}
        report["splits"][s]={"queries":len(ql),"gallery":len(pl),"classes":len(by_class),"metrics":metrics,
                             "diagnosis":diagnosis,"vs_native":deltas,"classes_AP_P":by_class}
        print(s,metrics,flush=True)
    atomic_json(root/(dataset+"-complete.json"),report)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    parser.add_argument("--dataset",choices=("sketchy","tuberlin"),required=True)
    options=parser.parse_args()
    run(options.workspace,options.dataset)
