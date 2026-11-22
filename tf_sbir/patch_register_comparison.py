"""Compare independently attributed neurons with the published high-norm-register criterion on the identical B/32 checkpoint."""

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from .concept_benchmark import atomic_json, checksum, read
from .crossdomain_benchmark import revision
from .encoding import load_frozen_clip, save_tensor
from .gallery_panel_resume import semantic_inputs
from .patch_mediation_inputs import attention
from .patch_response_mediation import FAMILY
from .data import Images, configure_image_worker, normal_transform
from torch.utils.data import DataLoader


MODES=("paper_patches_75","code_with_cls_75","code_default_threshold30","maximum_patch","top_three_patches")


@torch.inference_mode()
def run(w):
    inputs=semantic_inputs(w,"sketchy")
    cfg,_,_,photos,*_=inputs
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    v=load_frozen_clip(cfg.clip_checkpoint,torch.device("cuda")).visual
    root=w/"results"/FAMILY
    old=torch.load(root/"activation-source.pt",weights_only=True,mmap=True)
    measures=read(root/"activation-source.json")
    hot={l:measures["samples"]["gallery_discovery"][f"hidden/{l}"]["gallery_discovery_units"][:8] for l in (2,3)}
    result,full={},{}
    for split in ("gallery_discovery","gallery_holdout"):
        ids=old[split]["ids"]
        ds=Images([photos[int(i)] for i in ids],normal_transform(),"rgb")
        dl=DataLoader(ds,batch_size=12,num_workers=cfg.workers,pin_memory=True,worker_init_fn=configure_image_worker)
        means=defaultdict(list);hits=defaultdict(list); norms=[]
        for px in dl:
            h=v._embeds(px.cuda(non_blocking=True).to(v.conv1.weight.dtype))
            values=[]
            for block in v.transformer.resblocks[:6]:
                post=attention(v,block,h)
                fc,act,out=tuple(block.mlp.children())
                a=act(fc(block.ln_2(post)))
                values.append(a.abs())
                h=post+block.ls_2(out(a))
            n=h.float().norm(dim=-1)
            norms.append(n.cpu())
            patchmask=torch.ones_like(n,dtype=torch.bool);patchmask[:,0]=False
            largest=n.masked_fill(~patchmask,-torch.inf).argsort(descending=True,stable=True)
            masks={"paper_patches_75":(n>75)&patchmask,"code_with_cls_75":n>75,"code_default_threshold30":n>30}
            for k,tag in ((1,"maximum_patch"),(3,"top_three_patches")):
                masks[tag]=torch.zeros_like(patchmask).scatter_(1,largest[:,:k],True)
            for name,mask in masks.items():
                q=mask.float()/mask.sum(1,keepdim=True).clamp_min(1)
                scores=torch.stack([torch.bmm(q[:,None],arr.float()).squeeze(1) for arr in values],1)
                means[name].append(scores.cpu());hits[name].append(mask.sum(1).cpu())
        total=torch.cat(norms)
        audit={}
        block_scores={}
        for name in MODES:
            counts=torch.cat(hits[name]);per_image=torch.cat(means[name])
            valid=counts>0
            if valid.any():
                score=per_image[valid].mean(0)
                giant=score.flatten().argsort(descending=True,stable=True)[:10]
                pairs=[(int(x)//3072+1,int(x)%3072) for x in giant]
                term={"global_top10":pairs,"global_top10_overlaps_sensitive_units":{str(l):[i for i in hot[l] if (l,i) in pairs] for l in hot},
                      "within_layer_top8_overlap":{},"within_layer_sensitive_median_rank":{}}
                for layer,chosen in hot.items():
                    order=score[layer-1].argsort(descending=True,stable=True)
                    rank=order.argsort()+1
                    term["within_layer_top8_overlap"][str(layer)]=int(torch.isin(order[:8],torch.tensor(chosen)).sum())
                    term["within_layer_sensitive_median_rank"][str(layer)]=float(rank[chosen].double().quantile(.5))
                block_scores[name]=score
            else:term={"global_top10":None}
            term.update({"qualifying_images":int(valid.sum()),"median_positions_per_qualifying_image":float(counts[valid].float().median()) if valid.any() else None})
            audit[name]=term
        result[split]={"images":len(ids),"units_from_separate_curvature_criterion":hot,"outlier_detector_block_one_based":6,
                      "image_patch_norm_max_quantiles":total[:,1:].amax(1).quantile(torch.tensor([0.,.5,.95,1.])).tolist(),"criteria":audit}
        full[split]={"ids":ids,"patch_norms_including_cls":total,"detector_means":block_scores,"counts":{k:torch.cat(x) for k,x in hits.items()}}
        print(split,[(k,x["qualifying_images"],x.get("within_layer_top8_overlap")) for k,x in audit.items()],flush=True)
    path=root/"register-comparison.pt"
    save_tensor(full,path)
    atomic_json(path.with_suffix(".json"),{"revision":revision(),"source":"Jiang et al., NeurIPS 2025, Section 3.2, top_layer 5 (zero based), strict threshold 75, top 10; criterion adapted to the original OpenAI B/32",
                "public_code":"https://github.com/nickjiang2378/test-time-registers/blob/main/shared/algorithms.py",
                "source_model_difference":"The authors' main demonstration uses OpenCLIP B/16. These are newly measured B/32 scores. Quantile and within-layer checks are diagnostic extensions.",
                "design":"Published absolute activation averaged over selected positions first, and over qualifying images second. No sparsity filter. Top 10 globally within the first six blocks.",
                "groups":result,"per_query_sha256":checksum(path),"device":torch.cuda.get_device_name()})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(p.parse_args().workspace)
