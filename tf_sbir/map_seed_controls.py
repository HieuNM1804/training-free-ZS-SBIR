"""Cross the initial CLIP signal with one fixed RGB-only gallery operator on Colab."""

import argparse
from pathlib import Path

import torch

from .capture_query_metrics import evaluate_with_details
from .concept_benchmark import atomic_json
from .concept_likelihood import ranked_scores
from .conditional_semantic_index import bare_words
from .crossdomain_benchmark import revision
from .encoding import save_tensor
from .foundation import load_words
from .map_study import Data
from .map_transfer_audit import import_measured, verify_decision
from .retrieval_diagnosis import global_inputs
from .semantic import read_words


def smooth(kernel, evidence, depth):
    def apply(x):
        for _ in range(depth):
            x = torch.sparse.mm(kernel.reverse,x)
        x = kernel.status @ x
        for _ in range(depth):
            x = torch.sparse.mm(kernel.local,x)
        return x
    mass = apply(evidence.new_ones((evidence.size(1),1))).clamp_min_(1e-30)
    scores = torch.empty_like(evidence)
    for begin in range(0,len(scores),384):
        x = evidence[begin:begin+384].T.contiguous()
        scores[begin:begin+x.size(1)] = (apply(x)/mass).T
    return scores


@torch.inference_mode()
def run(w):
    decision,locks = verify_decision(w/"results")
    name="cross-diffusion-map"
    module=import_measured(w,name,locks[name])
    config=module.Config(**locks[name]["winner"])
    out=w/"results/map-study/seed-ablation"
    for ds in ("sketchy","tuberlin"):
        args,_,queries,_,photos,raw,rgb=global_inputs(w,ds)
        for split,(qi,ql) in queries.items():
            pi,pl=photos[split]
            q,p=raw[qi].cuda(),rgb[pi].cuda()
            kernel=module.Kernel(p,module.nearest_gallery(p,config.neighbors),config.neighbors,config.bandwidth)
            table={}
            for seed in ("rgb","bare","domain"):
                if seed=="rgb":
                    evidence=q@p.T
                else:
                    words=read_words(args.vocabulary)
                    bare,scale=bare_words(args,words)
                    source,target=(v.cuda() for v in ((bare,bare) if seed=="bare" else load_words(args,words)))
                    evidence=Data(q,p,source,target,scale).base
                for mode in ("initial","smoothed"):
                    if mode=="initial":
                        scores=evidence
                    else:
                        old_precision=torch.get_float32_matmul_precision()
                        try:
                            torch.set_float32_matmul_precision("high")
                            scores=smooth(kernel,evidence,config.iterations)
                        finally:
                            torch.set_float32_matmul_precision(old_precision)
                    def rows():
                        for block in scores.split(96):
                            yield from ranked_scores(block).cpu()
                    label=seed+"-"+mode
                    (ap,prec,mk,pk),per_query=evaluate_with_details(rows(),ql,pl,split,split+" "+label,4)
                    save_tensor(per_query,out/"per-query"/f"{split}-{label}.pt")
                    table[label]={"mAP":float(ap),"precision":float(prec),"map_k":mk,"p_k":pk}
                    del scores
                del evidence
            if config.mode=="rgb":
                location="heldout" if ds=="tuberlin" else "sketchy"
                ref=torch.load(w/"results"/name/location/"per-query"/f"{split}-{locks[name]['winner_id']}.pt",weights_only=True)
                obs=torch.load(out/"per-query"/f"{split}-domain-smoothed.pt",weights_only=True)
                diff=(ref-obs).abs()
                table["recomputed_official_mean_abs_delta"]=diff.mean(0).tolist()
                table["recomputed_official_max_abs_delta"]=diff.amax(0).tolist()
                if float(diff.mean(0).max())>5e-5:
                    raise ValueError("Independent decomposition does not reproduce the RGB graph")
            result={"joint_sha256":decision["sha256"],"config":locks[name]["winner"],"audit_revision":revision(),
                    "selection":False,"queries":len(ql),"gallery":len(pl),"scores":table}
            atomic_json(out/f"{split}.json",result)
            del kernel
            print(split,table,flush=True)
    atomic_json(out/"complete.json",{"joint_sha256":decision["sha256"],"complete":True})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(p.parse_args().workspace)
