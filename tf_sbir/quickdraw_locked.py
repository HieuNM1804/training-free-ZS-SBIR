"""Evaluate the pre-existing Sketchy mAP winner on all QuickDraw unseen sketches on Colab."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
from importlib import util
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import torch
from torch.nn import functional as F

from .concept_benchmark import atomic_json, checksum, read
from .conditional_semantic_index import bare_words
from .cross_diffusion_map import Config, Index
from .crossdomain_benchmark import digest, revision
from .crossdomain_inputs import CHECKPOINT
from .data import UNSEEN_CLASSES
from .encoding import load_global_inputs, save_tensor
from .foundation import load_words
from .map_study import Data
from .quickdraw_inputs import paths
from .retrieval_evidence import load_kd
from .semantic import read_words
from .shared_concept_likelihood import emission, target_inputs


FAMILY = "cross-diffusion-map"
JOINT_SHA = "90eee74f8730e3e76fe17ac93fef4b48648f91b04bcc8371e6c296cbcca7158d"
SPLIT_SHA = "de129ecf410271ecb69ad614a57b4a8ab567bb448876d022e2338f1b2185a89c"
KD_FUNCTION_SHA = "9310e11f47f6fafb54c64060709df8a302165e54e2a8812c97bdb85193c15139"


def get_lock(workspace):
    root = workspace/"results"
    joint = read(root/"map-study/joint-freeze.json")
    signature = joint.pop("sha256")
    if signature!=JOINT_SHA or digest(joint)!=signature or joint["primary"]!=FAMILY:
        raise ValueError("Incorrect pre-existing Sketchy method selection")
    family = read(root/FAMILY/"freeze.json")
    signature = family.pop("sha256")
    entry = joint["families"][FAMILY]
    if (digest(family)!=signature or signature!=entry["freeze"] or family["winner_id"]!=entry["name"]
            or digest(family["winner"])[:16]!=entry["name"] or family["winner"]["mode"]!="rgb"):
        raise ValueError("Incorrect frozen parameters")
    return family,signature,joint["utc"]


def compare_original(w,lock,config,q,p,source,target,scale):
    # Keep the exact implementation that produced the Sketchy ranks as the reference.
    module_path = w/"worktrees"/("map-"+lock["revision"][:12])/"tf_sbir/cross_diffusion_map.py"
    spec = util.spec_from_file_location("tf_sbir.quickdraw_frozen_reference",module_path)
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    qids = torch.linspace(0,len(q)-1,12).long()
    pids = torch.linspace(0,len(p)-1,768,device=p.device).long()
    data = Data(q[qids].cuda(),p[pids],source,target,scale)
    original = module.Index(data).score(module.Config(**lock["winner"]))
    batched = Index(SimpleNamespace(p=data.p),rgb_only=True).score(config,data.base)
    identical = torch.equal(original.argsort(1,descending=True),batched.argsort(1,descending=True))
    if not identical or not torch.allclose(original,batched,atol=1e-7,rtol=1e-6):
        raise ValueError("Batched RGB operator differs from the frozen implementation")
    return {"queries":len(qids),"photos":len(pids),"identical_complete_ranks":identical,
            "max_score_difference":float((original-batched).abs().max()),"frozen_source_sha256":checksum(module_path)}


class OfficialMetrics:
    def __init__(self, function, photo_labels):
        self.function,self.labels = function,photo_labels
        self.query = torch.tensor([[1.,0.]])
        fixed = torch.linspace(.75,-.75,len(photo_labels))
        self.gallery = torch.stack((fixed,(1.-fixed.square()).sqrt()),1)
        # Only the label order varies. These cosine values realize the exact
        # method ranking with no ties after KD-SBIR's internal affine transform.
        scores = (F.cosine_similarity(self.query,self.gallery)+1.)*.5
        if not bool((scores[:-1]>scores[1:]).all()):
            raise ValueError("The cosine representation did not preserve strict rank order")

    def __call__(self,orders,labels):
        values = torch.empty(len(labels),2)
        for i,(order,label) in enumerate(zip(orders,labels,strict=True)):
            ap,precision,mk,pk = self.function(self.query,self.gallery,label.view(1),self.labels[order],"quickdraw")
            if (mk,pk)!=(0,200):
                raise ValueError("Original KD-SBIR returned an unexpected QuickDraw protocol")
            values[i,0],values[i,1] = ap,precision
        return values


def score_shard(q,labels,source,a,scale,index,config,metrics,pool,photo_labels):
    futures = []
    independently_computed = []
    rank = torch.arange(1,len(photo_labels)+1,device=photo_labels.device)
    for begin in range(0,len(q),384):
        block = q[begin:begin+384].cuda()
        y = labels[begin:begin+len(block)]
        evidence = emission(block,source,scale) @ a.T
        scores = index.score(config,evidence)
        if not torch.isfinite(scores).all():
            raise ValueError("Non-finite gallery scores")
        order = scores.argsort(dim=1,descending=True,stable=True)
        correct = photo_labels[order]==y.cuda()[:,None]
        precision_by_rank = correct.cumsum(1)/rank
        ap = (correct*precision_by_rank).sum(1)/correct.sum(1)
        p200 = correct[:,:200].float().mean(1)
        independently_computed.append(torch.stack((ap,p200),1).cpu())
        host = order.to(device="cpu",dtype=torch.int32)
        futures.extend(pool.submit(metrics,b,l) for b,l in zip(host.split(24),y.split(24),strict=True))
    official = torch.cat([job.result() for job in futures])
    difference = (official-torch.cat(independently_computed)).abs().amax(0)
    if float(difference.max())>2e-6:
        raise ValueError("Vectorized AP/P disagree with the original KD-SBIR evaluator")
    return official,difference


@torch.inference_mode()
def run(w,workers):
    lock,freeze,selection_utc = get_lock(w)
    classes = UNSEEN_CLASSES["quickdraw"]
    if digest(classes)!=SPLIT_SHA:
        raise ValueError("Unseen class list differs from D:/distill")
    meta = read(w/"quickdraw-data.json")
    args,qpaths,ql,ppaths,pl = paths(w,448,10)
    q,p,_ = load_global_inputs(args,qpaths,ppaths,False)
    p = p.cuda()
    if (len(q),len(p),len(ql.unique()),len(pl.unique()))!=(91491,56407,30,30):
        raise ValueError("Expected the entire QuickDraw v1 unseen query set and gallery")
    words = read_words(args.vocabulary)
    if len(words)!=20931:
        raise ValueError("Different public vocabulary")
    _,scale = bare_words(args,words)
    source,target = (t.cuda() for t in load_words(args,words))
    function,kd_file = load_kd(w)
    if hashlib.sha256(inspect.getsource(function).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:
        raise ValueError("Different original KD-SBIR function")
    config = Config(**lock["winner"])
    check = compare_original(w,lock,config,q,p,source,target,scale)
    index = Index(SimpleNamespace(p=p),rgb_only=True)
    index.prepare(config)
    a = target_inputs(p,target,scale)
    a /= a.sum(1,keepdim=True)
    metrics = OfficialMetrics(function,pl)
    glabels = pl.cuda()
    folder = w/"results/quickdraw-locked-map"
    protocol = {"dataset":"quickdraw","source":meta["source"],"archive_sha256":meta["archive_sha256"],
                "image_manifest_sha256":meta["image_manifest_sha256"],"class_list_sha256":SPLIT_SHA,
                "family":FAMILY,"configuration":lock["winner"],"measured_method_revision":lock["revision"],
                "revision":revision(),"joint_sha256":JOINT_SHA,"freeze_sha256":freeze,"selection_utc":selection_utc,
                "selection_data":["sketchy_1","sketchy_2"],"checkpoint_sha256":CHECKPOINT,
                "external_words":len(words),"external_words_sha256":checksum(args.vocabulary),
                "kd_reference_file_sha256":kd_file,"kd_function_sha256":KD_FUNCTION_SHA}
    bound = digest(protocol)
    chunks,max_errors = [],[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for begin in range(0,len(q),3072):
            end = min(len(q),begin+3072)
            shard = folder/"shards"/f"{begin:06d}-{end:06d}.pt"
            if shard.is_file():
                entry = torch.load(shard,map_location="cpu",weights_only=True)
                if entry["protocol"]!=bound or entry["official"].shape!=(end-begin,2):
                    raise ValueError("Partial results have a different protocol")
            else:
                observed,error = score_shard(q[begin:end],ql[begin:end],source,a,scale,index,config,metrics,pool,glabels)
                entry = {"protocol":bound,"official":observed,"max_AP_P_error":error}
                save_tensor(entry,shard)
            chunks.append(entry["official"])
            max_errors.append(entry["max_AP_P_error"])
            print(f"QuickDraw KD-SBIR {end}/{len(q)} sketches on all {len(p)} gallery photos",flush=True)
    values = torch.cat(chunks)
    if values.shape!=(len(ql),2) or not torch.isfinite(values).all():
        raise ValueError("Incomplete QuickDraw scores")
    ap,prec = values.mean(0).tolist()
    pt = folder/"per-query.pt"
    save_tensor({"AP":values[:,0],"P200":values[:,1],"labels":ql,"gallery_labels":pl},pt)
    with (folder/"per-class.csv").open("w",newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["class","queries","gallery","mAP_all","P200"])
        for i,category in enumerate(classes):
            mask = ql==i
            writer.writerow([category,int(mask.sum()),int((pl==i).sum()),*values[mask].mean(0).tolist()])
    result = protocol|{"utc":datetime.now(timezone.utc).isoformat(),"queries":len(q),"gallery":len(p),
             "classes":len(classes),"mAP":ap,"precision":prec,"map_k":0,"p_k":200,
             "original_kd_queries":len(values),"verification_max_abs_AP_P":torch.stack(max_errors).amax(0).tolist(),
             "gpu":torch.cuda.get_device_name(),"peak_allocated_MiB":torch.cuda.max_memory_allocated()/2**20,
             "frozen_code_comparison":check,"per_query_sha256":checksum(pt),"per_class_sha256":checksum(folder/"per-class.csv")}
    atomic_json(folder/"quickdraw.json",result)
    with zipfile.ZipFile(folder/"quickdraw-evidence.zip","w",compression=zipfile.ZIP_DEFLATED) as archive:
        for path in (folder/"quickdraw.json",folder/"per-class.csv",pt):
            archive.write(path,path.name)
    print("QuickDraw mAP@all",ap,"P@200",prec,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    parser.add_argument("--metric-workers",type=int,default=10)
    options=parser.parse_args()
    run(options.workspace,options.metric_workers)
