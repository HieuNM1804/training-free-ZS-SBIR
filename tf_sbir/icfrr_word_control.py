"""ICFRR public reranker on the same CLIP embeddings, vocabulary seed and KD evaluator."""

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import os
from pathlib import Path
import urllib.request

import torch

from .concept_benchmark import atomic_json, checksum
from .conditional_agreement_study import Replay
from .conditional_semantic_index import SIZES, bare_words
from .crossdomain_benchmark import SPLIT_HASHES, digest, revision
from .crossdomain_inputs import CHECKPOINT
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .foundation import load_words
from .quickdraw_locked import KD_FUNCTION_SHA
from .retrieval_diagnosis import global_inputs
from .retrieval_evidence import load_kd, paired_delta
from .semantic import read_words
from .shared_concept_likelihood import emission


FAMILY = "icfrr-word-control"
UPSTREAM = "e01bf1be481881a0b1288e86a587a5aff6e06acb"
SOURCE_SHA = "fc836bf6eac3312c9bf8cc0f256ce1b584c6f939841d84d53312e108c5e0ba87"
KG, KQ, BETA, STEPS = 256, 250, .5, 5


def upstream_function():
    url = "https://raw.githubusercontent.com/finlay-hudson/ICFRR/" + UPSTREAM + "/src/rerank.py"
    with urllib.request.urlopen(url, timeout=30) as response:
        source = response.read()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA:
        raise ValueError("Unexpected upstream source")
    tree = ast.parse(source)
    names = {"_get_scores_based_off_im_dom", "reranking_without_each_element"}
    tree.body = [node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    if len(tree.body) != 2:
        raise ValueError("Upstream function definitions not found")
    scope = {"torch":torch}
    exec(compile(tree, url, "exec"),scope)
    return scope["reranking_without_each_element"]


def gallery_index(photos):
    result = []
    self_rank0 = 0
    for start in range(0,len(photos),256):
        d = torch.cdist(photos[start:start+256],photos)
        order = d.argsort(1)
        self_rank0 += int((order[:,0] == torch.arange(start,start+len(d),device=d.device)).sum())
        result.append(order[:,1:1+KG].contiguous())
    return torch.cat(result), self_rank0


def iterate(seed, neighbors):
    original, order = seed.sort(dim=1, descending=True)
    weights = torch.linspace(1,0,seed.size(1),device=seed.device)[:KG]
    votes = weights.expand(len(seed),KQ,KG).reshape(len(seed),-1)
    yield 0, order
    for step in range(1, STEPS+1):
        ids = neighbors[order[:,:KQ]].reshape(len(seed),-1)
        bonus = torch.zeros_like(seed).scatter_add_(1,ids,votes) * (BETA/KQ)
        # The upstream code keeps the first sorted score at each rank across iterations.
        rank = (original + bonus.gather(1,order)).sort(dim=1,descending=True).indices
        order = order.gather(1,rank)
        yield step, order


def check_reference(photos, queries, upstream):
    p = photos[:513]
    seeds = -torch.cdist(queries[:4],p)
    d = torch.cdist(p,p)
    ranks = d.argsort(1).argsort(1)-1
    ids = d.argsort(1)[:,1:1+KG]
    original, order = seeds.sort(1,descending=True)
    details = []
    for step, fast in iterate(seeds,ids):
        if step:
            order = upstream(order,original,ranks,KG=KG,KQ=KQ,beta=BETA)
        top_agreement = float((fast[:,:100].sort(1).values == order[:,:100].sort(1).values).all(1).float().mean())
        rank_agreement = float((fast == order).float().mean())
        details.append({"step":step, "whole_rank_fraction":rank_agreement,"exact_top100_set_fraction":top_agreement})
        if top_agreement < 1 or rank_agreement < .99:
            raise ValueError("Vectorized reranker differs from published functions on a real-gallery fixture")
    return details


@torch.inference_mode()
def run(w,workers):
    out = w/"results"/FAMILY
    kd,_ = load_kd(w)
    if hashlib.sha256(inspect.getsource(kd).strip().encode()).hexdigest()!=KD_FUNCTION_SHA:
        raise ValueError("Unexpected evaluator")
    upstream = upstream_function()
    for dataset in ("sketchy","tuberlin"):
        if dataset=="tuberlin":
            atomic_json(out/"freeze.json", {"revision":revision(),"upstream":UPSTREAM,
                        "kg":KG,"kq":KQ,"beta":BETA,"iterations":STEPS,
                        "scores_before_transfer":{s:checksum(out/(s+".json")) for s in ("sketchy_1","sketchy_2")}})
        cfg,_,queries,_,galleries,qraw,praw = global_inputs(w,dataset)
        words = read_words(cfg.vocabulary)
        _,scale = bare_words(cfg,words)
        source,target=(v.cuda() for v in load_words(cfg,words))
        for split,(qi,ql) in queries.items():
            pi,pl = galleries[split]
            if (len(ql),len(pl),len(pl.unique()))!=SIZES[split] or digest(UNSEEN_CLASSES[split])!=SPLIT_HASHES[split]:
                raise ValueError("Unexpected benchmark split")
            q,p = qraw[qi].cuda(),praw[pi].cuda()
            official = check_reference(p,q,upstream)
            ids, self_rank0 = gallery_index(p)
            probs = torch.softmax((p @ target.T) * scale, 1)
            replay = Replay(kd,pl,split)
            values = {}
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for start in range(0,len(q),96):
                    query = q[start:start+96]
                    labels = ql[start:start+len(query)]
                    lexical = emission(query,source,scale) @ probs.T
                    seeds = {"rgb":-torch.cdist(query,p),"word":lexical,
                             "word_scaled":lexical/lexical.amax(1,keepdim=True)}
                    for name,seed in seeds.items():
                        for step,order in iterate(seed,ids):
                            if step in (0,1,5) and (name!="word_scaled" or step>0):
                                key = name+"_"+str(step)
                                values.setdefault(key,[]).append(replay.score(order,labels,pool))
                    if start%1920==0:
                        print(split,start+len(query),"/",len(q),flush=True)
            values = {k:torch.cat(v) for k,v in values.items()}
            if any(v.shape!=(len(ql),2) for v in values.values()):
                raise ValueError("Incomplete scores")
            path = out/(split+".pt")
            save_tensor({"query_labels":ql,"metrics":values},path)
            record = {"protocol":{"split":split,"original_kd_queries_per_method":len(q),"gallery":len(p),
                                  "classes":len(pl.unique()),"checkpoint":CHECKPOINT,"kd_sha256":KD_FUNCTION_SHA,
                                  "vocab_sha256":checksum(cfg.vocabulary),"upstream":UPSTREAM,"revision":revision(),
                                  "gpu":torch.cuda.get_device_name()},"map_k":replay.cutoffs[0],"p_k":replay.cutoffs[1],
                      "scores":{k:dict(zip(("mAP","precision"),v.mean(0).tolist())) for k,v in values.items()},
                      "vs_direct_words":{k:paired_delta(v,values["word_0"],ql) for k,v in values.items()},
                      "actual_gallery_fixture":official,"self_ranked_first":self_rank0,"per_query_sha256":checksum(path)}
            feedback=torch.load(w/"results"/"gallery-lexical-feedback"/(split+".pt"),weights_only=True)
            if not torch.equal(ql,feedback["query_labels"]):
                raise ValueError("Feedback and ICFRR query orders differ")
            record["diffusion_family_vs_icfrr"]={
                name:{k:paired_delta(feedback["metrics"][name],v,ql) for k,v in values.items() if k.endswith("_5")}
                for name in ("diffusion","inverse_2","inverse_4","distinct_witness","independent_words")}
            atomic_json(out/(split+".json"),record)
            print(split,record["scores"],flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    parser.add_argument("--workers",type=int,default=min(10,len(os.sched_getaffinity(0))))
    args=parser.parse_args()
    run(args.workspace,args.workers)
