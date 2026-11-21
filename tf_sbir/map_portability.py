"""Stress-test the locked inductive photo index and public word bank on Colab."""

import argparse
from pathlib import Path
import zipfile

import torch

from .capture_query_metrics import evaluate_with_details
from .concept_benchmark import atomic_json, checksum, read
from .concept_likelihood import ranked_scores
from .conditional_semantic_index import bare_words
from .crossdomain_benchmark import revision
from .data import UNSEEN_CLASSES
from .encoding import save_tensor
from .foundation import load_words
from .map_diagnosis import map_values
from .map_followup import signed
from .retrieval_diagnosis import global_inputs
from .retrieval_evidence import paired_delta, tensor
from .semantic import read_words
from .shared_concept_likelihood import emission, fit_prior, posterior, target_inputs


FAMILY = "neighborhood-calibrated-likelihood"
PHOTO_SEED = 15517
WORD_SEED = 9241


def partition(n, seed):
    permutation = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    return [part.sort().values for part in permutation.tensor_split(2)]


def reference_graph(photos, ref_ids, config):
    count = len(photos)
    k = config["k"]
    reference = photos[ref_ids]
    positions = torch.full((count,), -1, dtype=torch.long, device=photos.device)
    positions[ref_ids] = torch.arange(len(reference), device=photos.device)
    columns, values = [], []
    for start in range(0, count, 512):
        similarities = photos[start:start + 512] @ reference.T
        own = positions[start:start + len(similarities)]
        rows = (own >= 0).nonzero().flatten()
        similarities[rows, own[rows]] = -torch.inf
        nearest = similarities.topk(k, dim=1)
        columns.append(nearest.indices)
        values.append(nearest.values.clamp_min(0))
    columns = torch.cat(columns).flatten().int()
    distances = 2 * (1 - torch.cat(values)).clamp_min(0)
    if config["weighting"] == "cosine":
        # Learn the kernel width only on the reference fold, even for new photos.
        width = .36 * distances[ref_ids].sqrt().mean().square().clamp_min(1e-12)
        weights = (-distances / width).softmax(1)
    else:
        weights = torch.full_like(distances, 1 / k)
    rows = torch.arange(0,(count + 1)*k,k,dtype=torch.int32,device=photos.device)
    return torch.sparse_csr_tensor(rows,columns,weights.flatten(),size=(count,len(ref_ids)),check_invariants=False)


def photo_emissions(photos, bank, scale):
    a = target_inputs(photos,bank,scale)
    a /= a.sum(1,keepdim=True)
    return a


def build_index(photos, ref_ids, target, scale, graph, config, given_prior=None):
    a = photo_emissions(photos[ref_ids], target, scale)
    if given_prior is None:
        priors, _ = fit_prior(a)
        prior = priors[config["estimator"]]
    else:
        prior = given_prior
    if config["calibration"] == "posterior":
        return torch.sparse.mm(graph, posterior(a, prior))
    return posterior(torch.sparse.mm(graph, a), prior)


def official(q, ql, pl, source, scale, index, split, mode, complements):
    smaller = {name: [] for name in complements}
    device_labels = pl.cuda()
    def rows():
        for first in range(0, len(q), 96):
            features = q[first:first + 96]
            scores = emission(features,source,scale) @ index.T
            if not torch.isfinite(scores).all():
                raise ValueError("Non-finite retrieval scores")
            for key, ids in complements.items():
                stats, _, _, _ = map_values(scores[:,ids], ql[first:first+len(features)].cuda(),device_labels[ids],split)
                smaller[key].append(stats.cpu())
            yield from ranked_scores(scores).cpu()
    (_,_,mk,pk), details = evaluate_with_details(rows(),ql,pl,split,f"{split} {mode}",5 if split!="sketchy_2" else 3)
    expected = (200,200) if split == "sketchy_2" else (0,100)
    if (mk,pk) != expected or details.shape != (len(ql),2):
        raise ValueError("Wrong retrieval evaluation scope")
    return details, {k: torch.cat(v) for k,v in smaller.items()}


def means(values):
    return dict(zip(("mAP","precision"),values.mean(0).tolist(),strict=True))


def name_effect(words, halves, classes, labels, half_details):
    def plain(term):
        return " ".join(term.casefold().replace("_"," ").replace("-"," ").split())
    banks=[{plain(words[i]) for i in ids.tolist()} for ids in halves]
    names=[plain(c) for c in classes]
    in_a,in_b=[torch.tensor([name in group for name in names]) for group in banks]
    eligible=(in_a ^ in_b)[labels]
    if not eligible.any():
        return {"classes":0,"queries":0}
    which=in_a[labels][:,None]
    present=torch.where(which,half_details[0],half_details[1])[eligible]
    absent=torch.where(which,half_details[1],half_details[0])[eligible]
    _,cluster=labels[eligible].unique(return_inverse=True)
    return {"classes":int((in_a ^ in_b).sum()),"queries":int(eligible.sum()),
            "present":means(present),"absent":means(absent),"present_minus_absent":paired_delta(present,absent,cluster)}


@torch.inference_mode()
def run(w):
    results=w/"results"
    decision,signature=signed(results/"map-study/joint-freeze.json")
    if decision["primary"]!=FAMILY:
        raise ValueError("Pair of perturbations was prespecified for the neighborhood model")
    lock,lock_sha=signed(results/FAMILY/"freeze.json")
    if decision["families"][FAMILY]["freeze"]!=lock_sha:
        raise ValueError("The method lock does not match the selection")
    config=lock["winner"]
    out=results/"map-portability"
    out.mkdir(exist_ok=True)
    report={"revision":revision(),"joint_sha256":signature,"configuration":config,"photo_seed":PHOTO_SEED,"word_seed":WORD_SEED,
            "protocol_sha256":checksum(Path(__file__).resolve().parents[1]/"docs/MAP_PORTABILITY_PROTOCOL_VI.md"),"splits":{}}
    sketchy_prior=None
    for dataset in ("sketchy","tuberlin"):
        args,_,queries,_,galleries,sketches,photos=global_inputs(w,dataset)
        words=read_words(args.vocabulary)
        source,target=(x.cuda() for x in load_words(args,words))
        _,scale=bare_words(args,words)
        word_halves=partition(len(words),WORD_SEED)
        if dataset=="sketchy":
            a=photo_emissions(photos.cuda(),target,scale)
            priors,stat=fit_prior(a)
            sketchy_prior=priors[config["estimator"]]
            report["source_prior"]={"unique_gallery_photos":len(photos),"fit":stat}
            del a,priors
        for split,(qi,ql) in queries.items():
            pi,pl=galleries[split]
            q,p=sketches[qi].cuda(),photos[pi].cuda()
            all_ids=torch.arange(len(p),device=p.device)
            all_words=torch.arange(len(words),device=p.device)
            halves=[h.cuda() for h in partition(len(p),PHOTO_SEED)]
            full_graph=reference_graph(p,all_ids,config)
            variants=[("full",all_ids,all_words,None),
                      ("photos_a",halves[0],all_words,None),("photos_b",halves[1],all_words,None),
                      ("words_a",all_ids,word_halves[0].cuda(),None),("words_b",all_ids,word_halves[1].cuda(),None)]
            if split=="tuberlin":
                variants.append(("sketchy_prior",all_ids,all_words,sketchy_prior))
            sub="heldout" if dataset=="tuberlin" else "sketchy"
            measured=tensor(results/FAMILY/sub/"per-query"/f"{split}-{lock['winner_id']}.pt")
            table,partial_base,word_results={},{},[]
            for mode,reference,vocab,given in variants:
                graph=full_graph if reference is all_ids else reference_graph(p,reference,config)
                index=build_index(p,reference,target[vocab],scale,graph,config,given)
                complement={}
                if mode in ("full","photos_a"): complement["photos_a"]=halves[1]
                if mode in ("full","photos_b"): complement["photos_b"]=halves[0]
                points,small=official(q,ql,pl,source[vocab],scale,index,split,mode,complement)
                record={"reference_photos":len(reference),"ranked_photos":len(pl),"words":len(vocab),"queries":len(ql),
                        **means(points),"delta_vs_frozen":paired_delta(points,measured,ql)}
                if mode=="full":
                    error=(points-measured).abs()
                    record["mean_abs_AP_P_difference"]=error.mean(0).tolist()
                    record["max_abs_AP_P_difference"]=error.amax(0).tolist()
                    if float(error.mean(0).max())>5e-5:
                        raise ValueError("Independent implementation did not reproduce the locked method")
                    partial_base=small
                if mode.startswith("photos_"):
                    record["only_held_out_images"]={"gallery":len(complement[mode]),"half_index":means(small[mode]),
                        "original_index":means(partial_base[mode]),"delta":paired_delta(small[mode],partial_base[mode],ql)}
                if mode.startswith("words_"): word_results.append(points)
                table[mode]=record
                save_tensor(points,out/"per-query"/f"{split}-{mode}.pt")
                del index,graph
                print(split,mode,record["mAP"],record["precision"],flush=True)
            report["splits"][split]={"experiments":table,
                "literal_name":name_effect(words,word_halves,UNSEEN_CLASSES[split],ql,word_results)}
            atomic_json(out/f"{split}.json",report["splits"][split])
            del full_graph
    atomic_json(out/"portability.json",report)
    artifact=w/"map-portability-evidence.zip"
    with zipfile.ZipFile(artifact,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(p for p in out.rglob("*") if p.is_file()):
            archive.write(path,path.relative_to(w))
    with zipfile.ZipFile(artifact) as archive:
        if archive.testzip() is not None:
            raise ValueError("Invalid portability archive")
        count=len(archive.namelist())
    atomic_json(w/"map-portability-archive-info.json",{"size":artifact.stat().st_size,"sha":checksum(artifact),"count":count})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(parser.parse_args().workspace)
