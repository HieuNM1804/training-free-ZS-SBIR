"""Independent checks and paired analyses of frozen, full-gallery retrieval results on Colab."""

import argparse
from collections import Counter
import csv
from dataclasses import asdict
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F

from .concept_benchmark import atomic_json, base_blocks, checksum, read
from .concept_likelihood import ConceptIndex, Config, photo_graph, ranked_scores
from .crossdomain_benchmark import SPLIT_HASHES, digest
from .crossdomain_inputs import CHECKPOINT
from .data import UNSEEN_CLASSES
from .foundation import load_words
from .retrieval_diagnosis import global_inputs, stratified_sample
from .reverse_prompt import PhotoIndex, Setting
from .semantic import read_words


def tensor(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def within_class_percentile(preference, pl):
    out = torch.zeros_like(preference)
    for label in pl.unique():
        ids = (pl == label).nonzero().flatten()
        sorted_ids = ids[preference[ids].argsort(stable=True)]
        out[sorted_ids] = torch.linspace(0., 1., len(ids), device=pl.device)
    return out


def distribution(top, ql, pl, names, percentile):
    labels = pl[top]
    wrong = labels != ql[:, None]
    hist = top[wrong].bincount(minlength=len(pl))
    hist_top = top[:, 0][wrong[:, 0]].bincount(minlength=len(pl))
    m = max(1, round(len(pl) * .01))
    common = Counter(names[k] for k in labels[:, 0][wrong[:, 0]].tolist())
    return {
        "top1_correct": int((~wrong[:, 0]).sum()), "errors_at_1": int(wrong[:, 0].sum()),
        "wrong_at_10": int(wrong.sum()), "unique_wrong_top1_photos": int((hist_top > 0).sum()),
        "wrong_top10_share_top1pct": float(hist.topk(m).values.sum() / hist.sum().clamp_min(1)),
        "wrong_top1_categories": common.most_common(4),
        "mean_within_class_style_percentile_wrong_top1": float(percentile[top[:, 0][wrong[:, 0]]].mean()),
    }


def projection_probe(q, p, ql, pl, direction, cutoff):
    q = F.normalize(q - (q @ direction)[:, None] * direction)
    photos = F.normalize(p - (p @ direction)[:, None] * direction)
    ids = (q @ photos.T).topk(cutoff).indices
    relevant = pl[ids] == ql[:, None]
    return {"precision": float(relevant.float().mean()), "top1": float(relevant[:, 0].float().mean())}


def native_evidence(q, p, ql, pl, paths, names, words, s):
    direction = F.normalize((words[0] - words[1]).mean(0), dim=0)
    style = p @ direction
    percentile = within_class_percentile(style, pl)
    top, first, positive = [], [], []
    all_ids = torch.arange(len(p), device=p.device)[None]
    for start in range(0, len(q), 128):
        scores = q[start:start + 128] @ p.T
        relevant = ql[start:start + len(scores), None] == pl
        maximum, correct = scores.masked_fill(~relevant, -torch.inf).max(1)
        first.append((1 + (scores > maximum[:, None]).sum(1)
                      + ((scores == maximum[:, None]) & (all_ids < correct[:, None])).sum(1)).cpu())
        positive.append(correct.cpu())
        top.append(scores.topk(10).indices)
    top, first = torch.cat(top), torch.cat(first)
    data = distribution(top, ql, pl, names, percentile)
    wrong = top[:, 0][pl[top[:, 0]] != ql]
    items = wrong.bincount(minlength=len(p)).topk(5)
    data["frequent_wrong_top1"] = [{"photo": "/".join(Path(paths[i]).parts[-3:]), "count": int(n),
                                    "within_class_style_percentile": float(percentile[i])}
                                   for i, n in zip(items.indices.tolist(), items.values.tolist())]
    data["mean_first_positive_rank"] = float(first.float().mean())
    data["median_first_positive_rank"] = int(first.median())
    sample = stratified_sample(ql.cpu(), 24, 1703)
    k = 200 if s == "sketchy_2" else 100
    random = torch.randn(len(direction), generator=torch.Generator().manual_seed(713)).to(p.device)
    data["remove_one_domain_direction_sample"] = projection_probe(q[sample], p, ql[sample], pl, direction, k)
    data["remove_one_random_direction_sample"] = projection_probe(q[sample], p, ql[sample], pl, F.normalize(random, dim=0), k)
    return data, top, first, torch.cat(positive), percentile


def paired_delta(after, before, labels):
    difference = (after.double() - before.double()) * 100
    counts = labels.bincount().double()
    sums = torch.zeros(len(counts), 2, dtype=torch.double).index_add_(0, labels, difference)
    draws = torch.randint(len(counts), (5000, len(counts)), generator=torch.Generator().manual_seed(8042))
    estimates = sums[draws].sum(1) / counts[draws].sum(1)[:, None]
    lo, hi = estimates.quantile(torch.tensor([.025, .975], dtype=torch.double), dim=0)
    return {name: {"delta_pp": float(difference[:, i].mean()), "ci95_class": [float(lo[i]), float(hi[i])],
                   "classes_up": int((sums[:, i] > 0).sum()), "classes": len(counts)}
            for i,name in enumerate(("mAP", "precision"))}


def load_kd(workspace):
    path = workspace / "inquiry_kd_reference.py"
    spec = importlib.util.spec_from_file_location("kd_reference_for_sbir", path)
    imported = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imported)
    return imported._retrieval_metrics, checksum(path)


def check_record(row, folder, ql, pl, dataset):
    if (row["queries"], row["gallery"], row["classes"]) != (len(ql),len(pl),len(ql.unique())):
        raise ValueError("Invalid gallery or sketch count")
    if (row["p_k"],row["map_k"]) != ((200,200) if dataset == "sketchy_2" else (100,0)):
        raise ValueError("Incorrect evaluator")
    if row["checkpoint_sha256"] != CHECKPOINT or row["split_sha256"] != SPLIT_HASHES[dataset]:
        raise ValueError("Invalid checkpoint or split")
    details = tensor(folder / "per-query" / (dataset + "-" + row["method"] + ".pt"))
    if details.shape != (len(ql), 2) or not torch.isfinite(details).all():
        raise ValueError("Invalid per-query measurements")
    if max(abs(details[:, i].mean().item() - row[k]) for i,k in enumerate(("mAP", "precision"))) > 1e-6:
        raise ValueError("Reported mean differs from retained queries")
    return details


def verify_locks(root):
    joint = read(root / "inquiry-joint-freeze.json")
    if joint.pop("sha256") != digest(joint):
        raise ValueError("Damaged joint freeze")
    for family, entry in joint["families"].items():
        lock = read(root / family / "freeze.json")
        if lock.pop("freeze_sha256") != entry["freeze_sha256"] or digest(lock) != entry["freeze_sha256"]:
            raise ValueError("Family checksum failed")
        for name,h in lock.get("sketchy_rows", lock.get("sketchy_scores_sha256",{})).items():
            if checksum(root / family / "sketchy" / name) != h:
                raise ValueError("Sketchy score changed after freeze")
        held = [read(p) for p in (root / family / "heldout").glob("tuberlin-*.json")]
        names = {digest(r["parameters"]) for r in held if r["method"] != "native-clip"}
        if (entry["tu_rows_prelock"] != 0 or not held or any(r["freeze_sha256"] != entry["freeze_sha256"]
                or r["revision"] != entry["revision"] or r["exploratory_unseen"] for r in held)
                or names != {digest(c) for c in lock["controls"].values()}):
            raise ValueError("Held-out scores do not match the predeclared controls")
    return joint


@torch.inference_mode()
def report(w):
    root = w / "results"
    joint = verify_locks(root)
    family = joint["primary"]
    kd, kd_hash = load_kd(w)
    directional = family == "reverse-prompt-diffusion"
    locked = read(root / family / "freeze.json")
    c = (Setting if directional else Config)(**locked["winner"])
    output = {"primary": family, "configuration": asdict(c), "kd_source_sha256": kd_hash,
              "joint_freeze_sha256": digest(joint), "splits": {}}
    all_rows, categories = [], []
    for dataset in ("sketchy", "tuberlin"):
        args, qp, queries, pp, gallery, raw, rgb = global_inputs(w, dataset)
        texts = tuple(t.to("cuda") for t in load_words(args, read_words(args.vocabulary)))
        for s, (qi, ql) in queries.items():
            pi, pl = gallery[s]
            sub = "heldout" if s == "tuberlin" else "sketchy"
            folder = root / family / sub
            all_details, controls = {}, {}
            for branch in joint["families"]:
                archive = root / branch / sub
                recorded_labels = tensor(archive / f"{s}-labels.pt")
                if not torch.equal(ql,recorded_labels["query"]) or not torch.equal(pl,recorded_labels["gallery"]):
                    raise ValueError("Query or photo order differs between methods")
                for fp in sorted(archive.glob(s+"-*.json")):
                    row = read(fp)
                    details = check_record(row, archive, ql, pl, s)
                    key = branch+"/"+row["method"]
                    all_details[key] = details
                    all_rows.append([s, branch, row["method"], row["precision"], row["mAP"], row["revision"], row["freeze_sha256"]])
                    if branch == family:
                        controls[digest(row["parameters"])] = key
            details = all_details[family+"/"+c.name]
            native = all_details[family+"/native-clip"]
            q, p = raw[qi].cuda(), rgb[pi].cuda()
            labels, glabels = ql.cuda(), pl.cuda()
            names = UNSEEN_CLASSES[s]
            paths, qpaths = [pp[i] for i in pi.tolist()], [qp[i] for i in qi.tolist()]
            native_stats, native_top, ranks, pos, pct = native_evidence(q, p, labels, glabels, paths, names, texts, s)
            if directional:
                idx = PhotoIndex(p, *texts)
                def score_rows(query):
                    return idx.scores(query, c)
                final_top = tensor(folder / "top10" / f"{s}-{c.name}.pt").cuda().long()
            else:
                idx = ConceptIndex(p, *texts, c.target_temp)
                graph = photo_graph(p, c.neighbors) if c.walk else None
                def score_rows(query):
                    return base_blocks(query, p, c, idx, graph)
                final_top = torch.cat([scores.topk(10).indices for scores in score_rows(q)])
            official_first = tensor(folder / "top1" / f"{s}-native-clip.pt").cuda().long()
            fixed, old = glabels[final_top[:,0]] == labels, glabels[official_first] == labels
            checks = {}
            ka, kp, mk, pk = kd(q, p, ql, pl, s)
            checks["original_kd_native_full"] = {"mAP": float(ka), "precision": float(kp), "map_k":mk,"p_k":pk,
                "max_abs_difference": max(abs(float(ka)-float(native[:,0].mean())),abs(float(kp)-float(native[:,1].mean())))}
            checks["native_top1_difference_dot_vs_cosine"] = int((official_first != native_top[:,0]).sum())
            chosen = stratified_sample(ql,1,830)
            scores = torch.cat(list(score_rows(q[chosen])))
            batch_diff = float((scores[0]-next(score_rows(q[chosen[:1]]))[0]).abs().max())
            checks["max_single_vs_batch_abs_score_first_query"] = batch_diff
            ranked = ranked_scores(scores).cpu()
            vector_query = torch.tensor([[1.,0.]])
            diffs = []
            for x, j in zip(ranked, chosen.tolist()):
                vec = torch.stack((x,(1-x.square()).sqrt()),dim=1)
                ca,cp,_,_ = kd(vector_query,vec,ql[j:j+1],pl,s)
                diffs.append(max(abs(float(ca)-float(details[j,0])),abs(float(cp)-float(details[j,1]))))
            checks["independent_rank_replay"] = {"queries":len(diffs),"max_abs_difference":max(diffs),
                                                  "within_1e-6":sum(d<=1e-6 for d in diffs)}
            del idx
            comparisons = {"native": paired_delta(details,native,ql)}
            for name, value in locked["controls"].items():
                comparisons[name] = paired_delta(details,all_details[controls[digest(value)]],ql)
            for other, entry in joint["families"].items():
                key = other+"/"+entry["scores"][0]["method"]
                comparisons[other] = paired_delta(details,all_details[key],ql)
            history = {}
            for file in sorted((w/"inquiry-historical").glob(s+"-*.json")):
                row = read(file)
                if row["checkpoint_sha256"] != CHECKPOINT or row["queries"] != len(ql) or row["gallery"] != len(pl):
                    raise ValueError("Historical measurement mismatched")
                old_details = tensor(file.with_suffix(".pt"))
                if old_details.shape != details.shape or max(abs(float(old_details[:,i].mean())-row[k])
                        for i,k in enumerate(("mAP","precision"))) > 1e-6:
                    raise ValueError("Historical result not validated")
                history[row["method"]] = {"precision":row["precision"],"mAP":row["mAP"],
                                            "delta":paired_delta(details,old_details,ql)}
            examples = []
            for label, mask in [("rescued",fixed & ~old),("regressed",old & ~fixed),("still_wrong",~old & ~fixed)]:
                ids = torch.where(mask.cpu())[0]
                for j in ids[torch.randperm(len(ids),generator=torch.Generator().manual_seed(840))[:4]].tolist():
                    path_name = lambda path: "/".join(Path(path).parts[-3:])
                    examples.append({"type":label,"query":path_name(qpaths[j]),"native_top1":path_name(paths[int(official_first[j])]),
                                     "first_correct_rank_native":int(ranks[j]),"first_correct_native":path_name(paths[int(pos[j])]),
                                     "method_top1":path_name(paths[int(final_top[j,0])])})
            for i, name in enumerate(names):
                subset = ql == i
                categories.append([s,name,int(subset.sum()),int((pl==i).sum()),*native[subset].mean(0).tolist(),
                                   *details[subset].mean(0).tolist()])
            output["splits"][s] = {"queries":len(ql),"gallery":len(pl),"classes":len(names),
                "native":native_stats,"method":distribution(final_top,labels,glabels,names,pct),
                "top1_rescued":int((fixed & ~old).sum()),"top1_regressed":int((old & ~fixed).sum()),
                "native_AP_P":native.mean(0).tolist(),"method_AP_P":details.mean(0).tolist(),
                "checks":checks,"paired_comparisons":comparisons,"historical":history,"examples":examples}
            print(s,output["splits"][s]["method_AP_P"],checks,flush=True)
    dest = root / "inquiry-report"
    atomic_json(dest / "evidence.json",output)
    for name,header,rows in [
        ("configurations.csv", ["split","family","configuration","precision","mAP","revision","freeze_sha256"],all_rows),
        ("by-class.csv", ["split","class","queries","photos_in_class","native_mAP","native_P","method_mAP","method_P"],categories)]:
        with (dest/name).open("w",newline="") as f:
            cw=csv.writer(f); cw.writerow(header); cw.writerows(rows)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    report(parser.parse_args().workspace)
