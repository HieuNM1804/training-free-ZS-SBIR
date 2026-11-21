"""Report full frozen scores, pair invariance, dictionary sensitivity and actual errors."""

import argparse
from pathlib import Path
from types import SimpleNamespace
import zipfile

import torch

from . import public_semantic_translation as translations
from . import public_word_calibration as calibration
from .capture_query_metrics import evaluate_with_details
from .concept_benchmark import atomic_json, checksum, read
from .concept_likelihood import ranked_scores
from .conditional_semantic_index import bare_words
from .crossdomain_benchmark import digest, revision
from .data import UNSEEN_CLASSES
from .encoding import global_cache_name
from .foundation import load_words
from .retrieval_diagnosis import global_inputs, stratified_sample
from .retrieval_evidence import paired_delta, tensor
from .semantic import read_words
from .shared_concept_likelihood import emission
from .strict_family_selection import FAMILY, UNIFORM, WINDOW


TRANSLATION = "public-semantic-translation"
NATIVE = "conditional-semantic-index"


def checked_lock(path):
    item = read(path)
    if item.pop("sha256") != digest(item):
        raise ValueError(f"Lock signature mismatch at {path}")
    return item


def archive(w):
    root = w/"results"
    name = "strict-public-evidence"
    final = w/(name+".zip")
    temp = final.with_suffix(".partial")
    files = [p for folder in (FAMILY,TRANSLATION,"strict-public-audit")
             for p in sorted((root/folder).rglob("*")) if p.is_file() and p.suffix in {".pt",".json",".csv"}]
    with zipfile.ZipFile(temp,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for p in files:
            z.write(p,p.relative_to(w).as_posix())
    temp.replace(final)
    atomic_json(w/(name+"-archive-info.json"),{"sha":checksum(final),"count":len(files),"size":final.stat().st_size})


def score_sample(scores, ql, pl, split):
    def rows():
        for block in scores.split(96):
            yield from ranked_scores(block).cpu()
    (_, _, _, _), details = evaluate_with_details(rows(), ql, pl, split, "dictionary " + split, 4)
    return details


class Scorer:
    def __init__(self, name, params, banks, scale, windows):
        self.name, self.scale = name, scale
        self.source, self.target, literal = banks
        self.q_weight = self.p_weight = None
        self.windows = windows
        self.params = params
        if name == TRANSLATION:
            self.channel = translations.Lexicon(literal, scale)
            self.config = translations.Config(**params)
        if name == FAMILY:
            self.config = calibration.Config(**params)
            model = calibration.Index(SimpleNamespace(source=self.source, target=self.target, scale=scale))
            self.q_weight, self.p_weight = model.weights(self.config)

    def score(self, q, p, window_ids):
        query = calibration.adjusted(emission(q, self.source, self.scale), self.q_weight, True)
        if self.name == TRANSLATION:
            query = query.lerp(self.channel.translate(query, self.config), self.config.mass)
        result = query.new_empty(len(query), len(p))
        step = 32 if self.name == WINDOW else 512
        for start in range(0, len(p), step):
            block = p[start:start+step]
            docs = (self.scale * (block @ self.target.T)).softmax(1)
            docs = calibration.adjusted(docs, self.p_weight, False)
            scores = query @ docs.T
            if self.name == WINDOW:
                view = self.windows[window_ids[start:start+len(block)]].to(block.device).float()
                posterior = (self.scale * (view.flatten(0, 1) @ self.target.T)).softmax(1)
                region = posterior.view(len(view), view.size(1), -1).mean(1)
                scores = scores.lerp(query @ region.T, self.params["mix"])
            result[:, start:start+len(block)] = scores
        return result


def full_results(root, split, ql, leaves, primary):
    entries, details = {}, {}
    for name, leaf in leaves.items():
        place = root / name / ("heldout" if split == "tuberlin" else "sketchy")
        file = place / f"{split}-{leaf}.json"
        row = read(file)
        values = tensor(place / "per-query" / f"{split}-{leaf}.pt")
        if values.shape != (len(ql), 2) or max(abs(float(values[:, i].mean())-row[k])
                    for i,k in enumerate(("mAP","precision"))) > 1e-6:
            raise ValueError(f"Invalid per-query data for {name} {split}")
        details[name] = values
        entries[name] = {k:row[k] for k in ("mAP","precision","queries","gallery","classes")}
        entries[name]["source_sha256"] = checksum(file)
        entries[name]["revision"] = row["revision"]
        if name in (FAMILY, TRANSLATION, WINDOW):
            audit = row["kd_check"]
            if max(audit["kd_minus_pipeline_max_abs_same_order_AP_P"]) != 0:
                raise ValueError("Official evaluator replay disagrees")
            entries[name]["kd_check"] = audit
            lock = checked_lock(root/name/"freeze.json")
            entries[name]["controls"] = {}
            for control, settings in lock["controls"].items():
                cid = digest(settings)[:16]
                r = read(place / f"{split}-{cid}.json")
                entries[name]["controls"][control] = {k:r[k] for k in ("mAP","precision")}
    for name in entries:
        entries[name]["primary_minus_method"] = paired_delta(details[primary], details[name], ql)
        entries[name]["method_minus_uniform"] = paired_delta(details[name], details[UNIFORM], ql)
    return entries, details


def short_path(path):
    return "/".join(Path(path).parts[-3:])


def error_pairs(scores, native_scores, qnames, pnames, ql, pl, words, q, p, source, target, scale):
    top = native_scores.argmax(1).cpu()
    relevant = ql[:,None] == pl[None,:]
    correct_photo = native_scores.masked_fill(~relevant.to(scores.device), -torch.inf).argmax(1).cpu()
    ids = torch.arange(len(q))
    wrong = pl[top] != ql
    flipped = (scores[ids,correct_photo] > scores[ids,top]).cpu()
    current = scores.argmax(1).cpu()
    by_type = {"native_wrong_now_top1_correct": wrong & (pl[current] == ql),
               "native_correct_now_top1_wrong": ~wrong & (pl[current] != ql),
               "stays_wrong_at_top1": wrong & (pl[current] != ql)}
    examples = {}
    for name, mask in by_type.items():
        pool = torch.where(mask)[0]
        draw = torch.randperm(len(pool),generator=torch.Generator().manual_seed(8592))[:3]
        examples[name] = []
        for j in pool[draw].tolist():
            candidates = [int(top[j]), int(correct_photo[j]), int(current[j])]
            qlikes = emission(q[j:j+1], source, scale)
            photo = (scale*(p[candidates]@target.T)).softmax(1)
            terms = (qlikes*photo).argmax(1).tolist()
            examples[name].append({"query":short_path(qnames[j]), "native_wrong_or_top":short_path(pnames[candidates[0]]),
                "native_best_positive":short_path(pnames[candidates[1]]), "new_top":short_path(pnames[candidates[2]]),
                "shared_exact_words_on_three_pairs":[words[i] for i in terms],
                "native_scores":[float(native_scores[j,i]) for i in candidates],
                "pair_method_scores":[float(scores[j,i]) for i in candidates]})
    return {"native_wrong":int(wrong.sum()),"native_wrong_pair_flipped":int((wrong&flipped).sum()),
            "corrected_top1":int(by_type["native_wrong_now_top1_correct"].sum()),
            "regressed_top1":int(by_type["native_correct_now_top1_wrong"].sum()), "examples":examples}


@torch.inference_mode()
def run(w):
    root = w/"results"
    joint = checked_lock(root/translations.DECISION)
    primary = joint["primary"]
    leaves = {n:v["name"] for n,v in joint["families"].items()} | {NATIVE:"native"}
    params = {} if primary == UNIFORM else checked_lock(root/primary/"freeze.json")["winner"]
    dest=root/"strict-public-audit"
    output={"revision":revision(),"primary":primary,"parameters":params,"criterion":joint["rule"],
            "decision":checksum(root/translations.DECISION),"sample_seed":4379,"words_split_seed":9241,"splits":{}}
    for dataset in ("sketchy","tuberlin"):
        a, qp, qs, pp, gs, raw, photos = global_inputs(w,dataset)
        words=read_words(a.vocabulary)
        literal, scale=bare_words(a,words)
        src,tgt=(x.cuda() for x in load_words(a,words))
        literal=literal.cuda()
        windows=None
        if primary==WINDOW:
            cache=a.cache/global_cache_name(a.root,pp,"photo","winclip-2-3-5-v1","fp16",a.clip_checkpoint)
            windows=torch.load(cache,weights_only=True,mmap=True)[f"masked{params['side']}"]
        for split,(qi,ql) in qs.items():
            pi,pl=gs[split]
            full, details=full_results(root,split,ql,leaves,primary)
            sample=stratified_sample(ql,24,4379)
            q=raw[qi[sample]].cuda()
            p=photos[pi].cuda()
            ql_sample=ql[sample]
            stats={}
            bank_ids=torch.randperm(len(words),generator=torch.Generator().manual_seed(9241))
            definitions={"full":(src,tgt,literal), "bare_prompts":(literal,literal,literal)}
            for i,section in enumerate(bank_ids.tensor_split(2)):
                definitions[f"half_{i}"]=(src[section],tgt[section],literal[section])
            baseline_sample=None
            for name,banks in definitions.items():
                scorer=Scorer(primary,params,banks,scale,windows)
                scores=scorer.score(q,p,pi)
                measured=score_sample(scores,ql_sample,pl,split)
                item={"mAP":float(measured[:,0].mean()),"precision":float(measured[:,1].mean()),"words":len(banks[0])}
                if name=="full":
                    baseline_sample=measured
                    official=details[primary][sample]
                    item["mean_minus_official_pp"]=(100*(measured-official).double().mean(0)).tolist()
                    if (measured.mean(0)-official.mean(0)).abs().max()>.0005:
                        raise ValueError("Independent reconstruction differs materially")
                    ids=stratified_sample(pl,8,9057).flip(0)
                    pairs=stratified_sample(ql_sample,2,1183).flip(0)
                    pruned=scorer.score(q[pairs],p[ids],pi[ids])
                    suberr=float((pruned-scores[pairs][:,ids]).abs().max())
                    singleerr=max(float((scorer.score(q[j:j+1],p[k:k+1],pi[k:k+1])[0,0]-scores[j,k]).abs())
                                  for j,k in zip(pairs[:5].tolist(),ids[:5].tolist()))
                    item["invariance"]={"max_subgallery_abs":suberr,"max_singleton_abs":singleerr,
                        "paired_query_count":len(pairs),"paired_photo_count":len(ids),"singletons":5}
                    if max(suberr,singleerr)>1e-4:
                        raise ValueError("Score unexpectedly depends on batch composition")
                    item["native_witnesses"]=error_pairs(scores,q@p.T,[qp[i] for i in qi[sample].tolist()],
                        [pp[i] for i in pi.tolist()],ql_sample,pl,words,q,p,src,tgt,scale)
                else:
                    item["versus_full"]=paired_delta(measured,baseline_sample,ql_sample)
                stats[name]=item
                del scorer,scores
            other={}
            if split=="tuberlin":
                norm=lambda name: name.replace("_"," ").replace("-"," ").lower().strip()
                seen={norm(c) for s in ("sketchy_1","sketchy_2") for c in UNSEEN_CLASSES[s]}
                unseen=[i for i,c in enumerate(UNSEEN_CLASSES[split]) if norm(c) not in seen]
                mask=torch.isin(ql,torch.tensor(unseen))
                other={"queries":int(mask.sum()),"classes":len(unseen),"all_gallery":len(pl),
                       "AP_P":{n:v[mask].mean(0).tolist() for n,v in details.items()}}
            output["splits"][split]={"full":full,"dictionary_sample":{"queries":len(q),"gallery":len(pl),
                                              "classes":len(ql.unique()),"conditions":stats},"distinct_names":other}
            atomic_json(dest/f"{split}.json",output["splits"][split])
            print(split, primary,full[primary]["mAP"],stats["full"]["invariance"],flush=True)
    atomic_json(dest/"report.json",output)
    archive(w)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace",type=Path,default=Path("/content/sbir"))
    run(p.parse_args().workspace)
