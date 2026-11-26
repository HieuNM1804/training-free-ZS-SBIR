"""Compare conditional completions with interpolated endpoints and original Sketchy identities."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F

from .benchmark import merge_inputs
from .concept_benchmark import atomic_json, checksum
from .conditional_completion import complete, full_donor_controls, random_frames
from .conditional_completion_study import (
    BATCH, DEFAULT_PROMPTS, DOCS, FAMILY, FRAMES, SEED, SKETCHY, Split, VOCABULARY_STUDY, class_interval,
    context, encode_prompt, evaluate, image_inputs, mean_percent, select,
)
from .crossdomain_benchmark import revision
from .crossdomain_inputs import runtime
from .encoding import load_frozen_clip, save_tensor
from .gallery_diffusion import nearest_gallery
from .query_partition_study import save_lock
from .retrieval_evidence import load_kd


WEIGHTS = (0., .0625, .125, .25, .5, .75, 1.)


def banks_for_controls(photos, text, scale, frames, r, k, weights):
    logits = scale * (photos @ text.T)
    native = logits.softmax(1)
    lower, upper = native.clone(), native.clone()
    endpoint = torch.zeros_like(native)
    chord = {w: torch.zeros_like(native) for w in weights}
    for basis in frames:
        observed = F.normalize(photos @ basis[:, :r], dim=1)
        neighbors, _ = nearest_gallery(observed, k)
        for start in range(0, len(photos), 96):
            sl = slice(start, min(start+96,len(photos)))
            ids = neighbors[sl]
            donor = native[ids]
            lower[sl] = torch.minimum(lower[sl], donor.amin(1))
            upper[sl] = torch.maximum(upper[sl], donor.amax(1))
            endpoint[sl].add_(donor.mean(1), alpha=1/len(frames))
            donor_logits = logits[ids]
            cosine = (photos[sl,None] * photos[ids]).sum(-1)
            for w, bank in chord.items():
                if w == 1.:
                    bank[sl].add_(native[sl], alpha=1/len(frames))
                elif w == 0.:
                    bank[sl].add_(donor.mean(1), alpha=1/len(frames))
                else:
                    norm = (w*w + (1-w)**2 + 2*w*(1-w)*cosine).sqrt()
                    blended = (w * logits[sl,None] + (1-w) * donor_logits) / norm[:,:,None]
                    bank[sl].add_(blended.softmax(-1).mean(1), alpha=1/len(frames))
    return chord, endpoint, lower, upper


def source_ids(split, paths):
    photos = {}
    for pos, index in enumerate(split.gallery_ids.tolist()):
        f = Path(paths[1][index])
        key = (f.parent.name, f.stem)
        if key in photos:
            raise ValueError("Ambiguous source photo")
        photos[key] = pos
    output = []
    for index in split.query_ids.tolist():
        f = Path(paths[0][index])
        stem, sep, number = f.stem.rpartition("-")
        output.append(photos.get((f.parent.name,stem),-1) if sep and number.isdigit() else -1)
    ids = torch.tensor(output)
    good = ids >= 0
    if not torch.equal(split.sketch_labels[good], split.photo_labels[ids[good]]):
        raise ValueError("Source photo category mismatch")
    return ids


def identities(q, bank, split, ids):
    """Rank the actual source within its true class and in the full gallery. Midrank handles ties."""
    good = torch.where(ids >= 0)[0]
    labels = split.photo_labels.cuda()
    out = []
    for idx in good.split(BATCH):
        source = ids[idx].cuda()
        scores = q[idx] @ bank.T.contiguous()
        native = scores.gather(1, source[:,None])
        same_class = labels[None] == labels[source,None]
        better = scores > native
        tied = scores == native
        class_rank = 1 + (better & same_class).sum(1) + ((tied & same_class).sum(1)-1) / 2
        all_rank = 1 + better.sum(1) + (tied.sum(1)-1) / 2
        win = 1 - (class_rank-1) / (same_class.sum(1)-1).clamp_min(1)
        out.append(torch.stack((win, 1/class_rank, class_rank<=10, 1/all_rank, all_rank<=10),1).cpu())
    return torch.cat(out)


def pack(split, text, kd, frames, lock, weights):
    q, native, photos, t, replay = context(split,text,kd)
    chord, donor, lower, upper = banks_for_controls(photos,t,split.scale,frames,lock["rank"],lock["k"],weights)
    controls = {f"chord-{w}":v for w,v in chord.items()}
    controls.update({f"posterior-{w}":w*native+(1-w)*donor for w in weights})
    return q,native,photos,t,replay,controls,donor,lower,upper


@torch.inference_mode()
def run():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    workspace = Path("/content/sbir")
    family = workspace/"results"/FAMILY
    root = family/"identity-audit"
    primary_lock = json.loads((family/"freeze.json").read_bytes())
    lock = primary_lock["config"]
    workers = min(4,len(os.sched_getaffinity(0)))
    cfg = runtime(workspace,"sketchy",batch=BATCH,workers=workers)
    words = json.loads((DOCS/VOCABULARY_STUDY/"freeze.json").read_bytes())["subsets"]["boxable"]["words"]
    model = load_frozen_clip(cfg.clip_checkpoint, torch.device("cuda"))
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    text = {p:encode_prompt(model,tokenizer,words,p) for p in DEFAULT_PROMPTS}
    kd,_ = load_kd(workspace)
    frames = random_frames(text[DEFAULT_PROMPTS[1]].size(1),FRAMES,SEED,torch.device("cuda"))
    data = image_inputs(cfg,SKETCHY,model)
    paths = [merge_inputs(cfg.root,SKETCHY,mode)[0] for mode in ("sketch","photo")]
    primary_key = primary_lock["sha256"]
    protocol = save_lock(root/"protocol.json",{"commit":revision(),"primary_lock":primary_key,"weights":WEIGHTS,
        "source_identity":"Match category and exact photo stem to each sketch; compare the original photo within its category.",
        "control_selection":"Each family separately maximizes equal-weight Sketchy mAP then precision. Use unchanged on TU.",
        "convex_witness":"Sum of word masses strictly above all endpoints, and missing mass below every endpoint. Endpoints include the owner and all its measured donors."})
    grids=[]
    for name in SKETCHY:
        split=Split.from_inputs(data,name)
        target=root/f"grid-{name}.json"
        if target.exists():
            grid=json.loads(target.read_bytes())
            if grid["protocol"]!=protocol:
                raise ValueError("Conflicting control grid")
            grids.append(grid["metrics"])
            continue
        q,native,photos,t,replay,controls,*rest=pack(split,text,kd,frames,lock,WEIGHTS)
        grid={key:mean_percent(evaluate(split,q,bank,replay)) for key,bank in controls.items()}
        atomic_json(target,{"protocol":protocol,"metrics":grid})
        grids.append(grid)
        del q,native,photos,t,replay,controls,rest
    chosen={arm:select(grids,arm) for arm in ("chord","posterior")}
    signature=save_lock(root/"freeze.json",{"commit":revision(),"protocol":protocol,"primary_lock":primary_key,
        "chosen":chosen,"grids":{s:checksum(root/f"grid-{s}.json") for s in SKETCHY}})
    print("IDENTITY AUDIT FROZEN",chosen,flush=True)
    weights=sorted({float(v.split("-")[1]) for v in chosen.values()})
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for name in (*SKETCHY,"tuberlin"):
            target=root/f"{name}.json"
            if target.exists():
                previous=json.loads(target.read_bytes())
                if previous["selection"]!=signature or checksum(target.with_suffix(".pt"))!=previous["per_query_sha256"]:
                    raise ValueError("Invalid matched-control result")
                continue
            if name=="tuberlin":
                del data
                cfg=runtime(workspace,"tuberlin",batch=BATCH,workers=workers)
                data=image_inputs(cfg,(name,),model)
            split=Split.from_inputs(data,name)
            q,native,photos,t,replay,controls,donor,lower,upper=pack(split,text,kd,frames,lock,weights)
            k=lock["k"]
            partial,_=complete(photos,t,split.scale,frames,lock["rank"],[k])
            primary=partial[k]
            above=(primary-upper).clamp_min(0).sum(1)
            below=(lower-primary).clamp_min(0).sum(1)
            sphere_rows={arm:evaluate(split,q,controls[key],replay,pool) for arm,key in chosen.items()}
            ids=source_ids(split,paths) if name in SKETCHY else None
            identity={}
            if ids is not None:
                index,_=nearest_gallery(photos,lock["donor_k"])
                full=full_donor_controls(photos,t,split.scale,index,[lock["donor_k"]])[lock["donor_k"]]
                for arm,bank in (("dot",native),("completion",primary),("matched_visible",donor),("full_donor_best",full),
                                 *((arm,controls[key]) for arm,key in chosen.items())):
                    identity[arm]=identities(q,bank,split,ids)
            artifact=target.with_suffix(".pt")
            save_tensor({"qi":split.query_ids,"pi":split.gallery_ids,"labels":split.sketch_labels,
                         "rows":sphere_rows,"source_indices":ids,"identity":identity,
                         "upper_gap":above.cpu(),"lower_gap":below.cpu()},artifact)
            summary={"metrics":{arm:mean_percent(v) for arm,v in sphere_rows.items()},
                     "certified_upper_mass_mean":above.mean().item(),
                     "fraction_above_0.001_upper_gap":(above>.001).float().mean().item(),
                     "fraction_above_0.01_upper_gap":(above>.01).float().mean().item(),
                     "fraction_beyond_lower_or_upper_0.001":(torch.maximum(above,below)>.001).float().mean().item(),
                     "identity_fields":["within_class_pair_win","within_class_MRR","within_class_R10","all_MRR","all_R10"],
                     "identity":{arm:mean_percent(v) for arm,v in identity.items()},
                     "identity_delta_vs_each":{arm:class_interval(100*(identity["completion"][:,0]-v[:,0]),
                                                   split.sketch_labels[ids>=0],seed=SEED) for arm,v in identity.items()},
                     "source_queries":int((ids>=0).sum()) if ids is not None else None,
                     "queries":len(q),"gallery":len(photos),"classes":len(split.sketch_labels.unique()),
                     "cutoffs":replay.cutoffs,"selection":signature,"per_query_sha256":checksum(artifact)}
            atomic_json(target,summary)
            print(name,"identity and interpolation",summary["metrics"],summary["identity"],flush=True)
            del controls,donor,lower,upper,primary,partial,identity,sphere_rows


if __name__=="__main__":
    run()
