"""Compare two-description nomination on human drawings and matched source photographs."""

import argparse
import json

import open_clip
from scipy.stats import rankdata
import torch
from torch.nn.functional import normalize

from .concept_benchmark import atomic_json, checksum
from .encoding import load_frozen_clip, save_tensor
from .kaggle_unseen_features import DEFAULT_CACHE, load_split
from .native_block_probe import SEED, deterministic
from .profiled_study import true_sources
from .semantic_donor_specificity import boot
from .wordnet_renaming_pilot import OUT, SPLITS, nominate
from .wordnet_renaming_text import describe, encode, inventory


@torch.inference_mode()
def run(name,revision,ceiling_revision):
    split,_,words = load_split(DEFAULT_CACHE,name)
    source = true_sources(split,DEFAULT_CACHE).cpu()
    if (source<0).any() or not torch.equal(split.sketch_labels,split.photo_labels[source]):
        raise ValueError("Wrong or missing exact source")
    src,inv = source.unique(return_inverse=True)
    class_of_src = split.photo_labels[src]
    labels = split.sketch_labels
    spans = []
    for i in (0,1):
        path=OUT/"ceiling"/f"{name}-{i}.pt"
        manifest=json.loads(path.with_suffix(".json").read_bytes())
        if manifest["revision"]!=ceiling_revision or manifest["sha256"]!=checksum(path):
            raise ValueError("Class utility table did not verify")
        spans.append(torch.load(path,weights_only=True,mmap=True))
    order=torch.cat([v["indices"] for v in spans]).argsort()
    table=torch.cat([v["values"] for v in spans])[order]
    lookup={t:i for i,t in enumerate(spans[0]["strings"])}
    if not all(torch.equal(v["classes"],labels) for v in spans):
        raise ValueError("Unexpected class labels")
    senses,_=inventory(words)
    canonical=[describe("photo",s.anonymous) for s in senses]
    class_ap=table[[lookup[t] for t in canonical],:,0].T.contiguous()
    # Ranks are computed over the actual menu of senses, keeping any tied AP together.
    quantile=torch.from_numpy((rankdata(class_ap.numpy(),method="average",axis=1)-.5)/len(senses))
    photos=normalize(split.photo[src].float(),dim=1)
    sketch=normalize(split.sketch.float(),dim=1)
    pooled=normalize(sketch+normalize(split.mirror.float(),dim=1),dim=1)
    model=load_frozen_clip(json.loads((DEFAULT_CACHE/"manifest.json").read_bytes())["checkpoint"],torch.device("cuda"))
    tok=open_clip.get_tokenizer("ViT-B-32")
    heads=torch.tensor([s.head for s in senses],device="cuda")
    cols=("first_AP","best_two_AP","better_percentile","contains_top_decile")
    rows,nominated={},{}
    for domain in ("drawing","photo"):
        bank=encode(model,tok,[describe(domain,s.anonymous) for s in senses])
        for key,image in (("sketch",sketch),("source",photos),("pooled",pooled)):
            if key=="pooled" and domain=="photo":
                continue
            pair=nominate(image,bank,heads).cpu()
            if key=="source":
                pair=pair[inv]
            values=class_ap[labels[:,None],pair]
            percentile=quantile[labels[:,None],pair].amax(1)
            tag=f"{key}_{domain}"
            nominated[tag]=pair
            rows[tag]=torch.stack((values[:,0],values.amax(1),percentile,(percentile>=.90).double()),dim=1)
    raw=[torch.load(OUT/f"{name}-{i}.pt",weights_only=True,mmap=True) for i in (0,1)]
    ord_pilot=torch.cat([v["positions"] for v in raw]).argsort()
    prior=torch.cat([v["pair"] for v in raw])[ord_pilot]
    agreement=(prior==nominated["pooled_drawing"]).all(1).double().mean().item()
    if agreement < .995:
        raise ValueError("Pilot candidate order not reproduced")
    unique,count=inv.unique(return_counts=True)
    if not torch.equal(unique,torch.arange(len(src))):
        raise ValueError("Non-contiguous subject indices")
    def subject_class_summary(delta):
        mean=torch.zeros(len(src),len(cols),dtype=torch.float64)
        mean.index_add_(0,inv,delta)
        mean/=count[:,None]
        byclass=torch.stack([mean[class_of_src==c].mean(0) for c in class_of_src.unique()])
        return {"query_points":(100*delta.mean(0)).tolist(),"balanced_subject_and_class":boot((100*byclass).numpy())}
    differences={}
    for domain in ("drawing","photo"):
        key=f"source_minus_sketch_same_{domain}_prompt"
        differences[key]=subject_class_summary(rows["source_"+domain]-rows["sketch_"+domain])
    for depiction in ("sketch","source"):
        key=f"photo_minus_drawing_prompt_same_{depiction}"
        differences[key]=subject_class_summary(rows[depiction+"_photo"]-rows[depiction+"_drawing"])
    output=OUT/"matched"/f"{name}.pt"
    save_tensor({"query_ids":split.query_ids,"source_gallery_ids":split.gallery_ids[source],"query_classes":labels,
                 "nominations":nominated,"rows":rows},output)
    atomic_json(output.with_suffix(".json"),{"split":name,"revision":revision,"ceiling_revision":ceiling_revision,
                "seed":SEED,"classes":len(labels.unique()),"sketches":len(labels),"distinct_source_photos":len(src),
                "columns":cols,"means_percent":{k:(100*v.mean(0)).tolist() for k,v in rows.items()},
                "paired":differences,"pilot_order_reproduction":agreement,"tensor_sha256":checksum(output),
                "device":torch.cuda.get_device_name()})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split",required=True,choices=SPLITS)
    parser.add_argument("--revision",required=True)
    parser.add_argument("--ceiling-revision",required=True)
    args=parser.parse_args()
    deterministic()
    run(args.split,args.revision,args.ceiling_revision)
