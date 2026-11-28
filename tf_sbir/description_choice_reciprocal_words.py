"""Build fixed reciprocal word-assignment questions from independent real subjects."""

import argparse
import json

import torch

from .concept_benchmark import atomic_json, checksum
from .description_block_choices import NativeReads, order_key
from .description_choice_blocks import OUT, PILOT, inputs
from .encoding import save_tensor
from .kaggle_unseen_features import DEFAULT_CACHE
from .native_block_probe import SEED, deterministic
from .semantic_boxable import DEFAULT_PROMPTS
from .wordnet_renaming_pilot import SPLITS


def pair_subjects(name, row):
    ids, group = row["source_ids"].unique(return_inverse=True)
    positions, classes = [], []
    for i in range(len(ids)):
        sketches = (group == i).nonzero().flatten().tolist()
        pos = min(sketches, key=lambda j: order_key("reciprocal-sketch", name, int(row["query_ids"][j])))
        positions.append(pos)
        classes.append(int(row["classes"][pos]))
    order = sorted(range(len(ids)), key=lambda i: (order_key("reciprocal-class", name, classes[i]),
                                                   order_key("reciprocal-source", name, int(ids[i]))))
    if len(order) % 2:
        raise ValueError("Expected an even source population")
    pairs = torch.tensor([positions[i] for i in order]).reshape(2,-1).T.contiguous()
    if (row["classes"][pairs[:,0]] == row["classes"][pairs[:,1]]).any() or len(row["source_ids"][pairs].unique()) != len(ids):
        raise ValueError("Reciprocal pairs must be disjoint and from different classes")
    return pairs


def noun_utility(name):
    chunks = []
    for i in range(2):
        p = PILOT / "ceiling" / f"{name}-{i}.pt"
        m = json.loads(p.with_suffix(".json").read_bytes())
        if m["revision"] != "50904116bfa71ad76e6ae6c65f5fec5a235d713d" or m["sha256"] != checksum(p):
            raise ValueError("Wrong official noun AP table")
        r = torch.load(p, weights_only=True, mmap=True)
        keep = r["indices"] < r["words"]
        chunks.append((r["indices"][keep], r["values"][keep]))
    order = torch.cat([k for k, _ in chunks])
    if not torch.equal(order.sort().values,torch.arange(601)):
        raise ValueError("Incomplete Open Images noun table")
    return torch.cat([v for _, v in chunks])[order.argsort()]


@torch.inference_mode()
def run(name, revision, native_revision, choices_revision):
    f = OUT / f"{name}-free.pt"
    m = json.loads(f.with_suffix(".json").read_bytes())
    if m["revision"] != choices_revision or m["sha256"] != checksum(f):
        raise ValueError("Wrong original predictions")
    r = torch.load(f, mmap=True, weights_only=True)
    pairs = pair_subjects(name, r)
    first = r["source"][:, -1, 0].long()
    catalog = json.loads((PILOT / "inventory.json").read_bytes())
    head = torch.tensor([s["head"] for s in catalog["senses"]])
    words = head[first[pairs]]
    text = torch.load(DEFAULT_CACHE/"text.pt", mmap=True, weights_only=True)
    meta, _, _ = inputs()
    if text["checkpoint"] != meta["sha256"]:
        raise ValueError("Wrong text feature checkpoint")
    axis = (text["text"][DEFAULT_PROMPTS[0]][words[:,0]] -
            text["text"][DEFAULT_PROMPTS[0]][words[:,1]]).cuda()
    utility = noun_utility(name)
    classes = r["classes"][pairs]
    differences = utility[words[:,:1], classes] - utility[words[:,1:], classes]
    ap = differences[:,:,0]
    eligible = ap[:,0]*ap[:,1] < 0
    orientation = ap[:,0].sign()
    outputs = {}
    counts = {}
    for mode, key in (("sketch", "query_ids"), ("photo", "source_ids")):
        reader = NativeReads(mode, native_revision)
        ids = r[key][pairs]
        end = reader.read(ids.flatten(), torch.tensor([12],device="cuda"))[2].reshape(len(ids),2,-1)
        scores = (end * axis[:,None,:]).sum(2).cpu()
        margin = scores[:,0] - scores[:,1]
        value = margin[eligible] * orientation[eligible]
        counts[mode] = {"correct": int((value>0).sum()), "wrong": int((value<0).sum()),
                        "ties": int((value==0).sum()), "accuracy": float(100*((value>0).double()+.5*(value==0)).mean())}
        outputs[mode] = scores
        del reader
    dest = OUT / f"reciprocal-{name}.pt"
    save_tensor({"pair_positions":pairs, "query_ids":r["query_ids"][pairs], "source_ids":r["source_ids"][pairs],
                 "classes":classes, "words":words, "name_KD_differences":differences, "eligible":eligible,
                 "scores":outputs},dest)
    atomic_json(dest.with_suffix(".json"),{"revision":revision,"native_revision":native_revision,
                "choices_revision":choices_revision,"seed":SEED,"split":name,"pairs":len(pairs),
                "distinct_words":int((words[:,0]!=words[:,1]).sum()),"reciprocal":int(eligible.sum()),
                "classes_represented":len(classes[eligible].unique()),"counts":counts,"raw_sha256":checksum(dest)})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=SPLITS,required=True)
    p.add_argument("--revision",required=True)
    p.add_argument("--native-revision",required=True)
    p.add_argument("--choices-revision",required=True)
    a = p.parse_args()
    deterministic()
    run(a.split,a.revision,a.native_revision,a.choices_revision)
