"""Describe frozen Semantic Dot top-1 errors using the best correct photo and the returned photo."""

import argparse
from collections import Counter

import torch
from torch.nn.functional import normalize

from .concept_benchmark import atomic_json
from .data import UNSEEN_CLASSES
from .kaggle_unseen_features import DEFAULT_CACHE, load_split
from .named_word_axis_audit import OUT, logits, text_readouts
from .native_block_probe import deterministic
from .semantic_boxable import DEFAULT_PROMPTS
from .semantic_operator_audit import SPLITS


@torch.inference_mode()
def run(name):
    split, text, words = load_split(DEFAULT_CACHE, name)
    photos = normalize(split.photo.float(), dim=1)
    zs, zp = logits(split, photos, *(text[k] for k in DEFAULT_PROMPTS))
    _, _, q, p = text_readouts(zs, zp)
    photo_words, query_words = p.argmax(1), q.argmax(1)
    gallery = split.photo_labels.cuda()
    classes = UNSEEN_CLASSES[name]
    bank = p.T.contiguous()
    examples, patterns = Counter(), Counter()
    for start in range(0, len(q), 384):
        sl = slice(start, start+384)
        scores = q[sl] @ bank
        truth = split.sketch_labels[sl].cuda()
        first = scores.argmax(1)
        best_true = scores.masked_fill(gallery[None, :] != truth[:, None], -torch.inf).argmax(1)
        batch = zip(truth.tolist(), gallery[first].tolist(), query_words[sl].tolist(),
                    photo_words[first].tolist(), photo_words[best_true].tolist(), strict=True)
        for label, retrieved, word, wrong_word, true_word in batch:
            if label == retrieved:
                continue
            patterns[word == wrong_word, word == true_word, wrong_word == true_word] += 1
            examples[classes[label], classes[retrieved], words[word], words[wrong_word], words[true_word]] += 1
    table = {"split": name, "queries": len(q), "wrong_top1": sum(patterns.values()),
             "patterns_querywrong_querytrue_wrongtrue": [(*key, n) for key, n in sorted(patterns.items())],
             "frequent_error_tuples": [(*key, n) for key, n in examples.most_common(12)]}
    atomic_json(OUT/"attribution"/f"errors-{name}.json", table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=SPLITS)
    args = parser.parse_args()
    deterministic()
    run(args.split)
