"""Public dictionary, neutral descriptions, and the two name assignments for the approved pilot."""

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
import re

import nltk
from nltk.corpus import wordnet as wn
import torch
from torch.nn.functional import normalize

from .concept_benchmark import checksum


WORDNET_SHA256 = "cbda5ea6eef7f36a97a43d4a75f85e07fccbb4f23657d27b4ccbc93e2646ab59"
PIECES = re.compile(r"[a-z0-9]+", re.I)


def canonical(text):
    return " ".join(PIECES.findall(text.lower()))


@dataclass(frozen=True)
class Sense:
    key: str
    head: int
    public_ids: tuple
    aliases: tuple
    definition: str
    anonymous: str


@lru_cache(maxsize=None)
def noun_forms(token):
    return {token, *(canonical(v) for v in wn._morphy(token, wn.NOUN))}


def anonymize(gloss, aliases):
    tokens = list(PIECES.finditer(gloss))
    vocabulary = set(aliases)
    longest = max(len(alias.split()) for alias in aliases)
    spans = []
    i = 0
    removed = 0
    while i < len(tokens):
        for end in range(min(len(tokens), i+longest), i, -1):
            segment = gloss[tokens[i].start():tokens[end-1].end()].lower()
            if not re.fullmatch(r"[a-z0-9 _-]+", segment):
                continue
            prefix = [t.group().lower() for t in tokens[i:end-1]]
            forms = noun_forms(tokens[end-1].group().lower())
            if any(" ".join([*prefix, last]) in vocabulary for last in forms):
                spans.append((tokens[i].start(),tokens[end-1].end()))
                removed += end-i
                i = end
                break
        else:
            i += 1
    if removed == len(tokens):
        return ""
    result = gloss
    for start, stop in reversed(spans):
        result = result[:start]+"object"+result[stop:]
    return result


def inventory(words):
    archive = Path(str(nltk.data.find("corpora/wordnet.zip")))
    if wn.get_version() != "3.0" or checksum(archive) != WORDNET_SHA256:
        raise ValueError("Unexpected WordNet corpus")
    public = {}
    for i, word in enumerate(words):
        public.setdefault(canonical(word), []).append(i)
    matches = {}
    for lemma in sorted(wn.all_lemma_names(pos=wn.NOUN)):
        ids = public.get(canonical(lemma))
        if ids is None:
            continue
        for syn in wn.synsets(lemma, pos=wn.NOUN):
            if any(canonical(v.name()) == canonical(lemma) for v in syn.lemmas()):
                matches.setdefault(syn.name(),set()).update(ids)
    senses, empty = [], []
    for key, group in sorted(matches.items()):
        syn = wn.synset(key)
        aliases = tuple(sorted({canonical(v.name()) for v in syn.lemmas()}))
        anon = anonymize(syn.definition(), aliases)
        if not anon:
            empty.append(key)
            continue
        ids = tuple(sorted(group))
        senses.append(Sense(key, ids[0], ids, aliases, syn.definition(), anon))
    used = {i for s in senses for i in s.public_ids}
    return senses, {"resource_sha256": WORDNET_SHA256, "words": len(words), "usable_public_names": len(used),
                    "missing_names": [word for i,word in enumerate(words) if i not in used], "empty": empty,
                    "senses": [asdict(v) for v in senses]}


def describe(domain, gloss, name=None):
    if domain == "drawing":
        subject = "an object" if name is None else "a "+name
        prompt = f"a drawing of {subject}."
    else:
        subject = "the object" if name is None else "the "+name
        prompt = f"a photo of {subject}."
    return f"{prompt} {gloss.rstrip('.')}."


def encode(model, tokenizer, sentences):
    unique = list(dict.fromkeys(sentences))
    encoded = []
    for start in range(0,len(unique),384):
        tokens = tokenizer(unique[start:start+384]).cuda()
        encoded.append(normalize(model.encode_text(tokens).float(), dim=1))
    indices = {s:i for i,s in enumerate(unique)}
    vectors = torch.cat(encoded)
    return vectors[[indices[s] for s in sentences]]


def pair_language(model, tokenizer, pairs, senses, words):
    sentences, usable = [], []
    for ai,bi in pairs:
        a,b = senses[ai],senses[bi]
        aliases = tuple(sorted(set(a.aliases) | set(b.aliases)))
        A,B = anonymize(a.definition,aliases), anonymize(b.definition,aliases)
        usable.append(bool(A and B and A != B))
        if not A or not B:
            A,B = a.anonymous,b.anonymous
        an,bn = words[a.head],words[b.head]
        sentences.extend((describe("drawing",A,an),describe("drawing",B,bn),
                          describe("drawing",A,bn),describe("drawing",B,an),
                          describe("drawing",A),describe("drawing",B),
                          describe("photo",A),describe("photo",B)))
    return encode(model,tokenizer,sentences).reshape(len(pairs),8,-1), torch.tensor(usable,device="cuda")
