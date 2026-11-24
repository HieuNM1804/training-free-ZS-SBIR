"""Trace the most strongly certified error for a query with a correct matched decision."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from .concept_benchmark import atomic_json
from .crossdomain_benchmark import revision
from .gallery_panel_resume import semantic_inputs
from .semantic import read_words
from .semantic_invariant_question import Question, solve_decision
from .semantic_invariant_question_study import FAMILY, word_rows


def matched_decisions(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["q"]].append(row)
    eligible = [p for p in grouped.values() if sorted(r["invariant"]["acc"] for r in p)==[0.,1.]]
    if not eligible:
        return []
    def wrong_strength(pair):
        wrong = next(r for r in pair if r["invariant"]["acc"]==0.)
        return (abs(wrong["invariant"]["margin"]), -wrong["q"])
    return sorted(max(eligible,key=wrong_strength),key=lambda r:r["invariant"]["acc"])


def trace(row, inputs, vocabulary):
    cfg, qpaths, _, ppaths, _, qf, pf, src, tgt, scale = inputs
    pos, neg, q = row["pos"],row["neg"],row["q"]
    a,b = sorted((pos,neg))
    photos = word_rows(pf, (a,b), (src,tgt), scale)
    sketch = word_rows(qf,(q,), (src,),scale)[q][0]
    test = Question(photos[a],photos[b],True)
    optimum = test.best
    tie = solve_decision(test.contrast,np.vstack((test.rows,sketch)))
    direction = 1 if sketch @ optimum.answer>0 else -1
    before = direction*sketch*optimum.answer
    change = direction*sketch*(optimum.answer-tie.answer)
    cost = .5*test.contrast*(optimum.answer-tie.answer)
    np.testing.assert_allclose(change.sum(),before.sum(),atol=1e-8)
    np.testing.assert_allclose(cost.sum(),abs(row["invariant"]["margin"]),atol=5e-7)
    def words(values):
        top = np.argsort(-values,kind="stable")[:5]
        return [dict(word=vocabulary[i],contribution=float(values[i]),
                     query_word_probability=float(sketch[i])) for i in top]
    return dict(query=q,positive=pos,negative=neg,query_class=Path(qpaths[q]).parent.name,
                relevant_class=Path(ppaths[pos]).parent.name,foil_class=Path(ppaths[neg]).parent.name,
                correct=bool(row["invariant"]["acc"]),signed_answer_for_relevant=row["invariant"]["answer"],
                reversal_cost_points=100*abs(row["invariant"]["margin"]),
                vote_words=words(before),word_changes_needed_to_tie=words(change),
                photo_discrimination_cost_words=words(cost),
                description="Word contributions are signed and additive. Individual words can have negative cost.")


@torch.inference_mode()
def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    work=Path("/content/sbir")
    root=work/"results"/FAMILY
    inputs=semantic_inputs(work,"sketchy")
    vocabulary=read_words(inputs[0].vocabulary)
    cases={}
    for split in ("sketchy_1","sketchy_2"):
        data=json.loads((root/(split+"-pairs.json")).read_text())
        cases[split]=[trace(r,inputs,vocabulary) for r in matched_decisions(data["boundary"])]
    atomic_json(root/"examples.json",dict(commit=revision(),seed=42,examples=cases))


if __name__=="__main__":
    run()
