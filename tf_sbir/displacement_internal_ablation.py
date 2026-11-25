"""Matched internal removals from the locked displacement-consensus ranker."""
import argparse
from functools import partial

import torch

from .gcwe_conditioned_donor_test import run
from .gcwe_displacement_consensus import GEOMETRY_ABLATIONS, score_update
from .gcwe_displacement_leave_one_out import locked_measure
from .geometry_word_flat import CONFIG, GLOBAL_WEIGHT, REGIONAL_WEIGHT, CandidateRegions, log_score
from .regional_intervention_study import gallery_word_prior
from .selective_concept_listener import compile_index


FAMILY = 'displacement-internal-ablation'
PRIMARY = 'same_partition'
REGIONAL_BANKS = {
    'without_regional_rgb_diffusion': ('visual', True),
    'without_regional_visual_prototypes': ('rgb', True),
    'without_regional_prior_weighting': ('visual_rgb', False),
}
INTERNAL_ABLATIONS = (
    'without_semantic_rgb_diffusion', *REGIONAL_BANKS, 'without_geometry_rgb_diffusion',
    'without_any_rgb_diffusion', 'without_candidate_region_selection',
    'without_regional_gallery_calibration', 'without_semantic_regional_flip',
    'only_larger_regions', 'only_smaller_regions',
)


class InternalReadouts:
    def __init__(self, gallery, target, source, scale, graph, heads, compact):
        prior = gallery_word_prior(gallery, target, scale)
        self.banks = {}
        for name, (transform, weight_prior) in REGIONAL_BANKS.items():
            indices, weights, _ = compile_index(
                gallery, target, scale, prior, len(heads), transform,
                weight_prior=weight_prior, graph=graph,
            )
            if not torch.equal(indices, heads):
                raise ValueError('Regional intervention changed the fixed concept support')
            self.banks[name] = weights
        restored = graph.score(CONFIG, self.banks['without_regional_rgb_diffusion'].T.contiguous()).T
        torch.testing.assert_close(restored / restored.sum(1, keepdim=True), compact, atol=3e-5, rtol=3e-4)
        self.source = source
        self.scale = scale
        self.graph = graph
        self.heads = heads

    @staticmethod
    def evidence_reader(likelihood, prior):
        # The same observation and calibration rule, with a fixed subset of views.
        named = log_score(likelihood) - log_score(likelihood.mean(3, keepdim=True))
        attention = (prior.log() + named).softmax(2)
        return log_score((attention * likelihood).sum(2).mean(1))

    def __call__(self, poses, crops, bank, raw, reader, regional, lex, delta, geometry_raw, updated):
        global_score = GLOBAL_WEIGHT * lex
        regional_score = REGIONAL_WEIGHT * regional
        global_direct = GLOBAL_WEIGHT * log_score(raw)
        geometry_direct = GLOBAL_WEIGHT * (log_score(updated) - log_score(geometry_raw))
        scores = {
            'without_semantic_rgb_diffusion': global_direct + regional_score + delta,
            'without_geometry_rgb_diffusion': global_score + regional_score + geometry_direct,
            'without_candidate_region_selection':
                global_score + REGIONAL_WEIGHT * reader.read('uniform', 1., 'gallery') + delta,
            'without_regional_gallery_calibration':
                global_score + REGIONAL_WEIGHT * reader.read('self', 1., 'raw') + delta,
        }
        regional_variants = {}
        for name, weights in self.banks.items():
            changed = CandidateRegions(poses[:, 0], poses[:, 1], crops[:, :, 0], crops[:, :, 1],
                                       self.source, self.heads, weights,
                                       self.source.new_empty((0, len(self.source))), raw)
            changed.prepare(self.scale)
            regional_variants[name] = REGIONAL_WEIGHT * changed.read('self', 1., 'gallery')
            scores[name] = global_score + regional_variants[name] + delta
        scores['without_any_rgb_diffusion'] = (
            global_direct + regional_variants['without_regional_rgb_diffusion'] + geometry_direct
        )
        native = (self.scale * reader.full[:, 0]).softmax(1) @ bank.T
        native_semantic = log_score(self.graph.score(CONFIG, native))
        native_regional = self.evidence_reader(reader.l[:, :1], reader.prior)
        scores['without_semantic_regional_flip'] = (
            GLOBAL_WEIGHT * native_semantic + REGIONAL_WEIGHT * native_regional + delta
        )
        for name, subset in (('only_larger_regions', slice(9, 13)), ('only_smaller_regions', slice(0, 9))):
            region = self.evidence_reader(reader.l[:, :, subset], reader.prior[:, :, subset])
            scores[name] = global_score + REGIONAL_WEIGHT * region + delta
        return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('sketchy', 'tuberlin'))
    args = parser.parse_args()
    for dataset in ((args.dataset,) if args.dataset else ('sketchy', 'tuberlin')):
        run(update_fn=partial(score_update, locked_measure, ablate_internal=True),
            family=FAMILY, count=None, dataset=dataset, controls=False, ablate_term=PRIMARY,
            internal_factory=InternalReadouts,
            comparisons=[(PRIMARY, name) for name in (*INTERNAL_ABLATIONS, *GEOMETRY_ABLATIONS)])
