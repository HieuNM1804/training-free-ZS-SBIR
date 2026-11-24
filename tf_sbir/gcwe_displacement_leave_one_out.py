"""Leave one term out of the locked displacement-GCWE model on each full benchmark."""
import argparse
from functools import partial

from .gcwe_conditioned_donor_test import run
from .gcwe_displacement_consensus import concentration_controls, score_update


def locked_measure(coupling, geometry):
    return {'same_partition': concentration_controls(coupling, geometry)['same_partition']}


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repeat-tu',action='store_true',help='Independently rescore TU with the same cache and seed.')
    repeat_tu=parser.parse_args().repeat_tu
    family='gcwe-displacement-replay-tu' if repeat_tu else 'gcwe-displacement-leave-one-out'
    for dataset in (('tuberlin',) if repeat_tu else ('sketchy','tuberlin')):
        run(update_fn=partial(score_update,locked_measure), family=family,
            count=None, dataset=dataset, controls=False, ablate_term='same_partition',
            comparisons=[('same_partition',arm) for arm in ('without_semantic','without_regional','semantic_regional')])
