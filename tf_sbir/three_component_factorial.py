from functools import partial
from .gcwe_conditioned_donor_test import run
from .gcwe_displacement_consensus import score_update
from .gcwe_displacement_leave_one_out import locked_measure

if __name__=='__main__':
    for dataset in ('sketchy','tuberlin'):
        run(update_fn=partial(score_update,locked_measure),family='three-component-factorial',
            count=None,dataset=dataset,controls=False,ablate_term='same_partition',
            comparisons=[('same_partition',name) for name in ('raw_clip','two_view_clip','word_only',
                         'semantic_only','regional_only','geometry_only','without_semantic',
                         'without_regional','semantic_regional')])
