from tmol.score.terms import *

from tmol.score.score_types import ScoreType
from tmol.score.terms.score_term_factory import ScoreTermFactory

# from tmol.score.ljlk.


def test_score_term_factory_smoke():
    term = ScoreTermFactory.create_term_for_score_type(ScoreType.fa_lj)
    assert term
