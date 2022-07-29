from tmol.score.terms import *  # noqa: F401, F403

from tmol.score.score_types import ScoreType
from tmol.score.terms.score_term_factory import ScoreTermFactory

# from tmol.score.ljlk.


def test_score_term_factory_smoke(default_database, torch_device):
    term = ScoreTermFactory.create_term_for_score_type(
        ScoreType.fa_lj, param_db=default_database, device=torch_device
    )
    assert term
