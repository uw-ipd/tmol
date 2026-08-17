from tmol.score.terms import *  # noqa: F401, F403

from tmol.score import ScoreType
from tmol.score.terms import ScoreTermFactory

# from tmol.score.ljlk.


def test_score_term_factory_smoke(default_database, torch_device):
    term = ScoreTermFactory.create_term_for_score_type(
        ScoreType.fa_ljrep, param_db=default_database, device=torch_device
    )
    assert term
