from tmol.score.terms import *  # noqa: F401, F403
from tmol.score.terms import ScoreTermFactory

import pytest

from tmol.score import ScoreType


@pytest.mark.parametrize("score_type", [ScoreType.fa_ljrep, ScoreType.omega])
def test_score_term_factory_smoke(default_database, torch_device, score_type):
    term = ScoreTermFactory.create_term_for_score_type(
        score_type, param_db=default_database, device=torch_device
    )
    assert term
