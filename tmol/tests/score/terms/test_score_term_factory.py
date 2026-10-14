from tmol.score.terms import (  # noqa: F401
    _backbone_torsion_creator,
    _cartbonded_creator,
    _constraint_creator,
    _disulfide_creator,
    _dunbrack_creator,
    _elec_creator,
    _genbonded_creator,
    _hbond_creator,
    _ljlk_creator,
    _lk_ball_creator,
    _metal_coordination_creator,
    _na_torsion_creator,
    _ref_creator,
)
from tmol.score import ScoreTermFactory

import pytest

from tmol import ScoreType


@pytest.mark.parametrize("score_type", [ScoreType.fa_ljrep, ScoreType.omega])
def test_score_term_factory_smoke(default_database, torch_device, score_type):
    term = ScoreTermFactory.create_term_for_score_type(
        score_type, param_db=default_database, device=torch_device
    )
    assert term
