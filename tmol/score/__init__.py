import toolz.functoolz
import torch
from typing import Optional
from tmol.database import ParameterDatabase


def _non_memoized_beta2016(
    device: torch.device, param_db: Optional[ParameterDatabase] = None
):
    from tmol.database import ParameterDatabase
    from .score_function import ScoreFunction
    from .score_types import ScoreType

    if param_db is None:
        param_db = ParameterDatabase.get_default()

    sfxn = ScoreFunction(param_db, device)
    sfxn.set_weight(ScoreType.fa_ljatr, 1.0)
    sfxn.set_weight(ScoreType.fa_ljrep, 0.55)
    sfxn.set_weight(ScoreType.fa_lk, 1.0)
    sfxn.set_weight(ScoreType.fa_elec, 1.0)
    sfxn.set_weight(ScoreType.hbond, 1.0)
    sfxn.set_weight(ScoreType.lk_ball_iso, -0.38)
    sfxn.set_weight(ScoreType.lk_ball, 0.92)
    sfxn.set_weight(ScoreType.lk_bridge, -0.33)
    sfxn.set_weight(ScoreType.lk_bridge_uncpl, -0.33)
    sfxn.set_weight(ScoreType.omega, 0.48)
    sfxn.set_weight(ScoreType.rama, 0.50)
    sfxn.set_weight(ScoreType.disulfide, 1.25)
    sfxn.set_weight(ScoreType.cart_lengths, 0.5)
    sfxn.set_weight(ScoreType.cart_angles, 0.5)
    sfxn.set_weight(ScoreType.cart_torsions, 0.5)
    sfxn.set_weight(ScoreType.cart_impropers, 0.5)
    sfxn.set_weight(ScoreType.cart_hxltorsions, 0.5)
    sfxn.set_weight(ScoreType.dunbrack_rot, 0.76)
    sfxn.set_weight(ScoreType.dunbrack_rotdev, 0.69)
    sfxn.set_weight(ScoreType.dunbrack_semirot, 0.78)
    sfxn.set_weight(ScoreType.ref, 1.0)

    return sfxn


@toolz.functoolz.memoize
def beta2016_score_function(
    device: torch.device, param_db: Optional[ParameterDatabase] = None
):
    """Return a ScoreFunction implementing the beta_nov2016 score function
    of Rosetta3.

    See:
    https://pubs.acs.org/doi/10.1021/acs.jctc.6b0081 and
    https://pubs.acs.org/doi/full/10.1021/acs.jctc.7b00125
    """

    return _non_memoized_beta2016(device, param_db)
