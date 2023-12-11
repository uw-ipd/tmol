import toolz.functoolz
import torch


@toolz.functoolz.memoize
def beta2016_score_function(
    device: torch.device, param_db: "Optional[ParameterDatabase]" = None
):
    from tmol.database import ParameterDatabase
    from .score_function import ScoreFunction
    from .score_types import ScoreType

    if param_db is None:
        param_db = ParameterDatabase.get_default()

    sfxn = ScoreFunction(param_db, device)
    sfxn.set_weight(ScoreType.fa_lj, 1.0)
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

    # When these terms come online, here are there weights
    # sfxn.set_weight(ScoreType.fa_dun_rot, 0.76)
    # sfxn.set_weight(ScoreType.fa_dun_dev, 0.69)
    # sfxn.set_weight(ScoreType.fa_dun_semi, 0.78)

    return sfxn
