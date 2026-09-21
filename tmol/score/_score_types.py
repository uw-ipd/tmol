from tmol.utility import AutoNumber


class ScoreType(AutoNumber):
    """Stable indices for energy terms in score-function weight tensors."""

    fa_ljatr = ()
    fa_ljrep = ()
    fa_lk = ()
    fa_elec = ()
    hbond = ()
    cart_lengths = ()
    cart_angles = ()
    cart_torsions = ()
    cart_impropers = ()
    cart_hxltorsions = ()
    constraint = ()
    disulfide = ()
    omega = ()
    rama = ()
    dunbrack_rot = ()
    dunbrack_rotdev = ()
    dunbrack_semirot = ()
    lk_ball_iso = ()
    lk_ball = ()
    lk_bridge = ()
    lk_bridge_uncpl = ()
    ref = ()
    gen_torsions = ()
    na_torsion = ()
    na_torsion_well = ()
    # keep this one last
    n_score_types = ()
