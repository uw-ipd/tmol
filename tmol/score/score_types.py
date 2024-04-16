from tmol.utility.auto_number import AutoNumber


class ScoreType(AutoNumber):
    cart_lengths = ()
    cart_angles = ()
    cart_torsions = ()
    cart_impropers = ()
    cart_hxltorsions = ()
    disulfide = ()
    fa_ljatr = ()
    fa_ljrep = ()
    fa_lk = ()
    fa_elec = ()
    hbond = ()
    lk_ball_iso = ()
    lk_ball = ()
    lk_bridge = ()
    lk_bridge_uncpl = ()
    omega = ()
    rama = ()
    ref = ()
    dunbrack_rot = ()
    dunbrack_rotdev = ()
    dunbrack_semirot = ()
    # keep this one last
    n_score_types = ()
