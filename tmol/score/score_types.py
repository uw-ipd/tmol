from tmol.utility.auto_number import AutoNumber


class ScoreType(AutoNumber):
    cart_lengths = ()
    cart_angles = ()
    cart_torsions = ()
    cart_impropers = ()
    cart_hxltorsions = ()
    constraint = ()
    disulfide = () # packer enabled
    fa_ljatr = () # packer enabled
    fa_ljrep = () # packer enabled
    fa_lk = () # packer enabled
    fa_elec = () # packer enabled
    hbond = () # packer enabled
    lk_ball_iso = ()
    lk_ball = ()
    lk_bridge = ()
    lk_bridge_uncpl = ()
    omega = () # packer enabled
    rama = () # packer enabled
    ref = () # packer enabled
    dunbrack_rot = () # packer enabled
    dunbrack_rotdev = () # packer enabled
    dunbrack_semirot = () # packer enabled
    # keep this one last
    n_score_types = ()
