"""Identifiers for weighted components of a TMol score function."""

from tmol.utility.auto_number import AutoNumber


class ScoreType(AutoNumber):
    """Enumerate score-function components in stable weight-vector order.

    ``n_score_types`` is a terminal size sentinel and is not an energy
    component. See the API score-term map for the implementation and meaning
    of every preceding member.
    """

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
