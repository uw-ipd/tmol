from enum import Enum


class AutoNumber(Enum):
    def __new__(cls):
        value = len(cls.__members__)
        obj = object.__new__(cls)
        obj._value_ = value
        return obj


class ScoreType(AutoNumber):
    cart_length = ()
    cart_angle = ()
    cart_torsion = ()
    fa_lj = ()
    fa_lk = ()
    fa_elec = ()
    hbond = ()
    lk_ball_iso = ()
    lk_ball = ()
    lk_bridge = ()
    lk_bridge_uncpl = ()
    omega = ()
    # keep this one last
    n_score_types = ()
