from enum import Enum


class AutoNumber(Enum):
    def __new__(cls):
        value = len(cls.__members__)
        obj = object.__new__(cls)
        obj._value_ = value
        return obj


class ScoreType(AutoNumber):
    fa_lj = ()
    fa_lk = ()
    fa_elec = ()
    hbond = ()
    lk_ball_iso = ()
    lk_ball = ()
    lk_bridge = ()
    lk_bridge_uncpl = ()
    # temp: duplicate lk ball for development ease
    lk_ball_iso2 = ()
    lk_ball2 = ()
    lk_bridge2 = ()
    lk_bridge_uncpl2 = ()
    # keep this one last
    n_score_types = ()
