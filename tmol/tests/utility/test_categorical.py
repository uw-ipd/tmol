import pytest

import pandas

import enum
from tmol.utility._categorical import _names_to_val_cat, _vals_to_name_cat


def test_categorical_conversion():
    class TE(enum.IntEnum):
        a = enum.auto()
        b = enum.auto()
        c = enum.auto()

    vals = [TE.a, TE.b, TE.c, TE.a, TE.b, TE.c, TE.a, TE.b, TE.c]
    names = list("abcabcabc")

    assert list(_names_to_val_cat(TE, names)) == vals
    assert list(_vals_to_name_cat(TE, vals)) == names

    missing_names = list("abcd")
    missing_mask = [False, False, False, True]

    assert list(pandas.isna(_names_to_val_cat(TE, missing_names))) == missing_mask

    assert (
        list(pandas.isna(_vals_to_name_cat(TE, [TE.a, TE.b, TE.c, -1]))) == missing_mask
    )


def test_flag_enum():
    class TF(enum.IntFlag):
        a = enum.auto()
        b = enum.auto()

    with pytest.raises(NotImplementedError):
        _vals_to_name_cat(TF, [TF.a, TF.b])
