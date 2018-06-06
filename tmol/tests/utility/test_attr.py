import pytest

import attr
from tmol.utility.attr import AttrMapping, AttrMutableMapping


def test_attr_mapping():
    @attr.s(auto_attribs=True)
    class FooBar(AttrMapping):
        f: int
        b: int

    params = dict(f=1, b=2)

    fb = FooBar(**params)

    assert dict(fb) == params

    with pytest.raises(KeyError):
        fb["invalid"]

    with pytest.raises(TypeError):
        fb["f"] = 5

    with pytest.raises(AttributeError):
        fb.update(f=3)

    with pytest.raises(AttributeError):
        fb.pop("f")

    @attr.s(auto_attribs=True)
    class MFooBar(AttrMutableMapping):
        f: int
        b: int

    mfb = MFooBar(**params)

    assert dict(mfb) == params

    with pytest.raises(KeyError):
        mfb["invalid"]

    mfb["f"] = 3
    assert dict(mfb) == dict(b=2, f=3)

    mfb.update(b=4)
    assert dict(mfb) == dict(f=3, b=4)

    with pytest.raises(TypeError):
        mfb.pop("f")
