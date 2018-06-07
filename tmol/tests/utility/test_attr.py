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

    # Mapping interface produces same mapping
    assert len(fb) == len(params)
    assert list(fb) == list(params)
    assert dict(fb) == params

    # Key error on access to undefined param
    with pytest.raises(KeyError):
        fb["invalid"]

    # set/del result in errorsj
    with pytest.raises(TypeError):
        fb["f"] = 5

    with pytest.raises(TypeError):
        del fb["f"]

    # MutableMapping components are undefined
    with pytest.raises(AttributeError):
        fb.update(f=3)

    with pytest.raises(AttributeError):
        fb.pop("f")

    @attr.s(auto_attribs=True)
    class MFooBar(AttrMutableMapping):
        f: int
        b: int

    mfb = MFooBar(**params)

    # Mapping interface produces same mapping
    assert len(fb) == len(params)
    assert list(fb) == list(params)
    assert dict(fb) == params

    # Key error on get/set to undefined param
    with pytest.raises(KeyError):
        mfb["invalid"]

    with pytest.raises(KeyError):
        mfb["invalid"] = 1

    # set sets value
    mfb["f"] = 3
    assert dict(mfb) == dict(b=2, f=3)

    # del not supported
    with pytest.raises(TypeError):
        del fb["f"]

    # Mutable mapping interface works for set operations
    mfb.update(b=4)
    assert dict(mfb) == dict(f=3, b=4)

    # Mutable mapping interface errors for del operations
    with pytest.raises(TypeError):
        mfb.pop("f")

    # Using AttrMapping on non-attr classes results in interface errors.
    class NotAttr(AttrMapping):
        f: int = 1

    with pytest.raises(TypeError):
        NotAttr["f"]
