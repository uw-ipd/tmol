import pytest
import attr

from functools import singledispatch
from tmol.utility.mixins import cooperative_superclass_factory
from tmol.utility.attr import AttrMapping


class Setup:
    @classmethod
    def setup(cls, val, **kwargs):
        return cls(**cooperative_superclass_factory(cls, "factory", val, **kwargs))


def test_cooperative_factory_function_update():
    """Test factory function returning "complex" object."""

    @attr.s(auto_attribs=True)
    class FooParams(AttrMapping):
        f: int

    @attr.s(auto_attribs=True)
    class Foo:
        f: int

        @staticmethod
        def factory(val, b):
            return FooParams(f=val * b)

    @attr.s(auto_attribs=True)
    class BarParams(AttrMapping):
        b: int

    @attr.s(auto_attribs=True)
    class Bar:
        b: int

        @staticmethod
        def factory(val):
            return BarParams(b=val)

    @attr.s(auto_attribs=True)
    class FooBar(Foo, Bar, Setup):
        pass

    result = FooBar.setup(10)
    assert result.b == 10
    assert result.f == 100


def test_cooperative_factory_kwargs():
    """Test cooperative_superclass_factory factory function kwargs."""

    @attr.s(auto_attribs=True)
    class Foo:
        f: int

        @staticmethod
        def factory(val, b, mult=1):
            return dict(f=val + b * mult)

    @attr.s(auto_attribs=True)
    class Bar:
        b: int

        @staticmethod
        def factory(val, **_):
            return dict(b=val)

    @attr.s(auto_attribs=True)
    class FooBar(Foo, Bar, Setup):
        pass

    @attr.s(auto_attribs=True)
    class BarFoo(Bar, Foo, Setup):
        pass

    # Test kwarg passthrough into factory functions
    result = FooBar.setup(1)
    assert result.b == 1
    assert result.f == 2

    result = FooBar.setup(1, mult=4)
    assert result.b == 1
    assert result.f == 5

    # Test reverse MRO ordering, calling Foo factory first means 'b' is unbound
    with pytest.raises(TypeError):
        BarFoo.setup(1)

    # Test that param values mask factory kwargs
    result = FooBar.setup(1, b=1000)
    assert result.b == 1
    assert result.f == 2

    @attr.s(auto_attribs=True)
    class BoundVal(Setup):
        v: int

        @staticmethod
        def factory(val):
            return dict(v=val)

    @attr.s(auto_attribs=True)
    class BoundKwarg(Setup):
        v: int

        @staticmethod
        def factory(val, v):
            return dict(v=v)

    @attr.s(auto_attribs=True)
    class UnboundKwarg(Setup):
        v: int

        @staticmethod
        def factory(val, **_):
            return dict()

    # Test binding of args into params
    result = BoundVal.setup(10)
    assert result.v == 10

    # Test binding of kwargs into params
    result = BoundKwarg.setup(10, v=100)
    assert result.v == 100

    # Test that factory kwarags are not passed through to init
    with pytest.raises(TypeError):
        result = UnboundKwarg.setup(10, v=100)


def test_cooperative_factory_dispatch():
    """Test cooperative_superclass_factory dispatch idiom."""

    @attr.s(auto_attribs=True)
    class Foo:
        f: int

        @staticmethod
        @singledispatch
        def factory(val, b):
            raise NotImplementedError

        @factory.__func__.register(int)
        def __extract_int(val, b):
            return dict(f=val + b)

        @factory.__func__.register(str)
        @factory.__func__.register(float)
        def __extract_cast(val, b):
            return dict(f=int(val) + b)

    @attr.s(auto_attribs=True)
    class Bar:
        b: int

        @staticmethod
        @singledispatch
        def factory(val, **_):
            raise NotImplementedError

        @factory.__func__.register(int)
        def __extract_int(val, **_):
            return dict(b=val)

        @factory.__func__.register(str)
        def __extract_cast(val, **_):
            return dict(b=int(val))

    @attr.s(auto_attribs=True)
    class FooBar(Foo, Bar, Setup):
        pass

    # `int` & `str` factory func is bound for both
    result = FooBar.setup(1)
    assert result.b == 1
    assert result.f == 2

    result = FooBar.setup("1")
    assert result.b == 1
    assert result.f == 2

    # `float` only specialized for Foo, Bar raises NotImplemeted
    with pytest.raises(NotImplementedError):
        result.setup(1.1)
