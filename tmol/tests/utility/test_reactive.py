import pytest

import attr

from tmol.utility.reactive import reactive_attrs, reactive_property, ReactiveProperty


def test_no_self():
    """does not bind 'self', including it is an error"""

    def foobar_self(self, a, b, c=1):
        pass

    with pytest.raises(ValueError):
        ReactiveProperty.from_function(foobar_self)


def test_params():
    """input parameters are inferred from signature"""

    def foobar_valid(a, b, c=1):
        pass

    rp = ReactiveProperty.from_function(foobar_valid)
    assert rp.name == "foobar_valid"
    assert rp.f_value == foobar_valid
    assert rp.parameters == ("a", "b", "c")


def test_kwarg_param_resolution():
    """kwarg names must be specified in property declaration"""

    def foobar_kwargs(a, **kwargs):
        pass

    # binds kwarg names as parameters
    rp = ReactiveProperty.from_function(foobar_kwargs, kwargs=("b", "c"))

    assert rp.name == "foobar_kwargs"
    assert rp.f_value == foobar_kwargs
    assert rp.parameters == ("a", "b", "c")

    # does "the right thing" if string name is provided
    rp = ReactiveProperty.from_function(foobar_kwargs, kwargs=("b"))

    assert rp.name == "foobar_kwargs"
    assert rp.f_value == foobar_kwargs
    assert rp.parameters == ("a", "b")

    # raises an error if not explicitly specified
    with pytest.raises(ValueError):
        ReactiveProperty.from_function(foobar_kwargs)

    # raises an error if the function parameters and kwarg names collide
    with pytest.raises(ValueError):
        ReactiveProperty.from_function(foobar_kwargs, ("a", "b", "c"))

    def foobar(a, b):
        pass

    # raises an error if specified w/o kwarg argument
    with pytest.raises(ValueError):
        ReactiveProperty.from_function(foobar, kwargs=("c", "b"))

    # even if they match the names of args
    with pytest.raises(ValueError):
        ReactiveProperty.from_function(foobar_kwargs, kwargs=("a", "b"))


def test_kwarg():
    @reactive_attrs(auto_attribs=True)
    class Foo:
        bar: str
        bat: str

        @reactive_property(kwargs=("bar", "bat"))
        def cat(**kwargs):
            assert set(kwargs) == set(("bar", "bat"))
            return kwargs["bar"] + kwargs["bat"]

    # kwargs are resolved from class propertie
    assert Foo("bar", "bat").cat == "barbat"


def test_dynamic_property():
    """A reactive property can be determined via build-time introspection."""

    class Foo:
        a: int
        b: int

        def total(**fields) -> int:
            return sum(fields.values())

    def dynamic_total(cls, **attrs_kwargs):
        components = tuple(Foo.__annotations__.keys())

        setattr(cls, "total", reactive_property(cls.total, kwargs=components))

        return reactive_attrs(
            cls,
            **attrs_kwargs,
        )

    DFoo = dynamic_total(Foo, auto_attribs=True)
    assert set(attr.fields_dict(DFoo)) == set(("a", "b", "_reactive_values"))
    assert DFoo.__reactive_props__["total"].name == "total"
    assert DFoo.__reactive_props__["total"].parameters == ("a", "b")
    assert Foo.__reactive_deps__ == {"a": ("total", ), "b": ("total", )}

    assert DFoo(1, 2).total == 3


def test_args_invalid():
    """*args can't be used with reactive properties"""

    def foobar_args(a, *args):
        pass

    with pytest.raises(ValueError):
        ReactiveProperty.from_function(foobar_args)


def test_binding_in_subclass():
    @reactive_attrs(auto_attribs=True)
    class Foo:
        n: int

        @reactive_property
        def bar(n):
            return "bar" * n

        @reactive_property
        def bat():
            return "bat"

        @reactive_property
        def bun(n):
            return "bun" * n

        def np(self):
            pass

    # Test that properties bind their target functions
    assert tuple(Foo.__reactive_props__) == ("bar", "bat", "bun")

    # The functions are exposed and have proper parameters
    assert Foo.__reactive_props__["bar"].name == "bar"
    assert Foo.__reactive_props__["bar"].parameters == ("n", )
    assert Foo.__reactive_props__["bar"].f_value(2) == "barbar"

    assert Foo.__reactive_props__["bun"].name == "bun"
    assert Foo.__reactive_props__["bun"].parameters == ("n", )
    assert Foo.__reactive_props__["bun"].f_value(2) == "bunbun"

    assert Foo.__reactive_props__["bat"].name == "bat"
    assert Foo.__reactive_props__["bat"].parameters == ()
    assert Foo.__reactive_props__["bat"].f_value() == "bat"

    # Dependencies are resolved from parameter names
    assert Foo.__reactive_deps__ == {"n": ("bar", "bun")}

    # Non-reactive properies are passed through
    assert not isinstance(Foo.np, ReactiveProperty)

    # Values are resolved from inputs
    assert Foo(n=2).bar == "barbar"

    ### Test subclassing of properties
    @reactive_attrs(auto_attribs=True)
    class SubFoo(Foo):
        minus_n: int

        @reactive_property
        def bar(n, minus_n):
            return "subbar" * (n - minus_n)

        @reactive_property
        def baz():
            return "baz"

    # Properties are resolved from class and superclass
    assert tuple(SubFoo.__reactive_props__) == ("bar", "baz", "bat", "bun")

    # The subclass properties override superclass values
    assert SubFoo.__reactive_props__["bar"].name == "bar"
    assert SubFoo.__reactive_props__["bar"].parameters == ("n", "minus_n")
    assert SubFoo.__reactive_props__["bar"].f_value(3, 1) == "subbarsubbar"

    # superclass is inherited
    assert SubFoo.__reactive_props__["bat"].name == "bat"
    assert SubFoo.__reactive_props__["bat"].parameters == ()
    assert SubFoo.__reactive_props__["bat"].f_value() == "bat"

    assert Foo.__reactive_props__["bun"].name == "bun"
    assert Foo.__reactive_props__["bun"].parameters == ("n", )
    assert Foo.__reactive_props__["bun"].f_value(2) == "bunbun"

    # new properites are picked up
    assert SubFoo.__reactive_props__["baz"].name == "baz"
    assert SubFoo.__reactive_props__["baz"].parameters == ()
    assert SubFoo.__reactive_props__["baz"].f_value() == "baz"

    # and non-reactive are passed through normally
    assert not isinstance(SubFoo.np, ReactiveProperty)

    # deps are resolved using overriden values
    assert SubFoo.__reactive_deps__ == {
        "n": ("bar", "bun"),
        "minus_n": ("bar", )
    }

    v = SubFoo(n=3, minus_n=1)
    assert v.bar == "subbarsubbar"
    assert v.bun == "bunbunbun"


def test_runtime_resolution():
    @reactive_attrs(auto_attribs=True)
    class Foo:
        n: int

        @reactive_property
        def bar(n):
            return "bar" * n

        @reactive_property
        def bad(m):
            return "bad" * m

    v = Foo(n=2)

    # Dependencies are resolved at runtime
    assert v.bar == "barbar"
    with pytest.raises(AttributeError):
        v.bad

    # including dynamic properties...
    v.m = 2
    assert v.bad == "badbad"


def test_calculation():
    @reactive_attrs(auto_attribs=True, slots=True)
    class Foo:
        i: int = 1

        @reactive_property
        def j(i):
            return i + 1

        @reactive_property
        def k(j):
            return j + 1

    t = Foo()

    assert t.i == 1

    assert not hasattr(t._reactive_values, "j")
    assert not hasattr(t._reactive_values, "k")

    assert t.j == 2

    assert getattr(t._reactive_values, "j") == 2
    assert not hasattr(t._reactive_values, "k")

    assert t.k == 3
    assert getattr(t._reactive_values, "j") == 2
    assert getattr(t._reactive_values, "k") == 3

    t.i = 10
    assert not hasattr(t._reactive_values, "j")
    assert not hasattr(t._reactive_values, "k")

    assert t.j == 11

    assert getattr(t._reactive_values, "j") == 11
    assert not hasattr(t._reactive_values, "k")

    assert t.k == 12
    assert getattr(t._reactive_values, "j") == 11
    assert getattr(t._reactive_values, "k") == 12

    del t.i
    assert not hasattr(t._reactive_values, "j")
    assert not hasattr(t._reactive_values, "k")

    with pytest.raises(AttributeError):
        t.j

    assert not hasattr(t._reactive_values, "j")
    assert not hasattr(t._reactive_values, "k")

    with pytest.raises(AttributeError):
        t.k

    assert not hasattr(t._reactive_values, "j")
    assert not hasattr(t._reactive_values, "k")


def test_change_observer():
    @reactive_attrs(auto_attribs=True, slots=True)
    class Foo:
        i: int

        @reactive_property
        def i_mod2(i):
            return i % 2

        @i_mod2.should_invalidate
        def _update_imod2(i_mod2, param, new_val):
            assert param == "i"
            if (new_val % 2) != i_mod2:
                return True
            else:
                return False

        @reactive_property
        def k(i_mod2):
            return str(i_mod2)

    t = Foo(i=1)
    assert not hasattr(t._reactive_values, "i_mod2")
    assert not hasattr(t._reactive_values, "k")

    assert t.i_mod2 == 1
    assert t.k == "1"

    assert getattr(t._reactive_values, "i_mod2") == 1
    assert getattr(t._reactive_values, "k") == "1"

    t.i = 3

    assert getattr(t._reactive_values, "i_mod2") == 1
    assert getattr(t._reactive_values, "k") == "1"

    t.i = 4
    assert not hasattr(t._reactive_values, "i_mod2")
    assert not hasattr(t._reactive_values, "k")

    assert t.i_mod2 == 0
    assert t.k == "0"

    assert getattr(t._reactive_values, "i_mod2") == 0
    assert getattr(t._reactive_values, "k") == "0"
