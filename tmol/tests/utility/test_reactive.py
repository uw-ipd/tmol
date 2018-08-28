import pytest

import attr

from tmol.utility.reactive import reactive_attrs, reactive_property, _ReactiveProperty


def test_no_self():
    """does not bind 'self', including it is an error"""

    def foobar_self(self, a, b, c=1):
        pass

    with pytest.raises(ValueError):
        _ReactiveProperty.from_function(foobar_self)


def test_params():
    """input parameters are inferred from signature"""

    def foobar_valid(a, b, c=1):
        pass

    rp = _ReactiveProperty.from_function(foobar_valid)
    assert rp.f == foobar_valid
    assert rp.parameters == ("a", "b", "c")


def test_kwarg_param_resolution():
    """kwarg names must be specified in property declaration"""

    def foobar_kwargs(a, **kwargs):
        pass

    # binds kwarg names as parameters
    rp = _ReactiveProperty.from_function(foobar_kwargs, kwargs=("b", "c"))

    assert rp.f == foobar_kwargs
    assert rp.parameters == ("a", "b", "c")

    # does "the right thing" if string, rather than tuple, is provided
    rp = _ReactiveProperty.from_function(foobar_kwargs, kwargs=("b"))

    assert rp.f == foobar_kwargs
    assert rp.parameters == ("a", "b")

    # raises an error if not explicitly specified
    with pytest.raises(ValueError):
        _ReactiveProperty.from_function(foobar_kwargs)

    # raises an error if the function parameters and kwarg names collide
    with pytest.raises(ValueError):
        _ReactiveProperty.from_function(foobar_kwargs, ("a", "b", "c"))

    def foobar(a, b):
        pass

    # raises an error if specified w/o kwarg argument
    with pytest.raises(ValueError):
        _ReactiveProperty.from_function(foobar, kwargs=("c", "b"))

    # even if they match the names of args
    with pytest.raises(ValueError):
        _ReactiveProperty.from_function(foobar_kwargs, kwargs=("a", "b"))


def test_kwarg():
    """The kwargs of a reactive prop are resolved at evaluation time."""

    @reactive_attrs(auto_attribs=True)
    class Foo:
        bar: str
        bat: str

        @reactive_property(kwargs=("bar", "bat"))
        def cat(**kwargs):
            assert set(kwargs) == set(("bar", "bat"))
            return kwargs["bar"] + kwargs["bat"]

    # kwargs are resolved from class properties
    assert Foo("bar", "bat").cat == "barbat"


def test_dynamic_property():
    """A property can be added dynamically via pre-transform inspection."""

    class Foo:
        a: int
        b: int

        def total(**fields) -> int:
            return sum(fields.values())

    # A decorator that inspects the class and sets up an additional prop.
    def dynamic_total(cls, **attrs_kwargs):
        components = tuple(Foo.__annotations__.keys())

        setattr(cls, "total", reactive_property(cls.total, kwargs=components))

        return reactive_attrs(cls, **attrs_kwargs)

    DFoo = dynamic_total(Foo, auto_attribs=True)
    assert set(attr.fields_dict(DFoo)) == set(("a", "b", "_reactive_values"))

    assert DFoo.total is DFoo.__reactive_props__["total"]
    assert DFoo.__reactive_props__["total"].name == "total"
    assert DFoo.__reactive_props__["total"].parameters == ("a", "b")
    assert Foo.__reactive_deps__ == {"a": ("total",), "b": ("total",)}

    assert DFoo(1, 2).total == 3


def test_args_invalid():
    """*args can't be used with reactive properties"""

    def foobar_args(a, *args):
        pass

    with pytest.raises(ValueError):
        _ReactiveProperty.from_function(foobar_args)


def test_property_override_in_subclass():
    """Reactive properties override standard properties in base classes."""

    @reactive_attrs
    class Foo:
        foo: str = attr.ib()

        @property
        def foo2(self):
            return self.foo + self.foo

        @reactive_property
        def foo3(foo):
            return foo + foo + foo

    @reactive_attrs
    class SubFoo(Foo):
        @reactive_property
        def foo2(foo):
            return foo + "two"

    # Foo has foo2 prop and foo3 reactive prop
    assert tuple(Foo.__reactive_props__) == ("foo3",)

    assert Foo.foo3 is Foo.__reactive_props__["foo3"]
    assert Foo.__reactive_props__["foo3"].name == "foo3"
    assert Foo.__reactive_props__["foo3"].parameters == ("foo",)
    assert Foo.__reactive_props__["foo3"].f("bar") == "barbarbar"

    fc = Foo("bar")
    assert fc.foo == "bar"
    assert fc.foo2 == "barbar"
    assert fc.foo3 == "barbarbar"

    # SubFoo has new foo2 reactive prop and resolves value correctly
    assert tuple(SubFoo.__reactive_props__) == ("foo2", "foo3")
    assert SubFoo.foo2 is SubFoo.__reactive_props__["foo2"]
    assert SubFoo.__reactive_props__["foo2"].name == "foo2"
    assert SubFoo.__reactive_props__["foo2"].parameters == ("foo",)
    # Resolve to SubFoo.foo2, returning "bartwo" not "barbar"
    assert SubFoo.__reactive_props__["foo2"].f("bar") == "bartwo"

    assert SubFoo.foo3 is Foo.__reactive_props__["foo3"]
    assert SubFoo.__reactive_props__["foo3"].name == "foo3"
    assert SubFoo.__reactive_props__["foo3"].parameters == ("foo",)
    assert SubFoo.__reactive_props__["foo3"].f("bar") == "barbarbar"

    sfc = SubFoo("bar")
    assert sfc.foo == "bar"
    assert sfc.foo2 == "bartwo"
    assert sfc.foo3 == "barbarbar"


def test_binding_in_subclass():
    """Reactive graph is an mro-based union of base and class props.

    The reactive property graph of a subclass is the mro-based union of the
    class's reactive properties with the reactive properties of its base
    classes. This allows override-by-name of superclass reactive properties.

    Reactive properties are resolved, like normal properties, via the mro, and
    can be accessed by name as class attributes.

    The full set class and inherited reactive properties are bound in the
    __reactive_props__ class member.
    """

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
    assert Foo.bar is Foo.__reactive_props__["bar"]
    assert Foo.__reactive_props__["bar"].name == "bar"
    assert Foo.__reactive_props__["bar"].parameters == ("n",)
    assert Foo.__reactive_props__["bar"].f(2) == "barbar"

    assert Foo.bun is Foo.__reactive_props__["bun"]
    assert Foo.__reactive_props__["bun"].name == "bun"
    assert Foo.__reactive_props__["bun"].parameters == ("n",)
    assert Foo.__reactive_props__["bun"].f(2) == "bunbun"

    assert Foo.bat is Foo.__reactive_props__["bat"]
    assert Foo.__reactive_props__["bat"].name == "bat"
    assert Foo.__reactive_props__["bat"].parameters == ()
    assert Foo.__reactive_props__["bat"].f() == "bat"

    # Dependencies are resolved from parameter names
    assert Foo.__reactive_deps__ == {"n": ("bar", "bun")}

    # Non-reactive properties are passed through
    assert not isinstance(Foo.np, _ReactiveProperty)

    # Values are resolved from inputs
    assert Foo(n=2).bar == "barbar"

    ### Test property override in subclass
    @reactive_attrs(auto_attribs=True)
    class SubFoo(Foo):
        minus_n: int

        @reactive_property
        def bar(n, minus_n):
            return "subbar" * (n - minus_n)

        @reactive_property
        def baz():
            return "baz"

    # Properties are resolved from class and superclass, order insensitive comparison
    assert sorted(SubFoo.__reactive_props__.keys()) == sorted(
        ("bar", "baz", "bat", "bun")
    )

    # The subclass properties override superclass values
    assert SubFoo.bar is SubFoo.__reactive_props__["bar"]
    assert SubFoo.__reactive_props__["bar"].name == "bar"
    assert SubFoo.__reactive_props__["bar"].parameters == ("n", "minus_n")
    assert SubFoo.__reactive_props__["bar"].f(3, 1) == "subbarsubbar"

    # superclass is inherited
    assert SubFoo.bat is SubFoo.__reactive_props__["bat"]
    assert SubFoo.bat is Foo.bat
    assert SubFoo.__reactive_props__["bat"].name == "bat"
    assert SubFoo.__reactive_props__["bat"].parameters == ()
    assert SubFoo.__reactive_props__["bat"].f() == "bat"

    assert SubFoo.bun is SubFoo.__reactive_props__["bun"]
    assert SubFoo.bun is Foo.bun
    assert SubFoo.__reactive_props__["bun"].name == "bun"
    assert SubFoo.__reactive_props__["bun"].parameters == ("n",)
    assert SubFoo.__reactive_props__["bun"].f(2) == "bunbun"

    # new properties are picked up
    assert SubFoo.baz is SubFoo.__reactive_props__["baz"]
    assert SubFoo.__reactive_props__["baz"].name == "baz"
    assert SubFoo.__reactive_props__["baz"].parameters == ()
    assert SubFoo.__reactive_props__["baz"].f() == "baz"

    # and non-reactive are passed through normally
    assert not isinstance(SubFoo.np, _ReactiveProperty)

    # deps are resolved using overridden values
    assert SubFoo.__reactive_deps__ == {"n": ("bar", "bun"), "minus_n": ("bar",)}

    v = SubFoo(n=3, minus_n=1)
    assert v.bar == "subbarsubbar"
    assert v.bun == "bunbunbun"


def test_runtime_resolution():
    """Dependencies are resolved on execution, and can resolve dynamic attributes."""

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

    # Dependencies are resolved at runtime, missing m
    assert v.bar == "barbar"
    with pytest.raises(AttributeError):
        v.bad

    # including dynamic properties, ok when m is added
    v.m = 2
    assert v.bad == "badbad"


def test_calculation():
    """Reactive props are evaled on pull, and push invalidated by upstream changes.

    Reactive properties are evaluated when requested and stored in the
    _reactive_values container. Upstream changes will propogate downward,
    invalidating dependent values.
    """

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

    ## Check for evaluation and caching as props are requested
    assert t.i == 1

    assert not hasattr(t._reactive_values, "j")
    assert not hasattr(t._reactive_values, "k")

    assert t.j == 2

    assert getattr(t._reactive_values, "j") == 2
    assert not hasattr(t._reactive_values, "k")

    assert t.k == 3
    assert getattr(t._reactive_values, "j") == 2
    assert getattr(t._reactive_values, "k") == 3

    ## Upstream change invalidates all dependent values
    t.i = 10
    assert not hasattr(t._reactive_values, "j")
    assert not hasattr(t._reactive_values, "k")

    assert t.j == 11

    assert getattr(t._reactive_values, "j") == 11
    assert not hasattr(t._reactive_values, "k")

    assert t.k == 12
    assert getattr(t._reactive_values, "j") == 11
    assert getattr(t._reactive_values, "k") == 12

    ## Upstream del invalidates all dependent values
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


def test_should_invalidate():
    """'should_invalidate' handler will block forward-invalidation if it returns False.

    The 'should_invalidate' handler for a given property will block removal of
    the target property from the _reactive_values container if it returns
    false. This will halt forward-prop of the invalidation event, also
    preventing removal of downstream properties that only depend on the
    intermediate reactive property.
    """

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

    # Check that the downstream properties are calculated and stored
    t = Foo(i=1)
    assert not hasattr(t._reactive_values, "i_mod2")
    assert not hasattr(t._reactive_values, "k")

    assert t.i_mod2 == 1
    assert t.k == "1"

    assert getattr(t._reactive_values, "i_mod2") == 1
    assert getattr(t._reactive_values, "k") == "1"

    # When 'should_invalidate' handler returns True the downstream props
    # remain in the container
    t.i = 3

    assert getattr(t._reactive_values, "i_mod2") == 1
    assert getattr(t._reactive_values, "k") == "1"

    # When 'should_invalidate' handler returns True the downstream props are
    # cleared from the container then successfully recalculated when requested.
    t.i = 4
    assert not hasattr(t._reactive_values, "i_mod2")
    assert not hasattr(t._reactive_values, "k")

    assert t.i_mod2 == 0
    assert t.k == "0"

    assert getattr(t._reactive_values, "i_mod2") == 0
    assert getattr(t._reactive_values, "k") == "0"


def test_docstring_forwarding():
    """ReactiveProperty docstrings match source function docs."""

    @reactive_attrs
    class Foo:
        foo = attr.ib()

        @reactive_property
        def bar(foo):
            """After foo we hit bar."""
            return foo + "bar"

    assert Foo.bar.__doc__ == Foo.bar.f.__doc__
    assert Foo.bar.__doc__ == "After foo we hit bar."
    assert Foo("foo").bar == "foobar"


def test_reactive_property_frozen():
    """ReactiveProperty instances can't be updated via getter/setter/deleter."""

    @reactive_attrs
    class Foo:
        @reactive_property
        def p():
            pass

    with pytest.raises(NotImplementedError):
        Foo.p.getter(lambda s: None)

    with pytest.raises(NotImplementedError):
        Foo.p.setter(lambda s, _: None)

    with pytest.raises(NotImplementedError):
        Foo.p.deleter(lambda s: None)
