import pytest

from tmol.utility.reactive import reactive_attrs, reactive_property, ReactiveProperty


def test_no_self():
    def foobar_self(self, a, b, c=1):
        pass

    with pytest.raises(ValueError):
        ReactiveProperty.from_function(foobar_self)

    def foobar_noreturn(a, b, c=1):
        pass

    #with pytest.raises(ValueError):
    ReactiveProperty.from_function(foobar_noreturn)

    def foobar_valid(a, b, c=1):
        pass

    rp = ReactiveProperty.from_function(foobar_valid)
    assert rp.name == "foobar_valid"
    assert rp.f_value == foobar_valid
    assert rp.parameters == ("a", "b", "c")


def test_binding():
    @reactive_attrs
    class Foo:
        @reactive_property
        def bar(n):
            return "bar" * n

        @reactive_property
        def bat():
            return "bat"

        def np(self):
            pass

    assert tuple(Foo.__reactive_props__) == ("bar", "bat")

    assert Foo.__reactive_props__["bar"].name == "bar"
    assert Foo.__reactive_props__["bar"].parameters == ("n", )
    assert Foo.__reactive_props__["bar"].f_value(2) == "barbar"

    assert Foo.__reactive_props__["bat"].name == "bat"
    assert Foo.__reactive_props__["bat"].parameters == ()
    assert Foo.__reactive_props__["bat"].f_value() == "bat"

    assert not isinstance(Foo.np, ReactiveProperty)
    assert Foo.__reactive_deps__ == {"n": ("bar", )}

    @reactive_attrs
    class SubFoo(Foo):
        @reactive_property
        def bar(n, minus_n):
            return "subbar" * (n - minus_n)

        @reactive_property
        def baz():
            return "baz"

    assert tuple(SubFoo.__reactive_props__) == ("bar", "baz", "bat")

    assert SubFoo.__reactive_props__["bar"].name == "bar"
    assert SubFoo.__reactive_props__["bar"].parameters == ("n", "minus_n")
    assert SubFoo.__reactive_props__["bar"].f_value(3, 1) == "subbarsubbar"

    assert SubFoo.__reactive_props__["bat"].name == "bat"
    assert SubFoo.__reactive_props__["bat"].parameters == ()
    assert SubFoo.__reactive_props__["bat"].f_value() == "bat"

    assert SubFoo.__reactive_props__["baz"].name == "baz"
    assert SubFoo.__reactive_props__["baz"].parameters == ()
    assert SubFoo.__reactive_props__["baz"].f_value() == "baz"

    assert not isinstance(SubFoo.np, ReactiveProperty)

    assert SubFoo.__reactive_deps__ == {"n": ("bar", ), "minus_n": ("bar", )}


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
