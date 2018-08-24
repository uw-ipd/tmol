import pytest
import attr

from tmol.utility.reactive import reactive_attrs, reactive_property

from tmol.score.score_components import (
    ScoreComponent,
    ScoreComponentClasses,
    IntraScoreGraph,
    InterScoreGraph,
)


@reactive_attrs
class IntraFoo(IntraScoreGraph):
    @reactive_property
    def total_foo(target):
        return target.foo


@reactive_attrs
class InterFoo(InterScoreGraph):
    @reactive_property
    def total_foo(target_i, target_j):
        return target_i.foo + target_j.foo


@attr.s
class Foo(ScoreComponent):
    total_score_components = ScoreComponentClasses(
        name="foo", intra_container=IntraFoo, inter_container=InterFoo
    )

    foo: int = attr.ib()


@attr.s
class JustInterFoo(ScoreComponent):
    total_score_components = ScoreComponentClasses(
        name="foo", intra_container=None, inter_container=InterFoo
    )

    foo: int = attr.ib()


@attr.s
class JustIntraFoo(ScoreComponent):
    total_score_components = ScoreComponentClasses(
        name="foo", intra_container=IntraFoo, inter_container=None
    )

    foo: int = attr.ib()


@reactive_attrs
class IntraBar(IntraScoreGraph):
    @reactive_property
    def total_bar(target):
        return target.bar


@reactive_attrs
class InterBar(InterScoreGraph):
    @reactive_property
    def total_bar(target_i, target_j):
        return target_i.bar + target_j.bar


@attr.s
class Bar(ScoreComponent):
    total_score_components = ScoreComponentClasses(
        name="bar", intra_container=IntraBar, inter_container=InterBar
    )

    bar: int = attr.ib()


def test_single_component():
    """Score component accessors generate passthrough classes single component value.

    The `ScoreComponent` ``intra_score`` and ``inter_score`` class generators
    create @reactive_attrs instances, binding the "total_{term}" properties of the
    provided ComponentAccessors.

    A "total" reactive property is defined, summing the provided "total_{term}"
    properties of each component.

    The accessors are independent, providing a single accessor (eg intra but
    not inter) allows access to that component, throwing an NotImplementedError
    for the other.
    """

    fb = Foo(foo="foo")
    fb2 = Foo(foo="foo2")

    assert fb.intra_score(fb).total == "foo"
    assert fb.intra_score(fb).total_foo == "foo"

    assert fb.inter_score(fb, fb).total == "foofoo"
    assert fb.inter_score(fb, fb).total_foo == "foofoo"

    assert fb.inter_score(fb, fb2).total == "foofoo2"
    assert fb.inter_score(fb, fb2).total_foo == "foofoo2"

    assert fb.inter_score(fb2, fb).total == "foo2foo"
    assert fb.inter_score(fb2, fb).total_foo == "foo2foo"

    # Check missing inter accessor
    inter_fb = JustInterFoo(foo="foo")
    inter_fb2 = JustInterFoo(foo="foo2")

    with pytest.raises(NotImplementedError):
        assert inter_fb.intra_score(fb)

    assert inter_fb.inter_score(inter_fb, inter_fb).total == "foofoo"
    assert inter_fb.inter_score(inter_fb, inter_fb2).total == "foofoo2"
    assert inter_fb.inter_score(inter_fb2, inter_fb).total == "foo2foo"

    # Check missing intra accessor
    intra_fb = JustIntraFoo(foo="foo")

    assert intra_fb.intra_score(intra_fb).total == "foo"

    with pytest.raises(NotImplementedError):
        assert intra_fb.inter_score(intra_fb, intra_fb)


def test_two_component():
    """Score component accessors sum multiple component values.

    The `ScoreComponent` ``intra_score`` and ``inter_score`` class generators
    create @reactive_attrs instances, binding the "total" static method of the
    all ComponentAccessors in the mro as reactive properties under the property
    names "total_{name}".

    A "total" reactive property is _also_ defined, which will sum the component
    values in the order provided in the mro.

    The accessors are independent, but must be defined for _all_ components in
    the mro.  Missing a single component implementation invalidates the
    accessor for the derived class, throwing a NotImplementedError.
    """

    @attr.s
    class FooBar(Foo, Bar):
        pass

    @attr.s
    class BarFoo(Bar, Foo):
        pass

    fb = FooBar(foo="foo", bar="bar")
    assert fb.intra_score(fb).total == "foobar"
    assert fb.intra_score(fb).total_foo == "foo"
    assert fb.intra_score(fb).total_bar == "bar"

    fb2 = FooBar(foo="foo2", bar="bar2")

    assert fb.inter_score(fb, fb2).total == "foofoo2barbar2"
    assert fb.inter_score(fb, fb2).total_foo == "foofoo2"
    assert fb.inter_score(fb, fb2).total_bar == "barbar2"

    # Set summation order from mro
    bf = BarFoo(foo="foo", bar="bar")
    bf.intra_score(bf).total == "barfoo"
    bf.intra_score(bf).total_foo == "foo"
    bf.intra_score(bf).total_bar == "bar"

    # Check missing inter accessor on single component
    @attr.s
    class JustInterFooBar(JustInterFoo, Bar):
        pass

    inter_fb = JustInterFooBar("foo", "bar")

    with pytest.raises(NotImplementedError):
        inter_fb.intra_score(inter_fb)

    assert inter_fb.inter_score(inter_fb, fb2).total == "foofoo2barbar2"
    assert inter_fb.inter_score(inter_fb, fb2).total_foo == "foofoo2"
    assert inter_fb.inter_score(inter_fb, fb2).total_bar == "barbar2"

    # Check missing intra accessor on single component
    @attr.s
    class JustIntraFooBar(JustIntraFoo, Bar):
        pass

    intra_fb = JustIntraFooBar("foo", "bar")

    assert intra_fb.intra_score(intra_fb).total == "foobar"
    assert intra_fb.intra_score(intra_fb).total_foo == "foo"
    assert intra_fb.intra_score(intra_fb).total_bar == "bar"

    with pytest.raises(NotImplementedError):
        intra_fb.inter_score(intra_fb, intra_fb)
