import pytest
import attr
import torch

from tmol.utility.reactive import reactive_attrs, reactive_property

from tmol.score.score_components import (
    _ScoreComponent,
    ScoreComponentClasses,
    IntraScore,
    InterScore,
)


@reactive_attrs
class IntraFoo(IntraScore):
    @reactive_property
    def total_foo(target):
        return target.foo


@reactive_attrs
class InterFoo(InterScore):
    @reactive_property
    def total_foo(target_i, target_j):
        return target_i.foo + target_j.foo


@attr.s
@_ScoreComponent.mixin
class Foo:
    total_score_components = ScoreComponentClasses(
        name="foo", intra_container=IntraFoo, inter_container=InterFoo
    )

    foo = attr.ib()


@attr.s
@_ScoreComponent.mixin
class JustInterFoo:
    total_score_components = ScoreComponentClasses(
        name="foo", intra_container=None, inter_container=InterFoo
    )

    foo = attr.ib()


@attr.s
@_ScoreComponent.mixin
class JustIntraFoo:
    total_score_components = ScoreComponentClasses(
        name="foo", intra_container=IntraFoo, inter_container=None
    )

    foo = attr.ib()


@reactive_attrs
class IntraBar(IntraScore):
    @reactive_property
    def total_bar(target):
        return target.bar


@reactive_attrs
class InterBar(InterScore):
    @reactive_property
    def total_bar(target_i, target_j):
        return target_i.bar + target_j.bar


@attr.s
@_ScoreComponent.mixin
class Bar:
    total_score_components = ScoreComponentClasses(
        name="bar", intra_container=IntraBar, inter_container=InterBar
    )

    bar = attr.ib()


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

    fb = Foo(foo=torch.tensor(1.0))
    fb2 = Foo(foo=torch.tensor(2.0))

    assert fb.intra_score().total == 1.0
    assert fb.intra_score().total_foo == 1.0

    assert fb.inter_score(fb).total == 2.0
    assert fb.inter_score(fb).total_foo == 2.0

    assert fb.inter_score(fb2).total == 3.0
    assert fb.inter_score(fb2).total_foo == 3.0

    assert fb2.inter_score(fb).total == 3.0
    assert fb2.inter_score(fb).total_foo == 3.0

    # Check missing inter accessor
    inter_fb = JustInterFoo(foo=torch.tensor(1.0))
    inter_fb2 = JustInterFoo(foo=torch.tensor(2.0))

    with pytest.raises(NotImplementedError):
        assert inter_fb.intra_score()

    assert inter_fb.inter_score(inter_fb).total == 2.0
    assert inter_fb.inter_score(inter_fb2).total == 3.0
    assert inter_fb2.inter_score(inter_fb).total == 3.0

    # Check missing intra accessor
    intra_fb = JustIntraFoo(foo=torch.tensor(1.0))

    assert intra_fb.intra_score().total == 1.0

    with pytest.raises(NotImplementedError):
        assert intra_fb.inter_score(intra_fb)


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

    fb = FooBar(foo=torch.tensor(1.0), bar=torch.tensor(2.0))
    assert fb.intra_score().total == 3.0
    assert fb.intra_score().total_foo == 1.0
    assert fb.intra_score().total_bar == 2.0

    fb2 = FooBar(foo=torch.tensor(3.0), bar=torch.tensor(4.0))

    assert fb.inter_score(fb2).total == 10.0
    assert fb.inter_score(fb2).total_foo == 4.0
    assert fb.inter_score(fb2).total_bar == 6.0

    # Check missing inter accessor on single component
    @attr.s
    class JustInterFooBar(JustInterFoo, Bar):
        pass

    inter_fb = JustInterFooBar(foo=torch.tensor(1.0), bar=torch.tensor(2.0))

    with pytest.raises(NotImplementedError):
        inter_fb.intra_score()

    assert inter_fb.inter_score(fb2).total == 10.0
    assert inter_fb.inter_score(fb2).total_foo == 4.0
    assert inter_fb.inter_score(fb2).total_bar == 6.0

    # Check missing intra accessor on single component
    @attr.s
    class JustIntraFooBar(JustIntraFoo, Bar):
        pass

    intra_fb = JustIntraFooBar(foo=torch.tensor(1.0), bar=torch.tensor(2.0))

    assert intra_fb.intra_score().total == 3.0
    assert intra_fb.intra_score().total_foo == 1.0
    assert intra_fb.intra_score().total_bar == 2.0

    with pytest.raises(NotImplementedError):
        intra_fb.inter_score(intra_fb)
