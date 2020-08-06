import attr
import tmol.extern.toposort as toposort
import pytest
import torch
from tmol.score.modules.bases import ScoreSystem, ScoreModule, ScoreMethod


def test_resolve_modules():
    """ScoreSystem.build_for resolves module dependencies."""

    class FooMethod(ScoreMethod):
        @staticmethod
        def depends_on():
            return {Foo}

        @classmethod
        def build_for(cls, val, system, **_kws):
            return cls(system=system)

        def intra_forward(self, system: ScoreSystem, coords: torch.Tensor):
            return {"foo", coords.sum()}

    class TestScoreModule(ScoreModule):
        @classmethod
        def depends_on(cls):
            return set()

        @classmethod
        def build_for(cls, val, system, **_kws):

            missing = expected_missing[cls]
            present = expected_present[cls]

            for mtype in missing:
                with pytest.raises(KeyError):
                    mtype.get(system)

            for ptype in present:
                assert ptype.get(system) is not None

            return cls(system=system)

    class Foo(TestScoreModule):
        @staticmethod
        def depends_on():
            return {Bar, Bat}

    class Bar(TestScoreModule):
        @staticmethod
        def depends_on():
            return {Bat, Baz}

    class Bat(TestScoreModule):
        pass

    class Baz(TestScoreModule):
        pass

    expected_present = {Foo: {Bar, Bat, Baz}, Bar: {Bat, Baz}, Bat: {}, Baz: {}}

    expected_missing = {Foo: {}, Bar: {Foo}, Bat: {Foo, Bar}, Baz: {Foo, Bar}}

    # FooMethod depends on Foo, which pulls in Bar, Bat, Baz
    system = ScoreSystem.build_for(object(), [FooMethod], {"foo": 1.0})
    assert set(system.methods) == {FooMethod}
    assert set(system.modules) == {Foo, Bar, Bat, Baz}


def test_cycle_detection():
    """ScoreSystem setup detects cyclic module dependencies and errors."""

    class CycleMethod(ScoreMethod):
        @staticmethod
        def depends_on():
            return {Cycle1}

    class Cycle1(ScoreModule):
        @staticmethod
        def depends_on():
            return {Cycle2}

    class Cycle2(ScoreModule):
        @staticmethod
        def depends_on():
            return {Cycle1}

    with pytest.raises(toposort.CircularDependencyError):
        ScoreSystem.build_for(object(), [CycleMethod], {})


def test_build_for_kwargs():
    """Methods and modules pull from a common set of 'build_for' kwargs."""

    @attr.s(auto_attribs=True, kw_only=True, slots=True)
    class KwargModule(ScoreModule):
        required_kwarg: bool

        @classmethod
        def depends_on(cls):
            return set()

        @classmethod
        def build_for(cls, val, system, *, required_kwarg: bool, **_kws):
            return cls(system=system, required_kwarg=required_kwarg)

    @attr.s(auto_attribs=True, kw_only=True, slots=True)
    class KwargMethod(ScoreMethod):
        required_kwarg: bool
        optional_kwarg: bool

        @staticmethod
        def depends_on():
            return {KwargModule}

        @classmethod
        def build_for(
            cls,
            val,
            system,
            *,
            required_kwarg: bool,
            optional_kwarg: bool = None,
            **_kws,
        ):
            return cls(
                system=system,
                required_kwarg=required_kwarg,
                optional_kwarg=optional_kwarg,
            )

        def intra_forward(self, coords):
            return {}

    # Missing a required kwarg
    with pytest.raises(TypeError):
        ScoreSystem.build_for(object(), [KwargMethod], weights={})

    system = ScoreSystem.build_for(
        object(), [KwargMethod], weights={}, required_kwarg=True
    )

    # Required kwargs, optionals provided as defaults
    assert system.methods[KwargMethod].required_kwarg is True
    assert system.methods[KwargMethod].optional_kwarg is None
    assert KwargModule.get(system).required_kwarg is True

    # Required kwargs and optionals provided
    system = ScoreSystem.build_for(
        object(), [KwargMethod], weights={}, required_kwarg=True, optional_kwarg=True
    )
    assert system.methods[KwargMethod].required_kwarg is True
    assert system.methods[KwargMethod].optional_kwarg is True
    assert KwargModule.get(system).required_kwarg is True
