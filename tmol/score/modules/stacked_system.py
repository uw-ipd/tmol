import attr
from attrs_strict import type_validator

from functools import singledispatch

from tmol.score.modules.bases import ScoreSystem, ScoreModule
from tmol.system.packed import PackedResidueSystem, PackedResidueSystemStack


@attr.s(slots=True, auto_attribs=True, kw_only=True, frozen=True)
class StackedSystem(ScoreModule):
    """Score graph component describing stacked system's "depth" and "size".

    A score graph is defined over a set of independent system layers. The
    number of layers defines the stacked "depth", and the maximum number of atoms
    per layer defines the system "size". Each layer is defined over the same
    maximum number of atoms, but systems may have varying number of null atoms.

    Atom indices are defined by a layer index, atom index (l, n) tuple.

    Attributes:
        stack_depth: The system stack depth, ``l``.
        system_size: The maximum number of atoms per layer, ``n``.
    """

    @staticmethod
    def depends_on():
        return set()

    @staticmethod
    @singledispatch
    def build_for(val: object, system: ScoreSystem, **_) -> "StackedSystem":
        """singledispatch hook to resolve system size."""
        raise NotImplementedError("StackedSystem.factory_for({val}, ...)")

    stack_depth: int = attr.ib(validator=type_validator())
    system_size: int = attr.ib(validator=type_validator())


@StackedSystem.build_for.register(ScoreSystem)
def _clone_for_score_system(old, system, **_) -> StackedSystem:
    return StackedSystem(
        system=system,
        stack_depth=StackedSystem.get(old).stack_depth,
        system_size=StackedSystem.get(old).system_size,
    )


@StackedSystem.build_for.register(PackedResidueSystem)
def stack_for_system(
    system: PackedResidueSystem, score_system: ScoreSystem, **_
) -> StackedSystem:
    return StackedSystem(
        system=score_system, stack_depth=1, system_size=int(system.system_size)
    )


@StackedSystem.build_for.register(PackedResidueSystemStack)
def stack_for_stacked_system(
    stack: PackedResidueSystemStack, score_system: ScoreSystem, **_
) -> StackedSystem:
    return StackedSystem(
        system=score_system,
        stack_depth=len(stack.systems),
        system_size=max(int(system.system_size) for system in stack.systems),
    )
