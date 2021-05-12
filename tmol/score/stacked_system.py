from functools import singledispatch

from tmol.types.functional import validate_args
from tmol.score.score_graph import score_graph
from tmol.system.packed import PackedResidueSystem, PackedResidueSystemStack


@score_graph
class StackedSystem:
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
    @singledispatch
    def factory_for(val, **_):
        """Overridable clone-constructor."""
        return dict(stack_depth=val.stack_depth, system_size=val.system_size)

    stack_depth: int
    system_size: int


@StackedSystem.factory_for.register(PackedResidueSystem)
@validate_args
def stack_params_for_system(system: PackedResidueSystem, **_):
    return dict(stack_depth=1, system_size=int(system.system_size))


@StackedSystem.factory_for.register(PackedResidueSystemStack)
@validate_args
def stack_params_for_stacked_system(stack: PackedResidueSystemStack, **_):
    return dict(
        stack_depth=len(stack.systems),
        system_size=max(int(system.system_size) for system in stack.systems),
    )
