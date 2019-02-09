from functools import singledispatch

from .score_graph import score_graph


@score_graph
class StackedSystem:
    """Score graph component describing stacked system's "depth" and "size".

    A score graph is defined over a set of independent system layers. The
    number layers defines the stacked "depth", and the maximum number of atoms
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
