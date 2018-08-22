from functools import singledispatch

from tmol.utility.reactive import reactive_attrs

from .factory import Factory


@reactive_attrs(auto_attribs=True)
class StackedSystem(Factory):
    """Score graph component describing stacked system's "depth" and "size".

    A score graph is defined over a set of independent system layers. The
    number layers defines the stacked "depth", and the maximum number of atoms
    per layer defines the system "size". Each layer is defined over the same
    maximum number of atoms, but systems may have varying number of null atoms.

    Atom indices are defined by a layer index, atom index (l, n) tuple.
    """

    @staticmethod
    @singledispatch
    def factory_for(val, **_):
        """Overridable clone-constructor.

        Initialize from ``val.device`` if possible, otherwise defaulting to cpu.
        """
        return dict(stack_depth=val.stack_depth, system_size=val.system_size)

    stack_depth: int
    system_size: int
