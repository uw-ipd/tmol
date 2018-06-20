from functools import singledispatch

from tmol.utility.reactive import reactive_attrs

from .factory import Factory


@reactive_attrs(auto_attribs=True)
class StackedSystem(Factory):
    @staticmethod
    @singledispatch
    def factory_for(val, **_):
        """Overridable clone-constructor.

        Initialize from `val.device` if possible, otherwise defaulting to cpu.
        """
        return dict(stack_depth=val.stack_depth, system_size=val.system_size)

    stack_depth: int
    system_size: int
