import tmol.utility.mixins as mixins


class Factory:
    """Mixin managing cooperative score graph factory functions.

    `Factory` manages cooperative evaluation of a set of component-specifc
    factory functions, defined via ``factory_for`` class/static methods. Each
    factory function should extract a set of graph __init__ kwargs from an
    input ``val``, defaulting to implementing a partial clone from ``val``
    attributes.

    Components factory functions *should*, if appropriate, allow for
    `singledispatch <functools.singledispatch>` based overload on the type of
    ``val``, allowing for customization of score graph initialization for new
    input types. See :py:mod:`tmol.system.score_support` for factory functions
    providing score graph initialization from residue systems.

    See `tmol.utility.mixins.cooperative_superclass_factory` for details of
    kwarg-to-parameter resolution.
    """

    @classmethod
    def build_for(cls, val, **kwargs):
        """Construct score graph for val, defaults to cloning val."""
        return cls(**cls.init_parameters_for(val, **kwargs))

    @classmethod
    def init_parameters_for(cls, val, **kwargs):
        """Get score graph params for val, defaults to cloning."""
        return mixins.cooperative_superclass_factory(
            cls,
            "factory_for",
            val,
            **kwargs,
        )
