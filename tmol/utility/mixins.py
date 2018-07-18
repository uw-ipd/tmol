"""Utility functions to support mixin classes."""

from typing import Dict, Any, NewType, Union, Callable, Type
from toolz.dicttoolz import merge

QualifiedName = NewType("QualifiedName", str)


def qualified_name(obj: Union[Type, Callable]) -> QualifiedName:
    """The fully qualified <module>.<name> for a class/function."""
    return f"{obj.__module__}.{obj.__qualname__}"


def gather_superclass_properies(
        obj: Any,
        property_name: str,
) -> Dict[QualifiedName, Any]:
    """Gather property values from all base classes of an object.

    Traverses the object's __mro__ searching for the given property name. The
    property fget is invoked for *every* property definition and the property
    values are returned as a mapping from class name to property value.
    """

    vals = dict()

    for base_class in obj.__class__.__mro__:
        prop = base_class.__dict__.get(property_name, None)
        if prop:
            vals[qualified_name(base_class)] = prop.fget(obj)

    return vals


def cooperative_superclass_factory(
        cls,
        factory_func_name,
        *args,
        **kwargs,
):
    """Gather class factory components from subclasses and create object.

    Traverses a class __mro__ in *reverse* order accumulating __init__
    parameters via calls to class-level factory functions. Each factory
    function generates __init__ parameters by inspecting the factory function
    args, kwargs & current parameters and returning a parameter dict. Params
    are accumulated from factories via `dict.update`, making the partial
    results of param generation availabe to up-MRO factory functions.

    Note that the factory functions receive all current params as kwargs. In
    cases when the total class MRO is unknown a factory function should accept,
    and likely ignore, unknown kwargs. (Eg. ``def factory(cls, known, **_)``)

    Returns a dict of kwarg params accumulated from factory functions.
    """

    params = dict()

    factory_functions = [
        getattr(base, factory_func_name)
        for base in reversed(cls.mro())
        if factory_func_name in base.__dict__
    ]

    for f in factory_functions:
        # merge so params mask kwargs
        params.update(f(*args, **merge(kwargs, params)))

    return params
