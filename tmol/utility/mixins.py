"""Utility functions to support mixin classes."""

from typing import Dict, Any, NewType, Union, Callable, Type

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
