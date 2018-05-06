"""
A prototype implementation of attrs-based "reactive" properties.

A containing class stores state divided into two types: "standard" attributes,
which represent source data, and "reactive" properties, which are derived from
other attributes and other properties.

Reactive props are calculated dynamically via a "pull" model, accessing a
property invokes its definition function to calculate the property value. The
value is cached within the containing object and is returned, if available, on
the next property access.

The definition function may depend on a set a attrs/props in the object, bound
as the input parameters of the value function. This values are bound from the
object *before* value function execution. As depdencies *may* be themselves
reactive, resulting in a backward traversal through reactive properties to
calculate the required value. DAG-ness is not mandated, but infinite recursion
will occur in the case of mutually interdependent property values. Caveat usor.

Reative props are invalidated via a "push" model. Deleting or changing a
property value will invalidate the store value for all declared dependent
properties. These dependents may themselves have dependents, resulting in a
forward traversal through reactive properties to invalidate all derived
property values. This allows recalcuation when the property is next requested.

Property invalidation may be controlled via an optional "should_invalidate"
function, which consumes (a) the current reactive property value, (b) the name
of the changed input value and (c) the updated input parameter value. This
function returns True if the reactive property should be invalidated in
response to the change and False if the reactive value should be preserved.

Subclassing semantics akin to "attr" can be used. A superclass *may* specifiy
property dependencies that are provided by a subclass. The standard MRO is used
to resolve the value function for any duplicate properties.
"""

import inspect
from collections import defaultdict
from typing import Callable, Any, Optional, Tuple

import attr


def reactive_attrs(maybe_cls=None, **attrs_kwargs):
    def wrap(cls):
        return attr.attrs(_setup_reactive(cls), **attrs_kwargs)

    if not maybe_cls:
        return wrap
    else:
        return wrap(maybe_cls)


def reactive_property(f, should_invalidate=None):
    prop = ReactiveProperty.from_function(f)
    prop.should_invalidate(should_invalidate)
    return prop


@attr.s(slots=True, auto_attribs=True)
class ReactiveProperty:
    name: str
    parameters: Tuple[str, ...]

    f_value: Callable

    _should_invalidate_check: Optional[Callable[[Any, str, Any], bool]] = None

    def should_invalidate(self, f=None):
        self._should_invalidate_check = f
        return f

    @classmethod
    def from_function(cls, fun):
        parameters = tuple(inspect.signature(fun).parameters)
        if "self" in parameters:
            raise ValueError(
                "Reactive property value function do not bind 'self' parameter."
            )

        return cls(
            name=fun.__name__,
            parameters=parameters,
            f_value=fun,
        )


def _setup_reactive(cls):
    cd = cls.__dict__

    reactive_props = {
        n: v
        for n, v in cd.items()
        if isinstance(v, ReactiveProperty)
    }

    for n in reactive_props:
        delattr(cls, n)

    for super_cls in cls.__mro__[1:-1]:  # Traverse the MRO and collect
        sub_props = getattr(super_cls, "__reactive_props__", None)
        if sub_props is not None:
            for p in sub_props:
                if p not in reactive_props:
                    reactive_props[p] = sub_props[p]

    reactive_deps = defaultdict(list)
    for p in reactive_props.values():
        for param in p.parameters:
            reactive_deps[param].append(p.name)

    ReactiveValues = attr.make_class(
        cls.__name__ + "ReactiveValues",
        {p: attr.ib(init=False)
         for p in reactive_props},
        slots=True
    )

    # Setup reactive property "result" attrs
    if "__annotations__" not in cls.__dict__:
        setattr(cls, "__annotations__", dict())
    setattr(
        cls, "_reactive_values",
        attr.ib(
            default=attr.Factory(ReactiveValues),
            init=False,
            cmp=False,
            repr=False
        )
    )
    cls.__annotations__["_reactive_values"] = ReactiveValues

    setattr(cls, "__reactive_props__", reactive_props)
    setattr(
        cls, "__reactive_deps__",
        {n: tuple(v)
         for n, v in reactive_deps.items()}
    )

    cls.__getattr__ = __reactive_getattr__
    cls.__setattr__ = __reactive_setattr__
    cls.__delattr__ = __reactive_delattr__

    #for p in reactive_props:
    #    prop_attr = attr.ib(init=False, repr=False, cmp=False, hash=False)

    #    prop_attr_name = "_" + p.name
    #    setattr(cls, prop_attr_name, prop_attr)
    #    cls.__annotations__[prop_attr_name] = p.f_type

    return cls


def invalidate_reactive_deps(obj, n, n_value=attr.NOTHING):
    if not hasattr(obj, "_reactive_values") or n not in obj.__reactive_deps__:
        return

    for dname in obj.__reactive_deps__[n]:
        if hasattr(obj._reactive_values, dname):
            prop = obj.__reactive_props__[dname]

            if prop._should_invalidate_check:
                current_prop = getattr(obj._reactive_values, dname)
                if not prop._should_invalidate_check(current_prop, n, n_value):
                    continue

            delattr(obj._reactive_values, dname)
            invalidate_reactive_deps(obj, dname)


def __reactive_getattr__(self, n):
    prop = self.__reactive_props__.get(n, None)

    if n is "_reactive_values" or prop is None:
        return object.__getattribute__(self, n)

    if hasattr(self._reactive_values, n):
        return getattr(self._reactive_values, n)

    val = prop.f_value(*(getattr(self, p) for p in prop.parameters))

    setattr(self._reactive_values, n, val)

    return val


def __reactive_setattr__(self, n, v):
    object.__setattr__(self, n, v)
    invalidate_reactive_deps(self, n, v)


def __reactive_delattr__(self, n):
    object.__delattr__(self, n)
    invalidate_reactive_deps(self, n)
