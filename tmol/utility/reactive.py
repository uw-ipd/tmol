"""Attrs-based "reactive" properties.

Overview
--------

`reactive` is a framework to describe the composition and execution of
functions as properties and attributes of a class. By storing function results
as object properties, rather than named variables, `reactive` allows you to
organize :doc:`pure functions <toolz:purity>` in a declarative syntax, then
automatically handles dependency resolution, on-demand execution and result
caching.

As a motivating example, `reactive` lets you write::

    @reactive_attrs(auto_attribs=True)
    class FactionalAnalysis:
        source: str

        montegue_faction: Sequence[str] = [
            "Romeo", "Benvolio", "Montegue", "Lady Montegue"
        ]
        capulet_faction: Sequence[str] = [
            "Juliet", "Tybalt", "Capulet", "Lady Capulet"
        ]

        @reactive_property
        def lines(source);
            '''Convert text of play to spoken lines.'''
            ...

        @reactive_property
        def soliloquies(lines):
            '''Group spoken lines and identify soliloquies.'''
            ...

        @reactive_property
        def montegue_verbosity(montegue_faction, soliloquies):
            '''Calcuate average verbosity of the Montegue faction.'''
            ...

        @reactive_property
        def capulet_verbosity(capulet_faction, soliloquies):
            '''Calcuate average verbosity of the Capulet faction.'''
            ...

then *does the right thing* when you ask for::

    analysis = FactionalAnalysis(source=act_1)
    analysis.montegue_verbosity / analysis.capulet_verbosity

Argument Resolution
~~~~~~~~~~~~~~~~~~~

`reactive` organizes a computation by binding input data and function results
as `attributes <attrs:overview>` of a "reactive object". Function *inputs* are
resolved on-demand by keyword argument name via `getattr` from reactive object
properties.  By eschewing `self`, `reactive` allows direct declaration of
inter-function dependencies.

For example, ``FactionalAnalysis`` results in the dependency graph:

.. graphviz::
    :align: center

    digraph {
        source [shape=box ]
        capulet_faction [shape=box ]
        montegue_faction [shape=box ]

        source -> lines
        lines -> soliloquies

        capulet_faction -> capulet_verbosity
        soliloquies -> capulet_verbosity

        montegue_faction -> montegue_verbosity
        soliloquies -> montegue_verbosity
    }

|

Result Caching
~~~~~~~~~~~~~~

Given the assumption of :doc:`functional purity <toolz:purity>`, `reactive`
caches the result of `reactive_property` evaluations from the on-demand
resolution of property arguments. This provides in a lazy pull-based execution
model, in which requesting a `reactive_property` value results in the
evaluation, and caching, of its antecedent dependencies.

In ``FactionalAnalysis``, ``analysis.capulet_verbosity`` results in in the
evaluations:

.. graphviz::
    :align: center

    digraph {
        source [shape=box ]
        capulet_faction [shape=box ]
        montegue_faction [shape=box ]

        lines [style=filled]
        soliloquies [style=filled]
        capulet_verbosity [style=filled]

        source -> lines [penwidth=5.0]
        lines -> soliloquies [penwidth=5.0]

        capulet_faction -> capulet_verbosity [penwidth=5.0]
        soliloquies -> capulet_verbosity [penwidth=5.0]

        montegue_faction -> montegue_verbosity
        soliloquies -> montegue_verbosity
    }

|

The subsequent ``analysis.montegue_verbosity`` reuses the cached
``soliloquies``, resulting in the evaluations:

.. graphviz::
    :align: center

    digraph {
        source [shape=box ]
        capulet_faction [shape=box ]
        montegue_faction [shape=box ]

        lines [style=filled]
        soliloquies [style=filled]
        capulet_verbosity [style=filled]
        montegue_verbosity [style=filled]

        source -> lines
        lines -> soliloquies

        capulet_faction -> capulet_verbosity
        soliloquies -> capulet_verbosity

        montegue_faction -> montegue_verbosity [penwidth=5.0]
        soliloquies -> montegue_verbosity [penwidth=5.0]
    }

|

Further access to any property returns the cached value.

Reactive Invalidation
~~~~~~~~~~~~~~~~~~~~~

`reactive` uses the declared dependency graph to invalidate dependent values if
an attribute value is changed, and forward-propogates the invalidation of
reactive properties to all dependents. This provides a lazy, push-based update
model, in which updating an attribute value results in the selective
invalidation of its full direct and indirect dependencies.

.. note:: `reactive` only tracks modification of attribute values by `setattr`,
    and does *not* track internal modification of attribute values. Updates to
    attributes must be propogated by property reassignment. Eg::

        # Possible, modify and assign
        analysis.capulet_faction.extend(["Sampson", "Gregory"])
        analysis.capulet_faction = analysis.capulet_faction

        # Preferred, assign updated value
        analysis.capulet_faction = analysis.capulet_faction + ["Sampson", "Gregory"]

In ``FactionalAnalysis`` an update of ``capulet_faction`` results in invalidations:

.. graphviz::
    :align: center

    digraph {
        source [shape=box ]
        capulet_faction [shape=tripleoctagon ]
        montegue_faction [shape=box ]

        lines [style=filled]
        soliloquies [style=filled]
        capulet_verbosity [style=dashed]
        montegue_verbosity [style=filled]

        source -> lines
        lines -> soliloquies

        capulet_faction -> capulet_verbosity [penwidth=5.0]
        soliloquies -> capulet_verbosity

        montegue_faction -> montegue_verbosity
        soliloquies -> montegue_verbosity
    }

|

A subsequent update of ``source`` results in the forward-invalidations:

.. graphviz::
    :align: center

    digraph {
        source [shape=tripleoctagon ]
        capulet_faction [shape=box ]
        montegue_faction [shape=box ]

        lines [style=dashed]
        soliloquies [style=dashed]
        capulet_verbosity
        montegue_verbosity [style=dashed]

        source -> lines [penwidth=5.0]
        lines -> soliloquies [penwidth=5.0]

        capulet_faction -> capulet_verbosity
        soliloquies -> capulet_verbosity [penwidth=5.0]

        montegue_faction -> montegue_verbosity
        soliloquies -> montegue_verbosity [penwidth=5.0]
    }

|


Class Composition
~~~~~~~~~~~~~~~~~

`reactive` classes can be combined via standard inheritance, with cross class
dependencies are resolved via by the standard :term:`method resolution order
<python:method resolution order>`. This enables a robust mixin-based
composition model in which function inputs are passed, by name, between loosely
coupled components.

For example, consider an extension of ``FactionalAnalysis`` to incorporate an
additional form of sentiment analysis::

    @reactive_attrs(auto_attribs=True)
    class Source:
        source: str

        montegue_faction: Sequence[str] = [
            "Romeo", "Benvolio", "Montegue", "Lady Montegue"
        ]
        capulet_faction: Sequence[str] = [
            "Juliet", "Tybalt", "Capulet", "Lady Capulet"
        ]

        @reactive_property
        def lines(source);
            '''Convert text of play to spoken lines.'''
            ...

    @reactive_attrs
    class Verbosity:
        @reactive_property
        def soliloquies(lines):
            '''Group spoken lines and identify soliloquies.'''
            ...

        @reactive_property
        def montegue_verbosity(montegue_faction, soliloquies):
            '''Calcuate average verbosity of the Montegue faction.'''
            ...

        @reactive_property
        def capulet_verbosity(capulet_faction, soliloquies):
            '''Calcuate average verbosity of the Capulet faction.'''
            ...

    @reactive_attrs
    class Sentiment:
        @reactive_property
        def modern_lines(lines):
            '''Convert from shakespearean to modern english.'''
            ...

        @reactive_property
        def line_sentiment(modern_lines):
            '''Black-box sentiment analysis.'''
            ...

        @reactive_property
        def montegue_sentiment(montegue_faction, line_sentiment):
            '''Calcuate average sentiment of the Montegue faction.'''
            ...

        @reactive_property
        def capulet_sentiment(capulet_faction, line_sentiment):
            '''Calcuate average sentiment of the Capulet faction.'''
            ...

    @reactive_attrs
    class GrudgeAnalysis(Sourde, Verbosity, Sentiment):
        @reactive_property
        def grudge(
            montegue_sentiment,
            montegue_verbosity,
            capulet_sentiment,
            capulet_verbosity,
        ):
            '''From ancient grudge break to new mutiny?'''
            pass

Resulting in the combined reactive dependencies:

.. graphviz::
    :align: center

    digraph {
        subgraph cluster_Source {
            label = "Source"

            source [shape=box ]
            capulet_faction [shape=box ]
            montegue_faction [shape=box ]

            lines
            source -> lines
        }

        subgraph cluster_Verbosity {
            label = "Verbosity"

            soliloquies
            capulet_verbosity
            montegue_verbosity

            lines -> soliloquies

            capulet_faction -> capulet_verbosity
            soliloquies -> capulet_verbosity

            montegue_faction -> montegue_verbosity
            soliloquies -> montegue_verbosity
        }

        subgraph cluster_Sentiment {
            label = "Sentiment"

            modern_lines
            line_sentiment

            montegue_sentiment
            capulet_sentiment

            lines -> modern_lines
            modern_lines -> line_sentiment

            line_sentiment -> montegue_sentiment
            montegue_faction -> montegue_sentiment

            line_sentiment -> capulet_sentiment
            capulet_faction -> capulet_sentiment
        }

        subgraph cluster_GrudgeAnalysis {
            label = "GrudgeAnalysis";

            grudge

            capulet_sentiment -> grudge
            capulet_verbosity -> grudge

            montegue_sentiment -> grudge
            montegue_verbosity -> grudge
        }


    }

|


Custom Invalidation
~~~~~~~~~~~~~~~~~~~

Forward-invalidation can be customized on a per-property basis via an optional
:py:meth:`should_invalidate` function. The function is provided (a) the current
reactive property value, (b) the name of the changed input value and (c) the
updated input parameter value, returning `True` if the reactive property should
be invalidated in response to the change and `False` if the reactive value
should be preserved.

.. note::
    ``should_invalidate`` *requires* the value of reevaluation of
    parameter dependencies to provide an updated input parameter value, causing an
    "eager" reevaluation of the subgraph preceeding the ``should_invalidate``
    property.

** Parameters
~~~~~~~~~~~~~

Functions with :term:`**keyword parameters <python:parameter>` *must*
specify a set of expected argument names when added as a `reactive_property`.


"""

import inspect
from collections import defaultdict
from typing import Callable, Any, Optional, Tuple, Union

import types

import attr
import toolz


def _code(
    argcount,
    kwonlyargcount,
    nlocals,
    stacksize,
    flags,
    codestring,
    constants,
    names,
    varnames,
    filename,
    name,
    firstlineno,
    lnotab,
    freevars,
    cellvars,
    **kwargs,
):
    return types.CodeType(
        argcount,
        kwonlyargcount,
        nlocals,
        stacksize,
        flags,
        codestring,
        constants,
        names,
        varnames,
        filename,
        name,
        firstlineno,
        lnotab,
        freevars,
        cellvars,
    )


def _code_attrs(c):
    cattrs = {n[3:]: getattr(c, n) for n in dir(c) if n.startswith("co_")}
    cattrs["constants"] = cattrs.pop("consts")
    cattrs["codestring"] = cattrs.pop("code")
    return cattrs


def _rename_code_object(func, new_name):
    cattrs = _code_attrs(func.__code__)
    cattrs["name"] = new_name

    return types.FunctionType(
        _code(**cattrs), func.__globals__, new_name, func.__defaults__, func.__closure__
    )


def reactive_attrs(maybe_cls=None, **attrs_kwargs):
    """A class decorator that initializes reactive properties over the target
    type.
    """

    def wrap(cls):
        return attr.attrs(_setup_reactive(cls), **attrs_kwargs)

    if not maybe_cls:
        return wrap
    else:
        return wrap(maybe_cls)


def reactive_property(func=None, *, should_invalidate=None, kwargs=None):
    """Decorate a function as a reactive property of a class.

    ..  warning::
        Does *not* do anything unless the class is also decorated with
        :func:`reactive_attrs`.

    Args:
        func: The target function. The reactive property name and
            dependencies will be inferred from the function name and signature.

        should_invalidate: A :func:`callable` controlling selective
            invalidation of the reactive property in response to dependency
            changes.

        kwargs: Name of expected keyword parameters iff ``func`` accepts
            :term:`**keyword parameters <python:parameter>`.
    """

    def bind(f):
        prop = _ReactiveProperty.from_function(f, kwargs=kwargs)
        prop.should_invalidate(should_invalidate)
        return prop

    if func is not None:
        return bind(func)
    else:
        return bind


@attr.s(auto_attribs=True)
class ReactiveProperty(property):
    """Post-init property managing a reactive property.

    `property` subclass representing a reactive property within a reactive
    class. This is a standard `property` with fget & fdel methods and
    reactive-level metadata.

    Attributes:
        name: Name within the reactive property graph, inferred from assigned
            name in class construction
        f: Pure function used to calculate the property value.
        parameters: Named parameters for f, to be resolved from reactive object.
        should_invalidate: An optional value invalidation check function for f.
    """

    name: str
    f: Callable
    parameters: Tuple[str, ...]
    should_invalidate: Optional[Callable[[Any, str, Any], bool]] = None

    def __attrs_post_init__(self):
        prop = self

        def fget(self):
            val = getattr(self._reactive_values, prop.name, attr.NOTHING)
            if val is not attr.NOTHING:
                return val

            val = prop.f(**{p: getattr(self, p) for p in prop.parameters})
            setattr(self._reactive_values, prop.name, val)

            return val

        def fdel(self):
            # fdel just clears the stored value, invalidation is handeled by
            # class-level __delattr__
            delattr(self._reactive_values, prop.name)

        property.__init__(
            self,
            fget=_rename_code_object(fget, "get_" + prop.name),
            fdel=_rename_code_object(fdel, "del_" + prop.name),
        )

        self.__doc__ = prop.f.__doc__

    def getter(self, _):
        raise NotImplementedError(
            "ReactiveProperty does not support getter modifications"
        )

    def setter(self, _):
        raise NotImplementedError(
            "ReactiveProperty does not support setter modifications"
        )

    def deleter(self, _):
        raise NotImplementedError(
            "ReactiveProperty does not support deleter modifications"
        )


@attr.s(slots=True, auto_attribs=True)
class _ReactiveProperty:
    """Pre-init representation of a reactive property.

    Stub representation of a reactive property before a reactive class is
    processed to generate the reactive property graph.

    Attributes:
        f: Pure function used to calculate the property value.
        parameters: Named parameters for f, to be resolved from reactive object.
        _should_invalidate: An optional value invalidation check function for f.
    """

    f: Callable
    parameters: Tuple[str, ...]
    _should_invalidate: Optional[Callable[[Any, str, Any], bool]] = None

    def should_invalidate(self, f=None):
        self._should_invalidate = f
        return f

    @classmethod
    def from_function(
        cls, fun: Callable, kwargs: Optional[Union[str, Tuple[str, ...]]] = None
    ):
        """Init property from function, inferring parameters from signature."""
        parameters = inspect.signature(fun).parameters.values()

        param_types = set(p.kind for p in parameters)

        ### Check for any positional-only arguments
        if any(
            param_type
            not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.VAR_KEYWORD,
            )
            for param_type in param_types
        ):
            raise ValueError(
                f"function signature contains invalid parameter type: {parameters}"
            )

        ### Get name of all keyword params
        parameter_names = tuple(
            p.name
            for p in parameters
            if p.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        )

        ### Check for "self" parameter
        if "self" in parameter_names:
            raise ValueError(
                "Reactive property value function do not bind 'self' parameter."
            )

        ### Check for **kwargs, and add request kwarg names if provided
        if inspect.Parameter.VAR_KEYWORD in param_types:
            if isinstance(kwargs, str):
                kwargs = (kwargs,)

            if kwargs is None:
                raise ValueError(
                    "Function binds **kwargs, but no kwarg names provided."
                )
            elif set(parameter_names).intersection(kwargs):
                raise ValueError(
                    f"Specified kwarg is already explicit parameter. "
                    f"parameters: {parameter_names} kwargs: {kwargs}"
                )

            parameter_names = parameter_names + tuple(kwargs)
        else:
            if kwargs is not None:
                raise ValueError(
                    "Function does not bind **kwargs, but kwarg names provided."
                )

        return cls(parameters=parameter_names, f=fun)


def _setup_reactive(cls):

    # Rather all _ReactiveProperty in this class and transform into ReactiveProperty
    reactive_props = {
        n: ReactiveProperty(
            name=n,
            f=v.f,
            parameters=v.parameters,
            should_invalidate=v._should_invalidate,
        )
        for n, v in cls.__dict__.items()
        if isinstance(v, _ReactiveProperty)
    }

    for n, v in reactive_props.items():
        setattr(cls, n, v)

    # Walk through the class bases in  MRO, adding reactive props *if new*
    for super_cls in cls.__mro__[1:]:
        if "__reactive_props__" not in super_cls.__dict__:
            continue

        for n, p in super_cls.__dict__["__reactive_props__"].items():
            if n in reactive_props:
                # Already resolved a property with this name
                continue

            if n not in super_cls.__dict__:
                # This is the super super's prop, will resolve later
                continue

            reactive_props[n] = p

    setattr(cls, "__reactive_props__", reactive_props)

    # Expand identified properties into reactive dependencies for this class
    reactive_deps = defaultdict(list)
    for p in reactive_props.values():
        for param in p.parameters:
            reactive_deps[param].append(p.name)

    setattr(cls, "__reactive_deps__", toolz.valmap(tuple, reactive_deps))

    # Create reactive value container type for this class
    ReactiveValues = attr.make_class(
        cls.__name__ + "ReactiveValues",
        {p: attr.ib(init=False) for p in reactive_props},
        slots=True,
    )

    # Add attrs attribute for the value container
    setattr(
        cls,
        "_reactive_values",
        attr.ib(
            default=attr.Factory(ReactiveValues), init=False, cmp=False, repr=False
        ),
    )

    setattr(cls, "__annotations__", cls.__dict__.get("__annotations__", dict()))
    cls.__annotations__["_reactive_values"] = ReactiveValues

    # Add setattr/delattr hooks for reactive updates
    cls.__setattr__ = __reactive_setattr__
    cls.__delattr__ = __reactive_delattr__

    return cls


def _invalidate_reactive_deps(obj, attribute_name, new_val=attr.NOTHING):
    if not hasattr(obj, "_reactive_values"):
        # No reactive value container yet, haven't finished __init__
        return
    if attribute_name not in obj.__reactive_deps__:
        # Update is not a dependency of any reactive property
        return

    # Walk reactive dependents
    for dname in obj.__reactive_deps__[attribute_name]:
        # If there's no value abort
        if not hasattr(obj._reactive_values, dname):
            continue

        # Otherwise get the reactive property
        prop = obj.__reactive_props__[dname]

        # Check it's should_invalidate function, if one is defined
        if prop.should_invalidate:
            current_prop = getattr(obj._reactive_values, dname)
            if not prop.should_invalidate(current_prop, attribute_name, new_val):
                continue

        # Then call through the reactive delete, triggering a recursive invalidation
        delattr(obj, dname)


def __reactive_setattr__(self, n, v):
    object.__setattr__(self, n, v)
    _invalidate_reactive_deps(self, n, v)


def __reactive_delattr__(self, n):
    object.__delattr__(self, n)
    _invalidate_reactive_deps(self, n)
