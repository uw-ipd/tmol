r"""Graph components managing dispatch of "intra" and "inter" layer scoring.

Score evaluation involves the interaction of three types:

(a) A `_ScoreComponent`, defining a single system state.
(b) An `IntraScore`, managing the total intra-system score for a
    single system.
(c) An `IntraScore`, managing the total inter-system score for a pair
    of systems.

The ``_ScoreComponent`` type is instantiated once for a group of related scoring
operations, and is responsible for initializing any static or reusable data
required to score a system. The system state (Eg: atomic coordinates) is
updated via assignment for each score operation, preserving the ``System``
object.

The ``[Intra|Inter]Score`` types are instantiated for each score operation, and
are responsible for evaluating the score for a single system state using state
data stored within the ``System`` object.  A single ``[Intra|Inter]Score``
object is created for every scoring pass, and is not reused.

.. aafig::

   +-------------+
   |"Input Model"|
   +-------------+
       |
       | "Initialized via factory function."
       V
   +------------------+
   |"ScoreComponent"  |
   |   - 'coords'     |
   |   - 'database'   |
   |   - '...'        |
   +------------------+
       | | |
       | | | "Initialized per score operation."
       V V V
   +-----------------------------+
   |"IntraScore"                 |
   |   - "target: ScoreComponent"|
   +-----------------------------+

|

All types are defined via a composition of multiple term-specific components,
and a term contributes a component to each of the three types under a common
term "name". The ``[Inter|Intra]Score`` component exposes a term-specific
``total_<term>`` property, which are summed to produce a final ``total``
property.

.. aafig::

   +-----------------------------------------+
   |"ScoreComponent"                         |
   |                                         |
   |               +---------+               |
   |            ---+ "coords"+---            |
   |           /   +----+----+   \           |
   |          |         |         |          |
   |  +-------v-+  +----v----+  +-v-------+  |
   |  |"Term A" |  |"Term B" |  |"Term C" |  |
   |  +--------++  +----+----+  ++--------+  |
   +-----------|--------|--------|-----------+
               |        |        |
   +-----------|--------|--------|-----------+
   |  +--------v+  +----v----+  +v--------+  |
   |  |"total_A"|  |"total_B"|  |"total_C"|  |
   |  +-------+-+  +----+----+  +-+-------+  |
   |          |         |         |          |
   |           \   +----v----+   /           |
   |            -->| "total" |<--            |
   |               +---------+               |
   |"IntraScore"                             |
   +-----------------------------------------+

|

To "simplify" the definition of concrete scoring classes from a composite of
score component base classes, the ``IntraScore`` and ``InterScore`` types are
dynamically derived from the ``_ScoreComponent`` type via inspection of the
``_ScoreComponent`` MRO, gathering base components for the ``IntraScore`` and
``InterScore`` classes. Note that this results in a unsettling inversion of
ownership between classes and instances: ``_ScoreComponent`` types define class
level references to their ``IntraScore`` and ``InterScore`` counterparts, but
the resulting ``intra_score` and ``inter_score`` *objects* contain references
to a target ``_ScoreComponent`` object.

.. aafig::

   +---------------------------+
   | ScoreComponent            |
   |                           <-+
   |  'intra_score_type: type' | |
   |  'inter_score_type: type' | |
   |                           | |
   +---+-----------------------+ |
       |                         |
       |               "References"
       |                         |
   "Defines and constructs"      |
       |                         |
       | +---------------------+ |
       | | IntraScore          | |
       | |                     | |
       +->  "target:         " +-+
       | |  "  ScoreComponent" | |
       | +---------------------+ |
       |                         |
       | +---------------------+ |
       | | InterScore          | |
       | |                     | |
       +->  "target_i:       " +-+
         |  "  ScoreComponent" |
         |  "target_j:       " |
         |  "  ScoreComponent" |
         +---------------------+

|
"""
from typing import Optional, Tuple, Dict
import operator
import collections.abc

from functools import singledispatch

import attr
import toolz

import torch

from tmol.utility.reactive import reactive_attrs, reactive_property


@attr.s
class IntraScore:
    """Base mixin for intra-system scoring.

    Base component for an intra-system score evaluation for a target. The
    target's ScoreComponent class will define a specific composite IntraScore
    class with term names defined by `ScoreComponentClasses`. See module
    documentation for details.

    Components contributing to the score _must_ define ``total_{name}``, which
    will be provied as keyword args to the score accessors defined in this
    class.  Contributing components *may* use ``reactive_attrs`` to provide
    component properties and the ``staticmethod`` score accessors defined below
    will be exposed via ``reactive_property``.
    """

    target: "_ScoreComponent" = attr.ib()

    @staticmethod
    def total(target, **component_totals):
        if target.component_weights is None:
            # no weights provided, simple sum components
            return toolz.reduce(operator.add, component_totals.values())
        else:
            # weights provided, use to rescale
            # Note:
            #  1) weights not provided in input dict are assumed == 0
            #  2) tags in input dict not used in
            #     current graph are silently ignored
            total_score = torch.zeros_like(next(iter(component_totals.values())))
            for comp, score in component_totals.items():
                if comp in target.component_weights:
                    total_score += target.component_weights[comp] * score
            return total_score


@attr.s
class InterScore:
    """Base mixin for inter-system scoring.

    Base component for an inter-system score evaluation for a target. The
    target's ScoreComponent class will define a specific composite InterScore
    class with term names defined by `ScoreComponentClasses`. See module
    documentation for details.

    Components contributing to the score _must_ define ``total_{name}``, which
    will be provied as keyword args to the score accessors defined in this
    class.  Contributing components *may* use ``reactive_attrs`` to provide
    component properties and the ``staticmethod`` score accessors defined below
    will be exposed via ``reactive_property``.
    """

    target_i: "_ScoreComponent" = attr.ib()
    target_j: "_ScoreComponent" = attr.ib()

    @staticmethod
    def total(target, **component_totals):
        if target.component_weights is None:
            # no weights provided, simple sum components
            return toolz.reduce(operator.add, component_totals.values())
        else:
            # weights provided, use to rescale
            # Note:
            #  1) weights not provided in input dict are assumed == 0
            #  2) tags in input dict not used in
            #     current graph are silently ignored
            total_score = torch.zeros_like(next(iter(component_totals.values())))
            for comp, score in component_totals.items():
                if comp in target.component_weights:
                    total_score += target.component_weights[comp] * score
            return total_score


@attr.s(slots=True, frozen=True, auto_attribs=True)
class ScoreComponentClasses:
    """The intra/inter graph class components for a ScoreComponent.

    Container for intra/inter graph components exposing a specific score term
    for a `_ScoreComponent`. Each ``_ScoreComponent``-based term implementation
    will expose one-or-more named terms via the ``total_score_components``
    class property, which are composed to generate the corresponding
    ``IntraScore`` and ``InterScore`` utility classes.

    Attributes:
        name: The term name, used to determine the container class properties
            presenting the calculated term value. The _must_ be unique within
            score composite class.
        intra_container: Intra-score type, this _may_ be a ``reactive_attrs``
            type and _must_ expose a ``total_{name}`` property.
        inter_container: inter-score type, this _may_ be a ``reactive_attrs``
            type and _must_ expose a ``total_{name}`` property.
    """

    name: str
    intra_container: Optional[type] = None
    inter_container: Optional[type] = None


class _ScoreComponent:
    """Mixin collection managing definition of inter/intra score containers.

    A mixin-base for all score term implementations managing definition of
    ``InterScore`` and ``IntraScore`` composite classes for all terms present
    in a ``ScoreComponent`` class and creation of ``inter_score`` and
    ``intra_score`` objects during score evaluation.

    A ``ScoreComponent``-derived term mixin _must_ provide a
    ``total_score_components`` class property, containing one or more
    ``ScoreComponentClasses`` of each provided score term.

    The ``ScoreComponent`` base mixin then exposes the ``inter_score`` and
    ``intra_score`` methods; factory functions for class-specific
    ``InterScore`` and ``IntraScore`` instances.
    """

    # Score component related data stored as dunder properties on the composite
    # class. Note that these are class specific, and should *not* be returned
    # from base classes. Ie. Check for existence in cls.__dict__ rather than using
    # hasattr.
    __resolved_score_components__: Optional[
        Tuple[Tuple[type, ScoreComponentClasses], ...]
    ]
    __resolved_intra_score_type__: Optional[type]
    __resolved_inter_score_type__: Optional[type]

    def intra_score(self) -> IntraScore:
        """Create intra-score container over this component."""
        return self._intra_score_type()(self)

    def inter_score(self: "_ScoreComponent", other: "_ScoreComponent") -> InterScore:
        """Create inter-score container for this component and other."""
        return self._inter_score_type()(self, other)

    @classmethod
    def _intra_score_type(cls) -> type:
        """Compose and create IntraScore type for all ScoreComponents in class."""

        if "__resolved_intra_score_type__" in cls.__dict__:
            return cls.__resolved_intra_score_type__

        # Walk through the list of "ScoreComponent" inheritors in the primary
        # class mro, collecting all the ScoreComponentAccessors. Check that
        # every ScoreComponentAccessor provides an intra_container implementation.
        score_component_accessors = []
        for base, component in cls._score_components():
            if component.intra_container is not None:
                score_component_accessors.append(component)
            else:
                raise NotImplementedError(
                    f"score component does not support intra score container.\n"
                    f"component class: {base}\n"
                    f"component: {component}"
                )

            assert hasattr(component.intra_container, f"total_{component.name}"), (
                f"component.intra_container does not provide 'total_{component.name}': "
                f"{component}"
            )

        # Collect the intra_container classes into a base list
        generated_accessor_bases = list(
            set(component.intra_container for component in score_component_accessors)
        )

        # Collect the intra_container.total accessor functions, renaming
        # into appropriate "total_{name}" accessors, and then add the "total"
        # reactive property performing the sum.
        generated_accessor_kwargs = {
            accessor: [
                f"{accessor}_{component.name}"
                for component in score_component_accessors
            ]
            for accessor in ("total",)
        }

        generated_accessor_props = {
            accessor: reactive_property(IntraScore.total, kwargs=tuple(subprops))
            for accessor, subprops in generated_accessor_kwargs.items()
        }

        # Perfom class declaration and reactive_attrs init of the generated
        # container class
        cls.__resolved_intra_score_type__ = reactive_attrs(
            type(
                cls.__name__ + "IntraContainer",
                tuple(generated_accessor_bases),
                generated_accessor_props,
            )
        )

        return cls.__resolved_intra_score_type__

    @classmethod
    def _inter_score_type(cls) -> type:
        """Compose and create InterScore type for all ScoreComponents in class."""
        if "__resolved_inter_score_type__" in cls.__dict__:
            return cls.__resolved_inter_score_type__

        # Walk through the list of "ScoreComponent" inheritors in the primary
        # class mro, collecting all the ScoreComponentAccessors. Check that
        # every ScoreComponentAccessor provides an inter_container implementation.
        score_component_accessors = []
        for base, component in cls._score_components():
            if component.inter_container is not None:
                score_component_accessors.append(component)
            else:
                raise NotImplementedError(
                    f"score component does not support inter score container.\n"
                    f"component class: {base}\n"
                    f"component: {component}"
                )

            assert hasattr(component.inter_container, f"total_{component.name}"), (
                f"component.inter_container does not provide 'total_{component.name}': "
                f"{component}"
            )

        # Collect the inter_container classes into a base list
        generated_accessor_bases = [
            component.inter_container for component in score_component_accessors
        ]

        # Collect the inter_container.total accessor functions, renaming
        # into appropriate "total_{name}" accessors, and then add the "total"
        # reactive property performing the sum.
        generated_accessor_kwargs = {
            accessor: [
                f"{accessor}_{component.name}"
                for component in score_component_accessors
            ]
            for accessor in ("total",)
        }

        generated_accessor_props = {
            accessor: reactive_property(InterScore.total, kwargs=tuple(subprops))
            for accessor, subprops in generated_accessor_kwargs.items()
        }

        # Perform class declaration and reactive_attrs init of the generated
        # container class
        cls.__resolved_inter_score_type__ = reactive_attrs(
            type(
                cls.__name__ + "InterContainer",
                tuple(generated_accessor_bases),
                generated_accessor_props,
            )
        )

        return cls.__resolved_inter_score_type__

    @classmethod
    def _score_components(cls):
        """Gather all ``total_score_components`` defined in class bases."""
        if "__resolved_score_components__" in cls.__dict__:
            return cls.__resolved_score_components__

        score_components = []
        for base in cls.mro():
            base_components = base.__dict__.get("total_score_components", None)
            if base_components is None:
                continue

            if not isinstance(base_components, collections.abc.Collection):
                base_components = (base_components,)

            if base_components:
                score_components.extend((base, c) for c in base_components)

        cls.__resolved_score_components__ = tuple(score_components)

        return cls.__resolved_score_components__

    @classmethod
    def mixin(cls, target):
        """Mixin _ScoreComponent interface into class."""
        target._score_components = classmethod(cls._score_components.__func__)

        target._inter_score_type = classmethod(cls._inter_score_type.__func__)
        target.inter_score = cls.inter_score

        target._intra_score_type = classmethod(cls._intra_score_type.__func__)
        target.intra_score = cls.intra_score

        return target
