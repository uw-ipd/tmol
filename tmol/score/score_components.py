from typing import Optional, Tuple
import operator

import attr
import toolz

from tmol.utility.reactive import reactive_attrs, reactive_property


@attr.s(slots=True, frozen=True, auto_attribs=True)
class ScoreComponentClasses:
    name: str
    intra_container: Optional[type]
    inter_container: Optional[type]


@attr.s
class IntraScoreGraph:
    target: "ScoreComponent" = attr.ib()


@attr.s
class InterScoreGraph:
    target_i: "ScoreComponent" = attr.ib()
    target_j: "ScoreComponent" = attr.ib()


class ScoreComponent:
    """Graph component managing dispatch of "intra" and "inter" layer scoring.

    Score dispatch involves the interaction of three components:

    (a) A ``System``, defining the basic data within a single system.
    (b) An ``IntraContainer``, managing the total intra-system score for a
        single system.
    (c) An ``InterContainer``, managing the total inter-system score for a pair
        of systems.

    To "simplify" the definition of concrete scoring classes from a composite
    of score component base classes, the ``IntraContainer`` and ``InterContainer``
    types are dynamically derived from the ``System`` type via inspection of the
    ``System`` MRO, gathering base components for the ``IntraContainer`` and
    ``InterContainer`` classes. Note that this results in a unsettling inversion
    of ownership between classes and instances: ``System`` component *classes*
    define class level references to their ``IntraContainer`` and
    ``InterContainer`` counterparts, but ``intra_container` and ``inter_container``
    *objects* contain references to their target ``system`` object.

    .. aafig::

       +---------------------------+
       | System                    |
       |                           |
       |  'intra_score_type: type' |
       |  'inter_score_type: type' |
       |                           |
       +---+-----------------------+
           |
           |  Defines via
           |  TotalScoreComponents
           |
           | +---------------------+
           | | IntraScoreContainer |
           | |                     |
           +->  'target: System'   |
           | |                     |
           | +---------------------+
           |
           | +---------------------+
           | | InterScoreContainer |
           | |                     |
           +->  'target_i: System' |
             |  'target_j: System' |
             |                     |
             +---------------------+

    Components contributing to inter/intra scores *must* make the component's
    score terms available by implementing the ``total_score_components``
    class-level property, containing a ``ScoreComponentClasses`` instance or
    collection of ``ScoreComponentClasses`` instances.

    """

    # Score component related data stored as dunder properties on the composite
    # class. Note that these are class specific, and should *not* be returned
    # from base classes. Ie. Check for existance in cls.__dict__ rather than using
    # hasattr.
    __resolved_score_components__: Optional[Tuple[type, ...]]
    __resolved_intra_score_type__: Optional[type]
    __resolved_inter_score_type__: Optional[type]

    @classmethod
    def intra_score(cls: "ScoreComponent", target: "ScoreComponent") -> IntraScoreGraph:
        """Create intra-score container for target."""
        return cls._intra_score_type()(target)

    @classmethod
    def inter_score(
        cls: "ScoreComponent", target_i: "ScoreComponent", target_j: "ScoreComponent"
    ) -> InterScoreGraph:
        """Create inter-score container for i/j."""
        return cls._inter_score_type()(target_i, target_j)

    @classmethod
    def _intra_score_type(cls):
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
        generated_accessor_bases = [
            component.intra_container for component in score_component_accessors
        ]

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
            accessor: reactive_property(
                lambda **components: toolz.reduce(operator.add, components.values()),
                kwargs=tuple(subprops),
            )
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
    def _inter_score_type(cls):
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
            accessor: reactive_property(
                lambda **components: toolz.reduce(operator.add, components.values()),
                kwargs=tuple(subprops),
            )
            for accessor, subprops in generated_accessor_kwargs.items()
        }

        # Perfom class declaration and reactive_attrs init of the generated
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
        if "__resolved_score_components__" in cls.__dict__:
            return cls.__resolved_score_components__

        score_components = []
        for base in cls.mro():
            base_components = base.__dict__.get("total_score_components", None)
            if isinstance(base_components, ScoreComponentClasses):
                base_components = (base_components,)

            if base_components:
                score_components.extend((base, c) for c in base_components)

        cls.__resolved_score_components__ = tuple(score_components)

        return cls.__resolved_score_components__
