from typing import Optional, Tuple

import attr

from tmol.utility.reactive import reactive_attrs, reactive_property


@attr.s(slots=True, frozen=True, auto_attribs=True)
class ScoreComponentClasses:
    name: str
    intra_container: Optional[type]
    inter_container: Optional[type]


def total(component_total):
    return component_total


class ScoreComponent:
    """Base class for intra & inter system score summation.

    Graph component managing dispatch of score components for "intra"
    (single-system) and "inter" (two-system) scoring.

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
       |   intra_score_type: type  |
       |   inter_score_type: type  |
       +----+----------------------+
           |
           |  Defines via
           |  TotalScoreComponents
           |
           | +---------------------+
           | | IntraScoreContainer |
           +->   target: System    |
           | +---------------------+
           |
           | +---------------------+
           | | InterScoreContainer |
           +->   target_i: System  |
             |   target_j: System  |
             +---------------------+

    Components contributing to inter/intra scores *must* make the component's
    score terms available by implementing the ``total_score_components``
    class-level property, containing a ``ScoreComponentClasses`` instance or
    collection of ``ScoreComponentClasses`` instances.

    """

    __resolved_score_components: Optional[Tuple[type, ...]] = None
    __resolved_intra_score_type: Optional[type] = None
    __resolved_inter_score_type: Optional[type] = None

    @classmethod
    def intra_score(cls, target):
        return cls._intra_score_type()(target)

    @classmethod
    def inter_score(cls, target_i, target_j):
        return cls._inter_score_type()(target_i, target_j)

    @classmethod
    def _intra_score_type(cls):
        if cls.__resolved_intra_score_type:
            return cls.__resolved_intra_score_type

        container_bases = []
        for base, component in cls._score_components():
            if component.intra_container is not None:
                container_bases.append(component.intra_container)
                continue

            raise NotImplementedError(
                f"score component does not support intra score container.\n"
                f"component class: {base}\n"
                f"component: {component}"
            )

        cls.__resolved_intra_score_type = reactive_attrs(
            type(
                cls.__name__ + "IntraContainer",
                tuple(container_bases),
                dict(total=reactive_property(total)),
            )
        )

        return cls.__resolved_intra_score_type

    @classmethod
    def _inter_score_type(cls):
        if cls.__resolved_inter_score_type:
            return cls.__resolved_inter_score_type

        container_bases = []
        for base, component in cls._score_components():
            if component.inter_container:
                container_bases.append(component.inter_container)
                continue

            raise NotImplementedError(
                f"score component does not support inter score container.\n"
                f"component class: {base}\n"
                f"component: {component}"
            )

        cls.__resolved_inter_score_type = reactive_attrs(
            type(
                cls.__name__ + "InterContainer",
                tuple(container_bases),
                dict(total=reactive_property(total)),
            )
        )

        return cls.__resolved_inter_score_type

    @classmethod
    def _score_components(cls):
        if cls.__resolved_score_components:
            return cls.__resolved_score_components

        score_components = []
        for base in cls.mro():
            base_components = base.__dict__.get("total_score_components", None)
            if isinstance(base_components, ScoreComponentClasses):
                base_components = (base_components,)

            if base_components:
                score_components.extend((base, c) for c in base_components)

        cls.__resolved_score_components = tuple(score_components)

        return cls.__resolved_score_components
