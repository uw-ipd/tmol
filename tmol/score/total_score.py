from typing import Dict, Tuple, Set

import attr
import toolz

from tmol.utility.reactive import reactive_attrs, reactive_property
from tmol.utility.mixins import gather_superclass_properies


@attr.s(slots=True, frozen=True, auto_attribs=True, cmp=True)
class ScoreComponentAttributes:
    name: str
    total: str


@reactive_attrs(auto_attribs=True)
class TotalScoreComponentsGraph:
    """Base graph for total score summation.

    Graph component handling summation of individual score terms into a final
    "total_score" scalar.

    Components contributing to the "total_score" scalar *must* make the
    component's score terms available by implementing the
    `component_total_score_terms` property, returning a
    `ScoreComponentAttributes` instance or collection of
    `ScoreComponentAttributes`.
    """

    total_score_terms: (Dict[str, Tuple[ScoreComponentAttributes, ...]]
                        ) = (attr.ib(init=False, repr=False))

    def __attrs_post_init__(self):
        total_score_terms = gather_superclass_properies(
            self, "component_total_score_terms"
        )

        def norm_component(val):
            if isinstance(val, ScoreComponentAttributes):
                return (val, )
            else:
                return tuple(val)

        total_score_terms = toolz.valmap(norm_component, total_score_terms)

        components = list(toolz.concat(total_score_terms.values()))

        assert len({c.name for c in components}) == len(components), \
            "Duplicate component names."

        self.total_score_terms = total_score_terms

        if hasattr(super(), "__attrs_post_init__"):
            super().__attrs_post_init__()

    @reactive_property
    def total_score_components(total_score_terms) -> Set[str]:
        """Total score component property names."""
        return set(c.total for c in toolz.concat(total_score_terms.values()))

    @property
    def total_score(self):
        #TODO asford cache/reactive invalidate? Setup in static pass?
        assert len(self.total_score_components) > 0
        return sum(
            getattr(self, component_name)
            for component_name in self.total_score_components
        )

    def step(self):
        """Recalculate total_score and gradients wrt/ dofs or coords.

        Does not reset dof or coord grads.
        """

        self.reset_total_score()

        # TODO asford broken by total score issue?
        self.total_score.backward()
        return self.total_score
