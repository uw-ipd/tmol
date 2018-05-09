import attr
import properties

from tmol.properties.array import VariableT
from tmol.properties.reactive import derived_from, cached


@attr.s(slots=True, frozen=True, auto_attribs=True, cmp=True)
class ScoreComponentAttributes:
    name: str
    total: str
    atomic: str


class TotalScoreComponentsGraph(properties.HasProperties):
    score_components = properties.Set(
        "total score components",
        prop=properties.Instance(
            "Score component property accessor", ScoreComponentAttributes
        ),
        default=set(),
        observe_mutations=True
    )

    @derived_from(
        "score_components",
        properties.Set("total score component property names"),
    )
    def total_score_components(self):
        return set(c.total for c in self.score_components)

    @cached(VariableT("sum of score_components"))
    def total_score(self):
        assert len(self.score_components) > 0
        return sum(
            getattr(self, component.total)
            for component in self.score_components
        )

    @properties.observer(properties.everything)
    def on_change(self, change):
        if change["name"] in self.total_score_components:
            self._set("total_score", properties.undefined)
