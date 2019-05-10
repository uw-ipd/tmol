from typing import Dict
from functools import singledispatch

from .score_graph import score_graph


@score_graph
class ScoreWeights:
    """Mixin for scoring system enabling per-term reweighing

    Stores a dictionary matching score terms (strings) to weights (reals)
    which are used in per-term reweighting in 'total' in both
    Intra and Inter scores
    """

    @staticmethod
    @singledispatch
    def factory_for(other, component_weights=None, **_):
        """`clone`-factory, extract weights from other."""
        return dict(component_weights=component_weights)

    # Source per-term weights
    component_weights: Dict[str, float]
