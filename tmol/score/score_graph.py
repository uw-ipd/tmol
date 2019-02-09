from toolz import compose
from tmol.utility.reactive import reactive_attrs

from .factory_mixin import _Factory
from .score_components import _ScoreComponent


def score_graph(cls=None, *, auto_attribs=True):
    """Decorate a reactive score graph class."""

    def _wrap(cls):
        return compose(
            reactive_attrs(auto_attribs=auto_attribs),
            _Factory.mixin,
            _ScoreComponent.mixin,
        )(cls)

    if cls is None:
        return _wrap
    else:
        return _wrap(cls)
