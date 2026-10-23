"""General-purpose helpers used throughout TMol."""

from . import _args  # noqa: F401
from . import _attr  # noqa: F401
from . import _auto_number  # noqa: F401
from . import _biotite_util  # noqa: F401
from . import _categorical  # noqa: F401
from . import _cpp_extension  # noqa: F401
from . import _cumsum  # noqa: F401
from ._device import resolve_device  # noqa: F401
from ._dicttoolz import _vals  # noqa: F401
from . import _log  # noqa: F401
from . import _mixins  # noqa: F401
from . import _numba  # noqa: F401
from . import _nvtx  # noqa: F401
from . import _units  # noqa: F401

from toolz import first


def _unique_val(vals):  # noqa: F811
    """Extract a single, unique value from a collection of values."""
    return _just_one(set(vals))


def _just_one(vals):  # noqa: F811
    """Extract a single value from a length one collection of values."""
    assert len(vals) == 1
    return first(vals)


__all__ = [
    "resolve_device",
]
