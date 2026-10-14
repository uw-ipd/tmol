"""General-purpose helpers used throughout TMol."""

from toolz import first
from ._device import resolve_device


def unique_val(vals):  # noqa: F811
    """Extract a single, unique value from a collection of values."""
    return just_one(set(vals))


def just_one(vals):  # noqa: F811
    """Extract a single value from a length one collection of values."""
    assert len(vals) == 1
    return first(vals)


__all__ = [
    "just_one",
    "unique_val",
    "resolve_device",
]
