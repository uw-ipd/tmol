"""Shared parser for unit-qualified angle scalars in tmol YAML files.

Upstream tmol used ``pint`` to parse strings like ``"180 deg"`` into radians.
We don't want pint as a runtime dependency, so the chemical and scoring DBs
both used to carry a private ``_parse_angle`` copy. This module is the single
home; both DBs import from here.

Forms accepted:
    - ``None`` -> ``None`` (passthrough; callers handle Optional themselves)
    - bare ``int`` / ``float`` -> already in radians, returned as ``float``
    - ``"<num>"`` -> ``float(num)`` (already radians)
    - ``"<num> rad|radian|radians"`` -> ``float(num)``
    - ``"<num> deg|degree|degrees"`` -> ``float(num) * pi / 180``
"""

from __future__ import annotations

import math
from typing import Any


def parse_angle(v: Any) -> float:
    """Parse a number or unit-qualified string into radians."""
    if v is None:
        return v  # type: ignore[return-value]
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        parts = v.strip().split()
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 2:
            val = float(parts[0])
            unit = parts[1].lower()
            if unit in ("rad", "radian", "radians"):
                return val
            if unit in ("deg", "degree", "degrees"):
                return val * math.pi / 180.0
            raise ValueError(f"Unknown angle unit: {unit!r} in {v!r}")
        raise ValueError(f"Cannot parse angle: {v!r}")
    raise TypeError(f"Cannot parse angle from type {type(v).__name__}: {v!r}")


__all__ = ["parse_angle"]


if __name__ == "__main__":
    # ponytail: single self-check, asserts replace a test file.
    assert parse_angle(None) is None
    assert parse_angle(1.5) == 1.5
    assert parse_angle(2) == 2.0 and isinstance(parse_angle(2), float)
    assert parse_angle("1.25") == 1.25
    assert parse_angle("0 rad") == 0.0
    assert parse_angle("180 deg") == math.pi
    assert abs(parse_angle("-90 degrees") + math.pi / 2) < 1e-12
    assert parse_angle("  3.14 radian ") == 3.14
    for bad in ("180 gradian", "1 2 3"):
        try:
            parse_angle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")
    try:
        parse_angle([1, 2])  # type: ignore[arg-type]
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError for list input")
    print("OK")
