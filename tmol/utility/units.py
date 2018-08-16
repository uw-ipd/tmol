"""`pint <pint:index>`-based unit support functions."""

import math

from typing import Union, Tuple

import pint

ureg = pint.UnitRegistry()
u = ureg.parse_expression


def parse_angle(
    angle: Union[float, str], lim: Tuple[float, float] = (-2 * math.pi, 2 * math.pi)
) -> float:
    """Parse an angle via :doc:`pint <pint:index>` and convert to radians.

    Args:
        angle: Unit-qualified angle or float value in radians.
        lim: Raise ValueError if outside [min, max] range in radians.

    Returns:
        Angle in radians.

    """

    if isinstance(angle, str):
        val = ureg.parse_expression(angle)
    else:
        val = angle

    if not isinstance(val, pint.unit.Number):
        # May have parsed to a just a number or a
        # quantity with units, if so convert to radians
        val.to(ureg.rad)

    if lim:
        minv, maxv = lim
        if minv is not None and val < minv:
            raise ValueError(f"angle: {angle!r} outside of allowed range: {lim}")
        if maxv is not None and val > maxv:
            raise ValueError(f"angle: {angle!r} outside of allowed range: {lim}")

    return float(val)
