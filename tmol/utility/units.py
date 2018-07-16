"""`pint`-based unit support functions."""

from typing import Union, Tuple

import pint

ureg = pint.UnitRegistry()
u = ureg.parse_expression


def parse_angle(
        angle: Union[float, str], lim: Tuple = (u("-2pi rad"), u("2pi rad"))
) -> float:
    """Parse an angle and return value in radians."""

    if isinstance(angle, str):
        val = u(angle)
    else:
        val = angle

    if not isinstance(val, pint.unit.Number):
        # May have parsed to a just a number or a
        # quantity with units, if so convert to radians
        val.to(ureg.rad)

    if lim:
        minv, maxv = lim
        if minv is not None and val < minv:
            raise ValueError(
                f"angle: {angle!r} outside of allowed range: {lim}"
            )
        if maxv is not None and val > maxv:
            raise ValueError(
                f"angle: {angle!r} outside of allowed range: {lim}"
            )

    return float(val)
