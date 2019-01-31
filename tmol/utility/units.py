"""`pint <pint:index>`-based unit support functions."""

import math

from typing import Union, Tuple, NewType

import pint
import cattr

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


def parse_bond_angle(v: Union[float, str]) -> float:
    """Parse a bond angle on the range [0, pi) via pint."""
    return parse_angle(v, lim=(u("0 rad"), u("pi rad")))


def parse_dihedral_angle(v) -> float:
    """Parse a dihedral angle on the range [-pi, pi) via pint."""
    return parse_angle(v, lim=(u("-pi rad"), u("pi rad")))


Angle = NewType("Angle", float)
cattr.register_structure_hook(Angle, lambda v, t: parse_angle(v))

BondAngle = NewType("BondAngle", float)
cattr.register_structure_hook(BondAngle, lambda v, t: parse_bond_angle(v))

DihedralAngle = NewType("DihedralAngle", float)
cattr.register_structure_hook(DihedralAngle, lambda v, t: parse_dihedral_angle(v))
