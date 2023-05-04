import pytest

from math import pi
import tmol.utility.units as units
import pint


def test_angle_parsing():
    valid_values = [
        ".5 pi",
        "1/2 pi rad",
        "90 deg",
        ".5 * 180 deg",
        "90 degrees",
        0.5 * pi,
        0.5 * units.ureg.pi,
        0.5 * units.ureg.pi * units.ureg.rad,
        90 * units.ureg.deg,
    ]

    invalid_values = [
        "3 pi rad",  # Out of range
        90,  # Degree, mistakenly out of range
        720 * units.ureg.deg,
        10 * units.ureg.rad,
        1.4 * units.ureg.angstrom,
        "180",  # Degree, mistakenly out of range
        "pi meters",
    ]

    for v in valid_values:
        assert units.parse_angle(v) == 1 / 2 * pi

    for i in invalid_values:
        with pytest.raises((pint.DimensionalityError, ValueError)):
            units.parse_angle(i)

    units.parse_angle("1/2 pi", (units.u("0"), units.u("pi")))
    units.parse_angle("1/2 pi rad", (units.u("0"), None))

    units.parse_angle("-1/2 pi rad", (None, units.u("pi")))
    with pytest.raises(ValueError):
        units.parse_angle("-1/2 pi rad", (units.u("0"), units.u("pi")))
    with pytest.raises(ValueError):
        units.parse_angle("-1/2 pi rad", (units.u("0"), None))
