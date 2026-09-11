import numpy
import pytest


@pytest.fixture(scope="module")
def dihedral_module():
    from tmol._load_ext import load_module

    return load_module(
        __name__,
        __file__,
        "dihedral.pybind.cpp",
        "tmol.tests.score.dunbrack._dihedral",
    )


@pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64])
@pytest.mark.parametrize("default_angle", [-numpy.pi / 3, 0.0, numpy.pi / 3])
def test_default_dihedral_overwrites_derivative_scratch(
    dihedral_module, dtype, default_angle
):
    measure = (
        dihedral_module.measure_default_float
        if dtype == numpy.float32
        else dihedral_module.measure_default_double
    )
    angle, derivatives = measure(default_angle)
    assert angle == dtype(default_angle)
    numpy.testing.assert_array_equal(derivatives, numpy.zeros((4, 3), dtype=dtype))
