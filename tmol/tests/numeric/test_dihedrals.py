from toolz.curried import compose, map
import numpy
import torch

from tmol.numeric.dihedrals import coord_dihedrals
from tmol.utility.units import parse_angle


def test_coord_dihedrals():
    coords = numpy.array(
        [
            [24.969, 13.428, 30.692],  # N
            [24.044, 12.661, 29.808],  # CA
            [22.785, 13.482, 29.543],  # C
            [21.951, 13.670, 30.431],  # O
            [23.672, 11.328, 30.466],  # CB
            [22.881, 10.326, 29.620],  # CG
            [23.691, 9.935, 28.389],  # CD1
            [22.557, 9.096, 30.459],  # CD2
            [numpy.nan, numpy.nan, numpy.nan],
        ]
    )

    dihedral_atoms = numpy.array(
        [[0, 1, 2, 3], [0, 1, 4, 5], [1, 4, 5, 6], [1, 4, 5, 7], [-1, 0, 1, 3]]
    )

    dihedrals = compose(numpy.array, list, map(parse_angle))(
        ["-71.21515 deg", "-171.94319 deg", "60.82226 deg", "-177.63641 deg", numpy.nan]
    )

    calc_dihedrals = coord_dihedrals(
        coords.index_select(0, dihedral_atoms[:, 0].squeeze()),
        coords.index_select(0, dihedral_atoms[:, 1].squeeze()),
        coords.index_select(0, dihedral_atoms[:, 2].squeeze()),
        coords.index_select(0, dihedral_atoms[:, 3].squeeze()),
    )

    numpy.testing.assert_allclose(calc_dihedrals.numpy(), dihedrals, atol=1e-5)
