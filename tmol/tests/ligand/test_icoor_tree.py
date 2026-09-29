"""The atom tree and internal coordinates TMol builds residue types from (moved from AtomWorks)."""

import biotite.structure.info as info
import numpy as np
import pytest
from atomworks.experimental.protonation.geometry import build_coordinates

from tmol.ligand._icoor_tree import (
    build_atom_tree,
    find_root_atom,
    icoor_geometry_from_coords,
    signed_dihedral_angle,
)


@pytest.mark.parametrize("res_name", ["HEM", "TRP", "ARG", "ATP", "PRO", "TYR"])
def test_a_structure_survives_the_internal_coordinate_round_trip(res_name):
    atoms = info.residue(res_name)
    bonds = [(int(i), int(j)) for i, j, _ in atoms.bonds.as_array()]
    heavy = atoms.element != "H"
    order, parent, grandparents = build_atom_tree(
        len(atoms), bonds, heavy, find_root_atom(atoms.coord, bonds, heavy)
    )
    geometry = icoor_geometry_from_coords(atoms.coord, order, parent, grandparents)

    rebuilt = np.full_like(atoms.coord, np.nan, dtype=np.float64)
    rebuilt[order[:3]] = atoms.coord[order[:3]]
    for position, index in enumerate(order[3:], start=3):
        gp, ggp = grandparents[index]
        assert len({index, parent[index], gp, ggp}) == 4
        rebuilt[index] = build_coordinates(
            rebuilt[parent[index]], rebuilt[gp], rebuilt[ggp], geometry[position]
        )

    np.testing.assert_allclose(rebuilt, atoms.coord, atol=1e-4)


def test_signed_dihedral_angle_is_the_negated_iupac_dihedral():
    square = [
        np.array(p, dtype=float) for p in ([1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 1, 1])
    ]
    assert signed_dihedral_angle(*square) == pytest.approx(np.pi / 2)
    assert signed_dihedral_angle(*square[::-1]) == pytest.approx(np.pi / 2)
    assert signed_dihedral_angle(*(p * [1, 1, -1] for p in square)) == pytest.approx(
        -np.pi / 2
    )
