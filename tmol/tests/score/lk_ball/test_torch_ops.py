from typing import List

import attr
import pytest

import numpy
import torch

from tmol.score.bonded_atom import IndexedBonds
from tmol.tests.autograd import gradcheck


@attr.s(auto_attribs=True)
class Case:
    coords: torch.tensor
    bonds: torch.tensor
    atom_type_names: List[str]
    num_water: List[int]


test_cases = dict(
    donor_donor=Case(
        coords=torch.DoubleTensor(
            [
                [-6.007, 4.706, -0.074],
                [-6.747, 4.361, 0.549],
                [-5.791, 5.657, 0.240],
                [-6.305, 4.706, -1.040],
                [-10.018, 6.062, -2.221],
                [-9.160, 5.711, -2.665],
                [-9.745, 6.899, -1.697],
                [-10.429, 5.372, -1.610],
            ]
        ),
        atom_type_names=[
            "Nlys",
            "Hpol",
            "Hpol",
            "Hpol",
            "Nlys",
            "Hpol",
            "Hpol",
            "Hpol",
        ],
        bonds=numpy.array([[0, 1], [0, 2], [0, 3], [4, 5], [4, 6], [4, 7]]),
        num_water=[3, 0, 0, 0, 3, 0, 0, 0],
    ),
    ring_sp3=Case(
        # test 3: ring acceptor--sp3 acceptor
        coords=torch.DoubleTensor(
            [
                [-5.250, -1.595, -2.543],  # SER CB
                [-6.071, -0.619, -3.193],  # SER OG
                [-5.489, 0.060, -3.542],  # SER HG
                [-10.628, 2.294, -1.933],  # HIS CG
                [-9.991, 1.160, -1.435],  # HIS ND1
                [-10.715, 0.960, -0.319],  # HIS CE1
            ]
        ),
        atom_type_names=["CH2", "OH", "Hpol", "CH0", "NhisDDepro", "Caro"],
        bonds=numpy.array([[0, 1], [1, 2], [3, 4], [4, 5]]),
        num_water=[0, 3, 0, 0, 1, 0],
    ),
    sp2_nonpolar=Case(
        coords=torch.DoubleTensor(
            [
                [-5.282, -0.190, -0.858],
                [-6.520, 0.686, -0.931],
                [-6.652, 1.379, -1.961],
                [-6.932, 4.109, -5.683],
            ]
        ),
        atom_type_names=["CH2", "COO", "OOC", "CH3"],
        bonds=numpy.array([[0, 1], [0, 2], [1, 2]]),
        num_water=[0, 0, 2, 0],
    ),
)


@pytest.mark.parametrize(
    "test_case", list(test_cases.values()), ids=list(test_cases.keys())
)
def test_water_generation(test_case, default_database):
    from tmol.score.lk_ball.potentials.compiled import AttachedWaters

    op = AttachedWaters.from_database(default_database, torch.device("cpu"))

    coords = test_case.coords.to(torch.float)
    indexed_bonds = IndexedBonds.from_bonds(IndexedBonds.to_directed(test_case.bonds))
    atom_types = op.atom_resolver.type_idx(test_case.atom_type_names)

    waters = op.for_bonded_atoms(coords, atom_types, indexed_bonds)

    assert list((~torch.isnan(waters[..., 0])).sum(dim=-1)) == test_case.num_water

    def gradf(coords):
        waters = op.for_bonded_atoms(coords, atom_types, indexed_bonds)
        return waters[~torch.isnan(waters)]

    # Increased eps for single-precision gradcheck.
    gradcheck(gradf, (coords.requires_grad_(True),), eps=3e-3)
