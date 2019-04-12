from typing import List, Dict, Tuple

from math import nan

import attr
import pytest

import numpy
import torch

from tmol.score.bonded_atom import IndexedBonds, bonded_path_length
from tmol.tests.autograd import gradcheck


@attr.s(auto_attribs=True)
class Case:
    name: str
    coords: torch.tensor
    bonds: torch.tensor
    atom_type_names: List[str]
    num_water: List[int]
    waters: Dict[int, torch.Tensor]
    scores: Dict[Tuple[int, int], torch.Tensor]


test_cases = dict(
    donor_donor=Case(
        name="donor_donor",
        coords=torch.tensor(
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
        waters={
            0: torch.tensor(
                [
                    [-8.0447, 3.7560, 1.6415],
                    [-5.4108, 7.3310, 0.7927],
                    [-6.8406, 4.7060, -2.7763],
                    [nan, nan, nan],
                ]
            ),
            4: torch.tensor(
                [
                    [-7.6573, 5.0963, -3.4426],
                    [-9.2644, 8.3723, -0.7746],
                    [-11.1698, 4.1283, -0.5087],
                    [nan, nan, nan],
                ]
            ),
        },
        scores={
            (0, 4): torch.tensor([0.1678, 0.0000, 0.1325, 0.3948]),
            (4, 0): torch.tensor([0.1678, 0.0000, 0.1325, 0.3948]),
        },
    ),
    ring_sp3=Case(
        # test 3: ring acceptor--sp3 acceptor
        name="ring_sp3",
        coords=torch.tensor(
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
        waters={
            1: torch.tensor(
                [
                    [-7.5179, -1.8575, -5.2835],
                    [-7.8731, 0.4868, -1.3149],
                    [-4.3565, 1.3813, -4.2211],
                    [nan, nan, nan],
                ]
            ),
            4: torch.tensor(
                [
                    [-7.8073, -0.3386, -2.4266],
                    [nan, nan, nan],
                    [nan, nan, nan],
                    [nan, nan, nan],
                ]
            ),
        },
        scores={
            (1, 3): [0.00385956, 0.0001626, 0.0, 0.0],
            (1, 4): [0.0271, 0.0271, 0.0133, 0.2450],
            (1, 5): [0.00369549, 0.0028072, 0.0, 0.0],
            (4, 0): [0.01360676, 0.0135272, 0.0, 0.0],
            (4, 1): [0.0631, 0.0631, 0.0309, 0.2450],
        },
    ),
    sp2_nonpolar=Case(
        name="sp2_nonpolar",
        coords=torch.tensor(
            [
                [-5.282, -0.190, -0.858],
                [-6.520, 0.686, -0.931],
                [-6.652, 1.379, -1.961],
                [-6.932, 4.109, -5.683],
            ]
        ),
        atom_type_names=["CH2", "COO", "OOC", "CH3"],
        bonds=numpy.array([[0, 1], [1, 2]]),
        num_water=[0, 0, 2, 0],
        waters={
            2: torch.tensor(
                [
                    [-5.1302, 1.7371, -4.3176],
                    [-8.5929, 3.2215, -2.8751],
                    [nan, nan, nan],
                    [nan, nan, nan],
                ]
            )
        },
        scores={(2, 3): [0.14107985, 0.04765878, 0., 0.]},
    ),
)


@pytest.mark.parametrize(
    "test_case", list(test_cases.values()), ids=list(test_cases.keys())
)
def test_water_generation(test_case, default_database):
    from tmol.score.lk_ball.torch_ops import AttachedWaters, LKBall

    water_op = AttachedWaters.from_database(default_database, torch.device("cpu"))
    lkb_op = LKBall.from_database(default_database, torch.device("cpu"))

    coords = test_case.coords.to(torch.float)
    indexed_bonds = IndexedBonds.from_bonds(IndexedBonds.to_directed(test_case.bonds))
    bpl = torch.from_numpy(
        bonded_path_length(indexed_bonds.bonds.numpy(), len(coords), 5)
    ).to(torch.float)
    atom_types = water_op.atom_resolver.type_idx(test_case.atom_type_names)

    waters = water_op.apply(coords, atom_types, indexed_bonds)

    assert list((~torch.isnan(waters[..., 0])).sum(dim=-1)) == test_case.num_water
    for i, ewaters in test_case.waters.items():
        torch.testing.assert_allclose(waters[i], ewaters)

    def water_gradf(coords):
        waters = water_op.apply(coords, atom_types, indexed_bonds)
        return waters[~torch.isnan(waters)]

    # Increased eps for single-precision gradcheck.
    gradcheck(water_gradf, (coords.requires_grad_(True),), eps=4e-3)

    ind, V = lkb_op.apply(coords, coords, waters, waters, atom_types, atom_types, bpl)
    scores = torch.sparse_coo_tensor(ind.detach(), V).to_dense()

    for (i, j), escores in test_case.scores.items():
        torch.testing.assert_allclose(scores[i][j], escores, rtol=1e-5, atol=1e-4)

    def score_gradf(coords):
        waters = water_op.apply(coords, atom_types, indexed_bonds)
        ind, V = lkb_op.apply(
            coords, coords, waters, waters, atom_types, atom_types, bpl
        )

        return V

    c2 = coords.detach().requires_grad_(True)

    gradcheck(score_gradf, (c2,), eps=1e-3, rtol=6e-3)
