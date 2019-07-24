from typing import List

import attr
import pytest

import numpy
import torch

from tmol.score.bonded_atom import IndexedBonds, bonded_path_length
from tmol.tests.autograd import gradcheck

from tmol.score.ljlk.params import LJLKParamResolver
from tmol.score.chemical_database import AtomTypeParamResolver

from tmol.score.lk_ball.script_modules import LKBallIntraModule, LKBallInterModule


@attr.s(auto_attribs=True)
class Case:
    name: str
    coords: torch.tensor
    bonds: torch.tensor
    atom_type_names: List[str]
    expected_score: torch.tensor
    split: int


test_cases = dict(
    donor_donor=Case(
        name="donor_donor",
        coords=torch.tensor(
            [[
                [-6.007, 4.706, -0.074],
                [-6.747, 4.361, 0.549],
                [-5.791, 5.657, 0.240],
                [-6.305, 4.706, -1.040],
                [-10.018, 6.062, -2.221],
                [-9.160, 5.711, -2.665],
                [-9.745, 6.899, -1.697],
                [-10.429, 5.372, -1.610],
            ]]
        ),
        atom_type_names=[[
            "Nlys",
            "Hpol",
            "Hpol",
            "Hpol",
            "Nlys",
            "Hpol",
            "Hpol",
            "Hpol",
        ]],
        bonds=numpy.array([[0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 4, 5], [0, 4, 6], [0, 4, 7]]),
        split=4,
        expected_score=torch.tensor([[0.3355, 0.0000, 0.2649, 0.7896]]),
    ),
    ring_sp3=Case(
        # test 3: ring acceptor--sp3 acceptor
        name="ring_sp3",
        coords=torch.tensor(
            [[
                [-5.250, -1.595, -2.543],  # SER CB
                [-6.071, -0.619, -3.193],  # SER OG
                [-5.489, 0.060, -3.542],  # SER HG
                [-10.628, 2.294, -1.933],  # HIS CG
                [-9.991, 1.160, -1.435],  # HIS ND1
                [-10.715, 0.960, -0.319],  # HIS CE1
            ]]
        ),
        atom_type_names=[["CH2", "OH", "Hpol", "CH0", "NhisDDepro", "Caro"]],
        bonds=numpy.array([[0, 0, 1], [0, 1, 2], [0, 3, 4], [0, 4, 5]]),
        split=3,
        expected_score=torch.tensor([[0.1114, 0.1067, 0.0442, 0.4900]]),
    ),
    sp2_nonpolar=Case(
        name="sp2_nonpolar",
        coords=torch.tensor(
            [[
                [-5.282, -0.190, -0.858],
                [-6.520, 0.686, -0.931],
                [-6.652, 1.379, -1.961],
                [-6.932, 4.109, -5.683],
            ]]
        ),
        atom_type_names=[["CH2", "COO", "OOC", "CH3"]],
        bonds=numpy.array([[0, 0, 1], [0, 1, 2]]),
        split=3,
        expected_score=torch.tensor([[0.1411, 0.0477, 0.0000, 0.0000]]),
    ),
)


@pytest.mark.parametrize(
    "test_case", list(test_cases.values()), ids=list(test_cases.keys())
)
def test_lkball_intra(test_case, torch_device, default_database):
    param_resolver = LJLKParamResolver.from_database(
        default_database.chemical, default_database.scoring.ljlk, torch_device
    )
    atom_type_resolver = AtomTypeParamResolver.from_database(
        default_database.chemical, torch_device
    )

    coords = test_case.coords.to(dtype=torch.float, device=torch_device)
    indexed_bonds = IndexedBonds.from_bonds(
        IndexedBonds.to_directed(test_case.bonds)
    ).to(torch_device)

    print("indexed_bonds.bonds", indexed_bonds.bonds.shape)
    print("indexed_bonds.bond_spans", indexed_bonds.bond_spans.shape)

    bpl = torch.from_numpy(
        bonded_path_length(indexed_bonds.bonds[0].cpu().numpy(), coords.shape[1], 5)
    ).to(dtype=torch.float, device=torch_device)[None, :]

    atom_types = atom_type_resolver.type_idx(test_case.atom_type_names)

    polars = torch.nonzero(
        atom_type_resolver.params.is_acceptor[atom_types]
        + atom_type_resolver.params.is_donor[atom_types]
    ).reshape((1, -1))
    occluders = torch.nonzero(
        1 - atom_type_resolver.params.is_hydrogen[atom_types]
    ).reshape((1, -1))

    op = LKBallIntraModule(param_resolver, atom_type_resolver)
    op.to(coords)

    val = op(
        coords,
        polars,
        occluders,
        atom_types,
        bpl,
        indexed_bonds.bonds,
        indexed_bonds.bond_spans,
    )

    print("val", val)
    print("test_case.expected_score", test_case.expected_score)

    torch.testing.assert_allclose(
        val.cpu(), test_case.expected_score, atol=1e-4, rtol=1e-3
    )

    def val(c):
        return op(
            c,
            polars,
            occluders,
            atom_types,
            bpl,
            indexed_bonds.bonds,
            indexed_bonds.bond_spans,
        )

    print("val", val)

    gradcheck(val, (coords.requires_grad_(True),), eps=1e-3, atol=5e-4)


@pytest.mark.parametrize(
    "test_case", list(test_cases.values()), ids=list(test_cases.keys())
)
def test_lkball_inter(test_case, torch_device, default_database):
    param_resolver = LJLKParamResolver.from_database(
        default_database.chemical, default_database.scoring.ljlk, torch_device
    )
    atom_type_resolver = AtomTypeParamResolver.from_database(
        default_database.chemical, torch_device
    )

    coords = test_case.coords.to(dtype=torch.float, device=torch_device)
    indexed_bonds = IndexedBonds.from_bonds(
        IndexedBonds.to_directed(test_case.bonds)
    ).to(torch_device)

    print("indexed_bonds.bonds", indexed_bonds.bonds.shape)
    print("indexed_bonds.bond_spans", indexed_bonds.bond_spans.shape)

    bpl = torch.from_numpy(
        bonded_path_length(indexed_bonds.bonds[0].cpu().numpy(), coords.shape[1], 5)
    ).to(dtype=torch.float, device=torch_device)[None, :]

    atom_types = atom_type_resolver.type_idx(test_case.atom_type_names)

    polarsI = torch.nonzero(
        atom_type_resolver.params.is_acceptor[atom_types[:, :test_case.split]]
        + atom_type_resolver.params.is_donor[atom_types[:, :test_case.split]]
    ).reshape((1,-1))
    occludersI = torch.nonzero(
        1 - atom_type_resolver.params.is_hydrogen[atom_types[:, :test_case.split]]
    ).reshape((1,-1))

    polarsJ = torch.nonzero(
        atom_type_resolver.params.is_acceptor[atom_types[:, test_case.split:]]
        + atom_type_resolver.params.is_donor[atom_types[:, test_case.split:]]
    ).reshape((1,-1))
    occludersJ = torch.nonzero(
        1 - atom_type_resolver.params.is_hydrogen[atom_types[:, test_case.split:]]
    ).reshape((1,-1))

    op = LKBallInterModule(param_resolver, atom_type_resolver)
    op.to(coords)

    val = op(
        coords[:, :test_case.split],
        polarsI,
        occludersI,
        atom_types[:, :test_case.split],
        coords[:, test_case.split:],
        polarsJ,
        occludersJ,
        atom_types[:, test_case.split:],
        bpl[:, :test_case.split, test_case.split:],
        indexed_bonds.bonds,
        indexed_bonds.bond_spans,
    )

    torch.testing.assert_allclose(
        val.cpu(), test_case.expected_score, atol=1e-4, rtol=1e-3
    )

    def val(c):
        return op(
            c,
            polarsI,
            occludersI,
            atom_types[:, :test_case.split],
            coords[:, test_case.split:],
            polarsJ,
            occludersJ,
            atom_types[:, test_case.split :],
            bpl,
            indexed_bonds.bonds,
            indexed_bonds.bond_spans,
        )

    gradcheck(
        val, (coords[: test_case.split].requires_grad_(True),), eps=1e-3, atol=5e-4
    )
