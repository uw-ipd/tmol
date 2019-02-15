import attr

import numpy
import torch

from tmol.score.cartbonded.torch_op import (
    CartBondedLengthOp,
    CartBondedAngleOp,
    CartBondedTorsionOp,
    CartBondedImproperOp,
    CartBondedHxlTorsionOp,
)
from tmol.score.cartbonded.params import CartBondedParamResolver
from tmol.score.cartbonded.identification import CartBondedIdentification

import tmol.database


@attr.s(auto_attribs=True)
class ScoreSetup:
    param_resolver: tmol.score.cartbonded.params.CartBondedParamResolver
    tcoords: torch.Tensor
    tbondlength_atom_indices: torch.Tensor
    tbondlength_indices: torch.Tensor
    tbondangle_atom_indices: torch.Tensor
    tbondangle_indices: torch.Tensor
    ttorsion_atom_indices: torch.Tensor
    ttorsion_indices: torch.Tensor
    timproper_atom_indices: torch.Tensor
    timproper_indices: torch.Tensor
    thxltorsion_atom_indices: torch.Tensor
    thxltorsion_indices: torch.Tensor

    @classmethod
    def from_fixture(cls, database, system, torch_device) -> "ScoreSetup":
        coords = system.coords
        tcoords = (
            torch.from_numpy(coords)
            .to(device=torch_device, dtype=torch.float)
            .requires_grad_(True)
        )[None, :]

        param_resolver = CartBondedParamResolver.from_database(
            database.scoring.cartbonded, torch_device
        )
        param_identifier = CartBondedIdentification.setup(
            database.scoring.cartbonded, system.bonds
        )

        atom_names = system.atom_metadata["atom_name"].copy()
        res_names = system.atom_metadata["residue_name"].copy()

        # bondlengths
        bondlength_atom_indices = param_identifier.lengths
        bondlength_indices = param_resolver.resolve_lengths(
            res_names[bondlength_atom_indices[:, 0]],  # use atm1 for resid
            atom_names[bondlength_atom_indices[:, 0]],
            atom_names[bondlength_atom_indices[:, 1]],
        )
        bondlength_defined = bondlength_indices != -1

        tbondlength_atom_indices = torch.from_numpy(
            bondlength_atom_indices[bondlength_defined]
        ).to(device=param_resolver.device, dtype=torch.int32)
        tbondlength_indices = torch.from_numpy(
            bondlength_indices[bondlength_defined]
        ).to(device=param_resolver.device, dtype=torch.int32)

        # bondangles
        bondangle_atom_indices = param_identifier.angles
        bondangle_indices = param_resolver.resolve_angles(
            res_names[bondangle_atom_indices[:, 1]],  # use atm2 for resid
            atom_names[bondangle_atom_indices[:, 0]],
            atom_names[bondangle_atom_indices[:, 1]],
            atom_names[bondangle_atom_indices[:, 2]],
        )
        bondangle_defined = bondangle_indices != -1

        tbondangle_atom_indices = torch.from_numpy(
            bondangle_atom_indices[bondangle_defined]
        ).to(device=param_resolver.device, dtype=torch.int32)
        tbondangle_indices = torch.from_numpy(bondangle_indices[bondangle_defined]).to(
            device=param_resolver.device, dtype=torch.int32
        )

        # torsions
        torsion_atom_indices = param_identifier.torsions
        torsion_indices = param_resolver.resolve_torsions(
            res_names[torsion_atom_indices[:, 1]],  # use atm2 for resid
            atom_names[torsion_atom_indices[:, 0]],
            atom_names[torsion_atom_indices[:, 1]],
            atom_names[torsion_atom_indices[:, 2]],
            atom_names[torsion_atom_indices[:, 3]],
        )
        torsion_defined = torsion_indices != -1

        ttorsion_atom_indices = torch.from_numpy(
            torsion_atom_indices[torsion_defined]
        ).to(device=param_resolver.device, dtype=torch.int32)
        ttorsion_indices = torch.from_numpy(torsion_indices[torsion_defined]).to(
            device=param_resolver.device, dtype=torch.int32
        )

        # impropers
        improper_atom_indices = param_identifier.impropers
        improper_indices = param_resolver.resolve_impropers(
            res_names[improper_atom_indices[:, 2]],  # use atm3 for resid
            atom_names[improper_atom_indices[:, 0]],
            atom_names[improper_atom_indices[:, 1]],
            atom_names[improper_atom_indices[:, 2]],
            atom_names[improper_atom_indices[:, 3]],
        )
        improper_defined = improper_indices != -1

        timproper_atom_indices = torch.from_numpy(
            improper_atom_indices[improper_defined]
        ).to(device=param_resolver.device, dtype=torch.int32)
        timproper_indices = torch.from_numpy(improper_indices[improper_defined]).to(
            device=param_resolver.device, dtype=torch.int32
        )

        # hxl torsions
        # combine resolved atom indices and bondangle indices
        hxltorsion_atom_indices = param_identifier.torsions
        hxltorsion_indices = param_resolver.resolve_hxltorsions(
            res_names[hxltorsion_atom_indices[:, 2]],  # use atm3 for resid
            atom_names[hxltorsion_atom_indices[:, 0]],
            atom_names[hxltorsion_atom_indices[:, 1]],
            atom_names[hxltorsion_atom_indices[:, 2]],
            atom_names[hxltorsion_atom_indices[:, 3]],
        )

        # remove undefined indices
        hxltorsion_defined = hxltorsion_indices != -1
        thxltorsion_atom_indices = torch.from_numpy(
            hxltorsion_atom_indices[hxltorsion_defined]
        ).to(device=param_resolver.device, dtype=torch.int32)
        thxltorsion_indices = torch.from_numpy(
            hxltorsion_indices[hxltorsion_defined]
        ).to(device=param_resolver.device, dtype=torch.int32)

        return cls(
            param_resolver=param_resolver,
            tcoords=tcoords,
            tbondlength_atom_indices=tbondlength_atom_indices,
            tbondlength_indices=tbondlength_indices,
            tbondangle_atom_indices=tbondangle_atom_indices,
            tbondangle_indices=tbondangle_indices,
            ttorsion_atom_indices=ttorsion_atom_indices,
            ttorsion_indices=ttorsion_indices,
            timproper_atom_indices=timproper_atom_indices,
            timproper_indices=timproper_indices,
            thxltorsion_atom_indices=thxltorsion_atom_indices,
            thxltorsion_indices=thxltorsion_indices,
        )


def test_cartbonded_length_op(default_database, ubq_system, torch_device):
    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)
    op = CartBondedLengthOp.from_param_resolver(s.param_resolver)

    V = op.score(s.tcoords[0, :], s.tbondlength_atom_indices, s.tbondlength_indices)

    numpy.testing.assert_allclose(V.detach().sum(), 37.78476, atol=1e-3, rtol=0)


def test_cartbonded_length_gradcheck(default_database, ubq_system, torch_device):
    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)
    op = CartBondedLengthOp.from_param_resolver(s.param_resolver)

    t_atm_indices = torch.arange(24, dtype=torch.long)

    def eval_cbl(coords_subset):
        coords = s.tcoords[0, ...].clone()
        coords[t_atm_indices] = coords_subset
        v = op.score(coords, s.tbondlength_atom_indices, s.tbondlength_indices)
        return v

    masked_coords = s.tcoords[0, t_atm_indices]
    torch.autograd.gradcheck(
        eval_cbl, (masked_coords.requires_grad_(True),), eps=1e-3, atol=1e-3
    )


def test_cartbonded_angle_op(default_database, ubq_system, torch_device):
    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)
    op = CartBondedAngleOp.from_param_resolver(s.param_resolver)

    V = op.score(s.tcoords[0, :], s.tbondangle_atom_indices, s.tbondangle_indices)

    numpy.testing.assert_allclose(V.detach().sum(), 183.578, atol=1e-3, rtol=0)


def test_cartbonded_angle_gradcheck(default_database, ubq_system, torch_device):
    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)
    op = CartBondedAngleOp.from_param_resolver(s.param_resolver)

    t_atm_indices = torch.arange(24, dtype=torch.long)

    def eval_cba(coords_subset):
        coords = s.tcoords[0, ...].clone()
        coords[t_atm_indices] = coords_subset
        v = op.score(coords, s.tbondangle_atom_indices, s.tbondangle_indices)
        return v

    masked_coords = s.tcoords[0, t_atm_indices]
    torch.autograd.gradcheck(
        eval_cba, (masked_coords.requires_grad_(True),), eps=1e-3, atol=1e-3
    )


def test_cartbonded_torsion_op(default_database, ubq_system, torch_device):
    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)
    op = CartBondedTorsionOp.from_param_resolver(s.param_resolver)

    V = op.score(s.tcoords[0, :], s.ttorsion_atom_indices, s.ttorsion_indices)

    numpy.testing.assert_allclose(V.detach().sum(), 50.5842, atol=1e-3, rtol=0)


def test_cartbonded_torsion_gradcheck(default_database, ubq_system, torch_device):
    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)
    op = CartBondedTorsionOp.from_param_resolver(s.param_resolver)

    t_atm_indices = torch.arange(24, dtype=torch.long)

    def eval_cbt(coords_subset):
        coords = s.tcoords[0, ...].clone()
        coords[t_atm_indices] = coords_subset
        v = op.score(coords, s.ttorsion_atom_indices, s.ttorsion_indices)
        return v

    masked_coords = s.tcoords[0, t_atm_indices]
    torch.autograd.gradcheck(
        eval_cbt, (masked_coords.requires_grad_(True),), eps=1e-3, atol=5e-3
    )  # needs higher tol...


def test_cartbonded_improper_op(default_database, ubq_system, torch_device):
    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)
    op = CartBondedImproperOp.from_param_resolver(s.param_resolver)

    V = op.score(s.tcoords[0, :], s.timproper_atom_indices, s.timproper_indices)

    numpy.testing.assert_allclose(V.detach().sum(), 9.43055, atol=1e-3, rtol=0)


def test_cartbonded_improper_gradcheck(default_database, ubq_system, torch_device):
    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)
    op = CartBondedImproperOp.from_param_resolver(s.param_resolver)

    t_atm_indices = torch.arange(24, dtype=torch.long)

    def eval_cbi(coords_subset):
        coords = s.tcoords[0, ...].clone()
        coords[t_atm_indices] = coords_subset
        v = op.score(coords, s.timproper_atom_indices, s.timproper_indices)
        return v

    masked_coords = s.tcoords[0, t_atm_indices]
    torch.autograd.gradcheck(
        eval_cbi, (masked_coords.requires_grad_(True),), eps=1e-3, atol=5e-3
    )  # needs higher tol...


def test_cartbonded_hxltorsion_op(default_database, ubq_system, torch_device):
    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)
    op = CartBondedHxlTorsionOp.from_param_resolver(s.param_resolver)

    V = op.score(s.tcoords[0, :], s.thxltorsion_atom_indices, s.thxltorsion_indices)

    numpy.testing.assert_allclose(V.detach().sum(), 47.4197, atol=1e-3, rtol=0)


def test_cartbonded_hxltorsion_gradcheck(default_database, ubq_system, torch_device):
    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)
    op = CartBondedHxlTorsionOp.from_param_resolver(s.param_resolver)

    res_names = ubq_system.atom_metadata["residue_name"].copy()
    atm_mask = (res_names == "SER") | (res_names == "THR") | (res_names == "TYR")
    t_atm_indices = torch.from_numpy(atm_mask.nonzero()[0]).to(
        device=torch_device, dtype=torch.long
    )
    t_atm_indices = t_atm_indices[:33]  # limit runtime

    def eval_cbh(coords_subset):
        coords = s.tcoords[0, ...].clone()
        coords[t_atm_indices] = coords_subset
        v = op.score(coords, s.thxltorsion_atom_indices, s.thxltorsion_indices)
        return v

    masked_coords = s.tcoords[0, t_atm_indices]
    torch.autograd.gradcheck(
        eval_cbh, (masked_coords.requires_grad_(True),), eps=1e-3, atol=5e-3
    )  # needs higher tol...
