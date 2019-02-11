import toolz
import attr

import numpy
import torch
import sparse

from tmol.score.cartbonded.torch_op import CartBondedLengthOp
from tmol.score.cartbonded.params import CartBondedParamResolver
from tmol.score.cartbonded.identification import CartBondedIdentification

import tmol.database

from tmol.utility.args import ignore_unused_kwargs


@attr.s(auto_attribs=True)
class ScoreSetup:
    param_resolver: tmol.score.cartbonded.params.CartBondedParamResolver
    tcoords: torch.Tensor
    tbondlength_atom_indices: torch.Tensor
    tbondlength_indices: torch.Tensor

    @classmethod
    def from_fixture(cls, database, system, torch_device) -> "ScoreSetup":
        coords = system.coords

        param_resolver = CartBondedParamResolver.from_database(
            database.scoring.cartbonded, torch_device
        )
        param_identifier = CartBondedIdentification.setup(
            database.scoring.cartbonded, system.bonds[:, :]
        )

        atom_names = system.atom_metadata["atom_name"].copy()
        res_names = system.atom_metadata["residue_name"].copy()

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
        tcoords = (
            torch.from_numpy(coords).to(device=torch_device).requires_grad_(True)
        )[None, :]

        return cls(
            param_resolver=param_resolver,
            tcoords=tcoords,
            tbondlength_atom_indices=tbondlength_atom_indices,
            tbondlength_indices=tbondlength_indices,
        )


def test_cartbonded_length_op(default_database, ubq_system, torch_device):
    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)
    op = CartBondedLengthOp.from_param_resolver(s.param_resolver)

    V = op.score(s.tbondlength_atom_indices, s.tbondlength_indices, s.tcoords[0, :])

    numpy.testing.assert_allclose(V.detach().sum(), 37.784897, atol=1e-4)


# torch gradcheck
def test_cartbonded_length_gradcheck(default_database, ubq_system, torch_device):
    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)
    op = CartBondedLengthOp.from_param_resolver(s.param_resolver)

    natoms = 32
    mask = (s.tbondlength_atom_indices[:, 0] < natoms) & (
        s.tbondlength_atom_indices[:, 1] < natoms
    )

    tbondlength_atom_indices = s.tbondlength_atom_indices[mask]
    tbondlength_indices = s.tbondlength_indices[mask]

    def eval_cbl(coords):
        v = op.score(tbondlength_atom_indices, tbondlength_indices, coords)
        return v

    coords = s.tcoords[0, :natoms]
    torch.autograd.gradcheck(eval_cbl, (coords.requires_grad_(True),), eps=1e-4)
