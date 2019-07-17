import attr

import numpy
import torch

from tmol.score.cartbonded.script_modules import CartBondedModule
from tmol.score.cartbonded.params import CartBondedParamResolver
from tmol.score.cartbonded.identification import CartBondedIdentification
from tmol.score.bonded_atom import IndexedBonds

import tmol.database


@attr.s(auto_attribs=True)
class ScoreSetup:
    param_resolver: tmol.score.cartbonded.params.CartBondedParamResolver
    tcoords: torch.Tensor
    tbondlength_indices: torch.Tensor
    tbondangle_indices: torch.Tensor
    ttorsion_indices: torch.Tensor
    timproper_indices: torch.Tensor
    thxltorsion_indices: torch.Tensor

    @classmethod
    def from_fixture(cls, database, system, torch_device) -> "ScoreSetup":
        coords = system.coords
        tcoords = (
            torch.from_numpy(coords)
            .to(device=torch_device, dtype=torch.float)
            .requires_grad_(True)
        )[None, :]

        system_size = numpy.max(system.bonds)
        indexed_bonds = IndexedBonds.from_bonds(
            IndexedBonds.to_directed(system.bonds), minlength=system_size
        )
        param_resolver = CartBondedParamResolver.from_database(
            database.scoring.cartbonded, torch_device
        )
        param_identifier = CartBondedIdentification.setup(
            database.scoring.cartbonded, indexed_bonds
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

        tbondlength_indices = torch.cat(
            [
                torch.tensor(bondlength_atom_indices[bondlength_defined]),
                torch.tensor(bondlength_indices[bondlength_defined, None]),
            ],
            dim=1,
        ).to(device=torch_device, dtype=torch.int64)

        # bondangles
        bondangle_atom_indices = param_identifier.angles
        bondangle_indices = param_resolver.resolve_angles(
            res_names[bondangle_atom_indices[:, 1]],  # use atm2 for resid
            atom_names[bondangle_atom_indices[:, 0]],
            atom_names[bondangle_atom_indices[:, 1]],
            atom_names[bondangle_atom_indices[:, 2]],
        )
        bondangle_defined = bondangle_indices != -1

        tbondangle_indices = torch.cat(
            [
                torch.tensor(bondangle_atom_indices[bondangle_defined]),
                torch.tensor(bondangle_indices[bondangle_defined, None]),
            ],
            dim=1,
        ).to(device=torch_device, dtype=torch.int64)

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

        ttorsion_indices = torch.cat(
            [
                torch.tensor(torsion_atom_indices[torsion_defined]),
                torch.tensor(torsion_indices[torsion_defined, None]),
            ],
            dim=1,
        ).to(device=torch_device, dtype=torch.int64)

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

        timproper_indices = torch.cat(
            [
                torch.tensor(improper_atom_indices[improper_defined]),
                torch.tensor(improper_indices[improper_defined, None]),
            ],
            dim=1,
        ).to(device=torch_device, dtype=torch.int64)

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
        thxltorsion_indices = torch.cat(
            [
                torch.tensor(hxltorsion_atom_indices[hxltorsion_defined]),
                torch.tensor(hxltorsion_indices[hxltorsion_defined, None]),
            ],
            dim=1,
        ).to(device=torch_device, dtype=torch.int64)

        return cls(
            param_resolver=param_resolver,
            tcoords=tcoords,
            tbondlength_indices=tbondlength_indices,
            tbondangle_indices=tbondangle_indices,
            ttorsion_indices=ttorsion_indices,
            timproper_indices=timproper_indices,
            thxltorsion_indices=thxltorsion_indices,
        )


def test_cartbonded_op(default_database, ubq_system, torch_device):
    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)
    op = CartBondedModule(s.param_resolver)

    V = op(
        s.tcoords[0, :],
        s.tbondlength_indices,
        s.tbondangle_indices,
        s.ttorsion_indices,
        s.timproper_indices,
        s.thxltorsion_indices,
    )

    numpy.testing.assert_allclose(
        V.detach().cpu(),
        [37.78476, 183.578, 50.5842, 9.43055, 47.4197],
        atol=1e-3,
        rtol=0,
    )


def test_cartbonded_gradcheck(default_database, ubq_system, torch_device):
    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)
    op = CartBondedModule(s.param_resolver)

    t_atm_indices = torch.arange(24, dtype=torch.long)

    def eval_cb(coords_subset):
        coords = s.tcoords[0, ...].clone()
        coords[t_atm_indices] = coords_subset
        v = op(
            coords,
            s.tbondlength_indices,
            s.tbondangle_indices,
            s.ttorsion_indices,
            s.timproper_indices,
            s.thxltorsion_indices,
        )
        return v

    masked_coords = s.tcoords[0, t_atm_indices]
    torch.autograd.gradcheck(
        eval_cb, (masked_coords.requires_grad_(True),), eps=1e-2, atol=2e-2
    )
