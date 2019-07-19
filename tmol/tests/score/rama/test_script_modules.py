import attr
import pandas

import numpy
import torch

from tmol.score.rama.script_modules import RamaScoreModule
from tmol.score.rama.params import RamaParamResolver, RamaParams


@attr.s(auto_attribs=True)
class ScoreSetup:
    param_resolver: RamaParamResolver
    params: RamaParams
    tcoords: torch.Tensor

    @classmethod
    def from_fixture(cls, database, system, torch_device) -> "ScoreSetup":
        coords = system.coords[None,:]
        tcoords = (
            torch.from_numpy(coords)
            .to(device=torch_device, dtype=torch.float)
            .requires_grad_(True)
        )
        res_names = system.atom_metadata["residue_name"].copy()

        param_resolver = RamaParamResolver.from_database(
            database.scoring.rama, torch_device
        )

        phis = numpy.array(
            [
                [
                    x["residue_index"],
                    x["atom_index_a"],
                    x["atom_index_b"],
                    x["atom_index_c"],
                    x["atom_index_d"],
                ]
                for x in system.torsion_metadata[
                    system.torsion_metadata["name"] == "phi"
                ]
            ]
        )
        psis = numpy.array(
            [
                [
                    x["residue_index"],
                    x["atom_index_a"],
                    x["atom_index_b"],
                    x["atom_index_c"],
                    x["atom_index_d"],
                ]
                for x in system.torsion_metadata[
                    system.torsion_metadata["name"] == "psi"
                ]
            ]
        )
        dfphis = pandas.DataFrame(phis)
        dfpsis = pandas.DataFrame(psis)
        phipsis = dfphis.merge(
            dfpsis, left_on=0, right_on=0, suffixes=("_phi", "_psi")
        ).values[:, 1:]

        ramatable_indices = param_resolver.resolve_ramatables(
            res_names[phipsis[:, 5]],  # psi atom 'b'
            res_names[phipsis[:, 7]],  # psi atom 'd'
        )[None,:]

        rama_defined = numpy.all(phipsis != -1, axis=1)
        tphi_atom_indices = torch.from_numpy(phipsis[rama_defined, :4][None,:]).to(
            device=param_resolver.device, dtype=torch.int32
        )
        tpsi_atom_indices = torch.from_numpy(phipsis[rama_defined, 4:][None,:]).to(
            device=param_resolver.device, dtype=torch.int32
        )
        tramatable_indices = torch.from_numpy(ramatable_indices[:,rama_defined]).to(
            device=param_resolver.device, dtype=torch.int32
        )

        params = RamaParams(
            phi_indices=tphi_atom_indices,
            psi_indices=tpsi_atom_indices,
            param_indices=tramatable_indices,
        )

        return cls(param_resolver=param_resolver, params=params, tcoords=tcoords)


# torch forward op
def test_rama_intra(default_database, ubq_system, torch_device):
    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)
    op = RamaScoreModule(s.params, s.param_resolver)

    V = op(s.tcoords)

    numpy.testing.assert_allclose(V.detach().cpu(), -12.743369, atol=1e-4)


# torch gradcheck
def test_rama_intra_gradcheck(default_database, ubq_system, torch_device):
    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)
    op = RamaScoreModule(s.params, s.param_resolver)

    atom_names = ubq_system.atom_metadata["atom_name"].copy()
    atm_mask = (atom_names == "N") | (atom_names == "CA") | (atom_names == "C")
    t_atm_indices = torch.from_numpy(atm_mask.nonzero()[0]).to(
        device=torch_device, dtype=torch.long
    )
    t_atm_indices = t_atm_indices[2]  # limit runtime

    def eval_rama(coords_subset):
        coords = s.tcoords.clone()
        coords[0,t_atm_indices] = coords_subset
        return op(coords)

    masked_coords = s.tcoords[0,t_atm_indices]
    torch.autograd.gradcheck(
        eval_rama, (masked_coords.requires_grad_(True),), eps=1e-3, atol=5e-3
    )

