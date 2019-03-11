import attr
import pandas

import numpy
import torch

from tmol.score.rama.torch_op import RamaOp
from tmol.score.rama.params import RamaParamResolver


@attr.s(auto_attribs=True)
class ScoreSetup:
    param_resolver: RamaParamResolver
    tcoords: torch.Tensor
    tphi_atom_indices: torch.Tensor
    tpsi_atom_indices: torch.Tensor
    tramatable_indices: torch.Tensor

    @classmethod
    def from_fixture(cls, database, system, torch_device) -> "ScoreSetup":
        coords = system.coords
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
        )

        rama_defined = numpy.all(phipsis != -1, axis=1)
        tphi_atom_indices = torch.from_numpy(phipsis[rama_defined, :4]).to(
            device=param_resolver.device, dtype=torch.int32
        )
        tpsi_atom_indices = torch.from_numpy(phipsis[rama_defined, 4:]).to(
            device=param_resolver.device, dtype=torch.int32
        )
        tramatable_indices = torch.from_numpy(ramatable_indices[rama_defined]).to(
            device=param_resolver.device, dtype=torch.int32
        )

        return cls(
            param_resolver=param_resolver,
            tcoords=tcoords,
            tphi_atom_indices=tphi_atom_indices,
            tpsi_atom_indices=tpsi_atom_indices,
            tramatable_indices=tramatable_indices,
        )


# torch forward op
def test_rama_intra(default_database, ubq_system, torch_device):
    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)
    func = RamaOp.from_param_resolver(s.param_resolver)

    batch_scores = func.intra(
        s.tcoords, s.tphi_atom_indices, s.tpsi_atom_indices, s.tramatable_indices
    ).detach()
    target_scores = torch.tensor(
        [
            -0.3400,
            -0.1960,
            -0.5714,
            -0.7717,
            0.0106,
            -0.7053,
            0.1066,
            -0.1264,
            -2.2468,
            -0.0938,
            -0.2140,
            -0.5902,
            0.0402,
            0.0593,
            0.5039,
            0.4051,
            0.0028,
            -0.2385,
            -0.7350,
            -0.2852,
            -0.5702,
            0.0485,
            -0.4191,
            1.2609,
            0.3812,
            -0.3369,
            -0.4135,
            -0.4174,
            -0.2849,
            0.9794,
            0.8338,
            -0.4215,
            1.3010,
            -2.3470,
            -1.1876,
            -1.0530,
            0.4878,
            -0.1165,
            -0.6052,
            -0.2483,
            0.2899,
            -0.5349,
            -1.1289,
            0.3683,
            1.0286,
            -0.6671,
            -0.0719,
            -0.2612,
            -0.7437,
            -0.0319,
            1.2596,
            0.5173,
            0.2713,
            -0.6701,
            -0.2783,
            -0.2306,
            0.5705,
            -0.1212,
            -1.4027,
            -0.8667,
            0.3735,
            0.4209,
            1.4554,
            -0.9150,
            -0.1044,
            -0.6698,
            -0.1469,
            -0.2920,
            -0.7826,
            -0.6802,
            0.4629,
            -0.7489,
            0.8685,
            -0.1668,
        ]
    )

    numpy.testing.assert_allclose(batch_scores, target_scores, atol=1e-4)
    numpy.testing.assert_allclose(batch_scores.sum(), -12.743369, atol=1e-4)


# torch gradcheck
def test_rama_intra_gradcheck(default_database, ubq_system, torch_device):
    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)
    func = RamaOp.from_param_resolver(s.param_resolver)

    atom_names = ubq_system.atom_metadata["atom_name"].copy()
    atm_mask = (atom_names == "N") | (atom_names == "CA") | (atom_names == "C")
    t_atm_indices = torch.from_numpy(atm_mask.nonzero()[0]).to(
        device=torch_device, dtype=torch.long
    )
    t_atm_indices = t_atm_indices[2]  # limit runtime

    def eval_rama(coords_subset):
        coords = s.tcoords.clone()
        coords[t_atm_indices] = coords_subset
        v = func.intra(
            coords, s.tphi_atom_indices, s.tpsi_atom_indices, s.tramatable_indices
        )
        return v

    masked_coords = s.tcoords[t_atm_indices]
    torch.autograd.gradcheck(
        eval_rama, (masked_coords.requires_grad_(True),), eps=1e-3, atol=5e-3
    )
