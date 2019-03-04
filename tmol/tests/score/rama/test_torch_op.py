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

        rama_database = database.scoring.rama
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
            0.0552,
            -0.0626,
            -0.2889,
            -0.9853,
            0.2743,
            -0.9261,
            -0.0722,
            -0.3270,
            -2.1849,
            0.2025,
            -0.3236,
            -0.7654,
            -0.1076,
            0.1122,
            0.6153,
            -0.0791,
            0.1724,
            -0.9218,
            -0.5010,
            0.1756,
            -0.2816,
            -0.1927,
            -0.3496,
            0.9000,
            -0.3277,
            -0.0564,
            -0.2469,
            -0.0718,
            -0.4862,
            0.2991,
            0.2121,
            -0.2941,
            0.3152,
            -2.3336,
            -0.7814,
            -1.3753,
            0.6006,
            -0.1282,
            -0.2308,
            0.0792,
            0.1944,
            -0.5775,
            -1.1942,
            -0.4540,
            0.3580,
            -1.3384,
            0.2310,
            0.0774,
            -0.3643,
            0.2283,
            0.1285,
            0.7094,
            0.0273,
            -0.6875,
            -0.4692,
            0.1738,
            0.2071,
            -0.1251,
            -1.6779,
            -0.7964,
            -0.0079,
            0.2252,
            0.6655,
            -0.5249,
            -0.2056,
            -0.5781,
            0.1301,
            -0.5100,
            -0.9151,
            -0.6002,
            -0.0441,
            -0.4600,
            0.1911,
            -2.4703,
        ]
    )

    numpy.testing.assert_allclose(batch_scores, target_scores, atol=1e-4)
    numpy.testing.assert_allclose(batch_scores.sum(), -21.141438, atol=1e-4)


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
