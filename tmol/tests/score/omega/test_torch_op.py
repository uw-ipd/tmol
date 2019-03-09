import attr
import pandas

import numpy
import torch

from tmol.score.omega.torch_op import OmegaOp


@attr.s(auto_attribs=True)
class ScoreSetup:
    tcoords: torch.Tensor
    tomega_atom_indices: torch.Tensor

    @classmethod
    def from_fixture(cls, database, system, torch_device) -> "ScoreSetup":
        coords = system.coords
        tcoords = (
            torch.from_numpy(coords)
            .to(device=torch_device, dtype=torch.float)
            .requires_grad_(True)
        )
        res_names = system.atom_metadata["residue_name"].copy()

        omegas = numpy.array(
            [
                [
                    x["atom_index_a"],
                    x["atom_index_b"],
                    x["atom_index_c"],
                    x["atom_index_d"],
                ]
                for x in system.torsion_metadata[
                    system.torsion_metadata["name"] == "omega"
                ]
            ]
        )

        omega_defined = numpy.all(omegas != -1, axis=1)
        tomega_atom_indices = torch.from_numpy(omegas[omega_defined, :]).to(
            device=torch_device, dtype=torch.int32
        )

        return cls(tcoords=tcoords, tomega_atom_indices=tomega_atom_indices)


# torch forward op
def test_omega_intra(default_database, ubq_system, torch_device):
    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)
    func = OmegaOp.from_device(torch_device)
    spring_constant = torch.tensor(32.8, device=torch_device, dtype=torch.float)

    batch_scores = func.intra(
        s.tcoords, s.tomega_atom_indices, spring_constant
    ).detach()
    target_scores = torch.tensor(
        [
            0.0286,
            0.4407,
            0.0019,
            0.1767,
            0.0355,
            0.0000,
            0.0406,
            0.0000,
            0.0004,
            0.2490,
            0.1473,
            0.0009,
            0.0342,
            0.0054,
            0.3605,
            0.0426,
            0.2806,
            0.0515,
            0.0603,
            0.0101,
            0.1819,
            0.1775,
            0.0900,
            0.0013,
            0.0191,
            0.0005,
            0.0986,
            0.1578,
            0.0001,
            0.0029,
            0.0074,
            0.1888,
            0.0002,
            0.1221,
            0.0008,
            0.0288,
            0.0051,
            0.0653,
            0.0166,
            0.0566,
            0.6242,
            0.0620,
            0.0595,
            0.0581,
            0.0684,
            0.0022,
            0.0553,
            0.2947,
            0.1665,
            0.0526,
            0.1730,
            0.0763,
            0.1244,
            0.0832,
            0.0797,
            0.0030,
            0.0006,
            0.0052,
            0.0001,
            0.1718,
            0.0997,
            0.0949,
            0.0026,
            0.2705,
            0.1654,
            0.2422,
            0.0409,
            0.1172,
            0.2422,
            0.0244,
            0.0760,
            0.0047,
            0.0013,
            0.0043,
            0.0061,
        ]
    )

    numpy.testing.assert_allclose(batch_scores, target_scores, atol=1e-4)
    numpy.testing.assert_allclose(batch_scores.sum(), 6.741275, atol=1e-4)


# torch gradcheck
def test_omega_intra_gradcheck(default_database, ubq_system, torch_device):
    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)
    func = OmegaOp.from_device(torch_device)

    atom_names = ubq_system.atom_metadata["atom_name"].copy()
    atm_mask = (atom_names == "N") | (atom_names == "CA") | (atom_names == "C")
    t_atm_indices = torch.from_numpy(atm_mask.nonzero()[0]).to(
        device=torch_device, dtype=torch.long
    )
    t_atm_indices = t_atm_indices[2]  # limit runtime
    spring_constant = torch.tensor(32.8, device=torch_device, dtype=torch.float)

    def eval_omega(coords_subset):
        coords = s.tcoords.clone()
        coords[t_atm_indices] = coords_subset
        v = func.intra(coords, s.tomega_atom_indices, spring_constant)
        return v

    masked_coords = s.tcoords[t_atm_indices]
    torch.autograd.gradcheck(
        eval_omega, (masked_coords.requires_grad_(True),), eps=1e-3, atol=5e-3
    )
