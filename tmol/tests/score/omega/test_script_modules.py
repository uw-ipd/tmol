import attr
import numpy
import torch

from tmol.score.omega.script_modules import OmegaScoreModule


@attr.s(auto_attribs=True)
class ScoreSetup:
    tcoords: torch.Tensor
    tomega_atom_indices: torch.Tensor
    tK: torch.Tensor

    @classmethod
    def from_fixture(cls, database, system, torch_device) -> "ScoreSetup":
        coords = system.coords
        tcoords = (
            torch.from_numpy(coords)
            .to(device=torch_device, dtype=torch.float)
            .requires_grad_(True)
        )

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
        tK = torch.tensor(32.8, device=torch_device, dtype=torch.float)

        return cls(tcoords=tcoords, tomega_atom_indices=tomega_atom_indices, tK=tK)


# torch forward op
def test_omega_intra(default_database, ubq_system, torch_device):
    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)
    op = OmegaScoreModule(s.tomega_atom_indices, s.tK)

    V = op.final(s.tcoords)

    numpy.testing.assert_allclose(V.detach().cpu(), 6.741275, atol=1e-4)


# torch gradcheck
def test_omega_intra_gradcheck(default_database, ubq_system, torch_device):
    s = ScoreSetup.from_fixture(default_database, ubq_system, torch_device)
    op = OmegaScoreModule(s.tomega_atom_indices, s.tK)

    atom_names = ubq_system.atom_metadata["atom_name"].copy()
    atm_mask = (atom_names == "N") | (atom_names == "CA") | (atom_names == "C")
    t_atm_indices = torch.from_numpy(atm_mask.nonzero()[0]).to(
        device=torch_device, dtype=torch.long
    )
    t_atm_indices = t_atm_indices[2]  # limit runtime

    def eval_omega(coords_subset):
        coords = s.tcoords.clone()
        coords[t_atm_indices] = coords_subset
        v = op.final(coords)
        return v

    masked_coords = s.tcoords[t_atm_indices]
    torch.autograd.gradcheck(
        eval_omega, (masked_coords.requires_grad_(True),), eps=1e-3, atol=5e-3
    )
