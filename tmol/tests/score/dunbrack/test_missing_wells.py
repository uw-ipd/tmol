"""Missing wells must use the same representative in either handedness."""

import attr
import pytest
import torch

from tmol.database.scoring._mirrored_dunbrack import mirror_rotameric_library
from tmol.io import extended_pose_stack_from_sequences
from tmol.kinematics import set_named_torsions
from tmol.score.dunbrack import DunbrackEnergyTerm
from tmol.score.dunbrack._params import DunbrackParamResolver
from tmol.tests.score.dunbrack.test_parameter_identity import setup, render


def test_missing_well_lookup_preserves_mirrors_and_defined_rows(
    default_database, torch_device
):
    database = default_database.scoring.dun
    for library in database.rotameric_libraries:
        data = library.rotameric_data
        # Names are deliberately uninformative; reflection follows metadata.
        mirrored = attr.evolve(mirror_rotameric_library(library), table_name="mirror")
        subset = attr.evolve(
            database,
            rotameric_libraries=(library, mirrored),
            semi_rotameric_libraries=(),
        )
        rot, _, all_chi = DunbrackParamResolver._create_rotind2tableinds(
            subset, torch_device
        )
        n = 3 ** data.rotamers.shape[1]
        assert bool((rot >= 0).all())
        assert bool((rot < data.rotamers.shape[0]).all())
        torch.testing.assert_close(rot[:n], rot[n:].flip(0))
        torch.testing.assert_close(rot, all_chi)
        strides = 3 ** torch.arange(data.rotamers.shape[1] - 1, -1, -1)
        indices = ((data.rotamers - 1) * strides).sum(1).to(torch_device)
        torch.testing.assert_close(
            rot[indices],
            torch.arange(data.nrotamers(), dtype=rot.dtype, device=torch_device),
        )
        aliases = data.rotamer_alias
        if aliases.numel():
            source, target = aliases.chunk(2, dim=1)
            source = ((source - 1) * strides).sum(1).long().to(torch_device)
            target = ((target - 1) * strides).sum(1).long().to(torch_device)
            torch.testing.assert_close(rot[source], rot[target])


@pytest.mark.parametrize(
    "name,chis",
    [
        ("LYS", [60.0, 60.0, -60.0, 60.0]),
        ("ARG", [180.0, -60.0, 60.0, -60.0]),
    ],
)
def test_missing_well_scores_and_gradients_mirror(
    default_database, torch_device, name, chis
):
    left = extended_pose_stack_from_sequences(f"AX[{name}]A", device=torch_device)
    left = set_named_torsions(
        left,
        [0] * 6,
        [1] * 6,
        ["phi", "psi", "chi1", "chi2", "chi3", "chi4"],
        [-60.0, -40.0, *chis],
    )
    right = extended_pose_stack_from_sequences(
        f"X[DALA]X[D{name}]X[DALA]", device=torch_device
    )
    mapping = []
    for block in range(3):
        lt = left.packed_block_types.active_block_types[
            int(left.block_type_ind[0, block])
        ]
        rt = right.packed_block_types.active_block_types[
            int(right.block_type_ind[0, block])
        ]
        left_names = {a.name: i for i, a in enumerate(lt.atoms)}
        lo, ro = int(left.block_coord_offset[0, block]), int(
            right.block_coord_offset[0, block]
        )
        for j, atom in enumerate(rt.atoms):
            i = lo + left_names[atom.name]
            right.coords[0, ro + j] = -left.coords[0, i]
            mapping.append((i, ro + j))

    scores, gradients = [], []
    for pose in (left, right):
        term = DunbrackEnergyTerm(default_database, torch_device)
        setup(term, pose)
        coords = pose.coords.detach().double().requires_grad_(True)
        energy = render(term, pose, True)(coords)[:, 0, 1, 1]
        gradient = torch.autograd.grad(energy.sum(), coords)[0]
        assert bool(torch.isfinite(energy).all()) and bool(
            torch.isfinite(gradient).all()
        )
        scores.append(energy)
        gradients.append(gradient)
    torch.testing.assert_close(scores[0], scores[1], rtol=1e-5, atol=1e-5)
    li, ri = zip(*mapping)
    torch.testing.assert_close(
        gradients[0][0, list(li)], -gradients[1][0, list(ri)], rtol=1e-5, atol=1e-4
    )
