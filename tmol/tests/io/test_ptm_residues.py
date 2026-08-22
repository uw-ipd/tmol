import pytest
import torch
import biotite.structure

from tmol.io import (
    biotite_from_pose_stack,
    extended_pose_stack_from_sequences,
    pose_stack_from_biotite,
)
from tmol.score import beta2016_score_function


PTMS = (
    ("SER", "phosphorylated", "SEP"),
    ("THR", "phosphorylated", "TPO"),
    ("TYR", "phosphorylated", "PTR"),
    ("LYS", "monomethylated", "MLZ"),
    ("LYS", "dimethylated", "MLY"),
    ("LYS", "trimethylated", "M3L"),
)

ROSETTA_IDEAL_BONDS = (
    ("SER", "phosphorylated", (("OG", "P", 1.615), ("P", "O1P", 1.597))),
    ("THR", "phosphorylated", (("OG1", "P", 1.613), ("P", "O1P", 1.551))),
    ("TYR", "phosphorylated", (("OH", "P", 1.608), ("P", "O1P", 1.503))),
    ("LYS", "monomethylated", (("NZ", "CM", 1.463),)),
    (
        "LYS",
        "dimethylated",
        (("NZ", "CH1", 1.474), ("NZ", "CH2", 1.474)),
    ),
    (
        "LYS",
        "trimethylated",
        (("NZ", "CM1", 1.482), ("NZ", "CM2", 1.482), ("NZ", "CM3", 1.482)),
    ),
)


@pytest.mark.parametrize("base,variant,name3", PTMS)
def test_ptm_biotite_round_trip_and_scoring(torch_device, base, variant, name3):
    pose, context = extended_pose_stack_from_sequences(
        f"AX[{base}:{variant}]A",
        device=torch_device,
        return_context=True,
    )

    block_type = pose.packed_block_types.active_block_types[pose.block_type_ind[0, 1]]
    assert block_type.name3 == name3
    assert block_type.io_equiv_class == base

    structure = biotite_from_pose_stack(pose, co=context.canonical_ordering)
    assert biotite.structure.get_residues(structure)[1].tolist() == [
        "ALA",
        name3,
        "ALA",
    ]

    round_trip = pose_stack_from_biotite(structure, torch_device)
    round_trip_type = round_trip.packed_block_types.active_block_types[
        round_trip.block_type_ind[0, 1]
    ]
    assert round_trip_type.name == f"{base}:{variant}"
    connections = round_trip.inter_residue_connections[0, :3, :, 0]
    assert torch.any(connections[0] == 1)
    assert torch.any(connections[1] == 0)
    assert torch.any(connections[1] == 2)
    assert torch.any(connections[2] == 1)

    coords = round_trip.coords.detach().clone().requires_grad_(True)
    score_function = beta2016_score_function(torch_device)
    score = score_function.render_whole_pose_scoring_module(round_trip)(coords)
    assert torch.isfinite(score).all()
    score.sum().backward()
    assert torch.isfinite(coords.grad).all()


def test_phosphate_pdb_atom_aliases_are_accepted(torch_device):
    pose, context = extended_pose_stack_from_sequences(
        "AX[SER:phosphorylated]A",
        device=torch_device,
        return_context=True,
    )
    structure = biotite_from_pose_stack(pose, co=context.canonical_ordering)
    for canonical, pdb_alias in (("O1P", "OP1"), ("O2P", "OP2"), ("O3P", "OP3")):
        structure.atom_name[
            (structure.res_name == "SEP") & (structure.atom_name == canonical)
        ] = pdb_alias

    round_trip = pose_stack_from_biotite(structure, torch_device)
    block_type = round_trip.packed_block_types.active_block_types[
        round_trip.block_type_ind[0, 1]
    ]
    assert block_type.name == "SER:phosphorylated"


@pytest.mark.parametrize("base,variant,bonds", ROSETTA_IDEAL_BONDS)
def test_ptm_ideal_bonds_match_rosetta(torch_device, base, variant, bonds):
    pose = extended_pose_stack_from_sequences(
        f"AX[{base}:{variant}]A", device=torch_device
    )
    block_type = pose.packed_block_types.active_block_types[pose.block_type_ind[0, 1]]
    expanded_coords, _ = pose.expand_coords()

    for atom1, atom2, expected_length in bonds:
        coord1 = expanded_coords[0, 1, block_type.atom_to_idx[atom1]]
        coord2 = expanded_coords[0, 1, block_type.atom_to_idx[atom2]]
        observed_length = torch.linalg.vector_norm(coord1 - coord2)
        torch.testing.assert_close(
            observed_length,
            torch.tensor(expected_length, device=torch_device),
            rtol=0,
            atol=2e-5,
        )
