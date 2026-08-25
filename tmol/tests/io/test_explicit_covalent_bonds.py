import biotite.structure as struc
import numpy as np
import pytest
import torch

from tmol.database import ParameterDatabase
from tmol.io import biotite_from_pose_stack, pose_stack_from_biotite
from tmol.io._covalent_bonds import (
    _explicit_cross_residue_bonds,
    augment_database_for_covalent_bonds,
)


def _crosslinked_protein(biotite_1ubq):
    structure = biotite_1ubq.copy()
    starts = struc.get_residue_starts(structure)
    structure = structure[: starts[3]].copy()
    starts = struc.get_residue_starts(structure)
    ends = np.append(starts[1:], structure.array_length())
    atom1 = next(i for i in range(starts[0], ends[0]) if structure.atom_name[i] == "O")
    atom2 = next(i for i in range(starts[2], ends[2]) if structure.atom_name[i] == "O")
    rows = [] if structure.bonds is None else structure.bonds.as_array().tolist()
    rows.append((atom1, atom2, int(struc.BondType.SINGLE)))
    structure.bonds = struc.BondList(
        structure.array_length(), np.asarray(rows, dtype=np.int32)
    )
    return structure, atom1, atom2


def test_explicit_nonpolymeric_bond_round_trip_and_score(biotite_1ubq, torch_device):
    structure, _, _ = _crosslinked_protein(biotite_1ubq)
    pose, context = pose_stack_from_biotite(
        structure, torch_device, no_optH=True, return_context=True
    )

    block_types = [
        pose.packed_block_types.active_block_types[int(ind)]
        for ind in pose.block_type_ind64[0]
        if ind >= 0
    ]
    assert block_types[0].name.endswith(":covalent_O")
    assert block_types[2].name.endswith(":covalent_O")
    conn1 = block_types[0].connection_to_cidx["covalent_O"]
    conn2 = block_types[2].connection_to_cidx["covalent_O"]
    assert tuple(pose.inter_residue_connections64[0, 0, conn1].tolist()) == (
        2,
        conn2,
    )
    assert tuple(pose.inter_residue_connections64[0, 2, conn2].tolist()) == (
        0,
        conn1,
    )
    assert int(pose.inter_block_bondsep64[0, 0, 2, conn1, conn2]) == 1

    from tmol import beta2016_score_function

    coords = pose.coords.detach().clone().requires_grad_(True)
    score = (
        beta2016_score_function(torch_device, param_db=context.parameter_database)
        .render_whole_pose_scoring_module(pose)(coords)
        .sum()
    )
    score.backward()
    assert torch.isfinite(score)
    assert torch.all(torch.isfinite(coords.grad))

    exported = biotite_from_pose_stack(pose, co=context.canonical_ordering)
    starts = struc.get_residue_starts(exported)
    atom_res = (
        np.searchsorted(starts, np.arange(exported.array_length()), side="right") - 1
    )
    cross = {
        (
            int(atom_res[a]),
            str(exported.atom_name[a]),
            int(atom_res[b]),
            str(exported.atom_name[b]),
        )
        for a, b, _ in exported.bonds.as_array()
        if atom_res[a] != atom_res[b]
    }
    assert (0, "O", 2, "O") in cross or (2, "O", 0, "O") in cross


def test_reusing_context_preserves_explicit_topology(biotite_1ubq, torch_device):
    structure, _, _ = _crosslinked_protein(biotite_1ubq)
    first, context = pose_stack_from_biotite(
        structure, torch_device, no_optH=True, return_context=True
    )
    moved = structure.copy()
    moved.coord += np.float32(0.25)
    second = pose_stack_from_biotite(moved, torch_device, no_optH=True, context=context)
    assert torch.equal(
        first.inter_residue_connections, second.inter_residue_connections
    )


def test_attachment_atom_cannot_have_two_partners(biotite_1ubq, torch_device):
    structure, atom1, _ = _crosslinked_protein(biotite_1ubq)
    starts = struc.get_residue_starts(structure)
    atom3 = next(
        i for i in range(starts[1], starts[2]) if structure.atom_name[i] == "O"
    )
    rows = structure.bonds.as_array().tolist()
    rows.append((atom1, atom3, int(struc.BondType.SINGLE)))
    structure.bonds = struc.BondList(
        structure.array_length(), np.asarray(rows, dtype=np.int32)
    )
    with pytest.raises(ValueError, match="more than one inter-residue bond"):
        pose_stack_from_biotite(structure, torch_device, no_optH=True)


def test_only_adjacent_c_n_bonds_are_treated_as_polymer(biotite_1ubq):
    structure = biotite_1ubq.copy()
    starts = struc.get_residue_starts(structure)
    structure = structure[: starts[3]].copy()
    starts = struc.get_residue_starts(structure)
    ends = np.append(starts[1:], structure.array_length())
    carbon = next(i for i in range(starts[0], ends[0]) if structure.atom_name[i] == "C")
    adjacent_nitrogen = next(
        i for i in range(starts[1], ends[1]) if structure.atom_name[i] == "N"
    )
    distant_nitrogen = next(
        i for i in range(starts[2], ends[2]) if structure.atom_name[i] == "N"
    )
    rows = [
        (carbon, adjacent_nitrogen, int(struc.BondType.SINGLE)),
        (carbon, distant_nitrogen, int(struc.BondType.SINGLE)),
    ]
    structure.bonds = struc.BondList(
        structure.array_length(), np.asarray(rows, dtype=np.int32)
    )

    bonds = _explicit_cross_residue_bonds(structure)
    assert len(bonds) == 1
    assert {endpoint[1] for endpoint in bonds[0]} == {"C", "N"}


def test_attachment_virtualizes_a_leaving_hydrogen(biotite_1ubq):
    """Adding an explicit bond does not leave an over-valent hydroxyl."""

    structure = biotite_1ubq.copy()
    starts = struc.get_residue_starts(structure)
    ser_residue = next(
        i for i, start in enumerate(starts) if structure.res_name[start] == "SER"
    )
    other_residue = ser_residue + 2
    ends = np.append(starts[1:], structure.array_length())
    atom1 = next(
        i
        for i in range(starts[ser_residue], ends[ser_residue])
        if structure.atom_name[i] == "OG"
    )
    atom2 = next(
        i
        for i in range(starts[other_residue], ends[other_residue])
        if structure.atom_name[i] == "O"
    )
    structure.bonds = struc.BondList(
        structure.array_length(),
        np.asarray([(atom1, atom2, int(struc.BondType.SINGLE))], dtype=np.int32),
    )

    database, variants = augment_database_for_covalent_bonds(
        structure, ParameterDatabase.get_default()
    )
    variant = next(
        residue
        for residue in database.chemical.residues
        if residue.name == variants[("SER", ("OG",))]
    )
    hg = next(atom for atom in variant.atoms if atom.name == "HG")
    assert hg.atom_type == "Vrt"
    assert "HG" in variant.properties.virtual
