"""Programmatic carbohydrate connection tests."""

import biotite.structure as struc
import biotite.structure.info as struc_info
import numpy as np
import torch

from tmol.io._covalent_bonds import (
    _carbohydrate_connection_roles,
    _explicit_cross_residue_bonds,
)


def _branched_glycan():
    residue_atoms = (
        ("ASN", ("ND2",)),
        ("NAG", ("C1", "O4", "O6")),
        ("MAN", ("C1",)),
        ("FUC", ("C1",)),
    )
    n_atoms = sum(len(atoms) for _, atoms in residue_atoms)
    array = struc.AtomArray(n_atoms)
    offset = 0
    atom_index = {}
    for residue_index, (residue_name, atoms) in enumerate(residue_atoms, start=1):
        end = offset + len(atoms)
        array.chain_id[offset:end] = "A"
        array.res_id[offset:end] = residue_index
        array.res_name[offset:end] = residue_name
        array.atom_name[offset:end] = atoms
        array.element[offset:end] = [
            "N" if atom.startswith("N") else "C" for atom in atoms
        ]
        for local_index, atom in enumerate(atoms):
            atom_index[(residue_name, atom)] = offset + local_index
        offset = end
    array.coord = np.arange(n_atoms * 3, dtype=float).reshape((n_atoms, 3))
    array.bonds = struc.BondList(
        n_atoms,
        np.asarray(
            [
                (
                    atom_index[("ASN", "ND2")],
                    atom_index[("NAG", "C1")],
                    int(struc.BondType.SINGLE),
                ),
                (
                    atom_index[("NAG", "O4")],
                    atom_index[("MAN", "C1")],
                    int(struc.BondType.SINGLE),
                ),
                (
                    atom_index[("NAG", "O6")],
                    atom_index[("FUC", "C1")],
                    int(struc.BondType.SINGLE),
                ),
            ],
            dtype=np.int32,
        ),
    )
    return array


def test_branched_carbohydrate_roles_follow_input_graph():
    array = _branched_glycan()
    roles = _carbohydrate_connection_roles(array, _explicit_cross_residue_bonds(array))

    by_residue_atom = {(key[3], atom): role for (key, atom), role in roles.items()}
    assert by_residue_atom[("NAG", "C1")] == "down"
    assert by_residue_atom[("NAG", "O4")] == "up"
    assert by_residue_atom[("MAN", "C1")] == "down"
    assert by_residue_atom[("NAG", "O6")] == "branch_O6"
    assert by_residue_atom[("FUC", "C1")] == "down"


def test_generated_glycopeptide_builds_scores_and_differentiates(torch_device):
    residues = []
    atom_indices = {}
    offset = 0
    for residue_index, name in enumerate(("ASN", "NAG", "MAN"), start=1):
        residue = struc_info.residue(name).copy()
        keep = residue.element != "H"
        if name == "ASN":
            keep &= residue.atom_name != "OXT"
        if name in ("NAG", "MAN"):
            keep &= residue.atom_name != "O1"
        residue = residue[keep]
        residue.chain_id[:] = "A"
        residue.res_id[:] = residue_index
        residue.coord[:, 0] += 4.0 * (residue_index - 1)
        for atom_index, atom_name in enumerate(residue.atom_name):
            atom_indices[(name, str(atom_name))] = offset + atom_index
        offset += len(residue)
        residues.append(residue)
    structure = struc.concatenate(residues)
    bonds = structure.bonds.as_array().tolist()
    bonds.extend(
        (
            (
                atom_indices[("ASN", "ND2")],
                atom_indices[("NAG", "C1")],
                int(struc.BondType.SINGLE),
            ),
            (
                atom_indices[("NAG", "O4")],
                atom_indices[("MAN", "C1")],
                int(struc.BondType.SINGLE),
            ),
        )
    )
    structure.bonds = struc.BondList(
        structure.array_length(), np.asarray(bonds, dtype=np.int32)
    )

    from tmol import beta2016_score_function
    from tmol.io import biotite_from_pose_stack, pose_stack_from_biotite

    pose, context = pose_stack_from_biotite(
        structure,
        torch_device,
        prepare_ligands=True,
        no_optH=True,
        return_context=True,
    )
    block_types = [
        pose.packed_block_types.active_block_types[int(block_type)]
        for block_type in pose.block_type_ind64[0]
        if block_type >= 0
    ]
    nag = next(block for block in block_types if block.name3 == "NAG")
    man = next(block for block in block_types if block.name3 == "MAN")
    assert nag.properties.polymer.polymer_type == "carbohydrate"
    assert {connection.name for connection in nag.connections} >= {"down", "up"}
    assert {connection.name for connection in man.connections} >= {"down"}

    coords = pose.coords.detach().clone().requires_grad_(True)
    score = (
        beta2016_score_function(torch_device, param_db=context.parameter_database)
        .render_whole_pose_scoring_module(pose)(coords)
        .sum()
    )
    score.backward()
    assert torch.isfinite(score)
    assert torch.isfinite(coords.grad).all()

    exported = biotite_from_pose_stack(pose, co=context.canonical_ordering)
    assert set(exported.res_name) >= {"ASN", "NAG", "MAN"}
    rebuilt = pose_stack_from_biotite(
        exported,
        torch_device,
        context=context,
        no_optH=True,
    )
    assert torch.equal(
        rebuilt.inter_residue_connections64, pose.inter_residue_connections64
    )
