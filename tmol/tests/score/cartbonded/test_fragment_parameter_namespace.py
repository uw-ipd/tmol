"""Splitting a ligand must preserve source bond/angle parameters across cuts."""

import attr
import pytest
import torch

from tmol.tests.ligand.test_fragmented_ligand_scoring import (
    _load_fixture,
    _annotate_at_bridge,
    _build,
)
from tmol.tests.score.cartbonded.test_explicit_connection_parameters import scoring_term
from tmol.database.scoring import CartBondedDatabase


@pytest.mark.parametrize("independent_records", [False, True])
def test_fragment_cut_energy_and_gradient_preserve_source_parameters(
    torch_device, independent_records
):
    structure, params_path, preparation = _load_fixture()
    structure = structure[structure.res_name == "LG1"]
    split = _annotate_at_bridge(structure, preparation)
    whole, whole_context, _ = _build(
        structure, params_path, torch_device, fragmented=False
    )
    fragmented, fragment_context, mapping = _build(
        split, params_path, torch_device, fragmented=True
    )
    assert fragmented.max_n_blocks == 2 and whole.max_n_blocks == 1
    modules = []
    for pose, context in ((whole, whole_context), (fragmented, fragment_context)):
        database = context.parameter_database
        source = database.scoring.cartbonded.residue_params["LG1"]
        rows = {"LG1": source}
        for bt in pose.packed_block_types.active_block_types:
            if bt.is_ligand_fragment:
                rows[bt.name] = attr.evolve(source) if independent_records else source
        database = attr.evolve(
            database,
            scoring=attr.evolve(
                database.scoring, cartbonded=CartBondedDatabase.from_cartres_dict(rows)
            ),
        )
        modules.append(
            scoring_term(pose, database).render_whole_pose_scoring_module(pose)
        )
    coordinates = [pose.coords.double().clone() for pose in (whole, fragmented)]
    indices = []
    for pose in (whole, fragmented):
        atom_map = {}
        for bi, ti in enumerate(pose.block_type_ind[0].tolist()):
            bt = pose.packed_block_types.active_block_types[ti]
            atom_map.update(
                (atom.name, int(pose.block_coord_offset[0, bi]) + ai)
                for ai, atom in enumerate(bt.atoms)
            )
        indices.append(atom_map)
    assert indices[0].keys() == indices[1].keys()
    last = max(mapping.entries, key=lambda entry: entry.block_ind)
    bt = fragmented.packed_block_types.active_block_types[
        int(fragmented.block_type_ind[0, last.block_ind])
    ]
    # Move one complete piece so cross-cut springs have substantial nonzero
    # energy and force. Equilibrium fixtures can hide a missing term.
    for atom in bt.atoms:
        for xyz, atom_map in zip(coordinates, indices):
            xyz[0, atom_map[atom.name]] += xyz.new_tensor([0.23, -0.11, 0.17])
    for xyz in coordinates:
        xyz.requires_grad_(True)
    energies = [module(xyz) for module, xyz in zip(modules, coordinates)]
    assert float(energies[0][:2].sum().detach()) > 1.0
    torch.testing.assert_close(energies[0], energies[1], rtol=1e-7, atol=1e-7)
    gradients = [
        torch.autograd.grad(energy.sum(), xyz)[0]
        for energy, xyz in zip(energies, coordinates)
    ]
    names = sorted(indices[0])
    aligned = [
        gradient[0, [atom_map[name] for name in names]]
        for gradient, atom_map in zip(gradients, indices)
    ]
    torch.testing.assert_close(aligned[0], aligned[1], rtol=1e-7, atol=1e-7)
