"""M13a-1 gates for flexible-pocket sidechain DOF plumbing."""

from __future__ import annotations


import attr
import numpy
import torch

from tmol.dock import select_pocket_residues
from tmol.dock.ga import LigandGeneLayout
from tmol.dock.ligand_dofs import build_ligand_docking_dofs
from tmol.dock.ga import (
    RandomPlacement,
    _block_heavy_coords,
    _canonicalize_genes,
    _random_initial_genes,
    crossover,
    mutate,
)
from tmol.dock.scoring_backend import (
    _LightweightGridObjective,
    _genes_from_raw_kinforest_dofs,
)
from tmol.kinematics import MinimizerMap
from tmol.kinematics import PoseStackKinematicsModule
from tmol.tests.dock.test_ligand_dof_roundtrip import (
    _ligand_block,
)
from tmol.pose import repeat_pose_stack
from tmol.score.gridscore._movable import (
    movable_block_mapping,
    movable_pose_stack,
    sidechain_context_blocks,
)


def _minimum_heavy_distance(pose_stack, receptor_block, ligand_block):
    receptor = _block_heavy_coords(pose_stack, receptor_block)
    ligand = _block_heavy_coords(pose_stack, ligand_block)
    return float(torch.amin(torch.cdist(receptor, ligand)))


def _dihedral(coords):
    a, b, c, d = coords
    ba = a - b
    bc = c - b
    cd = d - c
    unit_bc = bc / torch.linalg.vector_norm(bc)
    v = ba - torch.dot(ba, unit_bc) * unit_bc
    w = cd - torch.dot(cd, unit_bc) * unit_bc
    return torch.atan2(torch.dot(torch.linalg.cross(unit_bc, v), w), torch.dot(v, w))


def _connected_chi_blocks(pose_stack, ligand_block, selected):
    candidates = []
    for block in selected.detach().cpu().tolist():
        block_type = pose_stack.block_type(0, block)
        if not block_type.n_sc_torsions:
            continue
        partners = pose_stack.inter_residue_connections64[0, block, :, 0]
        for partner in partners[partners >= 0].detach().cpu().tolist():
            if partner == ligand_block:
                continue
            partner_type_index = int(pose_stack.block_type_ind64[0, partner])
            if partner_type_index < 0:
                continue
            partner_type = pose_stack.block_type(0, partner)
            if (
                partner_type.properties.polymer.is_polymer
                and partner_type.n_sc_torsions
            ):
                candidates.append(
                    (
                        _minimum_heavy_distance(pose_stack, block, ligand_block),
                        tuple(sorted((block, partner))),
                    )
                )
    assert candidates, "cdk2 pocket must have a contacting chi next to another chi"
    return min(candidates)[1]


def _pdb_residue_id(pose_stack, block):
    return (
        str(pose_stack.pdb_info.chain_labels[0, block]),
        str(pose_stack.pdb_info.residue_labels[0, block]),
        str(pose_stack.pdb_info.residue_insertion_codes[0, block]),
        pose_stack.block_type(0, block).name3,
    )


def _one_sidechain_movable_pose(full_pose, same_chain):
    ligand_block = _ligand_block(full_pose)
    selected = select_pocket_residues(full_pose, ligand_block, cutoff=6.0)
    selected_block = _connected_chi_blocks(full_pose, ligand_block, selected)[0]
    source = full_pose
    if same_chain:
        chain_id = full_pose.chain_id.clone()
        chain_id64 = full_pose.chain_id64.clone()
        chain_id[:, ligand_block] = chain_id[:, selected_block]
        chain_id64[:, ligand_block] = chain_id64[:, selected_block]
        source = attr.evolve(full_pose, chain_id=chain_id, chain_id64=chain_id64)
    movable = movable_pose_stack(source, ligand_block, (selected_block,))
    _, compact_sidechains, _ = movable_block_mapping(
        source, ligand_block, (selected_block,)
    )
    return movable, compact_sidechains[0]


def test_movable_pose_preserves_retained_polymer_connections_and_k0_bytes(
    cdk2_pose_and_db,
):
    pose_stack, _ = cdk2_pose_and_db
    ligand_block = _ligand_block(pose_stack)
    selected = select_pocket_residues(pose_stack, ligand_block, cutoff=6.0)
    connected_pair = _connected_chi_blocks(pose_stack, ligand_block, selected)

    movable = movable_pose_stack(pose_stack, ligand_block, connected_pair)
    _, compact_sidechains, compact_context = movable_block_mapping(
        pose_stack, ligand_block, connected_pair
    )
    context_blocks = sidechain_context_blocks(pose_stack, ligand_block, connected_pair)
    assert movable.max_n_blocks == 3 + len(context_blocks)
    assert (
        movable.block_type_ind64[0, 0] == pose_stack.block_type_ind64[0, ligand_block]
    )
    first_sidechain, second_sidechain = compact_sidechains
    assert torch.any(
        movable.inter_residue_connections64[0, first_sidechain, :, 0]
        == second_sidechain
    )
    assert torch.any(
        movable.inter_residue_connections64[0, second_sidechain, :, 0]
        == first_sidechain
    )
    assert context_blocks
    for compact_block, source_block in zip(compact_context, context_blocks):
        assert (
            movable.block_type_ind64[0, compact_block]
            == pose_stack.block_type_ind64[0, source_block]
        )

    ligand_only = movable_pose_stack(pose_stack, ligand_block)
    zero_pocket = movable_pose_stack(pose_stack, ligand_block, ())
    for field in (
        "coords",
        "block_coord_offset",
        "block_coord_offset64",
        "inter_residue_connections",
        "inter_residue_connections64",
        "block_type_ind",
        "block_type_ind64",
        "chain_id",
        "chain_id64",
    ):
        assert torch.equal(getattr(ligand_only, field), getattr(zero_pocket, field))
    for field in ("near_blocks", "bondsep"):
        assert torch.equal(
            getattr(ligand_only.inter_block_bondsep, field),
            getattr(zero_pocket.inter_block_bondsep, field),
        )
    keep = torch.tensor(
        [ligand_block, *sorted(set(connected_pair) | set(context_blocks))],
        dtype=torch.int64,
    )
    assert torch.equal(
        movable.inter_block_bondsep.to_dense(),
        pose_stack.inter_block_bondsep.to_dense()[:, keep[:, None], keep[None, :]],
    )
    for field in (
        "residue_labels",
        "residue_insertion_codes",
        "chain_labels",
        "atom_occupancy",
        "atom_b_factor",
    ):
        assert numpy.array_equal(
            getattr(ligand_only.pdb_info, field),
            getattr(zero_pocket.pdb_info, field),
        )


def test_flexible_sidechain_gene_roundtrip_and_isolation(cdk2_pose_and_db):
    full_pose, _ = cdk2_pose_and_db
    ligand_block = _ligand_block(full_pose)
    selected = select_pocket_residues(full_pose, ligand_block, cutoff=6.0)
    selected_blocks = _connected_chi_blocks(full_pose, ligand_block, selected)

    pose_stack = movable_pose_stack(full_pose, ligand_block, selected_blocks)
    _, compact_sidechains, _ = movable_block_mapping(
        full_pose, ligand_block, selected_blocks
    )
    docking_dofs = build_ligand_docking_dofs(
        pose_stack, ligand_block=0, sidechain_blocks=compact_sidechains
    )
    layout = LigandGeneLayout.from_pose_stack(pose_stack, docking_dofs)
    genes = layout.genes_from_pose_stack(pose_stack)
    rebuilt = layout.pose_stack_from_genes(genes)

    assert docking_dofs.sidechain_blocks == compact_sidechains
    assert docking_dofs.sidechain_torsions
    assert {mapping.block for mapping in docking_dofs.sidechain_torsions} == set(
        compact_sidechains
    )
    assert all(
        mapping.dof_name == "phi_c" for mapping in docking_dofs.sidechain_torsions
    )
    for mapping_index, mapping in enumerate(docking_dofs.sidechain_torsions):
        influenced_atoms = layout.sidechain_atom_indices[
            0, layout.sidechain_atom_influence[mapping_index]
        ]
        block_start = int(pose_stack.block_coord_offset64[0, mapping.block])
        block_end = block_start + pose_stack.block_type(0, mapping.block).n_atoms
        required_local_atom = int(
            pose_stack.block_type(0, mapping.block).ordered_torsions[
                mapping.torsion_index, 3, 0
            ]
        )
        assert influenced_atoms.numel() > 0
        assert bool(torch.all(influenced_atoms >= block_start))
        assert bool(torch.all(influenced_atoms < block_end))
        assert bool(torch.any(influenced_atoms == block_start + required_local_atom))
    assert genes.shape[1] == (
        6 + len(docking_dofs.torsions) + len(docking_dofs.sidechain_torsions)
    )
    kin_module = PoseStackKinematicsModule(pose_stack, docking_dofs.fold_forest)
    minmap = MinimizerMap(pose_stack, kin_module.kmd, docking_dofs.move_map)
    assert int(torch.sum(minmap.dof_mask)) == genes.shape[1]
    backend_genes = _genes_from_raw_kinforest_dofs(
        pose_stack,
        docking_dofs,
        kin_module.kmd.forest,
        minmap.dof_mask,
        layout.base_raw_dofs,
        layout.ligand_nodes,
    )
    torch.testing.assert_close(backend_genes, genes, rtol=0.0, atol=0.0)

    ligand_indices = (
        pose_stack.block_coord_offset64[:, 0, None] + layout.ligand_local_atoms
    )
    ligand_error = torch.amax(
        torch.abs(
            rebuilt.coords[:, ligand_indices[0]]
            - pose_stack.coords[:, ligand_indices[0]]
        )
    )
    sidechain_error = torch.amax(
        torch.abs(
            rebuilt.coords[:, layout.sidechain_atom_indices[0]]
            - pose_stack.coords[:, layout.sidechain_atom_indices[0]]
        )
    )
    for block in docking_dofs.sidechain_blocks:
        block_type = pose_stack.block_type(0, block)
        backbone_local = torch.tensor(
            [
                atom_index
                for atom_index, atom in enumerate(block_type.atoms)
                if atom.name in {"N", "CA", "C", "O", "OXT"}
            ],
            dtype=torch.int64,
        )
        backbone_indices = (
            pose_stack.block_coord_offset64[:, block, None] + backbone_local
        )
        torch.testing.assert_close(
            rebuilt.coords[:, backbone_indices[0]],
            pose_stack.coords[:, backbone_indices[0]],
            rtol=0.0,
            atol=0.0,
        )
    print(
        "cdk2 flexible round-trip blocks/residues, ligand max_A, sidechain max_A:",
        selected_blocks,
        [_pdb_residue_id(full_pose, block) for block in selected_blocks],
        float(ligand_error),
        float(sidechain_error),
    )
    assert float(ligand_error) < 1.0e-4
    assert float(sidechain_error) < 1.0e-4

    sidechain_perturbed = genes.clone()
    sidechain_gene = 6 + len(docking_dofs.torsions)
    sidechain_perturbed[:, sidechain_gene] += 0.2
    sidechain_pose = layout.pose_stack_from_genes(sidechain_perturbed)
    sidechain_delta = torch.linalg.vector_norm(
        sidechain_pose.coords - rebuilt.coords, dim=2
    )[0]
    changed_sidechain = torch.nonzero(
        sidechain_delta > 1.0e-8, as_tuple=False
    ).flatten()
    mapping = docking_dofs.sidechain_torsions[0]
    mapping_start = int(pose_stack.block_coord_offset64[0, mapping.block])
    mapping_end = mapping_start + pose_stack.block_type(0, mapping.block).n_atoms
    allowed_sidechain = {
        atom
        for atom in layout.sidechain_atom_indices[0].cpu().tolist()
        if mapping_start <= atom < mapping_end
    }
    assert changed_sidechain.numel() > 0
    assert set(changed_sidechain.cpu().tolist()) <= allowed_sidechain
    assert not bool(torch.any(sidechain_delta[ligand_indices[0]] > 0))
    other_block = next(
        block for block in docking_dofs.sidechain_blocks if block != mapping.block
    )
    other_start = int(pose_stack.block_coord_offset64[0, other_block])
    other_end = other_start + pose_stack.block_type(0, other_block).n_atoms
    assert not bool(torch.any(sidechain_delta[other_start:other_end] > 0))
    torsion_local = torch.tensor(
        pose_stack.block_type(0, mapping.block).ordered_torsions[
            mapping.torsion_index, :, 0
        ],
        dtype=torch.int64,
    )
    torsion_atoms = pose_stack.block_coord_offset64[0, mapping.block] + torsion_local
    before = _dihedral(rebuilt.coords[0, torsion_atoms])
    after = _dihedral(sidechain_pose.coords[0, torsion_atoms])
    logical_delta = torch.atan2(torch.sin(after - before), torch.cos(after - before))
    logical_slope = float(logical_delta / 0.2)
    print("selected sidechain logical-chi slope:", logical_slope)
    assert abs(logical_slope - 1.0) < 5.0e-3

    ligand_perturbed = genes.clone()
    ligand_perturbed[:, 6] += 0.2
    ligand_pose = layout.pose_stack_from_genes(ligand_perturbed)
    ligand_delta = torch.linalg.vector_norm(ligand_pose.coords - rebuilt.coords, dim=2)[
        0
    ]
    changed_ligand = torch.nonzero(ligand_delta > 1.0e-8, as_tuple=False).flatten()
    allowed_ligand = set(ligand_indices[0].cpu().tolist())
    assert changed_ligand.numel() > 0
    assert set(changed_ligand.cpu().tolist()) <= allowed_ligand
    assert not bool(torch.any(ligand_delta[layout.sidechain_atom_indices[0]] > 0))
    print(
        "changed atoms for one receptor chi / one ligand chi:",
        changed_sidechain.cpu().tolist(),
        changed_ligand.cpu().tolist(),
    )


def test_ga_operators_handle_appended_sidechain_chis(cdk2_pose_and_db):
    full_pose, _ = cdk2_pose_and_db
    one_pose, compact_sidechain = _one_sidechain_movable_pose(
        full_pose, same_chain=False
    )
    pose_stack = repeat_pose_stack(one_pose, 4)
    docking_dofs = build_ligand_docking_dofs(
        pose_stack, ligand_block=0, sidechain_blocks=(compact_sidechain,)
    )
    layout = LigandGeneLayout.from_pose_stack(pose_stack, docking_dofs)
    genes = layout.genes_from_pose_stack(pose_stack)
    expected_width = (
        6 + len(docking_dofs.torsions) + len(docking_dofs.sidechain_torsions)
    )
    assert docking_dofs.sidechain_torsions
    assert genes.shape == (4, expected_width)
    sidechain_start = 6 + len(docking_dofs.torsions)

    outside = genes.clone()
    outside[:, 6:] += 5.0 * torch.pi
    canonical = _canonicalize_genes(outside)

    generator = torch.Generator(device=genes.device).manual_seed(4815)
    mutated = mutate(
        genes,
        generator=generator,
        rtmutation_rate=0.0,
        torsmutation_rate=1.0,
        torsion_width=2.0 * torch.pi,
    )
    parent2 = genes.clone()
    parent2[:, 6:] -= 5.0 * torch.pi
    crossed = crossover(
        outside,
        parent2,
        generator=generator,
    )
    randomized = _random_initial_genes(
        layout,
        pocket_bounds=None,
        placement=RandomPlacement(),
        generator=generator,
    )

    for operator_name, operated in (
        ("canonicalize", canonical),
        ("mutate", mutated),
        ("crossover", crossed),
        ("randomize", randomized),
    ):
        assert operated.shape == genes.shape
        assert torch.all(operated[:, sidechain_start:] > -torch.pi)
        assert torch.all(operated[:, sidechain_start:] <= torch.pi)
        decoded = layout.pose_stack_from_genes(operated)
        pose_indices = torch.arange(pose_stack.n_poses)[:, None]
        ligand_indices = (
            pose_stack.block_coord_offset64[:, 0, None]
            + layout.ligand_local_atoms[None, :]
        )
        # Extreme ligand torsions and random rigid-body placements have a
        # pre-existing non-idempotent parameterization (F53b), and operated
        # genes are never re-encoded by the GA.  Their required invariant here
        # is a valid finite decode.
        assert torch.all(torch.isfinite(decoded.coords[pose_indices, ligand_indices]))

        recovered = layout.genes_from_pose_stack(decoded)
        recovered_decoded = layout.pose_stack_from_genes(recovered)
        sidechain_error = torch.amax(
            torch.abs(
                recovered_decoded.coords[pose_indices, layout.sidechain_atom_indices]
                - decoded.coords[pose_indices, layout.sidechain_atom_indices]
            ),
            dim=(1, 2),
        )
        torch.testing.assert_close(
            recovered_decoded.coords[pose_indices, layout.sidechain_atom_indices],
            decoded.coords[pose_indices, layout.sidechain_atom_indices],
            rtol=0.0,
            atol=1.0e-8,
        )
        assert bool(torch.all(sidechain_error < 1.0e-8))
        print(
            "GA sidechain decode/encode/decode per-pose max_A:",
            operator_name,
            sidechain_error.tolist(),
        )


def test_grid_final_result_writes_sidechain_coordinates(cpu_grid_case):
    case = cpu_grid_case
    full_pose = case.reference
    selected = select_pocket_residues(full_pose, case.ligand_block, cutoff=6.0)
    selected_blocks = _connected_chi_blocks(full_pose, case.ligand_block, selected)
    pose_stack = movable_pose_stack(full_pose, case.ligand_block, selected_blocks)
    _, compact_sidechains, _ = movable_block_mapping(
        full_pose, case.ligand_block, selected_blocks
    )
    docking_dofs = build_ligand_docking_dofs(
        pose_stack, ligand_block=0, sidechain_blocks=compact_sidechains
    )
    layout = LigandGeneLayout.from_pose_stack(pose_stack, docking_dofs)
    genes = layout.genes_from_pose_stack(pose_stack)
    genes[:, 6 + len(docking_dofs.torsions)] += 0.2
    raw_dofs = layout.raw_dofs_from_genes(genes)
    expected = layout.pose_stack_from_genes(genes)

    objective = _LightweightGridObjective(
        case.backend.for_ligand_block(0),
        pose_stack,
        docking_dofs,
        torch.float64,
        raw_dofs,
    )
    actual = objective.final_result(objective.initial_masked_dofs)

    torch.testing.assert_close(
        actual.pose_stack.coords[:, layout.sidechain_atom_indices[0]],
        expected.coords[:, layout.sidechain_atom_indices[0]],
        rtol=0.0,
        atol=1.0e-9,
    )
    torch.testing.assert_close(
        actual.pose_stack.coords, expected.coords, rtol=0.0, atol=1.0e-9
    )
    # Jump translations are unique. Jump rotations are checked through the
    # coordinates above because their raw parameterization is ambiguous; named
    # torsions remain unique modulo 2*pi.
    torch.testing.assert_close(actual.genes[:, :3], genes[:, :3], rtol=0.0, atol=1.0e-9)
    tors_diff = actual.genes[:, 6:] - genes[:, 6:]
    wrapped = torch.atan2(torch.sin(tors_diff), torch.cos(tors_diff))
    torch.testing.assert_close(
        wrapped, torch.zeros_like(wrapped), rtol=0.0, atol=1.0e-9
    )
