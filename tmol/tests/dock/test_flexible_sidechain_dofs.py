"""M13a-1 gates for flexible-pocket sidechain DOF plumbing."""

from __future__ import annotations

from types import SimpleNamespace

import attr
import numpy
import pytest
import torch

from tmol.dock import (
    LigandGeneLayout,
    build_ligand_docking_dofs,
    select_pocket_residue_blocks,
)
from tmol.dock.ga import (
    RandomPlacement,
    _block_heavy_coords,
    _canonicalize_genes,
    _ligand_only_pose_stack,
    _movable_block_mapping,
    _movable_pose_stack,
    _sidechain_context_blocks,
    _random_initial_genes,
    crossover,
    mutate,
)
from tmol.dock.dock_loop import _repeat_pose_stack
from tmol.dock.ligand_dofs import (
    _block_kinforest_nodes,
    _local_torsion_atoms,
)
from tmol.dock.scoring_backend import (
    _LightweightGridObjective,
    _genes_from_raw_kinforest_dofs,
)
from tmol.kinematics.move_map import MinimizerMap
from tmol.kinematics.fold_forest import EdgeType
from tmol.kinematics.script_modules import PoseStackKinematicsModule
from tmol.tests.dock.test_ligand_dof_roundtrip import (
    _ligand_block,
    cdk2_pose_and_db,
)
from tmol.tests.dock.test_grid_backend import cpu_grid_case


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
                        _minimum_heavy_distance(
                            pose_stack, block, ligand_block
                        ),
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
    selected = select_pocket_residue_blocks(full_pose, ligand_block, cutoff=6.0)
    selected_block = _connected_chi_blocks(full_pose, ligand_block, selected)[0]
    source = full_pose
    if same_chain:
        chain_id = full_pose.chain_id.clone()
        chain_id64 = full_pose.chain_id64.clone()
        chain_id[:, ligand_block] = chain_id[:, selected_block]
        chain_id64[:, ligand_block] = chain_id64[:, selected_block]
        source = attr.evolve(full_pose, chain_id=chain_id, chain_id64=chain_id64)
    movable = _movable_pose_stack(source, ligand_block, (selected_block,))
    _, compact_sidechains, _ = _movable_block_mapping(
        source, ligand_block, (selected_block,)
    )
    return movable, compact_sidechains[0]


def _one_sidechain_layout(full_pose, same_chain=False):
    pose_stack, compact_sidechain = _one_sidechain_movable_pose(
        full_pose, same_chain
    )
    docking_dofs = build_ligand_docking_dofs(
        pose_stack, ligand_block=0, sidechain_blocks=(compact_sidechain,)
    )
    return (
        pose_stack,
        docking_dofs,
        LigandGeneLayout.from_pose_stack(pose_stack, docking_dofs),
    )


def _block_atom_indices(pose_stack, block):
    n_atoms = pose_stack.block_type(0, block).n_atoms
    return (
        pose_stack.block_coord_offset64[:, block, None]
        + torch.arange(n_atoms, dtype=torch.int64, device=pose_stack.device)[
            None, :
        ]
    )


def _block_kinforest_indices(layout, block):
    pose_stack = layout.template
    return _block_kinforest_nodes(
        pose_stack,
        block,
        pose_stack.block_type(0, block).n_atoms,
        layout.kin_module.kmd.forest.id,
    )


def _assert_gene_roundtrip(layout, genes, atol):
    decoded = layout.pose_stack_from_genes(genes)
    assert torch.all(torch.isfinite(decoded.coords))
    recovered = layout.genes_from_pose_stack(decoded)
    recovered_decoded = layout.pose_stack_from_genes(recovered)
    per_pose_coordinate_error = torch.amax(
        torch.abs(recovered_decoded.coords - decoded.coords), dim=(1, 2)
    )
    torch.testing.assert_close(
        recovered_decoded.coords, decoded.coords, rtol=0.0, atol=atol
    )
    torch.testing.assert_close(
        recovered[:, :3], genes[:, :3], rtol=0.0, atol=atol
    )
    # The three raw jump-rotation values are not unique for a physical
    # rotation. Coordinate equivalence above is their round-trip invariant;
    # only named ligand and sidechain torsions are unique modulo 2*pi.
    angular_delta = recovered[:, 6:] - genes[:, 6:]
    wrapped_delta = torch.atan2(torch.sin(angular_delta), torch.cos(angular_delta))
    torch.testing.assert_close(
        wrapped_delta, torch.zeros_like(wrapped_delta), rtol=0.0, atol=atol
    )
    assert bool(torch.all(per_pose_coordinate_error < atol))
    return per_pose_coordinate_error


def test_cdk2_pocket_selection_is_heavy_atom_based_and_deterministic(
    cdk2_pose_and_db,
):
    pose_stack, _ = cdk2_pose_and_db
    ligand_block = _ligand_block(pose_stack)

    first = select_pocket_residue_blocks(pose_stack, ligand_block, cutoff=6.0)
    second = select_pocket_residue_blocks(pose_stack, ligand_block, cutoff=6.0)

    assert torch.equal(first, second)
    assert first.numel() > 0
    assert torch.all(first[1:] > first[:-1])
    selected = set(first.detach().cpu().tolist())
    rows = []
    for block in range(pose_stack.max_n_blocks):
        block_type_index = int(pose_stack.block_type_ind64[0, block])
        if block_type_index < 0 or block == ligand_block:
            continue
        block_type = pose_stack.packed_block_types.active_block_types[
            block_type_index
        ]
        if not block_type.properties.polymer.is_polymer:
            assert block not in selected
            continue
        distance = _minimum_heavy_distance(pose_stack, block, ligand_block)
        assert (block in selected) == (distance <= 6.0)
        if block in selected:
            rows.append((block, _pdb_residue_id(pose_stack, block), distance))
    print("cdk2 6A pocket blocks (block, chain/resid/icode/name3, min_A):", rows)


def test_movable_pose_preserves_retained_polymer_connections_and_k0_bytes(
    cdk2_pose_and_db,
):
    pose_stack, _ = cdk2_pose_and_db
    ligand_block = _ligand_block(pose_stack)
    selected = select_pocket_residue_blocks(pose_stack, ligand_block, cutoff=6.0)
    connected_pair = _connected_chi_blocks(pose_stack, ligand_block, selected)

    movable = _movable_pose_stack(pose_stack, ligand_block, connected_pair)
    _, compact_sidechains, compact_context = _movable_block_mapping(
        pose_stack, ligand_block, connected_pair
    )
    context_blocks = _sidechain_context_blocks(
        pose_stack, ligand_block, connected_pair
    )
    assert movable.max_n_blocks == 3 + len(context_blocks)
    assert (
        movable.block_type_ind64[0, 0]
        == pose_stack.block_type_ind64[0, ligand_block]
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

    ligand_only = _ligand_only_pose_stack(pose_stack, ligand_block)
    zero_pocket = _movable_pose_stack(pose_stack, ligand_block, ())
    for field in (
        "coords",
        "block_coord_offset",
        "block_coord_offset64",
        "inter_residue_connections",
        "inter_residue_connections64",
        "inter_block_bondsep",
        "inter_block_bondsep64",
        "block_type_ind",
        "block_type_ind64",
        "chain_id",
        "chain_id64",
    ):
        assert torch.equal(getattr(ligand_only, field), getattr(zero_pocket, field))
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
    selected = select_pocket_residue_blocks(full_pose, ligand_block, cutoff=6.0)
    selected_blocks = _connected_chi_blocks(full_pose, ligand_block, selected)

    pose_stack = _movable_pose_stack(full_pose, ligand_block, selected_blocks)
    _, compact_sidechains, _ = _movable_block_mapping(
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
        block
        for block in docking_dofs.sidechain_blocks
        if block != mapping.block
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
    torsion_atoms = (
        pose_stack.block_coord_offset64[0, mapping.block] + torsion_local
    )
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


def test_same_chain_ligand_jump_does_not_move_receptor(cdk2_pose_and_db):
    """Regression: before M13a-1b, the exact sidechain check failed here."""

    full_pose, _ = cdk2_pose_and_db
    pose_stack, docking_dofs, layout = _one_sidechain_layout(
        full_pose, same_chain=True
    )
    assert torch.all(pose_stack.chain_id64[:, 0] != pose_stack.chain_id64[:, 1])
    edges = docking_dofs.fold_forest.edges[0, : docking_dofs.fold_forest.n_edges[0]]
    root_blocks = set(
        edges[edges[:, 0] == int(EdgeType.root_jump), 2].tolist()
    )
    assert {0, 1} <= root_blocks
    genes = layout.genes_from_pose_stack(pose_stack)
    moved_genes = genes.clone()
    moved_genes[:, 0] += 1.0
    moved = layout.pose_stack_from_genes(moved_genes)

    base_kin_coords = layout.kin_module(layout.raw_dofs_from_genes(genes))
    moved_kin_coords = layout.kin_module(layout.raw_dofs_from_genes(moved_genes))
    receptor_nodes = _block_kinforest_indices(layout, 1)
    torch.testing.assert_close(
        moved_kin_coords[receptor_nodes],
        base_kin_coords[receptor_nodes],
        rtol=0.0,
        atol=0.0,
    )

    torch.testing.assert_close(
        moved.coords[:, layout.sidechain_atom_indices[0]],
        pose_stack.coords[:, layout.sidechain_atom_indices[0]],
        rtol=0.0,
        # Inverse-to-forward kinematics has an approximately 1e-15 A fp64
        # reconstruction floor; this is not receptor motion.
        atol=1.0e-12,
    )
    receptor_atoms = _block_atom_indices(pose_stack, 1)
    torch.testing.assert_close(
        moved.coords[:, receptor_atoms[0]],
        pose_stack.coords[:, receptor_atoms[0]],
        rtol=0.0,
        # Inverse-to-forward kinematics has an approximately 1e-15 A fp64
        # reconstruction floor; this is not receptor motion.
        atol=1.0e-12,
    )


@pytest.mark.parametrize("same_chain", [False, True])
def test_all_ligand_jump_genes_leave_receptor_fixed(cdk2_pose_and_db, same_chain):
    full_pose, _ = cdk2_pose_and_db
    pose_stack, _, layout = _one_sidechain_layout(full_pose, same_chain)
    genes = layout.genes_from_pose_stack(pose_stack)
    moved_genes = genes.clone()
    moved_genes[:, :6] += torch.tensor(
        [0.8, -0.5, 0.3, 0.12, -0.09, 0.07], dtype=genes.dtype
    )
    moved = layout.pose_stack_from_genes(moved_genes)

    base_kin_coords = layout.kin_module(layout.raw_dofs_from_genes(genes))
    moved_kin_coords = layout.kin_module(layout.raw_dofs_from_genes(moved_genes))
    receptor_nodes = _block_kinforest_indices(layout, 1)
    torch.testing.assert_close(
        moved_kin_coords[receptor_nodes],
        base_kin_coords[receptor_nodes],
        rtol=0.0,
        atol=0.0,
    )

    receptor_atoms = _block_atom_indices(pose_stack, 1)
    torch.testing.assert_close(
        moved.coords[:, receptor_atoms[0]],
        pose_stack.coords[:, receptor_atoms[0]],
        rtol=0.0,
        # Inverse-to-forward kinematics has an approximately 1e-15 A fp64
        # reconstruction floor; this is not receptor motion.
        atol=1.0e-12,
    )
    ligand_atoms = _block_atom_indices(pose_stack, 0)
    ligand_delta = torch.linalg.vector_norm(
        moved.coords[:, ligand_atoms[0]] - pose_stack.coords[:, ligand_atoms[0]],
        dim=2,
    )
    assert bool(torch.any(ligand_delta > 1.0e-6))


def test_ga_operators_handle_appended_sidechain_chis(cdk2_pose_and_db):
    full_pose, _ = cdk2_pose_and_db
    one_pose, compact_sidechain = _one_sidechain_movable_pose(
        full_pose, same_chain=False
    )
    pose_stack = _repeat_pose_stack(one_pose, 4)
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
                recovered_decoded.coords[
                    pose_indices, layout.sidechain_atom_indices
                ]
                - decoded.coords[pose_indices, layout.sidechain_atom_indices]
            ),
            dim=(1, 2),
        )
        torch.testing.assert_close(
            recovered_decoded.coords[
                pose_indices, layout.sidechain_atom_indices
            ],
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


@pytest.mark.parametrize(
    "uaids, expected",
    [
        (((0, -1), (1, 0), (2, -1), (3, -1)), "reaches another block"),
        (((0, -1), (1, -1), (-1, -1), (3, -1)), "unresolved atom"),
    ],
)
def test_sidechain_torsion_locality_error_is_generic(uaids, expected):
    block_type = SimpleNamespace(
        ordered_torsions=[uaids],
        torsions=[SimpleNamespace(name="chi1")],
    )

    with pytest.raises(ValueError, match=expected) as exc_info:
        _local_torsion_atoms(block_type, 0)

    assert "ligand torsion" not in str(exc_info.value)


@pytest.mark.parametrize(
    "dtype,n_poses,atol",
    [
        (dtype, n_poses, atol)
        for dtype, atol in ((torch.float64, 1.0e-8), (torch.float32, 1.0e-4))
        for n_poses in (1, 2, 3)
    ],
)
def test_flexible_sidechain_roundtrip_dtype_and_batch(
    cdk2_pose_and_db, dtype, n_poses, atol
):
    full_pose, _ = cdk2_pose_and_db
    one_pose, compact_sidechain = _one_sidechain_movable_pose(
        full_pose, same_chain=False
    )
    one_pose = attr.evolve(one_pose, coords=one_pose.coords.to(dtype=dtype))
    pose_stack = _repeat_pose_stack(one_pose, n_poses)
    docking_dofs = build_ligand_docking_dofs(
        pose_stack, ligand_block=0, sidechain_blocks=(compact_sidechain,)
    )
    layout = LigandGeneLayout.from_pose_stack(pose_stack, docking_dofs)
    genes = layout.genes_from_pose_stack(pose_stack)
    rebuilt = layout.pose_stack_from_genes(genes)

    native_coordinate_error = torch.zeros((n_poses,), dtype=dtype)
    for block in range(pose_stack.max_n_blocks):
        atom_indices = _block_atom_indices(pose_stack, block)
        pose_indices = torch.arange(n_poses)[:, None]
        coordinate_error = torch.amax(
            torch.abs(
                rebuilt.coords[pose_indices, atom_indices]
                - pose_stack.coords[pose_indices, atom_indices]
            ),
            dim=(1, 2),
        )
        native_coordinate_error = torch.maximum(
            native_coordinate_error, coordinate_error
        )
        torch.testing.assert_close(
            rebuilt.coords[pose_indices, atom_indices],
            pose_stack.coords[pose_indices, atom_indices],
            rtol=0.0,
            atol=atol,
        )
    identical_errors = _assert_gene_roundtrip(layout, genes, atol=atol)
    if n_poses > 1:
        distinct_genes = genes.clone()
        row_offsets = torch.arange(n_poses, dtype=dtype)[:, None]
        distinct_genes[:, 0:1] += 0.1 * row_offsets
        sidechain_gene = 6 + len(docking_dofs.torsions)
        distinct_genes[:, sidechain_gene : sidechain_gene + 1] += (
            0.05 * row_offsets
        )
        distinct_errors = _assert_gene_roundtrip(layout, distinct_genes, atol=atol)
        assert distinct_errors.shape == (n_poses,)
    else:
        distinct_errors = identical_errors
    print(
        "flexible round-trip dtype/n_poses/native_per_pose_A/"
        "identical_per_pose_A/distinct_per_pose_A:",
        dtype,
        n_poses,
        native_coordinate_error.tolist(),
        identical_errors.tolist(),
        distinct_errors.tolist(),
    )


def test_zero_pocket_dof_and_gene_path_is_byte_identical(cdk2_pose_and_db):
    full_pose, _ = cdk2_pose_and_db
    ligand_block = _ligand_block(full_pose)
    pose_stack = _ligand_only_pose_stack(full_pose, ligand_block)

    historical = build_ligand_docking_dofs(pose_stack, 0)
    explicit_zero = build_ligand_docking_dofs(pose_stack, 0, sidechain_blocks=())
    historical_layout = LigandGeneLayout.from_pose_stack(pose_stack, historical)
    zero_layout = LigandGeneLayout.from_pose_stack(pose_stack, explicit_zero)
    historical_genes = historical_layout.genes_from_pose_stack(pose_stack)
    zero_genes = zero_layout.genes_from_pose_stack(pose_stack)

    assert historical.sidechain_blocks == explicit_zero.sidechain_blocks == ()
    assert historical.sidechain_torsions == explicit_zero.sidechain_torsions == ()
    assert torch.equal(historical_layout.torsion_nodes, zero_layout.torsion_nodes)
    assert torch.equal(
        historical_layout.torsion_dof_types, zero_layout.torsion_dof_types
    )
    assert torch.equal(historical_layout.torsion_signs, zero_layout.torsion_signs)
    assert torch.equal(historical_layout.torsion_offsets, zero_layout.torsion_offsets)
    assert torch.equal(historical_genes, zero_genes)
    assert torch.equal(
        historical_layout.pose_stack_from_genes(historical_genes).coords,
        zero_layout.pose_stack_from_genes(zero_genes).coords,
    )


def test_grid_final_result_writes_sidechain_coordinates(cpu_grid_case):
    case = cpu_grid_case
    full_pose = case.reference
    selected = select_pocket_residue_blocks(
        full_pose, case.ligand_block, cutoff=6.0
    )
    selected_blocks = _connected_chi_blocks(
        full_pose, case.ligand_block, selected
    )
    pose_stack = _movable_pose_stack(
        full_pose, case.ligand_block, selected_blocks
    )
    _, compact_sidechains, _ = _movable_block_mapping(
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
    torch.testing.assert_close(
        actual.genes[:, :3], genes[:, :3], rtol=0.0, atol=1.0e-9
    )
    tors_diff = actual.genes[:, 6:] - genes[:, 6:]
    wrapped = torch.atan2(torch.sin(tors_diff), torch.cos(tors_diff))
    torch.testing.assert_close(
        wrapped, torch.zeros_like(wrapped), rtol=0.0, atol=1.0e-9
    )
