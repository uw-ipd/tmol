"""M13a-1 gates for flexible-pocket sidechain DOF plumbing."""

from __future__ import annotations


import attr
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
from tmol.tests.dock.test_ligand_dof_roundtrip import (
    _ligand_block,
)
from tmol.pose import repeat_pose_stack
from tmol.score.gridscore._movable import (
    movable_block_mapping,
    movable_pose_stack,
)


def _minimum_heavy_distance(pose_stack, receptor_block, ligand_block):
    receptor = _block_heavy_coords(pose_stack, receptor_block)
    ligand = _block_heavy_coords(pose_stack, ligand_block)
    return float(torch.amin(torch.cdist(receptor, ligand)))


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
