import copy
import os
import time

import attr
import torch

from tmol.pose import PDBInfo, PoseStack, PoseStackBuilder
from tmol.score import ScoreFunction

from tmol.pack import (
    PackerTask,
    SetPackerTask,
    PackerEnergyTables,
    run_simulated_annealing,
    impose_top_rotamer_assignments,
)
from tmol.pack.rotamer import build_rotamers
from tmol.utility._device import synchronize_device


def pack_rotamers(
    pose_stack: PoseStack,
    sfxn: ScoreFunction,
    task: PackerTask,
    verbose: bool = False,
) -> PoseStack:
    """Optimize side-chain conformers for a pose stack.

    Args:
        pose_stack: Poses whose task-enabled blocks will be packed.
        sfxn: Score function used to rank rotamer assignments.
        task: Allowed block types, conformers, and packing positions.
        verbose: Print synchronized stage timings when true.

    Returns:
        A new pose stack containing the lowest-ranked assignment per pose.
    """

    max_poses_per_chunk = _max_poses_per_packing_chunk(pose_stack)
    if pose_stack.n_poses > max_poses_per_chunk:
        packed_chunks = []
        for first_pose in range(0, pose_stack.n_poses, max_poses_per_chunk):
            last_pose = min(first_pose + max_poses_per_chunk, pose_stack.n_poses)
            chunk_pose_stack = _slice_pose_stack_for_packing(
                pose_stack, first_pose, last_pose
            )
            chunk_task = _slice_packer_task(task, first_pose, last_pose)
            packed_chunk = pack_rotamers(
                chunk_pose_stack, sfxn, chunk_task, verbose=verbose
            )
            packed_chunks.append(packed_chunk)
        combined_pose_stack = PoseStackBuilder.from_poses(
            packed_chunks, pose_stack.device
        )
        return combined_pose_stack

    if verbose:
        synchronize_device(pose_stack.device)
    start_time = time.perf_counter()

    task = SetPackerTask.from_packer_task(task)
    pbt = pose_stack.packed_block_types

    pose_stack, rotamer_set = build_rotamers(pose_stack, task, pbt.chem_db)
    if verbose:
        synchronize_device(pose_stack.device)
    end_time1 = time.perf_counter()

    (
        packer_energy_tables,
        rotamer_for_nonmolten_block,
        n_molten_blocks_per_pose,
        bc_rot_offset_for_molten_block,
        bc_rot_to_orig_rot,
        end_time2,
        end_time3,
    ) = _calculate_packer_energies(pose_stack, sfxn, rotamer_set, task, verbose=verbose)

    if verbose:
        synchronize_device(pose_stack.device)
    end_time4 = time.perf_counter()

    _, rotamer_assignments = run_simulated_annealing(packer_energy_tables)
    if verbose:
        synchronize_device(pose_stack.device)
    end_time5 = time.perf_counter()

    new_pose_stack = impose_top_rotamer_assignments(
        pose_stack,
        rotamer_set,
        rotamer_for_nonmolten_block,
        n_molten_blocks_per_pose,
        bc_rot_offset_for_molten_block,
        bc_rot_to_orig_rot,
        rotamer_assignments,
    )
    if verbose:
        synchronize_device(pose_stack.device)
    end_time6 = time.perf_counter()

    if verbose:
        print(
            f"pack_rotamers {end_time6 - start_time: .2f}"
            + f" build rots: {end_time1 - start_time: .2f} calcRPEs: {end_time2 - end_time1: .2f}"
            + f" build IG: {end_time3 - end_time2: .2f} build IG part2: {end_time4 - end_time3: .2f}"
            + f" run SA: {end_time5 - end_time4: .2f} pose ctor: {end_time6 - end_time5: .2f}"
        )

    return new_pose_stack


_SMALL_PACKING_POSE_CHUNK = 25
_DEFAULT_PACKING_POSE_CHUNK = 10
_ESTIMATED_PACKING_BYTES_PER_BLOCK = 2 * 1024 * 1024
_PACKING_FREE_MEMORY_FRACTION = 0.25
_CPU_INTERACTION_GRAPH_CHUNK_SIZE = 16
_CUDA_INTERACTION_GRAPH_CHUNK_SIZE = 32


def _interaction_graph_chunk_size(device: torch.device) -> int:
    """Select the benchmarked backend-specific sparse-table chunk width."""
    return (
        _CUDA_INTERACTION_GRAPH_CHUNK_SIZE
        if device.type == "cuda"
        else _CPU_INTERACTION_GRAPH_CHUNK_SIZE
    )


def _max_poses_per_packing_chunk(pose_stack: PoseStack) -> int:
    """Return a bounded, memory-aware native-packer batch size.

    Native score dispatches use signed 32-bit per-chunk indices. Keeping the
    batch bounded prevents a wide pose stack from overflowing those indices
    and also limits transient score/index storage. The size cutoff and memory
    estimate are deliberately conservative: packing scratch grows with both
    residue count and rotamer density. The environment override is primarily
    for deterministic tests and performance tuning.
    """
    configured = os.environ.get("TMOL_PACK_MAX_POSES_PER_CHUNK")
    if configured is not None:
        try:
            chunk_size = int(configured)
        except ValueError as error:
            raise ValueError(
                "TMOL_PACK_MAX_POSES_PER_CHUNK must be a positive integer; "
                f"got {configured!r}"
            ) from error
        if chunk_size <= 0:
            raise ValueError(
                "TMOL_PACK_MAX_POSES_PER_CHUNK must be a positive integer; "
                f"got {configured!r}"
            )
        return chunk_size

    max_n_blocks = pose_stack.max_n_blocks
    chunk_size = (
        _SMALL_PACKING_POSE_CHUNK
        if max_n_blocks <= 256
        else _DEFAULT_PACKING_POSE_CHUNK
    )

    if pose_stack.device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info(pose_stack.device)
        memory_budget = int(free_bytes * _PACKING_FREE_MEMORY_FRACTION)
        estimated_bytes_per_pose = (
            max(1, max_n_blocks) * _ESTIMATED_PACKING_BYTES_PER_BLOCK
        )
        memory_limited_chunk = max(1, memory_budget // estimated_bytes_per_pose)
        chunk_size = min(chunk_size, memory_limited_chunk)
    return chunk_size


_PACKER_TASK_POSE_TENSORS = (
    "is_real_block",
    "per_block_orig_block_type",
    "per_block_n_considered_block_types",
    "per_block_considered_block_types",
    "per_block_considered_block_types_is_orig",
    "restrict_to_repacking_masks",
    "per_block_is_block_type_allowed",
    "per_block_conformer_sampler_allowed",
    "per_block_chi_expansion",
)


def _slice_pose_stack_for_packing(
    pose_stack: PoseStack, first_pose: int, last_pose: int
) -> PoseStack:
    """Make a cheap contiguous view for an unconstrained packing chunk.

    The packer reads the input topology and coordinates but returns a new pose
    stack, so ordinary FastRelax chunks can share input storage safely. Fall
    back to the general builder for the uncommon metadata types whose pose
    indices need remapping.
    """
    if pose_stack.split_block_mapping:
        return PoseStackBuilder.from_poses(
            [pose_stack.split(i) for i in range(first_pose, last_pose)],
            pose_stack.device,
        )

    chunk_constraint_set = None
    if pose_stack.constraint_set is not None:
        constraints = pose_stack.constraint_set
        constraint_poses = constraints.constraint_atoms[:, :, 0]
        selected = torch.where(
            ((constraint_poses >= first_pose) & (constraint_poses < last_pose)).any(
                dim=1
            )
        )[0]
        chunk_atoms = constraints.constraint_atoms[selected].clone()
        chunk_atom_poses = chunk_atoms[:, :, 0]
        real_atoms = chunk_atom_poses != -1
        chunk_atom_poses[real_atoms] -= first_pose
        chunk_atoms[:, :, 0] = chunk_atom_poses
        chunk_unique_blocks = constraints.constraint_unique_blocks[selected].clone()
        chunk_unique_blocks[:, 0] -= first_pose
        chunk_constraint_set = attr.evolve(
            constraints,
            n_poses=last_pose - first_pose,
            constraint_function_inds=constraints.constraint_function_inds[selected],
            constraint_atoms=chunk_atoms,
            constraint_params=constraints.constraint_params[selected],
            constraint_num_unique_blocks=constraints.constraint_num_unique_blocks[
                selected
            ],
            constraint_unique_blocks=chunk_unique_blocks,
        )

    pdb_info = pose_stack.pdb_info
    chunk_pdb_info = PDBInfo(
        residue_labels=pdb_info.residue_labels[first_pose:last_pose].copy(),
        residue_insertion_codes=pdb_info.residue_insertion_codes[
            first_pose:last_pose
        ].copy(),
        chain_labels=pdb_info.chain_labels[first_pose:last_pose].copy(),
        atom_occupancy=pdb_info.atom_occupancy[first_pose:last_pose].copy(),
        atom_b_factor=pdb_info.atom_b_factor[first_pose:last_pose].copy(),
    )

    def view(tensor):
        return tensor[first_pose:last_pose].detach()

    return PoseStack(
        packed_block_types=pose_stack.packed_block_types,
        coords=view(pose_stack.coords),
        block_coord_offset=view(pose_stack.block_coord_offset),
        block_coord_offset64=view(pose_stack.block_coord_offset64),
        inter_residue_connections=view(pose_stack.inter_residue_connections),
        inter_residue_connections64=view(pose_stack.inter_residue_connections64),
        inter_block_bondsep=view(pose_stack.inter_block_bondsep),
        inter_block_bondsep64=view(pose_stack.inter_block_bondsep64),
        block_type_ind=view(pose_stack.block_type_ind),
        block_type_ind64=view(pose_stack.block_type_ind64),
        chain_id=view(pose_stack.chain_id),
        chain_id64=view(pose_stack.chain_id64),
        pdb_info=chunk_pdb_info,
        constraint_set=chunk_constraint_set,
        device=pose_stack.device,
        split_block_mapping=None,
    )


def _slice_packer_task(task: PackerTask, first_pose: int, last_pose: int) -> PackerTask:
    """Shallow-copy a configured task and slice all pose-major tensors."""
    chunk_task = copy.copy(task)
    for attribute in _PACKER_TASK_POSE_TENSORS:
        value = getattr(task, attribute)
        setattr(chunk_task, attribute, value[first_pose:last_pose])
    chunk_task.real_block_pose, chunk_task.real_block_block = torch.nonzero(
        chunk_task.is_real_block, as_tuple=True
    )
    return chunk_task


def _calculate_packer_energies(pose_stack, sfxn, rotamer_set, task, verbose=False):
    from tmol.pack.compiled import build_interaction_graph

    pbt = pose_stack.packed_block_types
    rotamer_scoring_module = sfxn.render_rotamer_scoring_module(pose_stack, rotamer_set)

    if pose_stack.device.type == "cuda":
        # CUDA graph construction atomically accumulates duplicate coordinates,
        # so it can consume raw int32 entries. Avoid constructing/coalescing a
        # PyTorch COO tensor: COO promotes coordinates to int64 and sorting can
        # briefly require another full-size copy.
        energy_indices, energy_values = rotamer_scoring_module.forward_sparse_entries(
            rotamer_set.coords
        )
    else:
        # The CPU interaction-graph accumulator is deliberately non-atomic.
        # Coalesce cross-layout duplicates before its parallel native loops.
        energies = rotamer_scoring_module(rotamer_set.coords).coalesce()
        energy_indices = energies.indices().to(torch.int32)
        energy_values = energies.values()

    if verbose:
        synchronize_device(pose_stack.device)
    end_time2 = time.perf_counter()

    chunk_size = _interaction_graph_chunk_size(pose_stack.device)

    (
        max_n_bump_checked_rotamers_per_pose_tensor,
        n_molten_blocks_per_pose,
        n_bc_rots_per_pose,
        bc_rot_offset_for_pose,
        n_bc_rots_for_molten_block,
        bc_rot_offset_for_molten_block,
        molten_block_ind_for_bc_rot,
        rotamer_for_nonmolten_block,
        bc_rot_to_orig_rot,
        bg_bg_energies,
        energy1b,
        chunk_pair_offset_for_block_pair,
        chunk_pair_offset,
        energy2b,
    ) = build_interaction_graph(
        task.bump_check,
        chunk_size,
        pbt.n_types,
        rotamer_set.n_rots_for_pose,
        rotamer_set.rot_offset_for_pose,
        rotamer_set.n_rots_for_block,
        rotamer_set.rot_offset_for_block,
        rotamer_set.pose_for_rot,
        rotamer_set.block_type_ind_for_rot,
        rotamer_set.block_ind_for_rot,
        energy_indices,
        energy_values,
        verbose,
    )
    if verbose:
        synchronize_device(pose_stack.device)
    end_time3 = time.perf_counter()

    packer_energy_tables = PackerEnergyTables(
        max_n_rotamers_per_pose=max_n_bump_checked_rotamers_per_pose_tensor.item(),
        pose_n_res=n_molten_blocks_per_pose,
        pose_n_rotamers=n_bc_rots_per_pose,
        pose_rotamer_offset=bc_rot_offset_for_pose,
        nrotamers_for_res=n_bc_rots_for_molten_block,
        oneb_offsets=bc_rot_offset_for_molten_block,
        res_for_rot=molten_block_ind_for_bc_rot,
        chunk_size=chunk_size,
        chunk_offset_offsets=chunk_pair_offset_for_block_pair,
        chunk_offsets=chunk_pair_offset,
        energy1b=energy1b,
        energy2b=energy2b,
    )

    return (
        packer_energy_tables,
        rotamer_for_nonmolten_block,
        n_molten_blocks_per_pose,
        bc_rot_offset_for_molten_block,
        bc_rot_to_orig_rot,
        end_time2,
        end_time3,
    )
