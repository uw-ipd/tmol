import contextlib
import copy
import os
import time

import attr
import torch

from tmol.pose._pose_stack import PoseStack
from tmol.pose._pose_stack_builder import PoseStackBuilder
from tmol.score._score_function import ScoreFunction

from tmol.pack._packer_task import PackerTask, SetPackerTask
from tmol.pack._datatypes import PackerEnergyTables
from tmol.pack._simulated_annealing import _run_simulated_annealing
from tmol.pack._impose_rotamers import (
    _impose_top_rotamer_assignments,
    _chosen_rotamer_for_block,
)
from tmol.pack.rotamer._build_rotamers import build_rotamers
from tmol.utility._device import _synchronize_device
from tmol.pack.rotamer._conjugated_groups import _write_group_members


def pack_rotamers(
    pose_stack: PoseStack,
    sfxn: ScoreFunction,
    task: PackerTask,
    verbose: bool = False,
    seed: int | None = None,
    anneal_device: torch.device | str | None = None,
) -> PoseStack:
    """Optimize side-chain conformers for a pose stack.

    The annealer draws from torch's generator on the pose device, so
    ``torch.manual_seed`` controls it on the CPU as on CUDA.

    Args:
        pose_stack: Poses whose task-enabled blocks will be packed.
        sfxn: Score function used to rank rotamer assignments.
        task: Allowed block types, conformers, and packing positions.
        verbose: Print synchronized stage timings when true.
        seed: Seed for this call only, leaving torch's global random state unchanged.
        anneal_device: Run annealing on this device instead of the pose
            stack's. Simulated annealing reads the interaction graph and
            writes rotamer indices, so it can run anywhere the tables can be
            put; on Apple silicon the tables cost almost nothing to move
            (9 ms for 174 MB of unified memory) and the Metal annealer
            searches both faster and far better than the host one. Left
            unset, host poses anneal on Metal when it is available, unless
            ``TMOL_MPS_OFFLOAD=0``.

    Returns:
        A new pose stack containing the lowest-ranked assignment per pose.
    """
    if seed is not None:
        cuda = pose_stack.device.type == "cuda"
        with torch.random.fork_rng(devices=[pose_stack.device] if cuda else []):
            torch.random.default_generator.manual_seed(seed)
            if cuda:
                with torch.cuda.device(pose_stack.device):
                    torch.cuda.manual_seed(seed)
            return pack_rotamers(
                pose_stack, sfxn, task, verbose=verbose, anneal_device=anneal_device
            )

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
                chunk_pose_stack,
                sfxn,
                chunk_task,
                verbose=verbose,
                anneal_device=anneal_device,
            )
            packed_chunks.append(packed_chunk)
        combined_pose_stack = PoseStackBuilder.from_poses(
            packed_chunks, pose_stack.device
        )
        return combined_pose_stack

    if verbose:
        _synchronize_device(pose_stack.device)
    start_time = time.perf_counter()

    task = SetPackerTask.from_packer_task(task)
    pbt = pose_stack.packed_block_types

    pose_stack, rotamer_set = build_rotamers(pose_stack, task, pbt.chem_db)
    if verbose:
        _synchronize_device(pose_stack.device)
    end_time1 = time.perf_counter()

    (
        packer_energy_tables,
        rotamer_for_nonmolten_block,
        n_molten_blocks_per_pose,
        bc_rot_offset_for_molten_block,
        bc_rot_to_orig_rot,
        collapse,
        _bg_bg_energies,
        end_time2,
        end_time3,
    ) = _calculate_packer_energies(pose_stack, sfxn, rotamer_set, task, verbose=verbose)

    if verbose:
        _synchronize_device(pose_stack.device)
    end_time4 = time.perf_counter()

    _, rotamer_assignments = _anneal_on(
        packer_energy_tables, anneal_device, pose_stack.device
    )
    if verbose:
        _synchronize_device(pose_stack.device)
    end_time5 = time.perf_counter()

    assignment_context = (
        torch.no_grad()
        if _should_stream_interaction_graph(pose_stack, rotamer_set, task)
        else contextlib.nullcontext()
    )
    with assignment_context:
        new_pose_stack = _impose_top_rotamer_assignments(
            pose_stack,
            rotamer_set,
            rotamer_for_nonmolten_block,
            n_molten_blocks_per_pose,
            bc_rot_offset_for_molten_block,
            bc_rot_to_orig_rot,
            rotamer_assignments,
        )
    if collapse is not None:
        # the packer chose for the representative; its group moves with it
        assignment = _chosen_rotamer_for_block(
            pose_stack,
            rotamer_for_nonmolten_block,
            n_molten_blocks_per_pose,
            bc_rot_offset_for_molten_block,
            bc_rot_to_orig_rot,
            rotamer_assignments[:, 0, :],
        )
        new_pose_stack = _write_group_members(
            new_pose_stack, rotamer_set, collapse, assignment
        )
    if verbose:
        _synchronize_device(pose_stack.device)
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

    chunk_pdb_info = pose_stack.pdb_info.sliced(first_pose, last_pose)

    def view(tensor):
        return tensor[first_pose:last_pose].detach()

    return PoseStack(
        packed_block_types=pose_stack.packed_block_types,
        coords=view(pose_stack.coords),
        block_coord_offset=view(pose_stack.block_coord_offset),
        block_coord_offset64=view(pose_stack.block_coord_offset64),
        inter_residue_connections=view(pose_stack.inter_residue_connections),
        inter_residue_connections64=view(pose_stack.inter_residue_connections64),
        inter_block_bondsep=pose_stack.inter_block_bondsep.select_poses(
            first_pose, last_pose
        ),
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


def _anneal_on(energy_tables, anneal_device, pose_device):
    """Anneal the interaction graph, optionally on a device of its own.

    The graph is already built when this runs, and annealing only reads it and
    returns rotamer indices, so the choice of device is a backend choice
    rather than a semantic one. Left unset, annealing stays where the poses
    are, except that host poses anneal on the Apple GPU when CPU scoring is
    offloading there too (see tmol.score._mps_offload).
    """
    if anneal_device is None and pose_device.type == "cpu":
        from tmol.score._mps_offload import mps_offload_enabled

        if mps_offload_enabled():
            anneal_device = "mps"
    if anneal_device is None:
        return _run_simulated_annealing(energy_tables)

    anneal_device = torch.device(anneal_device)
    if anneal_device == pose_device:
        return _run_simulated_annealing(energy_tables)

    # Not TensorGroup.to: the table group carries plain ints beside its
    # tensors, which that method rejects.
    moved = attr.evolve(
        energy_tables,
        **{
            name: value.to(anneal_device)
            for name, value in attr.asdict(energy_tables, recurse=False).items()
            if torch.is_tensor(value)
        },
    )
    scores, assignments = _run_simulated_annealing(moved)
    return scores.to(pose_device), assignments.to(pose_device)


def _should_stream_interaction_graph(pose_stack, rotamer_set, task) -> bool:
    """Select the non-differentiable bounded-memory CUDA packing path."""
    return (
        pose_stack.device.type == "cuda"
        and not rotamer_set.correlated_groups
        and not task.bump_check
    )


def _calculate_packer_energies(pose_stack, sfxn, rotamer_set, task, verbose=False):
    from tmol.pack.compiled._compiled import _build_interaction_graph
    from tmol.pack.rotamer._conjugated_groups import (
        _collapse_group_energies,
        _collapse_group_rotamers,
    )

    pbt = pose_stack.packed_block_types
    rotamer_scoring_module = sfxn.render_rotamer_scoring_module(pose_stack, rotamer_set)
    stream_interaction_graph = _should_stream_interaction_graph(
        pose_stack, rotamer_set, task
    )

    n_unique_entries = -1
    device_entries = None
    if stream_interaction_graph:
        energy_indices = None
        energy_values = None
    elif pose_stack.device.type == "cuda" and not rotamer_set.correlated_groups:
        # CUDA graph construction atomically accumulates duplicate coordinates,
        # so it can consume raw int32 entries. Avoid constructing/coalescing a
        # PyTorch COO tensor: COO promotes coordinates to int64 and sorting can
        # briefly require another full-size copy.
        energy_indices, energy_values = rotamer_scoring_module.forward_sparse_entries(
            rotamer_set.coords
        )
    elif (
        hasattr(rotamer_scoring_module, "forward_device_entries")
        and not rotamer_set.correlated_groups
        and not task.bump_check
    ):
        # A scorer whose table is computed on the GPU (see
        # tmol.score._mps_offload) keeps it there, and the graph is built
        # beside it; see _build_device_interaction_graph.
        device_entries = rotamer_scoring_module.forward_device_entries(
            rotamer_set.coords
        )
        energy_indices = None
        energy_values = None
    elif (
        hasattr(rotamer_scoring_module, "unique_prefix_nnz")
        and not rotamer_set.correlated_groups
    ):
        # A split scorer reports where its distinct-key prefix ends (see the
        # contract below) for its raw int32 entries too, so take those
        # directly and skip the int64 COO round trip.
        energy_indices, energy_values = rotamer_scoring_module.forward_sparse_entries(
            rotamer_set.coords
        )
        n_unique_entries = rotamer_scoring_module.unique_prefix_nnz
        if n_unique_entries is None:
            energies = torch.sparse_coo_tensor(
                energy_indices, energy_values, rotamer_scoring_module.shape
            ).coalesce()
            energy_indices = energies.indices().to(torch.int32)
            energy_values = energies.values()
            n_unique_entries = -1
    else:
        # The CPU interaction-graph accumulator is deliberately non-atomic, so
        # it may only run its parallel passes over entries whose keys are
        # distinct. Coalescing guarantees that globally, and is what happens
        # unless the scoring module can say where the duplicates start.
        #
        # A module may expose `unique_prefix_nnz`: the number of leading
        # entries of the table it just returned that are known to carry
        # distinct (pose, rot1, rot2) keys. A split scorer knows this without
        # looking -- it concatenates one internally-unique layout ahead of the
        # rest -- and the graph builder then runs parallel over the prefix and
        # serial over the tail, which is cheaper than sorting the whole table.
        energies = rotamer_scoring_module(rotamer_set.coords)
        unique_prefix = getattr(rotamer_scoring_module, "unique_prefix_nnz", None)
        if unique_prefix is None:
            energies = energies.coalesce()
            n_unique_entries = -1
        else:
            n_unique_entries = int(unique_prefix)
        # The underscore accessors: an uncoalesced table is exactly what the
        # prefix contract describes, and the checked accessors refuse it.
        energy_indices = energies._indices().to(torch.int32)
        energy_values = energies._values()

    # a group of covalently joined blocks is one choice, not several: fold its
    #    members onto a representative so the packer cannot pick a conformer
    #    for one that disagrees with its neighbour
    groups = rotamer_set.correlated_groups
    collapsed = _collapse_group_rotamers(pose_stack, rotamer_set, groups)
    if collapsed is not None:
        collapse, group_of_rot, conformer_of_rot = collapsed
        energies = _collapse_group_energies(
            energies, collapse, group_of_rot, conformer_of_rot
        )
        energy_indices = energies.indices().to(torch.int32)
        energy_values = energies.values()
        # collapse_group_energies returns a coalesced table of its own.
        n_unique_entries = -1
        n_rots_for_block = collapse.compact_n_rots_for_block
        rot_offset_for_block = collapse.compact_rot_offset_for_block
        block_ind_for_rot = collapse.compact_block_ind_for_rot
        pose_for_rot = collapse.compact_pose_for_rot
        block_type_ind_for_rot = collapse.compact_block_type_ind_for_rot
        n_rots_for_pose = collapse.compact_n_rots_for_pose
        rot_offset_for_pose = collapse.compact_rot_offset_for_pose
    else:
        collapse = None
        n_rots_for_block = rotamer_set.n_rots_for_block
        rot_offset_for_block = rotamer_set.rot_offset_for_block
        block_ind_for_rot = rotamer_set.block_ind_for_rot
        pose_for_rot = rotamer_set.pose_for_rot
        block_type_ind_for_rot = rotamer_set.block_type_ind_for_rot
        n_rots_for_pose = rotamer_set.n_rots_for_pose
        rot_offset_for_pose = rotamer_set.rot_offset_for_pose

    if verbose:
        _synchronize_device(pose_stack.device)
    end_time2 = time.perf_counter()

    chunk_size = _interaction_graph_chunk_size(pose_stack.device)

    graph_inputs = (
        pbt.n_types,
        n_rots_for_pose,
        rot_offset_for_pose,
        n_rots_for_block,
        rot_offset_for_block,
        pose_for_rot,
        block_type_ind_for_rot,
        block_ind_for_rot,
    )
    if stream_interaction_graph:
        graph_result = _build_streaming_interaction_graph(
            rotamer_scoring_module,
            rotamer_set.coords,
            chunk_size,
            graph_inputs,
            verbose,
        )
    elif device_entries is not None:
        graph_result = _build_device_interaction_graph(
            device_entries, chunk_size, graph_inputs, verbose
        )
    else:
        graph_result = _build_interaction_graph(
            task.bump_check,
            chunk_size,
            *graph_inputs,
            energy_indices,
            energy_values,
            n_unique_entries,
            verbose,
        )
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
        neighbor_row_offsets,
        neighbor_blocks,
        neighbor_chunk_offset_offsets,
        chunk_pair_offset,
        energy2b,
    ) = graph_result
    if verbose:
        _synchronize_device(pose_stack.device)
    end_time3 = time.perf_counter()

    if collapse is not None:
        # the graph was built on the compacted numbering; everything downstream
        #    addresses the rotamer set itself, so hand back original indices
        bc_rot_to_orig_rot = collapse.compact_to_orig[bc_rot_to_orig_rot]
        is_bg = rotamer_for_nonmolten_block != -1
        rotamer_for_nonmolten_block = torch.where(
            is_bg,
            collapse.compact_to_orig[rotamer_for_nonmolten_block.clamp(min=0)],
            rotamer_for_nonmolten_block,
        )

    packer_energy_tables = PackerEnergyTables(
        max_n_rotamers_per_pose=max_n_bump_checked_rotamers_per_pose_tensor.item(),
        pose_n_res=n_molten_blocks_per_pose,
        pose_n_rotamers=n_bc_rots_per_pose,
        pose_rotamer_offset=bc_rot_offset_for_pose,
        nrotamers_for_res=n_bc_rots_for_molten_block,
        oneb_offsets=bc_rot_offset_for_molten_block,
        res_for_rot=molten_block_ind_for_bc_rot,
        chunk_size=chunk_size,
        neighbor_row_offsets=neighbor_row_offsets,
        neighbor_blocks=neighbor_blocks,
        neighbor_chunk_offset_offsets=neighbor_chunk_offset_offsets,
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
        collapse,
        bg_bg_energies,
        end_time2,
        end_time3,
    )


@torch.no_grad()
def _build_device_interaction_graph(entries, chunk_size, graph_inputs, verbose):
    """Build the interaction graph on the device that holds the energy table.

    The streaming builder's passes, run over a table that is already
    materialized on the GPU rather than re-scored page by page: the entry-wise
    passes (block adjacency, chunk occupancy, energy accumulation) run there,
    one entry per thread with atomics, while the topology steps between them
    -- per block and per chunk pair, a few thousand rows -- run on the host.
    The rotamer numbering and everything addressed by it stay on the host as
    the CPU builder leaves them; the one- and two-body energy tables stay on
    the GPU, where the annealer reads them.
    """
    from tmol.pack.compiled._compiled import (
        _accumulate_interaction_graph_entries,
        _build_interaction_graph,
        _finalize_interaction_graph_chunk_topology,
        _finalize_interaction_graph_topology,
        _initialize_interaction_graph_topology,
        _note_interaction_graph_chunk_topology,
        _note_interaction_graph_topology,
    )

    (
        _max_n_block_types,
        _n_rots_for_pose,
        _rot_offset_for_pose,
        n_rots_for_block,
        rot_offset_for_block,
        _pose_for_rot,
        _block_type_ind_for_rot,
        block_ind_for_rot,
    ) = graph_inputs
    device = entries[0][1].device if entries else torch.device("cpu")
    empty_indices = torch.empty((3, 0), dtype=torch.int32)
    empty_values = torch.empty(0, dtype=torch.float32)

    # The empty build fixes the kept/fixed rotamer numbering on the host.
    base = list(
        _build_interaction_graph(
            False,
            chunk_size,
            *graph_inputs,
            empty_indices,
            empty_values,
            -1,
            verbose,
        )
    )
    block_adjacency, orig_block_to_molten = _initialize_interaction_graph_topology(
        n_rots_for_block, base[4], empty_values
    )

    def up(tensor):
        return tensor.to(device)

    n_rots_for_block_d = up(n_rots_for_block)
    rot_offset_for_block_d = up(rot_offset_for_block)
    block_ind_for_rot_d = up(block_ind_for_rot)
    orig_block_to_molten_d = up(orig_block_to_molten)
    n_bc_rots_for_molten_block_d = up(base[4])

    # Pass one: which molten blocks interact.
    block_adjacency = up(block_adjacency)
    for indices, values in entries:
        (block_adjacency,) = _note_interaction_graph_topology(
            block_ind_for_rot_d,
            orig_block_to_molten_d,
            block_adjacency,
            indices,
            values,
        )
    chunk_topology = list(
        _finalize_interaction_graph_topology(
            chunk_size, base[4], block_adjacency.cpu(), empty_values
        )
    )
    del block_adjacency

    # Pass two: which rotamer chunks inside those block pairs interact.
    neighbor_row_offsets_d = up(chunk_topology[0])
    neighbor_blocks_d = up(chunk_topology[1])
    chunk_adjacency = up(chunk_topology[4])
    for indices, values in entries:
        (chunk_adjacency,) = _note_interaction_graph_chunk_topology(
            chunk_size,
            n_rots_for_block_d,
            rot_offset_for_block_d,
            block_ind_for_rot_d,
            orig_block_to_molten_d,
            n_bc_rots_for_molten_block_d,
            neighbor_row_offsets_d,
            neighbor_blocks_d,
            up(chunk_topology[3]),
            chunk_adjacency,
            indices,
            values,
        )
    base[11], base[12], base[13] = chunk_topology[:3]
    base[14], base[15] = _finalize_interaction_graph_chunk_topology(
        chunk_size,
        base[4],
        chunk_topology[0],
        chunk_topology[1],
        chunk_topology[2],
        chunk_topology[3],
        chunk_adjacency.cpu(),
        empty_values,
    )
    del chunk_topology, chunk_adjacency

    # Pass three: accumulate the energies.
    bg_bg_energies = up(base[9])
    energy1b = up(base[10])
    energy2b = up(base[15])
    rotamer_for_nonmolten_block_d = up(base[7])
    bc_rot_offset_for_molten_block_d = up(base[5])
    neighbor_chunk_offset_offsets_d = up(base[13])
    chunk_offsets_d = up(base[14])
    for indices, values in entries:
        bg_bg_energies, energy1b, energy2b = _accumulate_interaction_graph_entries(
            chunk_size,
            n_rots_for_block_d,
            rot_offset_for_block_d,
            block_ind_for_rot_d,
            orig_block_to_molten_d,
            rotamer_for_nonmolten_block_d,
            n_bc_rots_for_molten_block_d,
            bc_rot_offset_for_molten_block_d,
            neighbor_row_offsets_d,
            neighbor_blocks_d,
            neighbor_chunk_offset_offsets_d,
            chunk_offsets_d,
            bg_bg_energies,
            energy1b,
            energy2b,
            indices,
            values,
        )
    base[9] = bg_bg_energies.cpu()
    base[10] = energy1b
    base[15] = energy2b
    return tuple(base)


@torch.no_grad()
def _build_streaming_interaction_graph(
    rotamer_scoring_module,
    coords,
    chunk_size,
    graph_inputs,
    verbose,
):
    """Build a CUDA interaction graph with three bounded score passes.

    The graph is inference-only. Running the passes under ``no_grad`` keeps
    each bounded page from retaining autograd state for the whole candidate
    set, which otherwise accumulates across FastRelax packing stages.
    """
    from tmol.pack.compiled._compiled import (
        _accumulate_interaction_graph_entries,
        _build_interaction_graph,
        _finalize_interaction_graph_chunk_topology,
        _finalize_interaction_graph_topology,
        _initialize_interaction_graph_topology,
        _note_interaction_graph_chunk_topology,
        _note_interaction_graph_topology,
    )

    (
        max_n_block_types,
        n_rots_for_pose,
        rot_offset_for_pose,
        n_rots_for_block,
        rot_offset_for_block,
        pose_for_rot,
        block_type_ind_for_rot,
        block_ind_for_rot,
    ) = graph_inputs
    empty_indices = torch.empty((3, 0), dtype=torch.int32, device=coords.device)
    empty_values = torch.empty(0, dtype=torch.float32, device=coords.device)

    # Empty no-bump construction establishes the canonical kept/fixed rotamer
    # numbering without retaining any score layout.
    base = list(
        _build_interaction_graph(
            False,
            chunk_size,
            max_n_block_types,
            n_rots_for_pose,
            rot_offset_for_pose,
            n_rots_for_block,
            rot_offset_for_block,
            pose_for_rot,
            block_type_ind_for_rot,
            block_ind_for_rot,
            empty_indices,
            empty_values,
            # The table is empty: this call establishes the kept/fixed rotamer
            # numbering and retains no score layout, so "all entries unique"
            # is both true and irrelevant.
            -1,
            verbose,
        )
    )
    block_adjacency, orig_block_to_molten = _initialize_interaction_graph_topology(
        n_rots_for_block,
        base[4],
        empty_values,
    )

    # Pass one records block topology only. Chunk support is allocated after
    # the sparse block graph is known, avoiding a global n_chunks**2 bitset.
    for _, indices, values in rotamer_scoring_module._iter_weighted_sparse_entries(
        coords, retain_shared_dispatch=False, topology_only=True
    ):
        (block_adjacency,) = _note_interaction_graph_topology(
            block_ind_for_rot,
            orig_block_to_molten,
            block_adjacency,
            indices,
            values,
        )
        del indices, values

    chunk_topology = list(
        _finalize_interaction_graph_topology(
            chunk_size,
            base[4],
            block_adjacency,
            empty_values,
        )
    )
    del block_adjacency

    # Pass two records chunk support inside each observed block edge.
    for _, indices, values in rotamer_scoring_module._iter_weighted_sparse_entries(
        coords, retain_shared_dispatch=False, topology_only=True
    ):
        (chunk_topology[4],) = _note_interaction_graph_chunk_topology(
            chunk_size,
            n_rots_for_block,
            rot_offset_for_block,
            block_ind_for_rot,
            orig_block_to_molten,
            base[4],
            chunk_topology[0],
            chunk_topology[1],
            chunk_topology[3],
            chunk_topology[4],
            indices,
            values,
        )
        del indices, values

    base[11], base[12], base[13] = chunk_topology[:3]
    base[14], base[15] = _finalize_interaction_graph_chunk_topology(
        chunk_size,
        base[4],
        chunk_topology[0],
        chunk_topology[1],
        chunk_topology[2],
        chunk_topology[3],
        chunk_topology[4],
        empty_values,
    )
    del chunk_topology

    # Pass three adds terms in canonical order. Native accumulation is
    # additive, preserving duplicate coordinates and CSR transpose symmetry.
    for _, indices, values in rotamer_scoring_module._iter_weighted_sparse_entries(
        coords, retain_shared_dispatch=False
    ):
        base[9], base[10], base[15] = _accumulate_interaction_graph_entries(
            chunk_size,
            n_rots_for_block,
            rot_offset_for_block,
            block_ind_for_rot,
            orig_block_to_molten,
            base[7],
            base[4],
            base[5],
            base[11],
            base[12],
            base[13],
            base[14],
            base[9],
            base[10],
            base[15],
            indices,
            values,
        )
        del indices, values
    return tuple(base)
