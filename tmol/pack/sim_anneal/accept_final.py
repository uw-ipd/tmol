import torch

# import numpy

from tmol.utility.tensor.common_operations import stretch, exclusive_cumsum2d_and_totals
from tmol.score.common.stack_condense import (
    condense_subset,
    take_values_w_sentineled_index,
)

from tmol.types.torch import Tensor
from tmol.types.functional import validate_args

# from tmol.chemical.restypes import Residue
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack


# from tmol.score.score_graph import score_graph
# from tmol.score.bonded_atom import BondedAtomScoreGraph
# from tmol.score.coordinates import CartesianAtomicCoordinateProvider
# from tmol.score.device import TorchDevice
# from tmol.score.score_components import ScoreComponentClasses, IntraScore
# from tmol.io.generic import to_pdb


@validate_args
def poses_from_assigned_rotamers(
    orig_poses: PoseStack,
    packed_block_types: PackedBlockTypes,
    pose_id_for_context: Tensor[torch.int32][:],
    context_coords: Tensor[torch.float32][:, :, 3],
    context_coord_offsets: Tensor[torch.int32][:, :],
    context_block_type: Tensor[torch.int32][:, :],
) -> PoseStack:
    pbt = packed_block_types
    device = pbt.device

    n_poses = context_coords.shape[0]
    max_context_coords_n_atoms = context_coords.shape[1]

    real_context_blocks = context_block_type != -1
    (real_context_block_context_ind, real_context_block_block_ind) = torch.nonzero(
        real_context_blocks, as_tuple=True
    )

    context_block_type64 = context_block_type.to(torch.int64)
    n_context_atoms = take_values_w_sentineled_index(
        pbt.n_atoms, context_block_type64, default_fill=0
    )
    n_atoms_offset, n_ats_total = exclusive_cumsum2d_and_totals(n_context_atoms)

    atom_begin = torch.zeros(
        (n_poses, max_context_coords_n_atoms), dtype=torch.int32, device=device
    )
    (nz_context_coord_offsets, _) = torch.nonzero(
        context_coord_offsets != -1, as_tuple=True
    )

    context_coord_offsets64 = context_coord_offsets.to(torch.int64)
    atom_begin[
        nz_context_coord_offsets, context_coord_offsets64[context_coord_offsets != -1]
    ] = 1
    cs_atom_begin = torch.cumsum(atom_begin, dim=1)
    block_for_atom = cs_atom_begin - 1

    context_for_atom64 = stretch(
        torch.arange(n_poses, dtype=torch.int64), max_context_coords_n_atoms
    ).view(n_poses, max_context_coords_n_atoms)
    block_type_for_atom64 = context_block_type64[
        context_for_atom64, block_for_atom
    ].view(n_poses, max_context_coords_n_atoms)

    block_n_atoms_for_atom = pbt.n_atoms[block_type_for_atom64]

    context_block_offset_for_atom = torch.gather(
        context_coord_offsets, dim=1, index=block_for_atom
    )

    block_ind_for_atom = (
        torch.remainder(
            torch.arange(
                n_poses * max_context_coords_n_atoms, dtype=torch.int32, device=device
            ),
            max_context_coords_n_atoms,
        ).view(n_poses, max_context_coords_n_atoms)
        - context_block_offset_for_atom
    )

    # we are building up to answering the question: what coordinates in the
    # context_coords tensor are real, and which ones are scratch space
    # used to hold the rotamers from the largest block types allowed at
    # each position. We have identified for each position (each "atom") in
    # the context_coords tensor what block it belongs to and how many atoms
    # are in that block. Then using a per-block arange, we can determine
    # which of those atoms are in the range [0..n_atoms) for that block.
    # The line below tells us which coordinates are real.
    context_atom_is_legit = block_ind_for_atom < block_n_atoms_for_atom

    condensed_coords = condense_subset(
        context_coords, context_atom_is_legit, default_fill=0.0
    )

    pid4c_64 = pose_id_for_context.to(torch.int64)

    return PoseStack(
        packed_block_types=packed_block_types,
        coords=condensed_coords,
        block_coord_offset=n_atoms_offset,
        block_coord_offset64=n_atoms_offset.to(torch.int64),
        inter_residue_connections=orig_poses.inter_residue_connections[pid4c_64],
        inter_residue_connections64=orig_poses.inter_residue_connections64[pid4c_64],
        inter_block_bondsep=orig_poses.inter_block_bondsep[pid4c_64],
        inter_block_bondsep64=orig_poses.inter_block_bondsep64[pid4c_64],
        block_type_ind=context_block_type,
        block_type_ind64=context_block_type64,
        device=orig_poses.device,
    )
