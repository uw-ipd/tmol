import torch

from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.io.details.compiled.compiled import gen_pose_hydrogens


def build_missing_hydrogens(
    coords: Tensor[torch.int32][:, :, :, 3],
    packed_block_types: PackedBlockTypes,
    chain_begin: Tensor[torch.int32][:, :],  # unused for now
    block_types: Tensor[torch.int32][:, :],
    missing: Tensor[torch.int32][:, :, :],
):
    # ok,
    # we're going to call gen_pose_hydrogens,
    # but first we need to prepare the input tensors
    # that are going to use
    device = packed_block_types.device
    n_poses = coords.shape[0]
    max_n_res = coords.shape[1]

    n_atoms = torch.zeros((n_poses, max_n_res), dtype=torch.int32, device=device)
    real_block_types = block_types != -1
    n_atoms[real_block_types] = packed_block_types.n_atoms[
        block_types[real_block_types]
    ]

    n_ats_inccumsum = torch.cumsum(n_ats, dim=1, dtype=torch.int32)
    max_n_ats = torch.max(n_ats_inccumsum[:, -1])
    block_coord_offset = torch.cat(
        (
            torch.zeros((n_poses, 1), dtype=torch.int32, device=device),
            n_ats_inccumsum[:, :-1],
        ),
        dim=1,
    )

    pose_like_coords = torch.zeros(
        (n_poses, max_n_ats, 3), dtype=torch.float32, device=device
    )
