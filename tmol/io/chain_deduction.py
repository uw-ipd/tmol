import numpy
import torch
import scipy
from tmol.pose.pose_stack import PoseStack
from tmol.types.functional import validate_args
from tmol.types.array import NDArray
from tmol.utility.ndarray.common_operations import exclusive_cumsum1d


def chain_inds_for_pose_stack(
    pose_stack: PoseStack,
) -> NDArray[numpy.int32][:, :]:
    n_poses = pose_stack.n_poses
    max_n_blocks = pose_stack.max_n_blocks
    max_n_conn = pose_stack.packed_block_types.max_n_conn

    bond_pairs = torch.full(
        (n_poses, max_n_blocks * max_n_conn * 2, 2),
        -1,
        dtype=torch.int32,
        device=pose_stack.device,
    )

    # 1. write down bond pairs forward
    bond_pairs[:, : max_n_blocks * max_n_conn, 0] = (
        torch.arange(max_n_blocks, dtype=torch.int32, device=pose_stack.device)
        .repeat_interleave(
            torch.full(
                (max_n_blocks,), max_n_conn, dtype=torch.int64, device=pose_stack.device
            )
        )
        .repeat((n_poses, 1))
    )
    bond_pairs[
        :, : max_n_blocks * max_n_conn, 1
    ] = pose_stack.inter_residue_connections[:, :, :, 0].view(
        n_poses, max_n_blocks * max_n_conn
    )

    # 2. and backward
    bond_pairs[
        :, max_n_blocks * max_n_conn :, 0
    ] = pose_stack.inter_residue_connections[:, :, :, 0].view(
        n_poses, max_n_blocks * max_n_conn
    )
    bond_pairs[:, max_n_blocks * max_n_conn :, 0] = (
        torch.arange(max_n_blocks, dtype=torch.int32, device=pose_stack.device)
        .repeat_interleave(
            torch.full(
                (max_n_blocks,), max_n_conn, dtype=torch.int64, device=pose_stack.device
            )
        )
        .repeat((n_poses, 1))
    )

    # and select out the set of real bonds
    is_real_bond = torch.all(bond_pairs != -1, dim=2)
    nz_real_bond_pose, _ = torch.nonzero(is_real_bond, as_tuple=True)

    # and now let's increment the residue indices by pose source
    bond_pairs = nz_real_bond_pose.unsqueeze(dim=1) * max_n_blocks + (
        bond_pairs[is_real_bond]
    )
    bond_pairs = bond_pairs.cpu().numpy()

    csr_bond_pairs = scipy.sparse.csr_matrix(
        (
            numpy.ones((bond_pairs.shape[0],), dtype=numpy.int32),
            (bond_pairs[:, 0], bond_pairs[:, 1]),
        ),
        shape=(n_poses * max_n_blocks, n_poses * max_n_blocks),
    )

    n_components, labels = scipy.sparse.csgraph.connected_components(
        csr_bond_pairs, directed=False, return_labels=True
    )

    labels = labels.reshape(n_poses, max_n_blocks)
    n_ccs = numpy.amax(labels, axis=1) + 1
    cc_offsets = exclusive_cumsum1d(n_ccs)
    labels = labels - cc_offsets.reshape(n_poses, 1)

    return labels
