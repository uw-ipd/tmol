import numpy
import torch
import scipy
from tmol.pose.pose_stack import PoseStack
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.types.functional import validate_args
from tmol.types.array import NDArray
from tmol.utility.ndarray.common_operations import exclusive_cumsum1d


@validate_args
def chain_inds_for_pose_stack(
    pose_stack: PoseStack,
) -> NDArray[numpy.int64][:, :]:
    """Label each residue by which chain it comes from, where "chain" is a group
    of polymer residues that are connected by certain sets of bonds (e.g. not
    disulfide bonds). This problem becomes one of finding the connected components
    of a graph and is handled using scipy's (CPU) code. Gap residues are given a
    chain ID of -1.
    """
    pbt = pose_stack.packed_block_types
    annotate_pbt_w_valid_connection_masks(pbt)

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
    is_valid_conn = torch.full(
        (n_poses, max_n_blocks, max_n_conn),
        -1,
        dtype=torch.bool,
        device=pose_stack.device,
    )
    real_blocks = pose_stack.block_type_ind != -1
    unreal_blocks = (
        torch.logical_not(real_blocks).cpu().numpy()
    )  # save for later when on CPU
    is_valid_conn[real_blocks] = pbt.connection_mask_for_chain_detection[
        pose_stack.block_type_ind64[real_blocks]
    ]

    conn_targets = torch.full(
        (n_poses, max_n_blocks, max_n_conn),
        -1,
        dtype=torch.int32,
        device=pose_stack.device,
    )
    conn_targets[is_valid_conn] = pose_stack.inter_residue_connections[:, :, :, 0][
        is_valid_conn
    ]
    conn_targets = conn_targets.view(n_poses, max_n_blocks * max_n_conn)
    res_ind_arange = (
        torch.arange(max_n_blocks, dtype=torch.int32, device=pose_stack.device)
        .repeat_interleave(
            torch.full(
                (max_n_blocks,), max_n_conn, dtype=torch.int64, device=pose_stack.device
            )
        )
        .repeat((n_poses, 1))
    )

    bond_pairs[:, : max_n_blocks * max_n_conn, 0] = res_ind_arange
    bond_pairs[:, : max_n_blocks * max_n_conn, 1] = conn_targets

    # 2. and backward
    bond_pairs[:, max_n_blocks * max_n_conn :, 0] = conn_targets
    bond_pairs[:, max_n_blocks * max_n_conn :, 1] = res_ind_arange

    # and select out the set of real bonds
    is_real_bond = torch.all(bond_pairs != -1, dim=2)
    nz_real_bond_pose, _ = torch.nonzero(is_real_bond, as_tuple=True)

    # and now let's increment the residue indices by pose source
    bond_pairs = nz_real_bond_pose.unsqueeze(dim=1) * max_n_blocks + (
        bond_pairs[is_real_bond]
    )

    # Caution: here forward, we will be on the CPU!
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
    # print("n_components", n_components)
    # print("labels", labels)

    labels = labels.reshape(n_poses, max_n_blocks)
    n_ccs = numpy.amax(labels, axis=1) - numpy.amin(labels, axis=1) + 1
    # print("n_ccs", n_ccs)
    cc_offsets = exclusive_cumsum1d(n_ccs)
    # print("cc_offsets", cc_offsets)
    labels = labels - cc_offsets.reshape(n_poses, 1)
    # print("labels", labels)
    # now re-label the gap residues with a chain ind of -1
    labels[unreal_blocks] = -1

    return labels


def annotate_pbt_w_valid_connection_masks(pbt: PackedBlockTypes):
    """We want to take the up-down polymeric connections between residues
    that have up-down connections and not other connections, unless
    otherwise instructed.

    The logic here is to take the up- and down-connections from
    polymeric residues as the ones that connect two residues part
    of the same chain. This would make the C->N connection along
    a protein backbone serve to say residues i and i+1 are part
    of the same chain without saying that a disulfide bond
    between residues i and j make them part of the same chain.
    (They are at that point a single molecule, but, conceptually
    still separate chains.)

    For non-polymeric residues, all their chemical bonds should
    be considered as connecting them to members of their same chain.

    The upshot is: if a polymeric residue is connected to a
    non-polymeric residue through one of its non-up/non-down
    connection points, the non-polymeric residue will still be
    considered part of the polymeric residue's chain. Either
    connection direction is sufficient to link two residues
    as part of the same chain.
    """
    if hasattr(pbt, "connection_mask_for_chain_detection"):
        return

    connection_masks = torch.zeros((pbt.n_types, pbt.max_n_conn), dtype=torch.bool)
    for i, bt in enumerate(pbt.active_block_types):
        if bt.properties.polymer.is_polymer:
            # for polymeric residues: only their up/down connections are
            # automatically considered part of chain connection identification
            if bt.up_connection_ind >= 0:
                connection_masks[i, bt.up_connection_ind] = True
            if bt.down_connection_ind >= 0:
                connection_masks[i, bt.down_connection_ind] = True
        else:
            # for non-polymeric residues, all their connecitons are
            # automatically
            connection_masks[i, : len(bt.connections)] = True

    setattr(pbt, "connection_mask_for_chain_detection", connection_masks.to(pbt.device))
