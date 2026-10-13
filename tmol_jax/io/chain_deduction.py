"""Chain deduction for ``PoseStack`` -- pure numpy port of ``tmol.io.chain_deduction``.

Upstream reference: ``tmol/io/chain_deduction.py``. The torch port walks
the inter-residue connection graph, marks valid "chain-building"
connections (up/down for polymers; all connections for small molecules),
and finds connected components via ``scipy.sparse.csgraph``. This JAX
port performs exactly the same algorithm with numpy arrays -- chain
identification is a discrete labelling task so there is nothing to JIT.

The primary API is :func:`chain_inds_for_pose_stack` (upstream name)
which returns an ``int64[n_poses, max_n_blocks]`` labelling. A more
convenient :func:`deduce_chains` wrapper returns ``[[start, end], ...]``
ranges per pose as called out in the task spec.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

# --------------------------------------------------------------------------- #
# PackedBlockTypes annotation                                                  #
# --------------------------------------------------------------------------- #


def annotate_pbt_w_valid_connection_masks(pbt: Any) -> np.ndarray:
    """Compute the ``(n_types, max_n_conn)`` bool mask of which
    connections count as "within-chain" for chain-detection.

    Polymeric residues only expose their ``up`` and ``down`` connections
    (so a disulfide does not make two cysteines "the same chain").
    Non-polymeric residues expose all their connections.

    The mask is cached as ``pbt.connection_mask_for_chain_detection`` on
    first computation -- matching the upstream pattern -- and returned as
    a numpy array for downstream use.
    """

    cached = getattr(pbt, "connection_mask_for_chain_detection", None)
    if cached is not None:
        return np.asarray(cached)

    n_types = int(getattr(pbt, "n_types", 0) or 0)
    max_n_conn = int(getattr(pbt, "max_n_conn", 0) or 0)
    mask = np.zeros((n_types, max_n_conn), dtype=bool)
    active = getattr(pbt, "active_block_types", ())
    for i, bt in enumerate(active):
        is_polymer = False
        try:
            is_polymer = bool(bt.properties.polymer.is_polymer)
        except AttributeError:
            is_polymer = False
        if is_polymer:
            up_ind = int(getattr(bt, "up_connection_ind", -1))
            down_ind = int(getattr(bt, "down_connection_ind", -1))
            if up_ind >= 0 and up_ind < max_n_conn:
                mask[i, up_ind] = True
            if down_ind >= 0 and down_ind < max_n_conn:
                mask[i, down_ind] = True
        else:
            n = min(len(getattr(bt, "connections", ()) or ()), max_n_conn)
            mask[i, :n] = True

    try:
        # Attach to the pbt so the next call is O(1). Best-effort:
        # frozen/slotted dataclasses may reject this and that's fine.
        object.__setattr__(pbt, "connection_mask_for_chain_detection", mask)
    except Exception:
        pass
    return mask


# --------------------------------------------------------------------------- #
# Chain labelling                                                              #
# --------------------------------------------------------------------------- #


def chain_inds_for_pose_stack(pose_stack: Any) -> np.ndarray:
    """Label each residue by chain, returning ``int64[n_poses, max_n_blocks]``.

    Residues sharing a "chain-building" bond (see
    :func:`annotate_pbt_w_valid_connection_masks`) receive the same
    chain index. Padding residues (``block_type_ind < 0``) are labelled
    ``-1``.
    """

    pbt = pose_stack.packed_block_types
    mask = annotate_pbt_w_valid_connection_masks(pbt)

    bti = np.asarray(pose_stack.block_type_ind)
    n_poses, max_n_blocks = bti.shape
    max_n_conn = int(getattr(pbt, "max_n_conn", 0) or 0)

    irc = np.asarray(pose_stack.inter_residue_connections)
    # irc shape: (n_poses, max_n_blocks, max_n_conn, 2) -- last dim = (other_block, other_conn)

    real_blocks = bti != -1

    # For each (p, b, c) we need the target block index of the connection,
    # gated by is_valid_conn (polymer up/down or non-polymer wildcard).
    is_valid_conn = np.zeros((n_poses, max_n_blocks, max_n_conn), dtype=bool)
    if np.any(real_blocks):
        safe_bti = np.where(real_blocks, bti, 0)
        is_valid_conn = np.where(
            real_blocks[..., None],
            mask[safe_bti],
            False,
        )

    # Build an edge list (pose*max_n_blocks offsets).
    if max_n_conn == 0 or max_n_blocks == 0:
        labels = np.full((n_poses, max_n_blocks), -1, dtype=np.int64)
        return labels

    conn_target_block = irc[..., 0]  # [P, B, C]

    # Compose edges: forward (src -> tgt) and backward (tgt -> src).
    pose_ind_grid = np.broadcast_to(
        np.arange(n_poses, dtype=np.int64)[:, None, None],
        (n_poses, max_n_blocks, max_n_conn),
    )
    res_ind_grid = np.broadcast_to(
        np.arange(max_n_blocks, dtype=np.int64)[None, :, None],
        (n_poses, max_n_blocks, max_n_conn),
    )
    src_flat_all = pose_ind_grid * max_n_blocks + res_ind_grid
    tgt_flat_all = pose_ind_grid * max_n_blocks + conn_target_block.astype(np.int64)

    is_edge = is_valid_conn & (conn_target_block != -1)
    src_list = src_flat_all[is_edge]
    tgt_list = tgt_flat_all[is_edge]

    # Add the reverse direction for an undirected graph.
    row_idx = np.concatenate([src_list, tgt_list])
    col_idx = np.concatenate([tgt_list, src_list])
    n_nodes = n_poses * max_n_blocks

    if row_idx.size == 0:
        # No bonds at all -- every residue is its own chain.
        labels = np.arange(n_nodes, dtype=np.int64).reshape(n_poses, max_n_blocks)
    else:
        data = np.ones(row_idx.size, dtype=np.int8)
        csr = scipy.sparse.csr_matrix(
            (data, (row_idx, col_idx)), shape=(n_nodes, n_nodes)
        )
        _, components = scipy.sparse.csgraph.connected_components(
            csr, directed=False, return_labels=True
        )
        labels = components.astype(np.int64).reshape(n_poses, max_n_blocks)

    # Re-number per-pose so the first chain of each pose is 0 (not a
    # globally-growing counter). This matches upstream's behaviour of
    # subtracting the min label per pose.
    for p in range(n_poses):
        row = labels[p]
        unique_real = np.unique(row[real_blocks[p]])
        remap = {lab: i for i, lab in enumerate(unique_real.tolist())}
        new_row = np.full_like(row, -1)
        for b in range(max_n_blocks):
            if real_blocks[p, b]:
                new_row[b] = remap[row[b]]
        labels[p] = new_row

    return labels


# --------------------------------------------------------------------------- #
# Public helper requested in the task spec                                     #
# --------------------------------------------------------------------------- #


def deduce_chains(
    pose_stack: Any,
) -> List[List[Tuple[int, int]]]:
    """Identify chain-break spans using inter-residue connections.

    Returns ``[[(start, end), ...], ...]`` -- a list of length
    ``n_poses``; for each pose, a list of ``(start, end)`` inclusive
    block-index ranges describing a contiguous run of residues that
    share a chain. Chains are emitted in residue-order (i.e. the order
    they appear along the block axis).

    "Chain break" here means the same algorithm as
    :func:`chain_inds_for_pose_stack`: any pair of consecutive blocks
    whose chain indices disagree kicks off a new chain.
    """

    labels = chain_inds_for_pose_stack(pose_stack)
    n_poses, max_n_blocks = labels.shape

    all_chains: List[List[Tuple[int, int]]] = []
    for p in range(n_poses):
        row = labels[p]
        chains_p: List[Tuple[int, int]] = []
        start = None
        current = None
        for b in range(max_n_blocks):
            lab = int(row[b])
            if lab < 0:
                if start is not None:
                    chains_p.append((start, b - 1))
                    start = None
                    current = None
                continue
            if start is None:
                start = b
                current = lab
            elif lab != current:
                chains_p.append((start, b - 1))
                start = b
                current = lab
        if start is not None:
            chains_p.append((start, max_n_blocks - 1))
        all_chains.append(chains_p)

    return all_chains


__all__ = [
    "annotate_pbt_w_valid_connection_masks",
    "chain_inds_for_pose_stack",
    "deduce_chains",
]
