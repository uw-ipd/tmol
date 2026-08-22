"""Utilities for installing explicit covalent bonds between pose blocks."""

from dataclasses import dataclass

import attr
import torch

from tmol.pose._pose_stack_builder import PoseStackBuilder


@dataclass(frozen=True)
class InterResidueConnection:
    """A reciprocal bond between named connection sites on two blocks."""

    pose_index: int
    block1: int
    connection1: str
    block2: int
    connection2: str


def connect_pose_blocks(pose_stack, connections):
    """Install explicit inter-residue bonds and rebuild bonded distances.

    Both block types must already expose the named connection sites.  This
    keeps chemistry declaration (residue types and variants) separate from
    topology declaration (which residues are actually bonded), as in Rosetta.
    The input pose is not modified.
    """

    connections = tuple(connections)
    if not connections:
        return pose_stack

    pbt = pose_stack.packed_block_types
    inter64 = pose_stack.inter_residue_connections64.clone()

    for bond in connections:
        if not 0 <= bond.pose_index < len(pose_stack):
            raise IndexError(f"pose index {bond.pose_index} is out of range")
        if bond.block1 == bond.block2:
            raise ValueError("an inter-residue connection requires two blocks")

        endpoints = (
            (bond.block1, bond.connection1),
            (bond.block2, bond.connection2),
        )
        resolved = []
        for block, connection in endpoints:
            if not 0 <= block < pose_stack.max_n_blocks:
                raise IndexError(f"block index {block} is out of range")
            bt_ind = int(pose_stack.block_type_ind64[bond.pose_index, block].item())
            if bt_ind < 0:
                raise ValueError(f"pose {bond.pose_index} block {block} is empty")
            bt = pbt.active_block_types[bt_ind]
            try:
                conn_ind = int(bt.connection_to_cidx[connection])
            except KeyError as exc:
                raise ValueError(
                    f"block {block} ({bt.name}) has no connection named "
                    f"'{connection}'"
                ) from exc
            if torch.any(inter64[bond.pose_index, block, conn_ind] != -1):
                raise ValueError(f"connection {block}:{connection} is already occupied")
            resolved.append((block, conn_ind, bt))

        (block1, conn1, bt1), (block2, conn2, bt2) = resolved
        type1 = bt1.connections[conn1].type
        type2 = bt2.connections[conn2].type
        if type1 != type2:
            raise ValueError(
                f"connection bond types disagree: {block1}:{bond.connection1} "
                f"is {type1}, {block2}:{bond.connection2} is {type2}"
            )
        inter64[bond.pose_index, block1, conn1] = torch.tensor(
            (block2, conn2), dtype=torch.int64, device=pose_stack.device
        )
        inter64[bond.pose_index, block2, conn2] = torch.tensor(
            (block1, conn1), dtype=torch.int64, device=pose_stack.device
        )

    real_res = pose_stack.block_type_ind64 >= 0
    pconn, offsets, block_n_conn, pose_n_pconn = (
        PoseStackBuilder._take_real_conn_conn_intrablock_pairs(
            pbt, pose_stack.block_type_ind64, real_res
        )
    )
    PoseStackBuilder._incorporate_inter_residue_connections_into_connectivity_graph(
        inter64, offsets, pconn
    )
    bondsep64 = PoseStackBuilder._calculate_interblock_bondsep_from_connectivity_graph(
        pbt, block_n_conn, pose_n_pconn, pconn
    )
    return attr.evolve(
        pose_stack,
        coords=pose_stack.coords.clone(),
        inter_residue_connections=inter64.to(torch.int32),
        inter_residue_connections64=inter64,
        inter_block_bondsep=bondsep64.to(torch.int32),
        inter_block_bondsep64=bondsep64,
    )
