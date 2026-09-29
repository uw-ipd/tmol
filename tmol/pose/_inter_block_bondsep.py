from typing import Sequence

import attr
import torch

from tmol.chemical import MAX_SIG_BOND_SEPARATION
from tmol.types import Tensor


@attr.s(auto_attribs=True, frozen=True)
class InterBlockBondsep:
    """Bond separations between the inter-block connections of nearby blocks.

    The dense table ``[pose, block1, block2, conn1, conn2]`` holds
    ``MAX_SIG_BOND_SEPARATION`` for almost every block pair, so only the pairs
    that some connection pair brings closer than that are stored. Row
    ``[pose, block1]`` lists those ``block2`` in ascending order, followed by at
    least one empty slot whose block is ``-1`` and whose separations are all the
    cap. Scanning a row up to the first slot that is ``block2`` or empty
    (``near_block_slot`` in ``tmol/score/common/count_pair.hh``) therefore
    finds the dense table's values for every block pair.

    Attributes:
        near_blocks: ``[pose, block1, slot, 2]`` int32 holding ``block2`` and
            the minimum separation over the pair's connections.
        bondsep: ``[pose, block1, slot, conn1, conn2]`` int8 separations between
            connection ``conn1`` of ``block1`` and ``conn2`` of ``block2``;
            connections a block type lacks hold the cap.
    """

    near_blocks: Tensor[torch.int32][:, :, :, 2]
    bondsep: Tensor[torch.int8][:, :, :, :, :]

    @property
    def n_poses(self) -> int:
        return self.near_blocks.shape[0]

    @property
    def max_n_blocks(self) -> int:
        return self.near_blocks.shape[1]

    @property
    def n_slots(self) -> int:
        return self.near_blocks.shape[2]

    @property
    def max_n_conn(self) -> int:
        return self.bondsep.shape[3]

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        """Shape of the equivalent dense table."""
        n, b, m = self.n_poses, self.max_n_blocks, self.max_n_conn
        return (n, b, b, m, m)

    @property
    def device(self) -> torch.device:
        return self.bondsep.device

    @classmethod
    def empty(
        cls,
        n_poses: int,
        max_n_blocks: int,
        max_n_conn: int,
        device: torch.device,
        n_slots: int = 1,
    ) -> "InterBlockBondsep":
        """Separations for blocks that are all at least the cap apart."""
        near_blocks = torch.full(
            (n_poses, max_n_blocks, n_slots, 2),
            MAX_SIG_BOND_SEPARATION,
            dtype=torch.int32,
            device=device,
        )
        near_blocks[..., 0] = -1
        bondsep = torch.full(
            (n_poses, max_n_blocks, n_slots, max_n_conn, max_n_conn),
            MAX_SIG_BOND_SEPARATION,
            dtype=torch.int8,
            device=device,
        )
        return cls(near_blocks=near_blocks, bondsep=bondsep)

    @classmethod
    def from_entries(
        cls,
        pose: Tensor[torch.int64][:],
        block1: Tensor[torch.int64][:],
        block2: Tensor[torch.int64][:],
        conn1: Tensor[torch.int64][:],
        conn2: Tensor[torch.int64][:],
        separation: Tensor[torch.int32][:],
        n_poses: int,
        max_n_blocks: int,
        max_n_conn: int,
    ) -> "InterBlockBondsep":
        """Build from the dense-table entries that are below the cap.

        Each ``(pose, block1, block2, conn1, conn2)`` may appear at most once;
        every entry not listed holds the cap.
        """
        device = separation.device
        n_rows = n_poses * max_n_blocks
        row = pose * max_n_blocks + block1
        pair_key, pair_of_entry = torch.unique(
            row * max_n_blocks + block2, sorted=True, return_inverse=True
        )
        pair_row = torch.div(pair_key, max_n_blocks, rounding_mode="floor")
        row_len = torch.bincount(pair_row, minlength=n_rows)
        n_slots = int(row_len.max()) + 1 if n_rows else 1
        row_start = torch.cumsum(row_len, 0) - row_len
        pair_slot = (
            torch.arange(pair_key.shape[0], dtype=torch.int64, device=device)
            - row_start[pair_row]
        )

        result = cls.empty(n_poses, max_n_blocks, max_n_conn, device, n_slots)
        near_blocks = result.near_blocks.view(n_rows, n_slots, 2)
        bondsep = result.bondsep.view(n_rows, n_slots, max_n_conn, max_n_conn)
        near_blocks[pair_row, pair_slot, 0] = (pair_key % max_n_blocks).to(torch.int32)
        bondsep[pair_row[pair_of_entry], pair_slot[pair_of_entry], conn1, conn2] = (
            separation.to(torch.int8)
        )
        if max_n_conn > 0:
            near_blocks[..., 1] = torch.amin(bondsep, dim=(2, 3)).to(torch.int32)
        return result

    @classmethod
    def from_connectivity(
        cls,
        distances: Tensor[torch.int32][:, :, :],
        offsets: Tensor[torch.int64][:, :],
        counts: Tensor[torch.int32][:, :],
        max_n_conn: int,
    ) -> "InterBlockBondsep":
        """Map shortest paths between pose connections onto block pairs.

        Args:
            distances: ``[pose, pconn, pconn]`` shortest paths between the
                pose's connections.
            offsets: ``[pose, block]`` first pose-connection index of each block.
            counts: ``[pose, block]`` number of connections of each block.
            max_n_conn: Width of the connection axes.
        """
        n_poses, max_n_blocks = counts.shape
        n_nodes = distances.shape[1]
        device = distances.device

        block_counts = counts.flatten().to(torch.int64).clamp(0, max_n_conn)
        node_row = torch.repeat_interleave(
            torch.arange(block_counts.shape[0], dtype=torch.int64, device=device),
            block_counts,
        )
        row_start = torch.cumsum(block_counts, 0) - block_counts
        node_conn = (
            torch.arange(node_row.shape[0], dtype=torch.int64, device=device)
            - row_start[node_row]
        )
        node = offsets.flatten()[node_row] + node_conn
        node_pose = torch.div(node_row, max_n_blocks, rounding_mode="floor")
        in_range = (node >= 0) & (node < n_nodes)
        block_of_node = torch.full(
            (n_poses, n_nodes), -1, dtype=torch.int64, device=device
        )
        conn_of_node = torch.zeros_like(block_of_node)
        block_of_node[node_pose[in_range], node[in_range]] = (
            node_row[in_range] % max_n_blocks
        )
        conn_of_node[node_pose[in_range], node[in_range]] = node_conn[in_range]

        pose, node1, node2 = torch.nonzero(
            distances < MAX_SIG_BOND_SEPARATION, as_tuple=True
        )
        block1 = block_of_node[pose, node1]
        block2 = block_of_node[pose, node2]
        real = (block1 >= 0) & (block2 >= 0)
        pose, node1, node2 = pose[real], node1[real], node2[real]
        return cls.from_entries(
            pose,
            block1[real],
            block2[real],
            conn_of_node[pose, node1],
            conn_of_node[pose, node2],
            distances[pose, node1, node2],
            n_poses,
            max_n_blocks,
            max_n_conn,
        )

    @classmethod
    def from_bonded_graph(
        cls,
        counts: Tensor[torch.int32][:, :],
        intra_separation: Tensor[torch.int32][:, :, :, :],
        connections: Tensor[torch.int64][:, :, :, 2],
    ) -> "InterBlockBondsep":
        """Shortest paths below the cap over the graph of inter-block connections.

        Each block's connections are joined by edges of their intra-block
        separation, and bonded connections of two blocks by edges of one, which
        replace an intra-block edge between the same connections. Only paths
        shorter than the cap are explored, so the work grows with the number of
        bonded neighbours rather than with the square of the connection count.

        Args:
            counts: ``[pose, block]`` number of connections of each block.
            intra_separation: ``[pose, block, conn1, conn2]`` bond separation
                between the atoms of a block's connections.
            connections: ``[pose, block, conn, 2]`` block and connection that
                each connection bonds to, ``-1`` if none.
        """
        n_poses, max_n_blocks, max_n_conn = connections.shape[:3]
        device = connections.device
        block_counts = counts.flatten().to(torch.int64)
        node_start = torch.cumsum(block_counts, 0) - block_counts
        n_nodes = int(block_counts.sum())
        if n_nodes == 0:
            return cls.empty(n_poses, max_n_blocks, max_n_conn, device)

        conn = torch.arange(max_n_conn, dtype=torch.int64, device=device)
        real = conn < counts[..., None]
        intra = (
            real[..., :, None]
            & real[..., None, :]
            & (intra_separation < MAX_SIG_BOND_SEPARATION)
        )
        pose, block, conn1, conn2 = torch.nonzero(intra, as_tuple=True)
        first = node_start[pose * max_n_blocks + block]
        intra_key = (first + conn1) * n_nodes + first + conn2
        intra_weight = intra_separation[pose, block, conn1, conn2].to(torch.int32)
        pose, block, conn1 = torch.nonzero(connections[..., 0] != -1, as_tuple=True)
        partner = connections[pose, block, conn1]
        inter_key = (node_start[pose * max_n_blocks + block] + conn1) * n_nodes + (
            node_start[pose * max_n_blocks + partner[:, 0]] + partner[:, 1]
        )

        edge_key, edge_of_key = torch.unique(
            torch.cat((intra_key, inter_key)), sorted=True, return_inverse=True
        )
        edge_weight = torch.empty(edge_key.shape, dtype=torch.int32, device=device)
        edge_weight[edge_of_key[: intra_key.shape[0]]] = intra_weight
        edge_weight[edge_of_key[intra_key.shape[0] :]] = 1
        edge_src = torch.div(edge_key, n_nodes, rounding_mode="floor")
        edge_dst = edge_key % n_nodes
        out_degree = torch.bincount(edge_src, minlength=n_nodes)
        out_start = torch.cumsum(out_degree, 0) - out_degree

        # extend every path improved in the last round by one edge, until none improves
        best_key, best_sep = edge_key, edge_weight
        front_src, front_node, front_sep = edge_src, edge_dst, edge_weight
        while front_src.shape[0] > 0:
            n_out = out_degree[front_node]
            path = torch.repeat_interleave(
                torch.arange(front_src.shape[0], device=device), n_out
            )
            path_start = torch.cumsum(n_out, 0) - n_out
            edge = out_start[front_node][path] + (
                torch.arange(path.shape[0], device=device) - path_start[path]
            )
            sep = front_sep[path] + edge_weight[edge]
            short = sep < MAX_SIG_BOND_SEPARATION
            cand_key = front_src[path][short] * n_nodes + edge_dst[edge][short]

            merged_key, merged_of = torch.unique(
                torch.cat((best_key, cand_key)), sorted=True, return_inverse=True
            )
            merged_sep = torch.full(
                merged_key.shape,
                MAX_SIG_BOND_SEPARATION,
                dtype=torch.int32,
                device=device,
            ).scatter_reduce(0, merged_of, torch.cat((best_sep, sep[short])), "amin")
            previous = torch.full_like(merged_sep, MAX_SIG_BOND_SEPARATION)
            previous[merged_of[: best_key.shape[0]]] = best_sep
            improved = merged_sep < previous
            best_key, best_sep = merged_key, merged_sep
            front_src = torch.div(merged_key[improved], n_nodes, rounding_mode="floor")
            front_node = merged_key[improved] % n_nodes
            front_sep = merged_sep[improved]

        node_row = torch.repeat_interleave(
            torch.arange(block_counts.shape[0], device=device), block_counts
        )
        node_conn = torch.arange(n_nodes, device=device) - node_start[node_row]
        src = torch.div(best_key, n_nodes, rounding_mode="floor")
        dst = best_key % n_nodes
        return cls.from_entries(
            torch.div(node_row[src], max_n_blocks, rounding_mode="floor"),
            node_row[src] % max_n_blocks,
            node_row[dst] % max_n_blocks,
            node_conn[src],
            node_conn[dst],
            best_sep,
            n_poses,
            max_n_blocks,
            max_n_conn,
        )

    @classmethod
    def concatenate(
        cls,
        parts: Sequence["InterBlockBondsep"],
        max_n_blocks: int,
        max_n_conn: int,
    ) -> "InterBlockBondsep":
        """Stack the poses of several tables, padding blocks and connections."""
        result = cls.empty(
            sum(part.n_poses for part in parts),
            max_n_blocks,
            max_n_conn,
            parts[0].device,
            max(part.n_slots for part in parts),
        )
        offset = 0
        for part in parts:
            n, b, k, m = part.n_poses, part.max_n_blocks, part.n_slots, part.max_n_conn
            result.near_blocks[offset : offset + n, :b, :k] = part.near_blocks
            result.bondsep[offset : offset + n, :b, :k, :m, :m] = part.bondsep
            offset += n
        return result

    def to_dense(self) -> Tensor[torch.int8][:, :, :, :, :]:
        """Materialize the ``[pose, block1, block2, conn1, conn2]`` table."""
        dense = torch.full(
            self.shape, MAX_SIG_BOND_SEPARATION, dtype=torch.int8, device=self.device
        )
        pose, block1, slot = torch.nonzero(self.near_blocks[..., 0] >= 0, as_tuple=True)
        block2 = self.near_blocks[pose, block1, slot, 0].to(torch.int64)
        dense[pose, block1, block2] = self.bondsep[pose, block1, slot]
        return dense

    def select_poses(self, start: int, stop: int) -> "InterBlockBondsep":
        """View of poses ``start`` to ``stop``."""
        return InterBlockBondsep(
            near_blocks=self.near_blocks[start:stop],
            bondsep=self.bondsep[start:stop],
        )

    def clone(self) -> "InterBlockBondsep":
        return InterBlockBondsep(
            near_blocks=self.near_blocks.clone(), bondsep=self.bondsep.clone()
        )

    def to(self, device: torch.device) -> "InterBlockBondsep":
        return InterBlockBondsep(
            near_blocks=self.near_blocks.to(device), bondsep=self.bondsep.to(device)
        )
