"""Pack protein coordinates and construct exact connection-graph distances."""

import jax.numpy as jnp
import numpy as np

from tmol_jax.chemical.constants import MAX_SIG_BOND_SEPARATION
from .pdb_info import PDBInfo
from .pose_stack import PoseStack


def _connect(connections, pose, left, right, left_conn, right_conn):
    if left_conn < 0 or right_conn < 0:
        raise ValueError("requested bond has no matching residue connection")
    if np.any(connections[pose, left, left_conn] >= 0) or np.any(
        connections[pose, right, right_conn] >= 0
    ):
        raise ValueError("a residue connection cannot carry two bonds")
    connections[pose, left, left_conn] = right, right_conn
    connections[pose, right, right_conn] = left, left_conn


def _connections(pbt, block_types, chains, disulfides):
    poses, blocks = block_types.shape
    connections = np.full((poses, blocks, max(pbt.max_n_conn, 2), 2), -1, np.int32)
    for pose, row in enumerate(block_types):
        for left in range(len(chains[pose]) - 1):
            right = left + 1
            if (
                row[left] < 0
                or row[right] < 0
                or chains[pose][left] != chains[pose][right]
            ):
                continue
            left_type, right_type = (
                pbt.active_block_types[row[i]] for i in (left, right)
            )
            up, down = left_type.up_connection_ind, right_type.down_connection_ind
            if up >= 0 and down >= 0:
                _connect(connections, pose, left, right, up, down)
        for left, right in disulfides[pose]:
            if not (0 <= left < blocks and 0 <= right < blocks) or left == right:
                raise ValueError("invalid disulfide residue pair")
            if row[left] < 0 or row[right] < 0:
                raise ValueError("disulfide endpoints cannot be padding")
            endpoints = [
                pbt.active_block_types[row[i]].connection_to_cidx.get("dslf", -1)
                for i in (left, right)
            ]
            _connect(connections, pose, left, right, *endpoints)
    return connections


def _bond_separation(pbt, block_types, connections):
    poses, blocks, slots, _ = connections.shape
    cutoff = MAX_SIG_BOND_SEPARATION
    result = np.full((poses, blocks, blocks, slots, slots), cutoff, np.int32)
    paths = np.asarray(pbt.intrares_path_distance)
    atoms = np.asarray(pbt.conn_atoms)
    counts = np.asarray(pbt.n_conn)
    for pose, row in enumerate(block_types):
        for block, kind in enumerate(row):
            if kind < 0:
                continue
            n = int(counts[kind])
            endpoints = atoms[kind, :n]
            distances = paths[kind][np.ix_(endpoints, endpoints)]
            result[pose, block, block, :n, :n] = np.where(
                distances < 0, cutoff, np.minimum(distances, cutoff)
            )
        for block, slot in zip(*np.nonzero(connections[pose, :, :, 0] >= 0)):
            partner, partner_slot = connections[pose, block, slot]
            result[pose, block, partner, slot, partner_slot] = 1
        flat = result[pose].transpose(0, 2, 1, 3).reshape(blocks * slots, -1).copy()
        for node in np.flatnonzero(flat.diagonal() == 0):
            np.minimum(flat, flat[:, node, None] + flat[None, node, :], out=flat)
        result[pose] = flat.reshape(blocks, slots, blocks, slots).transpose(0, 2, 1, 3)
    return result


def _validate_rows(block_types, coordinates, chains, numbers, names, disulfides):
    poses = len(block_types)
    if not poses or any(
        len(rows) != poses for rows in (coordinates, chains, numbers, names, disulfides)
    ):
        raise ValueError("pose inputs must have the same nonzero batch length")
    for types, labels, nums, residue_names in zip(block_types, chains, numbers, names):
        if any(len(row) != len(types) for row in (labels, nums, residue_names)):
            raise ValueError("residue metadata must match each pose's block count")
    if not any(len(row) for row in block_types):
        raise ValueError("pose batch contains no residues")


class PoseStackBuilder:
    @classmethod
    def from_block_types_and_coords(
        cls,
        pbt,
        block_type_inds_per_pose,
        coords_per_pose,
        chain_id_per_pose,
        res_num_per_pose,
        res_name_per_pose=None,
        disulfide_pairs_per_pose=None,
    ):
        """Pack aligned per-pose arrays; negative block types denote padding."""
        rows = [np.asarray(row, dtype=np.int32) for row in block_type_inds_per_pose]
        if res_name_per_pose is None:
            res_name_per_pose = [
                [
                    pbt.active_block_types[kind].name3 if kind >= 0 else "UNK"
                    for kind in row
                ]
                for row in rows
            ]
        disulfides = (
            [()] * len(rows)
            if disulfide_pairs_per_pose is None
            else disulfide_pairs_per_pose
        )
        _validate_rows(
            rows,
            coords_per_pose,
            chain_id_per_pose,
            res_num_per_pose,
            res_name_per_pose,
            disulfides,
        )
        if any(np.any(row >= pbt.n_types) for row in rows):
            raise ValueError("block type index is outside the residue palette")
        block_types = np.full((len(rows), max(map(len, rows))), -1, np.int32)
        for pose, row in enumerate(rows):
            block_types[pose, : len(row)] = row
        counts = np.asarray(pbt.n_atoms)[np.maximum(block_types, 0)]
        counts = np.where(block_types >= 0, counts, 0)
        cumulative = np.cumsum(counts, axis=1, dtype=np.int32)
        offsets = cumulative - counts
        totals = cumulative[:, -1]
        coords = np.zeros((len(rows), int(totals.max()), 3), np.float32)
        for pose, source in enumerate(coords_per_pose):
            source = np.asarray(source, dtype=np.float32)
            if source.shape != (totals[pose], 3):
                raise ValueError(
                    "coordinate shape does not match the residue atom counts"
                )
            coords[pose, : totals[pose]] = source
        connections = _connections(pbt, block_types, chain_id_per_pose, disulfides)
        distances = _bond_separation(pbt, block_types, connections)
        numbers = np.full(block_types.shape, -1, np.int32)
        for pose, row in enumerate(res_num_per_pose):
            numbers[pose, : len(row)] = row
        records = [
            [["ATOM"] * int(counts[p, b]) for b in range(len(rows[p]))]
            for p in range(len(rows))
        ]
        metadata = PDBInfo(
            chain_id=[[str(label) for label in row] for row in chain_id_per_pose],
            res_num=numbers,
            res_name=[list(row) for row in res_name_per_pose],
            atom_record_type=records,
        )
        return PoseStack(
            packed_block_types=pbt,
            coords=jnp.asarray(coords),
            block_coord_offset=jnp.asarray(offsets),
            inter_residue_connections=jnp.asarray(connections),
            inter_block_bondsep=jnp.asarray(distances),
            block_type_ind=jnp.asarray(block_types),
            pdb_info=metadata,
        )
