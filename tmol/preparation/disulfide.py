"""Prepare declared disulfide topology without importing a tensor backend.

The input offsets address a single flattened coordinate slab.  An adapter
therefore accounts for its backend's pose stride before calling this function,
then uploads the completed integer and parameter arrays exactly once.
Coordinates are neither accepted nor inspected here: disulfide edges come only
from reciprocal, explicitly declared inter-block connections.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

DISULFIDE_TOPOLOGY_SCHEMA_VERSION: Final = 1

DISULFIDE_PARAMETER_FIELDS: Final = (
    "d_location",
    "d_scale",
    "d_shape",
    "a_logA",
    "a_kappa",
    "a_mu",
    "dss_logA1",
    "dss_kappa1",
    "dss_mu1",
    "dss_logA2",
    "dss_kappa2",
    "dss_mu2",
    "dcs_logA1",
    "dcs_mu1",
    "dcs_kappa1",
    "dcs_logA2",
    "dcs_mu2",
    "dcs_kappa2",
    "dcs_logA3",
    "dcs_mu3",
    "dcs_kappa3",
    "wt_dih_ss",
    "wt_dih_cs",
    "wt_ang",
    "wt_len",
    "shift",
    "dss_mixed_logA1",
    "dss_mixed_kappa1",
    "dss_mixed_mu1",
    "dss_mixed_logA2",
    "dss_mixed_kappa2",
    "dss_mixed_mu2",
)

_MIRRORED_PARAMETER_COLUMNS: Final = np.asarray([8, 11, 13, 16, 19, 32], dtype=np.intp)


def _immutable(value: object, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.shape != shape:
        raise ValueError(f"expected shape {shape}; got {array.shape}")
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(), dtype=dtype).reshape(shape)


def _integer_array(value: object, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must have an integer dtype")
    return array


@dataclass(frozen=True, slots=True)
class DisulfideTopologyPlan:
    """Immutable, versioned arrays consumed by disulfide backend adapters.

    ``atom_indices_and_parameter_rows`` is int32 ``[n_disulfides, 8]``.
    Its columns are ``CA1, CB1, S1, S2, CB2, CA2, params1, params2``.
    ``parameter_rows`` is float32 ``[n_block_types, 33]``; its last column
    records chirality and the other columns follow
    :data:`DISULFIDE_PARAMETER_FIELDS`.
    """

    schema_version: int
    atom_indices_and_parameter_rows: np.ndarray
    parameter_rows: np.ndarray

    def __post_init__(self) -> None:
        if self.schema_version != DISULFIDE_TOPOLOGY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported disulfide topology schema: {self.schema_version}"
            )
        pairs = np.asarray(self.atom_indices_and_parameter_rows)
        params = np.asarray(self.parameter_rows)
        if pairs.ndim != 2 or pairs.shape[1] != 8:
            raise ValueError(f"pair rows must have shape [n, 8]; got {pairs.shape}")
        if params.ndim != 2 or params.shape[1] != 33:
            raise ValueError(
                f"parameter rows must have shape [n_types, 33]; got {params.shape}"
            )
        object.__setattr__(
            self,
            "atom_indices_and_parameter_rows",
            _immutable(pairs, np.dtype(np.int32), pairs.shape),
        )
        object.__setattr__(
            self,
            "parameter_rows",
            _immutable(params, np.dtype(np.float32), params.shape),
        )


def _prepare_parameter_rows(
    chirality: Sequence[object], global_parameters: Mapping[str, object]
) -> np.ndarray:
    chirality = np.asarray(chirality)
    if chirality.ndim != 1:
        raise ValueError(f"chirality must have shape [n_types]; got {chirality.shape}")
    try:
        values = np.asarray(
            [float(global_parameters[name]) for name in DISULFIDE_PARAMETER_FIELDS]
            + [1.0],
            dtype=np.float32,
        )
    except KeyError as error:
        raise ValueError(
            f"missing global disulfide parameter {error.args[0]!r}"
        ) from error
    if not np.isfinite(values).all():
        raise ValueError("global disulfide parameters must be finite scalars")
    rows = np.tile(values, (chirality.shape[0], 1))
    mirrored = np.asarray(chirality == "d", dtype=bool)
    rows[np.ix_(mirrored, _MIRRORED_PARAMETER_COLUMNS)] *= -1
    return rows


def _validate_declared_connections(connections: np.ndarray, real: np.ndarray) -> None:
    n_poses, n_blocks, n_connections = connections.shape[:3]
    for pose in range(n_poses):
        for block in range(n_blocks):
            for connection in range(n_connections):
                other, other_connection = map(int, connections[pose, block, connection])
                if other == -1 and other_connection == -1:
                    continue
                if other < 0 or other_connection < 0:
                    raise ValueError(
                        "connection endpoints must both be -1 or non-negative"
                    )
                if not real[pose, block]:
                    raise ValueError("padded blocks may not declare connections")
                if other >= n_blocks or other_connection >= n_connections:
                    raise ValueError("connection endpoint is out of bounds")
                if other == block:
                    raise ValueError(
                        "inter-block connections may not target the same block"
                    )
                if not real[pose, other]:
                    raise ValueError("connection endpoint targets a padded block")
                reciprocal = connections[pose, other, other_connection]
                if tuple(map(int, reciprocal)) != (block, connection):
                    raise ValueError(
                        "connection endpoints must be reciprocal: "
                        f"pose {pose}, ({block}, {connection}) -> "
                        f"({other}, {other_connection})"
                    )


def _collect_pair_rows(
    block_types: np.ndarray,
    offsets: np.ndarray,
    connections: np.ndarray,
    enabled: np.ndarray,
    downstream: np.ndarray,
) -> list[list[int]]:
    rows: list[list[int]] = []
    visited: set[tuple[int, int, int, int, int]] = set()
    int32 = np.iinfo(np.int32)
    for pose, pose_block_types in enumerate(block_types):
        for block, block_type_value in enumerate(pose_block_types):
            block_type = int(block_type_value)
            if block_type < 0:
                continue
            for connection in np.flatnonzero(enabled[block_type]):
                other, other_connection = map(int, connections[pose, block, connection])
                if other < 0:
                    continue
                other_type = int(block_types[pose, other])
                if not enabled[other_type, other_connection]:
                    continue
                endpoint = (block, int(connection))
                partner = (other, other_connection)
                edge = (pose, *sorted((endpoint, partner)))
                edge_key = (edge[0], *edge[1], *edge[2])
                if edge_key in visited:
                    continue
                visited.add(edge_key)

                left_local = downstream[block_type, connection, :3]
                right_local = downstream[other_type, other_connection, :3]
                if np.any(left_local < 0) or np.any(right_local < 0):
                    raise ValueError(
                        "a scored disulfide has an invalid downstream atom"
                    )
                left = int(offsets[pose, block]) + left_local[::-1]
                right = int(offsets[pose, other]) + right_local
                atoms = np.concatenate((left, right))
                if np.any(atoms > int32.max):
                    raise ValueError("flattened disulfide atom index exceeds int32")
                rows.append([*(int(atom) for atom in atoms), block_type, other_type])
    return rows


def prepare_disulfide_topology(
    block_types: np.ndarray,
    block_coord_offsets: np.ndarray,
    inter_block_connections: np.ndarray,
    disulfide_enabled_connections: np.ndarray,
    first_atoms_downstream_of_connection: np.ndarray,
    block_type_chirality: Sequence[object],
    global_parameters: Mapping[str, object],
) -> DisulfideTopologyPlan:
    """Prepare scoring rows from declared, reciprocal topology edges.

    ``block_coord_offsets`` must already index a flattened coordinate slab. For
    a backend storing coordinates as ``[pose, max_atoms, 3]``, adapters add
    ``pose * max_atoms`` before calling this function.
    """

    block_types = _integer_array(block_types, "block_types")
    offsets = _integer_array(block_coord_offsets, "block_coord_offsets")
    connections = _integer_array(inter_block_connections, "inter_block_connections")
    downstream = _integer_array(
        first_atoms_downstream_of_connection,
        "first_atoms_downstream_of_connection",
    )
    enabled = np.asarray(disulfide_enabled_connections, dtype=bool)

    if block_types.ndim != 2:
        raise ValueError(
            f"block_types must have shape [pose, block]; got {block_types.shape}"
        )
    if offsets.shape != block_types.shape:
        raise ValueError(
            f"block_coord_offsets must have shape {block_types.shape}; got {offsets.shape}"
        )
    if (
        connections.ndim != 4
        or connections.shape[:2] != block_types.shape
        or connections.shape[-1] != 2
    ):
        raise ValueError(
            "inter_block_connections must have shape [pose, block, connection, 2]"
        )
    n_types = len(block_type_chirality)
    n_connections = connections.shape[2]
    if enabled.shape != (n_types, n_connections):
        raise ValueError(
            "disulfide_enabled_connections must have shape "
            f"{(n_types, n_connections)}; got {enabled.shape}"
        )
    if (
        downstream.ndim != 3
        or downstream.shape[:2] != enabled.shape
        or downstream.shape[2] < 3
    ):
        raise ValueError(
            "first_atoms_downstream_of_connection must have shape "
            f"[n_types, {n_connections}, at_least_3]"
        )

    parameter_rows = _prepare_parameter_rows(block_type_chirality, global_parameters)
    real = block_types >= 0
    if np.any(block_types[real] >= n_types):
        raise ValueError("block_types contains an out-of-range block type")
    if np.any(offsets[real] < 0):
        raise ValueError(
            "real blocks must have non-negative flattened coordinate offsets"
        )

    # Validate the complete declared connection graph first, including ordinary
    # non-disulfide connections. Padding must be exactly (-1, -1).
    _validate_declared_connections(connections, real)
    rows = _collect_pair_rows(block_types, offsets, connections, enabled, downstream)

    pair_rows = np.asarray(rows, dtype=np.int32).reshape((-1, 8))
    return DisulfideTopologyPlan(
        schema_version=DISULFIDE_TOPOLOGY_SCHEMA_VERSION,
        atom_indices_and_parameter_rows=pair_rows,
        parameter_rows=parameter_rows,
    )
