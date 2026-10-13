from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from tmol_jax.pack import (
    PackerEnergyTables,
    RepackingRotamers,
    impose_repacking_assignments,
    repack_from_trace,
)
from tmol_jax.pose import PoseStack


def _pose_and_rotamers():
    pbt = SimpleNamespace(n_atoms=jnp.asarray([2, 1], dtype=jnp.int32), max_n_atoms=2)
    pose = PoseStack(
        packed_block_types=pbt,
        coords=jnp.zeros((2, 4, 3), dtype=jnp.float32),
        block_coord_offset=jnp.asarray([[0, 2], [0, 1]], dtype=jnp.int32),
        inter_residue_connections=jnp.full((2, 2, 1, 2), -1, dtype=jnp.int32),
        inter_block_bondsep=jnp.zeros((2, 2, 2, 1, 1), dtype=jnp.int32),
        block_type_ind=jnp.asarray([[0, 1], [1, -1]], dtype=jnp.int32),
    )

    def x(value):
        return [value, 0, 0]

    rotamers = RepackingRotamers(
        pose_for_rot=jnp.asarray([0, 0, 0, 1, 1], dtype=jnp.int32),
        block_ind_for_rot=jnp.asarray([0, 0, 1, 0, 0], dtype=jnp.int32),
        block_type_ind_for_rot=jnp.asarray([0, 0, 1, 1, 0], dtype=jnp.int32),
        coord_offset_for_rot=jnp.asarray([0, 2, 4, 5, 6], dtype=jnp.int32),
        coords=jnp.asarray(
            [x(10), x(11), x(20), x(21), x(30), x(40), x(50), x(51)],
            dtype=jnp.float32,
        ),
    )
    return pose, rotamers


def test_impose_repacking_assignments_mixed_and_padded():
    pose, rotamers = _pose_and_rotamers()
    result = impose_repacking_assignments(pose, rotamers, [[1, 2], [3, -1]])
    np.testing.assert_array_equal(
        result.coords[..., 0],
        [[20, 21, 30, 0], [40, 0, 0, 0]],
    )
    assert result.packed_block_types is pose.packed_block_types
    np.testing.assert_array_equal(result.block_type_ind, pose.block_type_ind)


@pytest.mark.parametrize(
    "assignments, message",
    [
        ([[1, 2], [2, -1]], "different pose/block"),
        ([[1, 2], [4, -1]], "residue-type changes"),
        ([[1, 2], [3, 0]], "padding blocks"),
        ([[1, 2], [-1, -1]], "valid rotamer assignment"),
    ],
)
def test_impose_repacking_assignments_rejects_invalid(assignments, message):
    pose, rotamers = _pose_and_rotamers()
    with pytest.raises(ValueError, match=message):
        impose_repacking_assignments(pose, rotamers, assignments)


def test_repack_from_trace_selects_and_imposes_best_trajectory():
    pose, rotamers = _pose_and_rotamers()
    tables = PackerEnergyTables(
        nrotamers_for_res=jnp.asarray([[2, 1], [1, 0]], dtype=jnp.int32),
        oneb_offsets=jnp.asarray([[0, 2], [3, 0]], dtype=jnp.int32),
        chunk_size=1,
        neighbor_row_offsets=jnp.zeros((5,), dtype=jnp.int32),
        neighbor_blocks=jnp.empty((0,), dtype=jnp.int32),
        neighbor_chunk_offset_offsets=jnp.empty((0,), dtype=jnp.int32),
        chunk_offsets=jnp.empty((0,), dtype=jnp.int32),
        energy1b=jnp.asarray([1, 5, 2, 4], dtype=jnp.float32),
        energy2b=jnp.empty((0,), dtype=jnp.float32),
    )
    initial = jnp.asarray([[[0, 0], [1, 0]], [[0, -1], [0, -1]]], dtype=jnp.int32)
    proposed_residue = jnp.zeros((1, 2, 2), dtype=jnp.int32)
    proposed_rotamer = jnp.asarray([[[1, 0], [0, 0]]], dtype=jnp.int32)
    packed, best, energies, accepted = repack_from_trace(
        pose,
        rotamers,
        tables,
        initial,
        proposed_residue,
        proposed_rotamer,
        jnp.ones((1, 2, 2), dtype=jnp.float32),
        jnp.zeros((1, 2, 2), dtype=jnp.float32),
        jnp.ones((1, 2, 2), dtype=jnp.bool_),
        base_candidate_assignments=jnp.asarray([[0, 2], [3, -1]], dtype=jnp.int32),
        candidate_for_table_rotamer=jnp.asarray([1, 0, 2, 3], dtype=jnp.int32),
        block_for_table_residue=jnp.asarray([[0, 1], [0, -1]], dtype=jnp.int32),
    )

    np.testing.assert_array_equal(best, [[0, 0], [0, -1]])
    np.testing.assert_array_equal(energies, [3, 4])
    np.testing.assert_array_equal(accepted, [[[False, True], [False, False]]])
    np.testing.assert_array_equal(
        packed.coords[..., 0], [[20, 21, 30, 0], [40, 0, 0, 0]]
    )
