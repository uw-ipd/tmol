import dataclasses
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

_MODULE_PATH = Path(__file__).parents[2] / "tmol/preparation/disulfide.py"
_SPEC = importlib.util.spec_from_file_location(
    "tmol_disulfide_preparation", _MODULE_PATH
)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

FIELDS = _MODULE.DISULFIDE_PARAMETER_FIELDS
DisulfideTopologyPlan = _MODULE.DisulfideTopologyPlan
prepare_disulfide_topology = _MODULE.prepare_disulfide_topology


def parameters():
    return {name: index + 0.25 for index, name in enumerate(FIELDS)}


def topology_inputs():
    block_types = np.array([[0, 1, -1], [2, 0, 1]], dtype=np.int32)
    offsets = np.array([[0, 10, -1], [100, 110, 120]], dtype=np.int64)
    connections = np.full((2, 3, 2, 2), -1, dtype=np.int32)
    connections[0, 0, 1] = (1, 0)
    connections[0, 1, 0] = (0, 1)
    connections[1, 1, 1] = (2, 0)
    connections[1, 2, 0] = (1, 1)
    enabled = np.array([[False, True], [True, False], [False, False]])
    downstream = np.full((3, 2, 4), -1, dtype=np.int32)
    downstream[0, 1, :3] = (5, 4, 3)
    downstream[1, 0, :3] = (2, 1, 0)
    return block_types, offsets, connections, enabled, downstream


def test_declared_edges_are_prepared_once_with_pilot_atom_ordering():
    plan = prepare_disulfide_topology(
        *topology_inputs(), ["l", "d", "achiral"], parameters()
    )

    np.testing.assert_array_equal(
        plan.atom_indices_and_parameter_rows,
        [[3, 4, 5, 12, 11, 10, 0, 1], [113, 114, 115, 122, 121, 120, 0, 1]],
    )
    assert plan.atom_indices_and_parameter_rows.dtype == np.int32
    assert plan.schema_version == 1


def test_reciprocal_self_block_edge_is_scored_once():
    connections = np.array([[[[0, 1], [0, 0]]]], dtype=np.int32)
    plan = prepare_disulfide_topology(
        np.array([[0]], dtype=np.int32),
        np.array([[10]], dtype=np.int32),
        connections,
        np.ones((1, 2), dtype=bool),
        np.array([[[0, 1, 2], [3, 4, 5]]], dtype=np.int32),
        ["l"],
        parameters(),
    )
    np.testing.assert_array_equal(
        plan.atom_indices_and_parameter_rows,
        [[12, 11, 10, 13, 14, 15, 0, 0]],
    )


def test_parameter_rows_match_pilot_layout_and_mirror_d_residues():
    plan = prepare_disulfide_topology(
        *topology_inputs(), ["l", "d", "achiral"], parameters()
    )
    base = np.array([parameters()[name] for name in FIELDS] + [1], dtype=np.float32)
    expected = np.tile(base, (3, 1))
    expected[1, [8, 11, 13, 16, 19, 32]] *= -1

    np.testing.assert_array_equal(plan.parameter_rows, expected)
    assert plan.parameter_rows.dtype == np.float32
    with pytest.raises(ValueError, match="read-only"):
        plan.parameter_rows[0, 0] = 0
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        plan.parameter_rows.flags.writeable = True
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.schema_version = 2


def test_empty_and_fully_padded_poses_produce_empty_pair_rows():
    empty = prepare_disulfide_topology(
        np.empty((0, 2), dtype=np.int32),
        np.empty((0, 2), dtype=np.int32),
        np.empty((0, 2, 1, 2), dtype=np.int32),
        np.zeros((1, 1), dtype=bool),
        np.zeros((1, 1, 3), dtype=np.int32),
        ["l"],
        parameters(),
    )
    assert empty.atom_indices_and_parameter_rows.shape == (0, 8)

    padded = prepare_disulfide_topology(
        np.full((2, 3), -1, dtype=np.int32),
        np.full((2, 3), -1, dtype=np.int32),
        np.full((2, 3, 1, 2), -1, dtype=np.int32),
        np.zeros((1, 1), dtype=bool),
        np.zeros((1, 1, 3), dtype=np.int32),
        ["l"],
        parameters(),
    )
    assert padded.atom_indices_and_parameter_rows.shape == (0, 8)


def test_rejects_nonreciprocal_and_out_of_bounds_endpoints():
    args = list(topology_inputs())
    args[2] = args[2].copy()
    args[2][0, 1, 0] = (-1, -1)
    with pytest.raises(ValueError, match="reciprocal"):
        prepare_disulfide_topology(*args, ["l", "d", "l"], parameters())

    args = list(topology_inputs())
    args[2] = args[2].copy()
    args[2][0, 0, 1] = (3, 0)
    with pytest.raises(ValueError, match="out of bounds"):
        prepare_disulfide_topology(*args, ["l", "d", "l"], parameters())


def test_skips_reciprocal_edges_without_two_disulfide_enabled_endpoints():
    block_types, offsets, connections, enabled, downstream = topology_inputs()
    enabled = enabled.copy()
    enabled[1, 0] = False
    plan = prepare_disulfide_topology(
        block_types,
        offsets,
        connections,
        enabled,
        downstream,
        ["l", "d", "l"],
        parameters(),
    )
    assert plan.atom_indices_and_parameter_rows.shape == (0, 8)


def test_rejects_invalid_scoring_atoms_and_missing_parameters():
    args = list(topology_inputs())
    args[4] = args[4].copy()
    args[4][0, 1, 2] = -1
    with pytest.raises(ValueError, match="invalid downstream atom"):
        prepare_disulfide_topology(*args, ["l", "d", "l"], parameters())

    missing = parameters()
    del missing[FIELDS[0]]
    with pytest.raises(ValueError, match="missing global disulfide parameter"):
        prepare_disulfide_topology(*topology_inputs(), ["l", "d", "l"], missing)
