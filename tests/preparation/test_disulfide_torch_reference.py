"""Cross-check neutral disulfide preparation against canonical Torch setup."""

import importlib.util
from pathlib import Path
import sys
import attr
import biotite.structure.io.pdbx as pdbx
import numpy as np
import torch

from tmol.database import ParameterDatabase
from tmol.io import pose_stack_from_biotite, pose_stack_from_pdb
from tmol.pose import PoseStackBuilder
from tmol.score.disulfide import DisulfideEnergyTerm
from tmol.tests.data import data_path
from tmol.tests.data.pdb import data as pdb_data

_MODULE_PATH = Path(__file__).parents[2] / "tmol/preparation/disulfide.py"
_SPEC = importlib.util.spec_from_file_location(
    "tmol_disulfide_preparation_torch_reference", _MODULE_PATH
)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
DISULFIDE_PARAMETER_FIELDS = _MODULE.DISULFIDE_PARAMETER_FIELDS
prepare_disulfide_topology = _MODULE.prepare_disulfide_topology

_MIRRORED_COLUMNS = [8, 11, 13, 16, 19, 32]


def _numpy(tensor):
    return tensor.detach().cpu().numpy()


def _canonical_pair_rows(block_types, offsets, connections, enabled, downstream):
    """Enumerate the exact upper-block traversal used by the Torch kernel."""
    rows = []
    poses = []
    for pose, pose_block_types in enumerate(block_types):
        for block1, block_type1 in enumerate(pose_block_types):
            if block_type1 < 0:
                continue
            for connection1 in np.flatnonzero(enabled[block_type1]):
                block2, connection2 = connections[pose, block1, connection1]
                if block2 < block1:
                    continue
                block_type2 = block_types[pose, block2]
                if block_type2 < 0 or not enabled[block_type2, connection2]:
                    continue
                atoms1 = (
                    offsets[pose, block1] + downstream[block_type1, connection1, 2::-1]
                )
                atoms2 = (
                    offsets[pose, block2] + downstream[block_type2, connection2, :3]
                )
                rows.append([*atoms1, *atoms2, block_type1, block_type2])
                poses.append(pose)
    return np.asarray(rows, dtype=np.int32).reshape((-1, 8)), np.asarray(poses)


def _neutral_plan_from_torch_setup(pose, database):
    term = DisulfideEnergyTerm(database, torch.device("cpu"))
    term.setup_packed_block_types(pose.packed_block_types)
    block_types, connections, enabled, downstream, canonical_params = (
        term.get_score_term_attributes(pose)
    )
    block_types = _numpy(block_types)
    connections = _numpy(connections)
    enabled = _numpy(enabled)
    downstream = _numpy(downstream)
    canonical_params = _numpy(canonical_params)

    # PoseStack offsets are pose-local. The neutral contract addresses one
    # flattened coordinate slab, matching PoseStack.rot_coord_offset.
    offsets = _numpy(pose.block_coord_offset).astype(np.int64)
    offsets += np.arange(len(pose), dtype=np.int64)[:, None] * pose.coords.shape[1]
    chirality = [
        block_type.properties.polymer.sidechain_chirality
        for block_type in pose.packed_block_types.active_block_types
    ]
    global_parameters = attr.asdict(database.scoring.disulfide.global_parameters)
    plan = prepare_disulfide_topology(
        block_types,
        offsets,
        connections,
        enabled,
        downstream,
        chirality,
        global_parameters,
    )
    expected_pairs, pair_poses = _canonical_pair_rows(
        block_types, offsets, connections, enabled, downstream
    )
    return plan, expected_pairs, pair_poses, canonical_params, chirality


def _mixed_chirality_pose(device):
    structure = pdbx.get_structure(
        pdbx.CIFFile.read(data_path("disulfide/6dmz_3_47_ld.cif")),
        model=1,
        include_bonds=True,
    )
    return pose_stack_from_biotite(structure, device, no_optH=True)


def test_neutral_plan_matches_canonical_torch_setup_for_protein_fixtures():
    device = torch.device("cpu")
    database = ParameterDatabase.get_default()
    no_disulfide = pose_stack_from_pdb(pdb_data["1UBQ"], device, residue_end=10)
    ll_disulfide = pose_stack_from_pdb(pdb_data["3PLC"], device)
    ld_disulfide = _mixed_chirality_pose(device)

    empty, expected_empty, empty_poses, _, _ = _neutral_plan_from_torch_setup(
        no_disulfide, database
    )
    assert expected_empty.shape == (0, 8)
    assert empty_poses.shape == (0,)
    assert empty.atom_indices_and_parameter_rows.shape == (0, 8)

    stacked = PoseStackBuilder.from_poses(
        [no_disulfide, ll_disulfide, ld_disulfide], device
    )
    assert (stacked.block_type_ind < 0).any(), "fixture must retain padded blocks"
    plan, expected_pairs, pair_poses, canonical_params, chirality = (
        _neutral_plan_from_torch_setup(stacked, database)
    )

    np.testing.assert_array_equal(plan.atom_indices_and_parameter_rows, expected_pairs)
    np.testing.assert_array_equal(plan.parameter_rows, canonical_params)
    assert not np.any(pair_poses == 0)
    assert np.any(pair_poses == 1)
    assert np.any(pair_poses == 2)

    ld_rows = expected_pairs[pair_poses == 2, -2:]
    assert any(
        {chirality[left], chirality[right]} == {"l", "d"} for left, right in ld_rows
    )
    ll_rows = expected_pairs[pair_poses == 1, -2:]
    assert all(chirality[left] == chirality[right] == "l" for left, right in ll_rows)

    base = np.asarray(
        [
            *(
                attr.asdict(database.scoring.disulfide.global_parameters)[name]
                for name in DISULFIDE_PARAMETER_FIELDS
            ),
            1.0,
        ],
        dtype=np.float32,
    )
    for index, handedness in enumerate(chirality):
        expected = base.copy()
        if handedness == "d":
            expected[_MIRRORED_COLUMNS] *= -1
        np.testing.assert_array_equal(plan.parameter_rows[index], expected)
