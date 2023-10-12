import pytest

from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack_builder import PoseStackBuilder


@pytest.fixture
def ubq_40_60_pose_stack(ubq_res, default_database, torch_device):
    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, ubq_res[:40], torch_device
    )
    p2 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, ubq_res[:60], torch_device
    )
    poses = PoseStackBuilder.from_poses([p1, p2], torch_device)
    return poses


@pytest.fixture
def fresh_default_packed_block_types(fresh_default_restype_set, torch_device):
    return PackedBlockTypes.from_restype_list(
        fresh_default_restype_set.chem_db,
        fresh_default_restype_set.residue_types,
        torch_device,
    )


@pytest.fixture
def two_ubq_poses(ubq_res, default_database, torch_device):
    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, ubq_res[:40], torch_device
    )
    p2 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, ubq_res[:60], torch_device
    )
    return PoseStackBuilder.from_poses([p1, p2], torch_device)
