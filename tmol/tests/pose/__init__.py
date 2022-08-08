import pytest

from tmol.pose.pose_stack import PoseStack
from tmol.pose.pose_stack_builder import PoseStackBuilder


@pytest.fixture
def ubq_40_60_pose_stack(ubq_res, torch_device):
    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        ubq_res[:40], torch_device
    )
    p2 = PoseStackBuilder.one_structure_from_polymeric_residues(
        ubq_res[:60], torch_device
    )
    poses = PoseStackBuilder.from_poses([p1, p2], torch_device)
    return poses
