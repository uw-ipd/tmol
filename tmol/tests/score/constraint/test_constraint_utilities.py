import torch
from tmol import (
    pose_stack_from_pdb,
    # beta2016_score_function,
    # canonical_form_from_pdb,
    # default_canonical_ordering,
    # default_packed_block_types,
    # pose_stack_from_canonical_form,
)
from tmol.score.constraint.utility import create_backbone_coordinate_constraints
from tmol.pose.pose_stack_builder import PoseStackBuilder


def test_create_mainchain_coordinate_constraints(
    ubq_pdb, default_database, torch_device
):
    pose_stack1 = pose_stack_from_pdb(ubq_pdb, torch_device)
    pose_stack10 = PoseStackBuilder.from_poses([pose_stack1] * 10, torch_device)

    pose_stack10 = create_backbone_coordinate_constraints(pose_stack10)
    assert pose_stack10.constraint_set is not None

    torch.testing.assert_close(
        pose_stack10.constraint_set.constraint_params[:3, 1:4],
        pose_stack10.coords[0, :3, :],
    )
