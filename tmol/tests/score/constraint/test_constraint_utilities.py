import torch
from tmol import pose_stack_from_file
from tmol.score.constraint import (
    constrain_all_ca,
    create_mainchain_coordinate_constraints,
)
from tmol.pose import PoseStackBuilder


def test_create_mainchain_coordinate_constraints(
    ubq_pdb, default_database, torch_device
):
    pose_stack1 = pose_stack_from_file(
        ubq_pdb, torch_device, no_optH=True, missing_density_distance_threshold=0.0
    )
    pose_stack10 = PoseStackBuilder.from_poses([pose_stack1] * 10, torch_device)

    pose_stack10 = create_mainchain_coordinate_constraints(pose_stack10)
    assert pose_stack10.constraint_set is not None

    torch.testing.assert_close(
        pose_stack10.constraint_set.constraint_params[:3, 1:4],
        pose_stack10.coords[0, :3, :],
    )


def test_constrain_all_ca(ubq_pdb, default_database, torch_device):
    pose_stack = pose_stack_from_file(
        ubq_pdb, torch_device, no_optH=True, missing_density_distance_threshold=0.0
    )

    constrained = constrain_all_ca(pose_stack)

    assert constrained.constraint_set is not None
    assert constrained.constraint_set.constraint_params.shape[1] == 5
    assert torch.all(constrained.constraint_set.constraint_params[:, 4] == 0.5)
