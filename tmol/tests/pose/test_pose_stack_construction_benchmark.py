import pytest

# from tmol.score.score_types import ScoreType
from tmol.pose.pose_stack import PoseStack


@pytest.mark.parametrize("n_poses", [1, 3, 10, 30, 100])
@pytest.mark.benchmark(group="pose_stack_construction")
def test_res_centric_score_benchmark(
    benchmark, n_poses, rts_ubq_res, default_database, torch_device
):
    pose_stack1 = PoseStack.one_structure_from_polymeric_residues(
        rts_ubq_res, torch_device
    )

    @benchmark
    def construct_pass():
        pose_stack_n = PoseStack.from_poses([pose_stack1] * n_poses, torch_device)
        return pose_stack_n
