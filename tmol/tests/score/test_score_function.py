import torch
import numpy

from tmol.score.score_function import ScoreFunction
from tmol.score.score_types import ScoreType
from tmol.pose.pose_stack import PoseStack


def test_pose_score_smoke(rts_ubq_res, default_database, torch_device):
    pose_stack1 = PoseStack.one_structure_from_polymeric_residues(
        rts_ubq_res[:4], torch_device
    )
    pose_stack100 = PoseStack.from_poses([pose_stack1] * 1, torch_device)

    sfxn = ScoreFunction(default_database, torch_device)
    sfxn.set_weight(ScoreType.fa_lj, 1.0)
    sfxn.set_weight(ScoreType.fa_lk, 0.8)

    scorer = sfxn.render_whole_pose_scoring_module(pose_stack100)

    scores = scorer(pose_stack100.coords)
    print("scores")
    print(scores)
    print(scores.shape)