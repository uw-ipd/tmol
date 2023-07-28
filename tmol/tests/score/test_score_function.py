from tmol.score.score_function import ScoreFunction
from tmol.score.score_types import ScoreType
from tmol.pose.pose_stack_builder import PoseStackBuilder


def test_pose_score_smoke(rts_ubq_res, default_database, torch_device):
    pose_stack1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        rts_ubq_res[:4], torch_device
    )
    pose_stack100 = PoseStackBuilder.from_poses([pose_stack1] * 100, torch_device)

    sfxn = ScoreFunction(default_database, torch_device)
    sfxn.set_weight(ScoreType.fa_ljatr, 1.0)
    sfxn.set_weight(ScoreType.fa_ljrep, 0.65)
    sfxn.set_weight(ScoreType.fa_lk, 0.8)

    scorer = sfxn.render_whole_pose_scoring_module(pose_stack100)

    scores = scorer(pose_stack100.coords)
    # print("scores")
    # print(scores)
    # print(scores.shape)

    assert scores is not None
