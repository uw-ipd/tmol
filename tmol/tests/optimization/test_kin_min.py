import torch
import numpy

from tmol import (
    PoseStack,
    run_kin_min,
    build_kinforest_network,
    beta2016_score_function,
    FoldForest,
    MoveMap,
)


def test_build_kinforest_sfxn_network_smoke(
    jagged_stack_of_465_res_ubqs: PoseStack,
    ff_3_jagged_ubq_465res_H: torch.Tensor,
    torch_device,
):
    pose_stack = jagged_stack_of_465_res_ubqs
    ff = FoldForest.from_edges(ff_3_jagged_ubq_465res_H)
    sfxn = beta2016_score_function(torch_device)
    mm = MoveMap.from_pose_stack(pose_stack)
    mm.move_all_jumps = True
    mm.move_all_named_torsions = True

    kf_sfxn_net = build_kinforest_network(pose_stack, sfxn, ff, mm)
    assert kf_sfxn_net is not None


def test_run_kin_min_smoke(
    jagged_stack_of_465_res_ubqs: PoseStack,
    ff_3_jagged_ubq_465res_H: torch.Tensor,
    torch_device,
):
    pose_stack = jagged_stack_of_465_res_ubqs
    ff = FoldForest.from_edges(ff_3_jagged_ubq_465res_H)
    sfxn = beta2016_score_function(torch_device)
    mm = MoveMap.from_pose_stack(pose_stack)
    mm.move_all_jumps = True
    mm.move_all_named_torsions = True

    wpsm = sfxn.render_whole_pose_scoring_module(pose_stack)

    start_score = wpsm(pose_stack.coords)
    # print("start score", start_score)
    minimized_pose_stack = run_kin_min(pose_stack, sfxn, ff, mm)
    end_score = wpsm(minimized_pose_stack.coords)
    # print("end score", end_score)

    # from tmol import write_pose_stack_pdb
    # dev = str(torch_device).partition(":")[0]
    # write_pose_stack_pdb(minimized_pose_stack, f"test_run_kin_min_smoke_{dev}.pdb")

    assert torch.all(end_score < start_score)
