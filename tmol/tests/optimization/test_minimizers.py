import torch

from tmol import (
    CartesianMinimizer,
    PoseStack,
    run_cart_min,
    run_kin_min,
    build_kinforest_network,
    beta2016_score_function,
    FoldForest,
    MoveMap,
)
import attrs


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
    minimized_pose_stack = run_kin_min(pose_stack, sfxn, ff, mm)
    end_score = wpsm(minimized_pose_stack.coords)

    assert torch.all(end_score < start_score)


def test_run_cart_min_smoke(
    jagged_stack_of_465_res_ubqs: PoseStack,
    torch_device,
):
    pose_stack = jagged_stack_of_465_res_ubqs
    sfxn = beta2016_score_function(torch_device)

    wpsm = sfxn.render_whole_pose_scoring_module(pose_stack)

    start_score = wpsm(pose_stack.coords)
    minimized_pose_stack = run_cart_min(
        pose_stack,
        sfxn,
        cuda_graph=torch_device.type == "cuda",
    )
    end_score = wpsm(minimized_pose_stack.coords)

    assert torch.all(end_score < start_score)


def test_cartesian_minimizer_reuses_compatible_network(
    jagged_stack_of_465_res_ubqs: PoseStack,
    torch_device,
):
    pose_stack = jagged_stack_of_465_res_ubqs
    sfxn = beta2016_score_function(torch_device)
    minimizer = CartesianMinimizer(cuda_graph=torch_device.type == "cuda")
    kwargs = {"max_iter": 3, "gradtol": 0.0, "atol": 0.0, "rtol": 0.0}

    first = minimizer(pose_stack, sfxn, optimizer_kwargs=kwargs)
    network = minimizer.network
    optimizer = minimizer.optimizer
    assert network is not None
    assert optimizer is not None
    second_input = attrs.evolve(pose_stack, coords=pose_stack.coords.clone())
    second = minimizer(second_input, sfxn, optimizer_kwargs=kwargs)

    assert minimizer.network is network
    assert minimizer.optimizer is optimizer
    assert minimizer.last_optimizer_reused
    torch.testing.assert_close(second.coords, first.coords)


def test_run_kin_min_torch_lbfgs(
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
    minimized_pose_stack = run_kin_min(
        pose_stack,
        sfxn,
        ff,
        mm,
        optimizer_cls=torch.optim.LBFGS,
        optimizer_kwargs={"lr": 1, "max_iter": 200, "line_search_fn": "strong_wolfe"},
    )
    end_score = wpsm(minimized_pose_stack.coords)

    assert torch.all(end_score < start_score)
