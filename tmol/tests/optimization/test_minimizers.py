import torch
import pytest
import attrs

from tmol.optimization._minimizers import (
    CartesianMinimizer,
    _build_kinforest_network,
    KinematicMinimizer,
    _run_min,
)
from tmol import (
    PoseStack,
    run_cart_min,
    run_kin_min,
    beta2016_score_function,
    FoldForest,
    MoveMap,
)


@pytest.mark.parametrize("per_pose", [False, True])
def test_run_min_value_only_matches_standard(per_pose):
    class QuadraticNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.x = torch.nn.Parameter(torch.tensor([8.0, -3.0], dtype=torch.float64))
            self.segment_ids = torch.arange(2, dtype=torch.int64)
            self.grad_modes = []

        def forward(self):
            self.grad_modes.append(torch.is_grad_enabled())
            return (self.x - self.x.new_tensor([1.0, 2.0])) ** 2

        def pose_stack_from_dofs(self):
            return self.x.detach().clone()

    results = []
    for value_only in (False, True):
        network = QuadraticNetwork()
        result = _run_min(
            network,
            optimizer_kwargs={
                "value_only_line_search": value_only,
                "max_iter": 6,
                "rtol": 1.0e-8,
                "atol": 1.0e-8,
                "gradtol": 1.0e-12,
                "history_size": 4,
            },
            per_pose=per_pose,
        )
        assert (False in network.grad_modes) == value_only
        torch.testing.assert_close(
            network.x.grad, 2 * (result - result.new_tensor([1.0, 2.0]))
        )
        results.append((result, network.x.grad.detach().clone()))
    for standard, value_only in zip(*results):
        torch.testing.assert_close(value_only, standard, rtol=0.0, atol=0.0)


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

    kf_sfxn_net = _build_kinforest_network(pose_stack, sfxn, ff, mm)
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


def test_kinematic_minimizer_reuses_compatible_network(
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
    minimizer = KinematicMinimizer(cuda_graph=torch_device.type == "cuda")
    kwargs = {"max_iter": 3, "gradtol": 0.0, "atol": 0.0, "rtol": 0.0}

    first = minimizer(pose_stack, sfxn, ff, mm, optimizer_kwargs=kwargs)
    network = minimizer.network
    optimizer = minimizer.optimizer
    assert network is not None
    assert optimizer is not None
    second_input = attrs.evolve(pose_stack, coords=pose_stack.coords.clone())
    second = minimizer(second_input, sfxn, ff, mm, optimizer_kwargs=kwargs)

    assert minimizer.network is network
    assert minimizer.optimizer is optimizer
    assert minimizer.last_optimizer_reused
    torch.testing.assert_close(second.coords, first.coords, atol=1e-4, rtol=1e-4)
