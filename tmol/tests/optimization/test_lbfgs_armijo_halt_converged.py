import torch

from tmol.optimization.lbfgs_armijo import LBFGS_Armijo_HaltConverged


def _make_optimizer(params, dof_pose_assignment, per_pose_eval_fn, n_poses, **kwargs):
    """Construct LBFGS_Armijo_HaltConverged with sensible defaults."""
    defaults = dict(gradtol=1e-3, max_iter=500, rtol=1e-5, atol=1e-5)
    defaults.update(kwargs)
    return LBFGS_Armijo_HaltConverged(
        [params],
        n_poses=n_poses,
        dof_pose_assignment=dof_pose_assignment,
        per_pose_eval_fn=per_pose_eval_fn,
        **defaults,
    )


def _quadratic_closure(optimizer, params, centers):
    """Closure for sum of independent (x_i - c_i)^2 terms."""

    def closure():
        optimizer.zero_grad()
        E = ((params - centers) ** 2).sum()
        E.backward()
        return E

    return closure


def test_halt_converged_both_poses_reach_minimum():
    """Both poses converge to their respective minima."""
    centers = torch.tensor([2.0, 5.0])
    params = torch.tensor([0.0, 0.0], requires_grad=True)
    dpa = torch.tensor([0, 1], dtype=torch.int64)

    def per_pose_eval():
        return (params.detach() - centers) ** 2

    opt = _make_optimizer(params, dpa, per_pose_eval, n_poses=2)
    opt.step(_quadratic_closure(opt, params, centers))

    result = params.detach()
    torch.testing.assert_close(result, centers, atol=1e-2, rtol=0)


def test_halt_converged_pose_at_minimum_is_frozen_and_stays():
    """A pose that starts at its minimum is frozen early and not displaced."""
    centers = torch.tensor([2.0, 5.0])
    # pose 0 starts exactly at its minimum; pose 1 starts far away
    params = torch.tensor([2.0, 0.0], requires_grad=True)
    dpa = torch.tensor([0, 1], dtype=torch.int64)

    def per_pose_eval():
        return (params.detach() - centers) ** 2

    opt = _make_optimizer(
        params,
        dpa,
        per_pose_eval,
        n_poses=2,
        pose_energy_atol=1e-4,
        n_iter_no_improve=3,
    )
    opt.step(_quadratic_closure(opt, params, centers))

    result = params.detach()
    # pose 0 must finish at (or very near) its starting minimum
    assert (
        abs(result[0].item() - 2.0) < 1e-4
    ), f"Pose 0 was displaced from its minimum: {result[0].item():.6f}"
    # pose 1 must converge
    assert (
        abs(result[1].item() - 5.0) < 1e-2
    ), f"Pose 1 did not converge: {result[1].item():.6f}"
    # pose 0 must have been recorded as converged
    state = opt.state[params]
    assert state["pose_converged"][0], "Pose 0 should have been halted"


def test_halt_converged_frozen_pose_recorded_in_state():
    """After minimization the state reflects which poses were halted."""
    centers = torch.tensor([0.0, 0.0, 10.0])
    # poses 0 and 1 start at their minima; pose 2 starts far away
    params = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
    dpa = torch.tensor([0, 1, 2], dtype=torch.int64)

    def per_pose_eval():
        return (params.detach() - centers) ** 2

    opt = _make_optimizer(
        params,
        dpa,
        per_pose_eval,
        n_poses=3,
        pose_energy_atol=1e-4,
        n_iter_no_improve=3,
    )
    opt.step(_quadratic_closure(opt, params, centers))

    state = opt.state[params]
    assert state["pose_converged"][0], "Pose 0 (at minimum) should have been halted"
    assert state["pose_converged"][1], "Pose 1 (at minimum) should have been halted"

    result = params.detach()
    assert (
        abs(result[2].item() - 10.0) < 1e-2
    ), f"Pose 2 did not converge: {result[2].item():.6f}"


def test_halt_converged_single_pose_behaves_like_base_optimizer():
    """n_poses=1 is a degenerate case and must still minimise correctly."""
    center = torch.tensor([3.0])
    params = torch.tensor([0.0], requires_grad=True)
    dpa = torch.tensor([0], dtype=torch.int64)

    def per_pose_eval():
        return (params.detach() - center) ** 2

    opt = _make_optimizer(params, dpa, per_pose_eval, n_poses=1)
    opt.step(_quadratic_closure(opt, params, center))

    assert abs(params.item() - 3.0) < 1e-2


def test_halt_converged_high_atol_prevents_freezing():
    """When pose_energy_atol is unreachably large no pose should be frozen."""
    centers = torch.tensor([2.0, 5.0])
    params = torch.tensor([0.0, 0.0], requires_grad=True)
    dpa = torch.tensor([0, 1], dtype=torch.int64)

    def per_pose_eval():
        return (params.detach() - centers) ** 2

    # atol of 1e6 means improvement is always < atol => n_no_improve always increments
    # ...but n_iter_no_improve=1000 means we'd need 1000 bad iters, which exceeds max_iter
    # Use atol=1e6 with n_iter_no_improve=1000 to prevent any freezing.
    opt = _make_optimizer(
        params,
        dpa,
        per_pose_eval,
        n_poses=2,
        pose_energy_atol=1e6,
        n_iter_no_improve=1000,
        max_iter=100,
    )
    opt.step(_quadratic_closure(opt, params, centers))

    state = opt.state[params]
    assert not any(
        state["pose_converged"]
    ), "No pose should have been frozen with unreachably large pose_energy_atol"
    # Optimizer still converges despite no per-pose freezing
    result = params.detach()
    torch.testing.assert_close(result, centers, atol=1e-2, rtol=0)


def test_halt_converged_fast_freeze_with_n_iter_no_improve_one():
    """n_iter_no_improve=1 freezes a pose after just one non-improving iteration."""
    centers = torch.tensor([2.0, 5.0])
    params = torch.tensor([2.0, 0.0], requires_grad=True)  # pose 0 already at min
    dpa = torch.tensor([0, 1], dtype=torch.int64)

    def per_pose_eval():
        return (params.detach() - centers) ** 2

    opt = _make_optimizer(
        params,
        dpa,
        per_pose_eval,
        n_poses=2,
        pose_energy_atol=1e-4,
        n_iter_no_improve=1,
    )
    opt.step(_quadratic_closure(opt, params, centers))

    state = opt.state[params]
    assert state["pose_converged"][
        0
    ], "Pose 0 should be frozen after 1 non-improving iteration"
    assert abs(params.detach()[0].item() - 2.0) < 1e-4


def test_halt_converged_asymmetric_curvature():
    """Stiff and soft quadratics both reach their minima.

    E0 = (x - 2)^2  (shallow)
    E1 = 100*(y - 5)^2  (steep; converges fast, gets frozen first)
    """
    params = torch.tensor([0.0, 0.0], requires_grad=True)
    dpa = torch.tensor([0, 1], dtype=torch.int64)
    centers = torch.tensor([2.0, 5.0])
    scale = torch.tensor([1.0, 100.0])

    def per_pose_eval():
        return scale * (params.detach() - centers) ** 2

    def closure():
        opt.zero_grad()
        E = (scale * (params - centers) ** 2).sum()
        E.backward()
        return E

    opt = _make_optimizer(
        params,
        dpa,
        per_pose_eval,
        n_poses=2,
        pose_energy_atol=1e-3,
        n_iter_no_improve=3,
    )
    opt.step(closure)

    result = params.detach()
    torch.testing.assert_close(result, centers, atol=0.05, rtol=0)


def test_halt_converged_many_poses():
    """Five independent quadratics all converge; poses that arrive first are frozen."""
    n_poses = 5
    centers = torch.arange(n_poses, dtype=torch.float) * 3.0  # 0, 3, 6, 9, 12
    # Start pose 0 at its minimum so it will be frozen quickly
    starts = centers.clone()
    starts[1:] = 0.0
    params = starts.clone().requires_grad_(True)
    dpa = torch.arange(n_poses, dtype=torch.int64)

    def per_pose_eval():
        return (params.detach() - centers) ** 2

    opt = _make_optimizer(
        params,
        dpa,
        per_pose_eval,
        n_poses=n_poses,
        pose_energy_atol=1e-4,
        n_iter_no_improve=3,
    )

    def closure():
        opt.zero_grad()
        E = ((params - centers) ** 2).sum()
        E.backward()
        return E

    opt.step(closure)

    result = params.detach()
    torch.testing.assert_close(result, centers, atol=0.05, rtol=0)

    state = opt.state[params]
    assert state["pose_converged"][0], "Pose 0 (started at minimum) should be halted"
