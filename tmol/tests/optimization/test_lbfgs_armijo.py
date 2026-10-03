import weakref
from copy import deepcopy

import torch
import pytest

from tmol.optimization import LBFGS_Armijo


def test_lbfgs_releases_closure_after_step(torch_device):
    x = torch.nn.Parameter(torch.tensor([2.0, -3.0], device=torch_device))
    optimizer = LBFGS_Armijo([x], max_iter=3)
    optimizer_ref = weakref.ref(optimizer)

    class Closure:
        def __call__(self):
            optimizer_ref().zero_grad()
            loss = x.square().sum()
            loss.backward()
            return loss

    closure = Closure()
    closure_ref = weakref.ref(closure)
    optimizer.step(closure)
    del closure
    assert closure_ref() is None
    del optimizer
    assert optimizer_ref() is None


class SimpleLJScore:
    def __init__(self, r_m=1.0, epsilon=1.0):
        self.r_m = r_m
        self.epsilon = epsilon

    def __call__(self, coords):
        N = coords.shape[0]
        row, col = torch.tril_indices(N, N, offset=-1)
        deltas = coords[row] - coords[col]
        dist = torch.norm(deltas, 2, -1)
        fd = self.r_m / dist

        fd2 = fd * fd
        fd6 = fd2 * fd2 * fd2
        fd12 = fd6 * fd6
        self.lj = self.epsilon * (fd12 - 3 * fd6)

        self.total_score = 2 * torch.sum(self.lj)

        return self


def test_lbfgs_requires_one_parameter_tensor():
    x = torch.nn.Parameter(torch.zeros(1))
    y = torch.nn.Parameter(torch.zeros(1))

    with pytest.raises(ValueError, match="exactly one parameter tensor"):
        LBFGS_Armijo([x, y])


def test_lbfgs_accepts_one_named_parameter():
    x = torch.nn.Parameter(torch.zeros(1))

    optimizer = LBFGS_Armijo([("x", x)])

    assert optimizer.param_groups[0]["params"][0] is x
    assert optimizer.param_groups[0]["param_names"] == ["x"]


def test_lbfgs_preserves_positional_arguments():
    x = torch.nn.Parameter(torch.tensor([2.0, -3.0]))
    segments = torch.tensor([0, 1])
    optimizer = LBFGS_Armijo(
        [x],
        0.5,
        3,
        1e-5,
        1e-6,
        1e-3,
        4,
        1e-9,
        False,
        segments,
        True,
        patience=2,
    )
    group = optimizer.param_groups[0]
    assert (group["lr"], group["max_iter"], group["history_size"]) == (0.5, 3, 4)
    assert (group["rtol"], group["atol"], group["gradtol"]) == (1e-5, 1e-6, 1e-3)
    assert group["patience"] == 2 and group["fixed_iterations"] is True
    assert optimizer._minstep == 1e-9 and optimizer.verbose is False
    assert optimizer._n_segments == 2


@pytest.mark.parametrize("patience", [0, -1, 0.5, 2.0, True, None])
def test_lbfgs_rejects_invalid_patience(patience):
    x = torch.nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError, match="patience must be a positive integer"):
        LBFGS_Armijo([x], patience=patience)


@pytest.mark.parametrize("name", ["rtol", "atol", "gradtol"])
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_lbfgs_rejects_invalid_tolerances(name, value):
    x = torch.nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError, match=f"{name} must be None or a finite"):
        LBFGS_Armijo([x], **{name: value})


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("rtol,atol", [(None, None), (0.0, None), (None, 0.0)])
def test_lbfgs_resumes_checkpoint_before_patience(dtype, rtol, atol):
    """Legacy None tolerances and one-step stopping survive a checkpoint load."""
    tolerance = float(torch.finfo(dtype).eps ** 0.5)
    x = torch.nn.Parameter(torch.tensor([3.0, -2.0, 1.0], dtype=dtype))
    optimizer = LBFGS_Armijo(
        [x],
        max_iter=1,
        history_size=1,
        gradtol=0.0,
        patience=1,
        rtol=tolerance if rtol is None else rtol,
        atol=tolerance if atol is None else atol,
    )

    def step(opt, param):
        def closure():
            opt.zero_grad()
            loss = (param.square() * param.new_tensor([1.0, 10.0, 100.0])).sum()
            loss.backward()
            return loss

        opt.step(closure)

    step(optimizer, x)
    checkpoint = deepcopy(optimizer.state_dict())
    # Recreate the old serialized schema after a real unfinished trajectory.
    group = checkpoint["param_groups"][0]
    del group["patience"]
    group.update(rtol=rtol, atol=atol)
    for state in checkpoint["state"].values():
        del state["small_steps"]
    resumed_x = torch.nn.Parameter(x.detach().clone())
    resumed = LBFGS_Armijo([resumed_x])
    resumed.load_state_dict(checkpoint)
    loaded = resumed.param_groups[0]
    assert loaded["patience"] == 1
    for name in ("rtol", "atol"):
        assert loaded[name] == optimizer.param_groups[0][name]
    # Loading must not rewrite the caller's checkpoint or the new defaults.
    assert "patience" not in checkpoint["param_groups"][0]
    assert resumed.defaults["patience"] == 5
    step(optimizer, x)
    step(resumed, resumed_x)
    torch.testing.assert_close(resumed_x, x)
    assert resumed.state[resumed_x]["n_iter"] == optimizer.state[x]["n_iter"]


def test_lbfgs_new_checkpoint_keeps_disabled_tolerances():
    x = torch.nn.Parameter(torch.ones(1))
    saved = LBFGS_Armijo([x], rtol=None, atol=None, gradtol=None, patience=7)
    resumed = LBFGS_Armijo([torch.nn.Parameter(torch.ones(1))])
    resumed.load_state_dict(saved.state_dict())
    group = resumed.param_groups[0]
    assert group["patience"] == 7
    assert all(group[name] is None for name in ("rtol", "atol", "gradtol"))


def test_lbfgs_zero_grad_supports_both_reset_modes():
    x = torch.nn.Parameter(torch.ones(2))
    optimizer = LBFGS_Armijo([x])

    x.sum().backward()
    optimizer.zero_grad(set_to_none=False)
    torch.testing.assert_close(x.grad, torch.zeros_like(x))

    x.sum().backward()
    optimizer.zero_grad()
    assert x.grad is None


def test_lbfgs_armijo():
    dtype = torch.float
    device = torch.device("cpu")

    Natoms = 100
    x = torch.randn(Natoms, 3, device=device, dtype=dtype, requires_grad=True)
    scorefunc = SimpleLJScore(r_m=1.0, epsilon=1.0)

    optimizer = LBFGS_Armijo([x], lr=1.0, rtol=1e-2, gradtol=1e-2)

    def closure():
        optimizer.zero_grad()
        E = scorefunc(10 * x)
        E.total_score.backward()
        return E.total_score

    score_start = closure()
    optimizer.step(closure)
    score_stop = closure()

    assert score_start > score_stop


def test_large_negative_gradient_does_not_converge():
    x = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = LBFGS_Armijo([x], max_iter=2, rtol=1e-6, atol=1e-6, gradtol=1.0)

    def closure():
        optimizer.zero_grad()
        loss = -10 * x.sum()
        loss.backward()
        return loss

    optimizer.step(closure)

    assert optimizer.state[x]["n_iter"] == 2
    torch.testing.assert_close(x.grad, torch.tensor([-10.0], dtype=x.dtype))


def test_short_run_allocates_only_reachable_history():
    x = torch.nn.Parameter(torch.tensor([3.0, -2.0]))
    optimizer = LBFGS_Armijo(
        [x], max_iter=3, history_size=128, rtol=0.0, atol=0.0, gradtol=0.0
    )

    def closure():
        optimizer.zero_grad()
        loss = (x * x).sum()
        loss.backward()
        return loss

    optimizer.step(closure)

    assert optimizer.state[x]["old_dirs_mat"].shape[0] == 2
    assert optimizer.state[x]["old_stps_mat"].shape[0] == 2


def test_reset_reuses_scratch_and_restarts_trajectory():
    initial = torch.tensor([3.0, -2.0])
    x = torch.nn.Parameter(initial.clone())
    optimizer = LBFGS_Armijo([x], max_iter=3, rtol=0.0, atol=0.0, gradtol=0.0)

    def closure():
        optimizer.zero_grad()
        loss = (x * x).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    first = x.detach().clone()
    history_ptr = optimizer.state[x]["old_dirs_mat"].data_ptr()
    with torch.no_grad():
        x.copy_(initial)
    optimizer.reset()
    optimizer.step(closure)

    torch.testing.assert_close(x, first)
    assert optimizer.state[x]["old_dirs_mat"].data_ptr() == history_ptr


def test_fixed_iterations_skips_early_convergence():
    x = torch.nn.Parameter(torch.zeros(2))
    optimizer = LBFGS_Armijo([x], max_iter=4, fixed_iterations=True)

    def closure():
        optimizer.zero_grad()
        loss = (x * x).sum()
        loss.backward()
        return loss

    optimizer.step(closure)

    assert optimizer.state[x]["n_iter"] == 4


def test_default_tolerances_do_not_stop_a_descending_run():
    """Many small gradients can still mean a large energy drop.

    Guards LBFGS_Armijo's defaults in optimization/_lbfgs_armijo.py, which had
    gradtol=1.0: a segment with max |dE/dx| <= 1 was converged however much
    energy each iteration still removed.
    """
    x = torch.nn.Parameter(torch.zeros(10000, dtype=torch.float64))
    optimizer = LBFGS_Armijo([x], max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = 0.25 * ((x - 1.0) ** 2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)

    torch.testing.assert_close(x.detach(), torch.ones_like(x), rtol=0, atol=1e-3)


@pytest.mark.parametrize("patience", [1, 3, 5])
def test_convergence_needs_patience_small_iterations(patience):
    """A segment converges only after patience consecutive small iterations.

    Guards _check_segment_convergence in optimization/_lbfgs_armijo.py, which
    stopped on the first iteration whose energy change was under atol or rtol.
    """
    x = torch.nn.Parameter(torch.tensor([3.0, -2.0, 1.0], dtype=torch.float64))
    scale = torch.tensor([1.0, 10.0, 100.0], dtype=torch.float64)
    # every iteration counts as small
    optimizer = LBFGS_Armijo([x], max_iter=50, atol=1e9, patience=patience)

    def closure():
        optimizer.zero_grad()
        loss = ((x * scale) ** 2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)

    assert optimizer.state[x]["n_iter"] == patience


@pytest.mark.xfail(reason="sparse tensor _copy failure in torch 1.6")
def test_lbfgs_armijo_sparse():
    indices = torch.LongTensor([[0, 0, 1], [0, 1, 1]])
    values = torch.FloatTensor([2, 3, 4])
    sizes = [2, 2]
    a = torch.sparse_coo_tensor(indices, values, sizes, requires_grad=True)

    optimizer = LBFGS_Armijo([a], lr=0.1, rtol=1e-8, atol=1e-8, gradtol=1e-8)

    def closure():
        optimizer.zero_grad()
        E = (a.coalesce().values().sum()) ** 2
        E.backward()
        return E

    score_start = closure()
    optimizer.step(closure)
    score_stop = closure()

    assert score_start > score_stop


def test_lbfgs_armijo_short_history():
    dtype = torch.float
    device = torch.device("cpu")

    Natoms = 20
    x = torch.randn(Natoms, 3, device=device, dtype=dtype, requires_grad=True)
    scorefunc = SimpleLJScore(r_m=1.0, epsilon=1.0)

    optimizer = LBFGS_Armijo([x], lr=1.0, rtol=1e-2, gradtol=1e-2, history_size=2)

    def closure():
        optimizer.zero_grad()
        E = scorefunc(10 * x)
        E.total_score.backward()
        return E.total_score

    score_start = closure()
    optimizer.step(closure)
    score_stop = closure()

    assert score_start > score_stop


@pytest.mark.parametrize("first_segment_size", [6, 7])
def test_lbfgs_armijo_wrapped_segment_histories(torch_device, first_segment_size):
    x = torch.nn.Parameter(
        torch.linspace(-2, 2, 12, dtype=torch.float64, device=torch_device)
    )
    segment_ids = torch.tensor(
        [0] * first_segment_size + [1] * (12 - first_segment_size), device=torch_device
    )
    curvature = torch.tensor(
        [1, 2, 4, 8, 1, 2, 4, 8, 4, 2, 1, 8],
        dtype=x.dtype,
        device=torch_device,
    )
    optimizer = LBFGS_Armijo(
        [x],
        segment_ids=segment_ids,
        history_size=2,
        max_iter=80,
        rtol=0,
        atol=0,
        gradtol=1e-8,
    )

    def closure():
        optimizer.zero_grad()
        terms = curvature * (x - 1).square()
        energies = torch.stack(
            (terms[:first_segment_size].sum(), terms[first_segment_size:].sum())
        )
        energies.sum().backward()
        return energies

    optimizer.step(closure)
    assert optimizer.state[x]["n_iter"] > 4
    torch.testing.assert_close(x, torch.ones_like(x), atol=1e-6, rtol=0)
