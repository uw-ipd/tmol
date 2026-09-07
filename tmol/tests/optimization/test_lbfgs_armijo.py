import torch
import pytest

from tmol.optimization import LBFGS_Armijo


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
