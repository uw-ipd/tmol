"""Per-segment L-BFGS: minimizing blocks together must match minimizing them alone."""

import torch

from tmol.optimization import LBFGS_Armijo, lbfgs_two_loop


def _history(m, n_segments, size, dtype, device, seed=0):
    """Random history with positive curvature in every slot."""
    gen = torch.Generator().manual_seed(seed)
    stps = torch.randn(m, n_segments, size, generator=gen, dtype=dtype)
    dirs = torch.randn(m, n_segments, size, generator=gen, dtype=dtype)
    dots = (stps * dirs).sum(-1)
    dirs = torch.where((dots >= 0).unsqueeze(-1), dirs, -dirs) + 0.5 * stps
    grad = torch.randn(n_segments, size, generator=gen, dtype=dtype)
    return grad.to(device), dirs.to(device), stps.to(device)


def test_batched_two_loop_matches_one_problem_at_a_time(torch_device):
    grad, dirs, stps = _history(7, 4, 13, torch.float64, torch_device)
    batched = lbfgs_two_loop(grad, dirs, stps)
    for p in range(grad.shape[0]):
        alone = lbfgs_two_loop(grad[p], dirs[:, p], stps[:, p])
        torch.testing.assert_close(batched[p], alone, rtol=1e-10, atol=1e-12)


def test_two_loop_ignores_zeroed_history_slots(torch_device):
    """A slot with no curvature for one problem must not affect any problem."""
    grad, dirs, stps = _history(7, 3, 9, torch.float64, torch_device)
    full = lbfgs_two_loop(grad, dirs, stps)

    dropped, problem = 3, 1
    dirs_z, stps_z = dirs.clone(), stps.clone()
    dirs_z[dropped, problem] = 0.0
    stps_z[dropped, problem] = 0.0
    zeroed = lbfgs_two_loop(grad, dirs_z, stps_z)

    keep = [i for i in range(dirs.shape[0]) if i != dropped]
    shorter = lbfgs_two_loop(
        grad[problem], dirs[keep][:, problem], stps[keep][:, problem]
    )
    torch.testing.assert_close(zeroed[problem], shorter, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(zeroed[0], full[0], rtol=1e-10, atol=1e-12)


def _quadratic(center, scale=1.0, offset=0.0):
    def energy(x):
        return offset + scale * 0.5 * ((x - center) ** 2).sum()

    return energy


def _rosenbrock(x):
    a, b = x[:-1], x[1:]
    return (100.0 * (b - a * a) ** 2 + (1.0 - a) ** 2).sum()


class _FakeGradient(torch.autograd.Function):
    """Constant energy that claims a downhill gradient: no step can ever help."""

    @staticmethod
    def forward(ctx, x):
        return torch.zeros((), dtype=x.dtype, device=x.device)

    @staticmethod
    def backward(ctx, grad_out):
        return None


def _inconsistent(x):
    """An objective whose gradient disagrees with its values."""
    return _FakeGradient.apply(x) + (x * 0.0).sum() + x.sum() - x.sum().detach()


class BlockProblem:
    """A sum of independent per-block objectives over disjoint DOFs."""

    def __init__(self, blocks, starts, device):
        self.blocks = blocks
        self.starts = [s.to(device) for s in starts]
        self.sizes = [s.numel() for s in self.starts]
        self.segment_ids = torch.cat(
            [
                torch.full((n,), i, dtype=torch.int64, device=device)
                for i, n in enumerate(self.sizes)
            ]
        )

    def energies(self, x):
        return torch.stack(
            [f(xi) for f, xi in zip(self.blocks, torch.split(x, self.sizes))]
        )

    def minimize_together(self, segmented=True, **kwargs):
        """Returns (optimizer, parameter, per-block parameters)."""
        x = torch.nn.Parameter(torch.cat(self.starts))
        if segmented:
            kwargs = dict(kwargs, segment_ids=self.segment_ids)
        optimizer = LBFGS_Armijo([x], **kwargs)

        def closure():
            optimizer.zero_grad()
            E = self.energies(x)
            E.sum().backward()
            return E if segmented else E.sum()

        optimizer.step(closure)
        return optimizer, x, list(torch.split(x.detach(), self.sizes))

    def minimize_alone(self, **kwargs):
        finals = []
        for block, start in zip(self.blocks, self.starts):
            x = torch.nn.Parameter(start.clone())
            optimizer = LBFGS_Armijo([x], **kwargs)

            def closure(block=block, x=x, optimizer=optimizer):
                optimizer.zero_grad()
                E = block(x)
                E.backward()
                return E

            optimizer.step(closure)
            finals.append(x.detach().clone())
        return finals


def _mixed_curvature(device):
    """Blocks that want very different step sizes: gentle, stiff, and a valley."""
    gen = torch.Generator().manual_seed(3)
    return BlockProblem(
        blocks=[
            _quadratic(3.0),
            _quadratic(0.0, scale=1.0e4),
            _rosenbrock,
        ],
        starts=[
            torch.randn(6, generator=gen, dtype=torch.float64),
            torch.randn(6, generator=gen, dtype=torch.float64),
            torch.full((8,), -1.2, dtype=torch.float64),
        ],
        device=device,
    )


def test_segmented_min_matches_one_block_at_a_time(torch_device):
    """The headline invariant: a stack minimizes exactly as its blocks do alone."""
    problem = _mixed_curvature(torch_device)
    kwargs = dict(lr=1.0, max_iter=50, gradtol=1e-10)

    _, _, together = problem.minimize_together(**kwargs)
    alone = problem.minimize_alone(**kwargs)

    for i, (a, b) in enumerate(zip(together, alone)):
        torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-9, msg=f"block {i}")


def test_unsegmented_min_does_not_match(torch_device):
    """Guard on the premise: one shared step size is what breaks the invariant."""
    problem = _mixed_curvature(torch_device)
    kwargs = dict(lr=1.0, max_iter=50, gradtol=1e-10)

    _, _, together = problem.minimize_together(segmented=False, **kwargs)
    alone = problem.minimize_alone(**kwargs)

    worst = max(float((a - b).abs().max()) for a, b in zip(together, alone))
    assert worst > 1e-3, worst


def test_constant_energy_offset_does_not_change_where_a_block_stops(torch_device):
    """Convergence must not depend on a constant part of the objective.

    A kinematic minimization carries bond and angle energies it cannot change,
    which used to enter the relative tolerance and stop the run thousands of
    times too early.
    """

    def problem_with(offset):
        return BlockProblem(
            blocks=[_quadratic(0.0, offset=offset), _rosenbrock],
            starts=[
                torch.ones(8, dtype=torch.float64),
                torch.full((10,), -1.2, dtype=torch.float64),
            ],
            device=torch_device,
        )

    kwargs = dict(lr=1.0, max_iter=300, gradtol=1e-10)
    _, _, plain = problem_with(0.0).minimize_together(**kwargs)
    _, _, offset = problem_with(1.0e8).minimize_together(**kwargs)

    # the offset lives entirely in block 0, so block 1 must be untouched by it
    torch.testing.assert_close(offset[1], plain[1], rtol=1e-6, atol=1e-8)
    assert _rosenbrock(offset[1]) < 1e-4, _rosenbrock(offset[1])


def test_energy_converged_block_stops_where_it_would_alone(torch_device):
    """A block must not keep moving while a slower stack-mate finishes."""
    problem = BlockProblem(
        blocks=[_quadratic(0.0, offset=1.0e8), _rosenbrock],
        starts=[
            torch.ones(8, dtype=torch.float64),
            torch.full((10,), -1.2, dtype=torch.float64),
        ],
        device=torch_device,
    )
    kwargs = dict(lr=1.0, max_iter=50, gradtol=0.0)
    together = problem.minimize_together(**kwargs)[2]
    alone = problem.minimize_alone(**kwargs)

    for i, (in_stack, by_itself) in enumerate(zip(together, alone)):
        torch.testing.assert_close(
            in_stack, by_itself, rtol=1e-8, atol=1e-9, msg=f"block {i}"
        )


def test_noise_floor_stops_before_grinding_on_noise(torch_device):
    """The floor scales with |f|, so a huge objective stops at a coarser dE."""
    coarse = BlockProblem(
        blocks=[_quadratic(0.0, offset=1.0e8)],
        starts=[torch.ones(8, dtype=torch.float64)],
        device=torch_device,
    )
    fine = BlockProblem(
        blocks=[_quadratic(0.0)],
        starts=[torch.ones(8, dtype=torch.float64)],
        device=torch_device,
    )
    kwargs = dict(lr=1.0, max_iter=300, gradtol=0.0, rtol=0.0, atol=0.0)

    coarse_opt, coarse_x, _ = coarse.minimize_together(**kwargs)
    fine_opt, fine_x, _ = fine.minimize_together(**kwargs)

    # both stop on the floor rather than at max_iter, and the offset objective
    # stops first because 100 * eps * 1e8 is a much larger absolute change
    assert coarse_opt.state[coarse_x]["n_iter"] < 300
    assert coarse_opt.state[coarse_x]["n_iter"] <= fine_opt.state[fine_x]["n_iter"]


def test_gradtol_does_not_freeze_nonstationary_segment(torch_device):
    """The gradient criterion alone must not freeze a large-gradient segment."""
    problem = BlockProblem(
        blocks=[_quadratic(0.0), _rosenbrock],
        starts=[
            torch.ones(4, dtype=torch.float64),
            torch.full((8,), -1.2, dtype=torch.float64),
        ],
        device=torch_device,
    )
    optimizer, x, _ = problem.minimize_together(
        lr=1.0,
        max_iter=20,
        gradtol=1e-8,
        # Isolate the gradient criterion tested here. Energy convergence is
        # independently covered by test_energy_converged_block_stops_where_it_would_alone.
        rtol=0.0,
        atol=0.0,
    )

    # the quadratic block reaches its minimum; the rosenbrock valley does not
    grads = torch.split(x.grad.detach(), problem.sizes)
    assert grads[0].abs().max() <= 1e-8, grads[0]
    assert grads[1].abs().max() > 1.0, grads[1]
    assert optimizer.state[x]["converged"].tolist() == [True, False]


def test_failed_line_search_retires_only_that_segment(torch_device):
    """A segment whose line search cannot succeed is retired, not retried forever.

    Its objective reports a downhill gradient but never changes value, so no step
    is ever accepted; it backtracks to the floor and fails every time it is
    asked. Retiring it after one restart keeps it from spending a full
    backtracking sweep per iteration for the rest of the run.
    """
    problem = BlockProblem(
        blocks=[_inconsistent, _rosenbrock],
        starts=[
            torch.ones(4, dtype=torch.float64),
            torch.full((8,), -1.2, dtype=torch.float64),
        ],
        device=torch_device,
    )
    optimizer, x, finals = problem.minimize_together(lr=1.0, max_iter=50, gradtol=1e-10)
    state = optimizer.state[x]

    assert state["stalled"].tolist() == [True, False]
    # a failing segment costs a full backtracking sweep (~20 evals); once retired
    # the per-iteration cost is the healthy segment's alone
    assert state["func_evals"] < 4 * state["n_iter"], state["func_evals"]
    # and the healthy segment still reaches the minimum it reaches alone
    alone = problem.minimize_alone(lr=1.0, max_iter=50, gradtol=1e-10)
    torch.testing.assert_close(finals[1], alone[1], rtol=1e-8, atol=1e-9)


def test_gradtol_uses_gradient_magnitude(torch_device):
    """A large negative gradient component must not read as converged."""
    # minimum at x = 10, so the gradient at the start is large and negative
    start = torch.zeros(4, dtype=torch.float64, device=torch_device)

    def run(**kwargs):
        x = torch.nn.Parameter(start.clone())
        optimizer = LBFGS_Armijo([x], lr=1.0, max_iter=50, gradtol=1.0, **kwargs)

        def closure():
            optimizer.zero_grad()
            E = ((x - 10.0) ** 2).sum()
            E.backward()
            return E

        optimizer.step(closure)
        return x.detach().clone()

    segment_ids = torch.zeros(4, dtype=torch.int64, device=torch_device)
    for minimized in (run(), run(segment_ids=segment_ids)):
        torch.testing.assert_close(
            minimized, torch.full_like(minimized, 10.0), rtol=1e-6, atol=1e-6
        )


def test_stack_size_does_not_change_a_block(torch_device):
    """One block alone must follow the path it follows in a stack.

    Everything the optimizer decides -- step size, history, convergence -- has
    to be per segment for this to hold, including when there is only one. Not
    bit-identical: batched reductions tile differently for one segment than for
    four, and a rosenbrock valley amplifies the last bits.
    """
    start = torch.full((8,), -1.2, dtype=torch.float64, device=torch_device)
    kwargs = dict(lr=1.0, max_iter=60, gradtol=1e-10)

    def run(n_copies):
        problem = BlockProblem(
            blocks=[_rosenbrock] * n_copies,
            starts=[start.clone() for _ in range(n_copies)],
            device=torch_device,
        )
        return problem.minimize_together(**kwargs)[2]

    alone = run(1)[0]
    for i, in_stack in enumerate(run(4)):
        torch.testing.assert_close(
            in_stack, alone, rtol=1e-8, atol=1e-9, msg=f"copy {i}"
        )


def test_mixed_stack_does_not_change_a_block(torch_device):
    """Nor may the *other* blocks in the stack change where a block lands."""
    kwargs = dict(lr=1.0, max_iter=60, gradtol=1e-10)
    start = torch.full((8,), -1.2, dtype=torch.float64, device=torch_device)

    alone = BlockProblem([_rosenbrock], [start.clone()], torch_device)
    mixed = BlockProblem(
        [_quadratic(0.0, scale=1.0e4, offset=1.0e8), _rosenbrock],
        [torch.ones(6, dtype=torch.float64), start.clone()],
        torch_device,
    )
    torch.testing.assert_close(
        mixed.minimize_together(**kwargs)[2][1],
        alone.minimize_together(**kwargs)[2][0],
        rtol=1e-8,
        atol=1e-9,
    )
