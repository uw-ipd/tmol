"""Batched L-BFGS for docking populations.

Every pose has the same number of freed DOFs, so the whole population is one
``[pose, dof]`` tensor and each iteration is a fixed sequence of device kernels:
no host synchronisation, no data-dependent shapes. The Armijo line search
evaluates ``n_trials`` geometrically shrinking steps of every pose in one batched
call and keeps the largest step that satisfies the sufficient-decrease test,
which is the step sequential backtracking would accept. On CUDA the whole
iteration is captured once as a CUDA graph and replayed.
"""

from __future__ import annotations

import threading
from typing import Callable

import torch

from tmol.dock._lbfgs_compiled import lbfgs_direction, lbfgs_update

# Entering a torch CUDA graph synchronizes the whole device, which is illegal while
# another thread captures: captures (once per minimizer) are serialized.
_CAPTURE_LOCK = threading.Lock()

# evaluate(x[trial, pose, dof]) -> (energy[trial, pose], gradient[trial, pose, dof])
Evaluate = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


class BatchedLBFGS:
    """Minimize ``pose`` independent smooth functions of ``dof`` variables."""

    def __init__(
        self,
        evaluate: Evaluate,
        x0: torch.Tensor,
        *,
        n_trials: int = 6,
        history: int = 10,
        atol: float = 1e-2,
        patience: int = 5,
        c1: float = 1e-4,
        graph: bool = False,
    ) -> None:
        self.evaluate = evaluate
        self.n_trials, self.history = n_trials, history
        self.atol, self.patience, self.c1 = atol, patience, c1
        n_poses, n_dofs = x0.shape
        kw = dict(dtype=x0.dtype, device=x0.device)
        self.x = x0.clone()
        self.f = torch.zeros(n_poses, **kw)
        self.g = torch.zeros_like(x0)
        self.s_hist = torch.zeros(n_poses, history, n_dofs, **kw)
        self.y_hist = torch.zeros_like(self.s_hist)
        self.rho = torch.zeros(n_poses, history, **kw)
        self.gamma = torch.ones(n_poses, **kw)
        self.shrink = torch.ones(n_poses, **kw)
        self.small = torch.zeros(n_poses, dtype=torch.int64, device=x0.device)
        self.active = torch.ones(n_poses, dtype=torch.bool, device=x0.device)
        self._graph = None
        self.trace: list = []
        self._reset(x0)
        if graph:
            self._capture()

    def _reset(self, x0: torch.Tensor) -> None:
        """Start a minimization from ``x0``: fresh history, energy and gradient."""
        self.x.copy_(x0)
        for state in (self.s_hist, self.y_hist, self.rho):
            state.zero_()
        self.gamma.fill_(1)
        self.shrink.fill_(1)
        self.small.zero_()
        self.active.fill_(True)
        energy, gradient = self.evaluate(self.x[None].expand(self.n_trials, -1, -1))
        self.f.copy_(energy[0])
        self.g.copy_(gradient[0])

    def _iterate(self) -> None:
        """One L-BFGS iteration: fused direction, batched line-search trials, fused update."""
        trial_x, alpha, gd = lbfgs_direction(
            self.x, self.g, self.s_hist, self.y_hist, self.rho, self.gamma,
            self.shrink, self.n_trials,
        )
        trial_f, trial_g = self.evaluate(trial_x)
        lbfgs_update(
            self.x, self.f, self.g, self.s_hist, self.y_hist, self.rho, self.gamma,
            self.shrink, self.small, self.active, trial_x, trial_f, trial_g, alpha,
            gd, self.c1, self.atol, self.patience,
        )

    def _capture(self) -> None:
        with _CAPTURE_LOCK:
            self._capture_locked()

    def _capture_locked(self) -> None:
        start = self.x.clone()
        stream = torch.cuda.Stream(device=self.x.device)
        stream.wait_stream(torch.cuda.current_stream(self.x.device))
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(3):  # warm up lazily built state outside the capture
                self._iterate()
        torch.cuda.current_stream(self.x.device).wait_stream(stream)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            self._graph, stream=stream, capture_error_mode="thread_local"
        ), torch.no_grad():
            self._iterate()
        self._start = start

    def minimize(self, x0: torch.Tensor, max_iter: int, check_every: int = 10) -> torch.Tensor:
        """Run up to ``max_iter`` iterations from ``x0``; returns the final DOFs."""
        with torch.no_grad():
            self._reset(x0)
            for iteration in range(1, max_iter + 1):
                if self._graph is not None:
                    self._graph.replay()
                else:
                    self._iterate()
                if iteration % check_every == 0:
                    n_active = int(self.active.sum())
                    self.trace.append((iteration, n_active))
                    if n_active == 0:
                        break
        return self.x.clone()
