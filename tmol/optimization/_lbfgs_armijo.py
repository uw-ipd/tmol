import torch
from types import SimpleNamespace
from torch.optim import Optimizer


def lbfgs_two_loop(grad, dirs, stps):
    """L-BFGS search direction H_k @ grad via the compact
    representation of Byrd, Nocedal & Schnabel, (Math. Prog. 63 (1994)):
        H_0 = I
        M = [[ R^-T (D + Y^T Y) R^-1, -R^-T ], [ -R^-1, 0 ]]
        H_k g = g + [S Y] M [S^T g ; Y^T g]

    Algebraically identical to the classic two-loop recursion, but all O(N*m)
    ops are parallelized.

    grad is (N) or (k, N) and dirs/stps are (m, N) or (m, k, N); in the latter
    case each of the k batches gets its own inverse-Hessian estimate.
    """
    unbatched = grad.dim() == 1
    if unbatched:
        grad = grad.unsqueeze(0)
        dirs = dirs.unsqueeze(1)
        stps = stps.unsqueeze(1)

    out_dtype = grad.dtype
    # Triangular solves are:
    #  a) imprecise on CUDA at float32
    #  b) small matrices (M x M) where M is history length (typically 128)
    # promote to float64
    S = stps.double()
    Y = dirs.double()
    g = -grad.double()
    a = torch.einsum("ipk,pk->pi", S, g)  # a_i = s_i . g
    b = torch.einsum("ipk,pk->pi", Y, g)  # b_i = y_i . g
    SY = torch.einsum("ipk,jpk->pij", S, Y)  # SY_ij = s_i . y_j
    YY = torch.einsum("ipk,jpk->pij", Y, Y)  # YY_ij = y_i . y_j
    R = torch.triu(SY)  # upper-triangular incl. diagonal
    D = SY.diagonal(dim1=-2, dim2=-1)  # D_i = s_i . y_i

    # An absent slot has an all-zero row and column; a unit diagonal there keeps
    # R invertible and leaves the search direction unchanged.
    R = R + torch.diag_embed((D == 0).to(R.dtype))

    # u = R^-1 a
    u = torch.linalg.solve_triangular(R, a.unsqueeze(-1), upper=True).squeeze(-1)
    # v = (D + Y^T Y) u - b
    v = torch.einsum("pij,pj->pi", YY, u) + D * u - b
    # p1 = R^-T v
    p1 = torch.linalg.solve_triangular(
        R.transpose(-2, -1), v.unsqueeze(-1), upper=False
    ).squeeze(-1)
    p2 = -u

    # result = g + S p1 + Y p2
    result = g + torch.einsum("pi,ipk->pk", p1, S) + torch.einsum("pi,ipk->pk", p2, Y)
    result = result.to(out_dtype)
    return result.squeeze(0) if unbatched else result


# per-segment line search states
_LS_DONE = 0
_LS_INCREASE = 1  # the step looked linear; trying a longer one
_LS_BACKTRACK = 2
_LS_FAILED = 3


def armijo_linesearch_segmented(
    func,
    derphi0,
    old_fval,
    alpha0,
    searching,
    factor=0.5,
    sigma_decrease=0.1,
    sigma_increase=0.8,
    minstep=1e-6,
):
    """Minimize over alpha, the function ``f(xk+alpha pk)``.

    Each segment gets its own alpha; one call to f evaluates all of them.

    Arguments:
        f (callable): Function to be minimized, f(step)
        derphi0 : (Tensor) directional derivative, per segment
        fval0 : (Tensor) func(0), the value of the function at the origin
        alpha0 : (Tensor) the initial stepsize, per segment
        searching : (Tensor) which segments take part in the search
        sigma_increase : (float) initial stepsize
                         [must be in (0,1) and >=sigma_decrease]
        sigma_decrease : (float) initial stepsize
                         [must be in (0,1) and <=sigma_increase]
        factor : (float) scalefactor in modifying stepsize [must be in (0,1)]
        minstep : (float) minimum stepsize to take

    Returns:
        stepsize - accepted stepsize, per segment
        f_val - final function value, per segment
        n_evals - number of calls to f
        status - per segment, one of the _LS_ codes
        trial_is_accepted - whether the last trial evaluated the accepted steps

    Notes
        See D.P. Bertsekas, Nonlinear Programming, 2nd ed, 1999, page 29.

        (fd) A few notes about this specific implementation:
        0) I believe this method was originally from Jim Havranek
        1) 'factor' corresponds roughly to 'beta', BUT on a successful initial step,
           factor is used to increase the stepsize.  When factor is used to decrease
           stepsize, factor^2 is used
        2) The stopping critera used is that in the paper, the first integer m>=0 s.t.:
             f(x_k) - f(x_k+beta^m*s*d_k) >= -sigma * beta^m * s * grad{f}(x_k) * d_k
           however, the two different values of sigma are used:
              * sigma_increase (0.8) is used to trigger an _increased_ stepsize
              * sigma_decrease (0.1) is _required_ or the step size is decreased
        3) in the code
              * 'alpha' corresponds to 's' in the text
              * 'factor' corresponds roughly to 'beta' in the text (see point 1)
    """
    phi0 = old_fval
    zeros = torch.zeros_like(alpha0)
    alpha = torch.where(searching, alpha0, zeros)
    # no step until one is accepted, so a rejected trial is never applied
    accepted = zeros.clone()
    phi_accepted = phi0.clone()
    status = torch.full(alpha0.shape, _LS_DONE, dtype=torch.int64, device=alpha0.device)

    phi = func(alpha)
    n_evals = 1

    # sigma_increase > sigma_decrease, so a linear-looking step also has
    # sufficient decrease; it is the one case where a longer step is tried
    linear = phi <= phi0 + alpha * sigma_increase * derphi0
    sufficient = phi <= phi0 + alpha * sigma_decrease * derphi0
    status = torch.where(searching & linear, _LS_INCREASE, status)
    status = torch.where(searching & ~linear & ~sufficient, _LS_BACKTRACK, status)
    took = searching & (linear | sufficient)
    accepted = torch.where(took, alpha, accepted)
    phi_accepted = torch.where(took, phi, phi_accepted)

    while True:
        active = (status == _LS_INCREASE) | (status == _LS_BACKTRACK)
        if not bool(active.any()):
            break

        trial = torch.where(status == _LS_INCREASE, alpha / factor, accepted)
        # see note above, decrease by factor^2
        trial = torch.where(status == _LS_BACKTRACK, alpha * factor * factor, trial)
        phi_trial = func(trial)
        n_evals += 1

        # longer step: keep it only if it beats the step we already have
        increasing = status == _LS_INCREASE
        better = increasing & (phi_trial < phi)
        accepted = torch.where(better, trial, accepted)
        phi_accepted = torch.where(better, phi_trial, phi_accepted)
        status = torch.where(increasing, _LS_DONE, status)

        backtracking = status == _LS_BACKTRACK
        armijo = backtracking & (phi_trial <= phi0 + trial * sigma_decrease * derphi0)
        accepted = torch.where(armijo, trial, accepted)
        phi_accepted = torch.where(armijo, phi_trial, phi_accepted)
        status = torch.where(armijo, _LS_DONE, status)

        # under the floor: accept anything that still went downhill, else fail
        floored = backtracking & ~armijo & (trial < minstep)
        downhill = floored & (phi_trial < phi0)
        accepted = torch.where(downhill, trial, accepted)
        phi_accepted = torch.where(downhill, phi_trial, phi_accepted)
        status = torch.where(downhill, _LS_DONE, status)
        failed = floored & ~downhill
        accepted = torch.where(failed, zeros, accepted)
        phi_accepted = torch.where(failed, phi0, phi_accepted)
        status = torch.where(failed, _LS_FAILED, status)
        for p in failed.nonzero(as_tuple=False).flatten().tolist():
            step = float(trial[p])
            finite = (
                (float(phi_trial[p]) - float(phi0[p])) / step if step else float("inf")
            )
            print(
                "Inaccurate G! Segment=",
                p,
                " Step=",
                step,
                " Deriv=",
                float(derphi0[p]),
                " Finite=",
                finite,
            )

        alpha = trial
        phi = phi_trial

    return accepted, phi_accepted, n_evals, status, bool(torch.equal(alpha, accepted))


class LBFGS_Armijo(Optimizer):
    """
    Implements L-BFGS algorithm with Armijo line search.
    All scaling and parameters taken directly from Rosetta

    Parameters:
        lr (float): learning rate (default: 1)
        max_iter (int): maximal number of iterations (default: 200)
        rtol (float): relative tolerance (default: 1e-6)
        atol (float): absolute tolerance (default: 0)
        gradtol (float): an absolute tolerance on max_i |df/dx_i| (default: 1)
        history_size (int): update history size (default: 128).
        segment_ids (Tensor): the segment (e.g. pose) each parameter element
            belongs to; each segment is minimized independently (default: one)
    """

    supports_segments = True

    def __init__(
        self,
        params,
        lr=1,
        max_iter=200,
        rtol=None,  # None => dtype-based default
        atol=None,  # None => dtype-based default
        gradtol=1.0,
        history_size=128,
        minstep=1e-12,
        verbose=False,
        segment_ids=None,
    ):
        defaults = dict(
            lr=lr,
            max_iter=max_iter,
            atol=atol,
            rtol=rtol,
            gradtol=gradtol,
            history_size=history_size,
        )
        super(LBFGS_Armijo, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError(
                "LBFGS doesn't support per-parameter options " "(parameter groups)"
            )

        self._params = self.param_groups[0]["params"]
        self._minstep = minstep
        self.verbose = verbose

        self._last_loss_vec = None
        self._sum_scratch = None
        if segment_ids is None:
            # the whole parameter as one segment: the same code path as a stack
            segment_ids = torch.zeros(
                self._params[0].numel(),
                dtype=torch.int64,
                device=self._params[0].device,
            )
        self._init_segments(segment_ids)

    def _init_segments(self, segment_ids):
        """Set up the mapping from parameter elements to independent blocks.

        Builds the index that scatters a parameter-shaped vector into a dense
        (n_segments, segment_size) layout, so all per-segment reductions are
        batched tensor ops rather than a loop over segments.
        """
        assert len(self._params) == 1, "segment_ids requires a single tensor"
        param = self._params[0]
        segment_ids = segment_ids.reshape(-1).to(torch.int64)
        assert (
            segment_ids.numel() == param.numel()
        ), "segment_ids needs one entry per parameter element"
        assert bool((segment_ids >= 0).all()), "segment_ids must be non-negative"

        n_segments = int(segment_ids.max().item()) + 1
        counts = torch.bincount(segment_ids, minlength=n_segments)
        assert bool((counts > 0).all()), "segment_ids must not skip a segment"
        segment_size = int(counts.max().item())

        # rank of each element within its own segment
        order = torch.argsort(segment_ids, stable=True)
        offsets = torch.cumsum(counts, 0) - counts
        rank = torch.empty_like(order)
        arange = torch.arange(segment_ids.numel(), device=order.device)
        rank[order] = arange - offsets[segment_ids[order]]

        self._segment_ids = segment_ids
        self._n_segments = n_segments
        self._segment_size = segment_size
        self._pad_index = segment_ids * segment_size + rank

    def _seg_sum(self, values):
        """Sum a parameter-shaped vector within each segment."""
        if self._sum_scratch is None or self._sum_scratch.dtype != values.dtype:
            self._sum_scratch = torch.zeros(
                self._n_segments * self._segment_size,
                dtype=values.dtype,
                device=values.device,
            )
        return self._pad(values, out=self._sum_scratch).sum(-1)

    def _seg_amax(self, values):
        """Maximum of a parameter-shaped vector within each segment."""
        out = torch.empty(self._n_segments, dtype=values.dtype, device=values.device)
        return out.scatter_reduce_(
            0, self._segment_ids, values, "amax", include_self=False
        )

    def _pad(self, values, out=None):
        """Scatter a parameter-shaped vector to (n_segments, segment_size)."""
        if out is None:
            out = torch.zeros(
                self._n_segments * self._segment_size,
                dtype=values.dtype,
                device=values.device,
            )
        else:
            out = out.view(-1)
            out.zero_()
        out[self._pad_index] = values
        return out.view(self._n_segments, self._segment_size)

    def _unpad(self, padded):
        """Gather a (n_segments, segment_size) tensor back to parameter shape."""
        return padded.reshape(-1)[self._pad_index]

    def _wrap_closure(self, closure):
        """Reduce a per-segment closure to a scalar, keeping the vector around.

        A closure may return one energy per segment; the line search needs the
        total, while the convergence tests want the per-segment values.
        """

        def wrapped():
            loss = closure()
            if (
                self._segment_ids is not None
                and torch.is_tensor(loss)
                and loss.numel() == self._n_segments
            ):
                self._last_loss_vec = loss.detach().reshape(self._n_segments)
                return loss.sum()
            self._last_loss_vec = None
            return loss

        return wrapped

    def _step_setup(self, closure):
        """Prepare for L-BFGS:
        read config
        initialize state (history buffers, x_ref, preallocate d and x_backup)
        evaluate initial loss and gradient
        returns a SimpleNamespace ctx"""
        # lbfgs only works w/ single parameter group
        assert len(self.param_groups) == 1
        assert len(self._params) == 1, "This version requires single tensor"

        group = self.param_groups[0]
        lr = group["lr"]
        max_iter = group["max_iter"]
        rtol = group["rtol"]
        atol = group["atol"]
        gradtol = group["gradtol"]
        history_size = group["history_size"]

        # dtype-based default
        #   float32 : eps~3.45e-4
        #   float64 : eps~1.49e-8
        dtype_based_tol = float(torch.finfo(self._params[0].dtype).eps ** 0.5)
        if rtol is None:
            rtol = dtype_based_tol
        if rtol < dtype_based_tol:
            print(f"  WARNING: rtol ({rtol}) is too low for dtype! ({dtype_based_tol})")
        if atol is None:
            atol = dtype_based_tol
        if atol < dtype_based_tol:
            print(f"  WARNING: atol ({atol}) is too low for dtype! ({dtype_based_tol})")

        param = self._params[0]

        # NOTE: LBFGS has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        state = self.state[param]
        state.setdefault("func_evals", 0)
        state.setdefault("n_iter", 0)

        # evaluate initial f(x)
        orig_loss = closure()
        loss = orig_loss.item()
        loss_vec = self._last_loss_vec
        state["func_evals"] += 1

        x = param.data.view(-1)
        flat_grad = param.grad.data.view(-1)

        # Preallocate on first call. `d` and `x_backup` must have stable
        # tensor identity so linefn (defined once in step) can capture them.
        L = x.numel()
        if state.get("d") is None:
            state["d"] = torch.empty_like(x)
            state["x_backup"] = torch.empty_like(x)
            if self._segment_ids is None:
                hist_shape = (history_size, L)
            else:
                hist_shape = (history_size, self._n_segments, self._segment_size)
                # scratch for the padded gradient handed to the two-loop
                state["grad_pad"] = torch.zeros(
                    hist_shape[1:], device=x.device, dtype=x.dtype
                )
            # zero-filled: unwritten slots and padding must not contribute
            state["old_dirs_mat"] = torch.zeros(
                hist_shape, device=x.device, dtype=x.dtype
            )
            state["old_stps_mat"] = torch.zeros(
                hist_shape, device=x.device, dtype=x.dtype
            )
            state["history_start"] = 0  # Circular buffer start index
            state["history_count"] = 0  # Number of items in history
            state["x_ref"] = x.clone()  # reference position for s computation
            flags = dict(dtype=torch.bool, device=x.device)
            # stationary segments; segments that gave up; segments owed a
            # steepest-descent restart after a failed line search
            state["converged"] = torch.zeros(self._n_segments, **flags)
            state["stalled"] = torch.zeros(self._n_segments, **flags)
            state["needs_reset"] = torch.zeros(self._n_segments, **flags)
            state["was_reset"] = torch.zeros(self._n_segments, **flags)

        return SimpleNamespace(
            # config
            max_iter=max_iter,
            lr=lr,
            rtol=rtol,
            atol=atol,
            gradtol=gradtol,
            history_size=history_size,
            # torch / state
            state=state,
            param=param,
            x=x,
            flat_grad=flat_grad,
            # preallocated scratch (stable refs captured by linefn)
            d=state["d"],
            x_backup=state["x_backup"],
            # cached across steps
            t=state.get("t"),
            prev_flat_grad=state.get("prev_flat_grad"),
            prev_loss=state.get("prev_loss"),
            prev_loss_vec=state.get("prev_loss_vec"),
            # history
            old_dirs_mat=state["old_dirs_mat"],
            old_stps_mat=state["old_stps_mat"],
            history_start=state["history_start"],
            history_count=state["history_count"],
            x_ref=state["x_ref"],
            converged=state["converged"],
            stalled=state["stalled"],
            needs_reset=state["needs_reset"],
            was_reset=state["was_reset"],
            gtd_seg=None,
            # current eval
            orig_loss=orig_loss,
            loss=loss,
            loss_vec=loss_vec,
            # line-search accounting
            ls_evals=0,
        )

    def _compute_search_direction(self, ctx):
        """L-BFGS update + two-loop recursion."""
        flat_grad = ctx.flat_grad
        d = ctx.d
        x = ctx.x

        if ctx.state["n_iter"] == 1:
            # initialize
            d.copy_(flat_grad).neg_()
            ctx.history_count = 0
        else:
            # do lbfgs update (update memory)
            y = flat_grad.sub(ctx.prev_flat_grad)
            s = x.sub(ctx.x_ref)  # cumulative displacement since last good step
            # a segment with no curvature of its own contributes no history
            keep = self._seg_sum(y * s) > 1e-6

            if bool(keep.any()):
                # updating memory - write directly into circular buffer
                if ctx.history_count < ctx.history_size:
                    # Still filling up the buffer
                    idx = ctx.history_count
                    ctx.history_count += 1
                else:
                    # Buffer full, overwrite oldest entry
                    idx = ctx.history_start
                    ctx.history_start = (ctx.history_start + 1) % ctx.history_size

                # advance the reference only for segments that took a good step
                keep_elem = keep[self._segment_ids]
                zero = torch.zeros((), dtype=y.dtype, device=y.device)
                self._pad(torch.where(keep_elem, y, zero), out=ctx.old_dirs_mat[idx])
                self._pad(torch.where(keep_elem, s, zero), out=ctx.old_stps_mat[idx])
                ctx.x_ref = torch.where(keep_elem, x, ctx.x_ref)

            # compute the approximate (L-BFGS) inverse Hessian
            if ctx.history_count == 0:
                # No history: use steepest descent direction
                d.copy_(flat_grad).neg_()
            else:
                # Create views old -> new
                if ctx.history_count < ctx.history_size:
                    old_dirs_view = ctx.old_dirs_mat[: ctx.history_count]
                    old_stps_view = ctx.old_stps_mat[: ctx.history_count]
                else:
                    # Buffer full, need to reorder: [start:end] + [0:start]
                    indices = torch.cat(
                        [
                            torch.arange(
                                ctx.history_start, ctx.history_size, device=x.device
                            ),
                            torch.arange(0, ctx.history_start, device=x.device),
                        ]
                    )
                    old_dirs_view = ctx.old_dirs_mat[indices]
                    old_stps_view = ctx.old_stps_mat[indices]

                grad_pad = self._pad(flat_grad, out=ctx.state["grad_pad"])
                d.copy_(
                    self._unpad(lbfgs_two_loop(grad_pad, old_dirs_view, old_stps_view))
                )

        self._restart_failed_segments(ctx)
        self._freeze_converged(ctx)

    def _restart_failed_segments(self, ctx):
        """Give a segment whose line search failed a fresh steepest descent.

        Its history is dropped (zeroed slots read as absent) and it is marked as
        restarted, so a second failure retires it instead of searching again.
        """
        if not bool(ctx.needs_reset.any()):
            ctx.was_reset = torch.zeros_like(ctx.needs_reset)
            return
        ctx.was_reset = ctx.needs_reset.clone()
        reset = ctx.needs_reset.nonzero(as_tuple=False).squeeze(-1)
        ctx.old_dirs_mat[:, reset, :] = 0.0
        ctx.old_stps_mat[:, reset, :] = 0.0
        reset_elem = ctx.needs_reset[self._segment_ids]
        ctx.d.copy_(torch.where(reset_elem, -ctx.flat_grad, ctx.d))
        ctx.x_ref = torch.where(reset_elem, ctx.x, ctx.x_ref)
        ctx.needs_reset = torch.zeros_like(ctx.needs_reset)

    def _inactive(self, ctx):
        """Segments that no longer move: converged or retired."""
        return ctx.converged | ctx.stalled

    def _freeze_converged(self, ctx):
        """Hold converged or retired segments still by zeroing their direction."""
        inactive = self._inactive(ctx)
        if not bool(inactive.any()):
            return
        ctx.d.mul_((~inactive).to(ctx.d.dtype)[self._segment_ids])

    def _directional_derivative(self, ctx, correct=True):
        """Directional derivative g . d, per segment and in total."""
        flat_grad, d = ctx.flat_grad, ctx.d
        gtd_seg = self._seg_sum(flat_grad * d)
        if correct:
            for check in (1, 2):
                bad = (gtd_seg > -1e-5) & ~self._inactive(ctx)
                if not bool(bad.any()):
                    break
                bad_elem = bad[self._segment_ids]
                if check == 1:
                    repaired = d * -torch.sign(flat_grad * d)
                else:
                    repaired = -flat_grad
                d.copy_(torch.where(bad_elem, repaired, d))
                gtd_seg = self._seg_sum(flat_grad * d)
        ctx.gtd_seg = gtd_seg
        return gtd_seg.sum().item()

    def _segment_line_search(self, ctx, linefn_vec):
        """Line search with an independent step size per segment."""
        # a segment with no downhill direction cannot search; any negative
        # slope is worth searching, however small, since the Armijo test
        # scales with it
        searching = ~self._inactive(ctx) & (ctx.gtd_seg < 0)
        # match Rosetta: start at 2x prev accepted step, capped at 1.0
        start_t = (ctx.t / 0.5).clamp(max=1.0)

        accepted, _, ls_evals, status, trial_is_accepted = armijo_linesearch_segmented(
            linefn_vec,
            ctx.gtd_seg,
            ctx.prev_loss_vec,  # energies at x_backup
            start_t,
            searching,
            factor=0.5,
            sigma_decrease=0.1,
            sigma_increase=0.8,
            minstep=self._minstep,
        )
        ctx.ls_evals = ls_evals

        ctx.x.copy_(ctx.x_backup).add_(ctx.d * accepted[self._segment_ids])
        if not trial_is_accepted:
            self._closure_fn()
        ctx.loss_vec = self._last_loss_vec
        ctx.loss = float(ctx.loss_vec.sum())

        # a failed search gets one steepest-descent restart, applied on the next
        # iteration so the other segments are not held up; then it is retired
        failed = status == _LS_FAILED
        # restart step length, as in the unsegmented rescue: 1/sqrt(|g.d|)
        retry_t = torch.clamp((-ctx.gtd_seg).clamp(min=self._minstep).rsqrt(), max=1.0)
        ctx.t = torch.where(failed, retry_t, accepted)
        ctx.t = torch.where(searching, ctx.t, start_t)
        if bool(failed.any()):
            ctx.stalled |= failed & ctx.was_reset
            ctx.needs_reset |= failed & ~ctx.was_reset

    def _check_segment_convergence(self, ctx, n_iter):
        """Freeze independently converged segments; stop when all are done.

        A segmented run must stop each segment at the same iteration where an
        equivalent one-segment run would stop. Otherwise an early-converged pose
        keeps moving while another pose finishes, making batch minimization
        depend on which other structures happen to share the stack.
        """
        newly_converged = self._seg_amax(ctx.flat_grad.abs()) <= ctx.gradtol
        if ctx.prev_loss_vec is not None:
            dE = (ctx.loss_vec - ctx.prev_loss_vec).abs()
            rdiff = 2 * dE / (ctx.loss_vec.abs() + ctx.prev_loss_vec.abs() + 1e-10)
            energy_converged = (dE <= ctx.atol) | (rdiff <= ctx.rtol)
            # A rejected zero step also has dE == 0, but it is not convergence:
            # the next iteration must try the scheduled steepest-descent reset.
            newly_converged |= energy_converged & ~ctx.needs_reset
        ctx.converged |= newly_converged
        done = self._inactive(ctx)

        n_done = int(done.sum().item())
        n_stalled = int(ctx.stalled.sum().item())
        if self.verbose:
            print(
                f"  iter {n_iter:4d}  E={ctx.loss:.6f}"
                f"  evals={ctx.ls_evals}"
                f"  done={n_done}/{self._n_segments}"
                + (f"  stalled={n_stalled}" if n_stalled else "")
            )
        if n_done == self._n_segments:
            if self.verbose:
                print(
                    f"  finished: {self._n_segments - n_stalled} converged,"
                    f" {n_stalled} stalled"
                )
            return True
        return False

    def step(self, closure):
        """
        The LBFGS minimization algorithm. Despite the name, this performs the full
        LBFGS minimization trajectory.

        Arguments:
            func (callable): a function that evaluates energy

        Returns:
            orig_loss: the energy (loss) following optimization
        """
        closure = self._wrap_closure(closure)
        self._closure_fn = closure
        ctx = self._step_setup(closure)

        x = ctx.x
        x_backup = ctx.x_backup
        d = ctx.d

        def linefn(alpha_vec):
            """Evaluate every segment at its own step size."""
            self.ls_func_evals += 1
            # Direct parameter update - eliminates _set_x_from_flat overhead
            x.copy_(x_backup).add_(d * alpha_vec[self._segment_ids])
            closure()
            return self._last_loss_vec

        current_evals = 1
        n_iter = 0
        while n_iter < ctx.max_iter:
            n_iter += 1
            ctx.state["n_iter"] += 1

            self._compute_search_direction(ctx)

            if ctx.prev_flat_grad is None:
                ctx.prev_flat_grad = ctx.flat_grad.clone()
            else:
                ctx.prev_flat_grad.copy_(ctx.flat_grad)
            ctx.prev_loss = ctx.loss
            ctx.prev_loss_vec = ctx.loss_vec

            # Armijo updates will track step length during optimization
            # thus, "learning rate" is only applied for the initial step
            if ctx.state["n_iter"] == 1:
                ctx.t = torch.full(
                    (self._n_segments,), ctx.lr, dtype=x.dtype, device=x.device
                )

            # directional derivative
            # (fd) this is some hacky stuff I put in R3 that is not typically part
            # (fd)   of lbfgs because the bfgs update had us frequently searching
            # (fd)   in positive grad directions
            # check 1: if dir. deriv. is positive, flip signs of positive components
            # check 2: if derivative is still positive, reset Hessian
            self._directional_derivative(ctx)

            # define the line search function
            # we do not need to compute gradients in here
            self.ls_func_evals = 0

            # Optimization: save original position and work directly with param.data
            x_backup.copy_(x)

            self._segment_line_search(ctx, linefn)

            ctx.flat_grad = ctx.param.grad.data.view(-1)  # Direct reference

            # update func eval
            current_evals += self.ls_func_evals
            ctx.state["func_evals"] += self.ls_func_evals

            if self._check_segment_convergence(ctx, n_iter):
                break

        if self.verbose:
            print(
                f"  LBFGS_Armijo done: {n_iter} iters,"
                f" {current_evals} func evals,"
                f" E={ctx.loss:.4f}"
            )

        # d and x_backup persist via preallocation in state; save the rest.
        ctx.state["t"] = ctx.t
        ctx.state["history_start"] = ctx.history_start
        ctx.state["history_count"] = ctx.history_count
        ctx.state["prev_flat_grad"] = ctx.prev_flat_grad
        ctx.state["prev_loss"] = ctx.prev_loss
        ctx.state["prev_loss_vec"] = ctx.prev_loss_vec
        ctx.state["x_ref"] = ctx.x_ref
        ctx.state["needs_reset"] = ctx.needs_reset
        ctx.state["was_reset"] = ctx.was_reset

        return ctx.orig_loss
