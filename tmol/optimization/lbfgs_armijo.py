import torch
from functools import reduce
from types import SimpleNamespace
from torch.optim import Optimizer


def armijo_linesearch(
    func,
    derphi0,
    old_fval,
    alpha0=1,
    factor=0.5,
    sigma_decrease=0.1,
    sigma_increase=0.8,
    minstep=1e-6,
):
    """Minimize over alpha, the function ``f(xk+alpha pk)``.

    Arguments:
        f (callable): Function to be minimized, f(step)
        derphi0 : (float) directional derivative
        fval0 : (float) func(0), the value of the function at the origin
        alpha0 : (float) the initial stepsize
        sigma_increase : (float) initial stepsize
                         [must be in (0,1) and >=sigma_decrease]
        sigma_decrease : (float) initial stepsize
                         [must be in (0,1) and <=sigma_increase]
        factor : (float) scalefactor in modifying stepsize [must be in (0,1)]
        minstep : (float) minimum stepsize to take

    Returns:
        stepsize - accepted stepsize
        f_val - final function value

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
    # evaluate phi(0) if not input
    n_evals = 0
    if old_fval is None:
        phi0 = func(0.0)
        n_evals += 1
    else:
        phi0 = old_fval

    # check armijo condition
    phi_a0 = func(alpha0)
    n_evals += 1

    # first, we check if we can increase the stepsize
    #     (if the func is still behaving linearly)

    if phi_a0 <= phi0 + alpha0 * sigma_increase * derphi0:
        # attempt to increase stepsize
        alpha1 = alpha0 / factor
        phi_a1 = func(alpha1)
        n_evals += 1
        if phi_a1 < phi_a0:
            return alpha1, phi_a1, n_evals, "increased"

        # step back
        return alpha0, phi_a0, n_evals, "step_back"

    # next, we check if we need to decrease the stepsize
    alpha1 = alpha0
    phi_a1 = phi_a0
    while phi_a1 > phi0 + alpha1 * sigma_decrease * derphi0:
        # (fd) check for search failure.  In R3, "Inaccurate G!" is reported
        # (fd) I have made a few modifications to this from R3
        #      (1) there is no relative stepsize check, only an absolute one
        #          (R3 checks that step is >=1e-5 times orig step and >=1e-12)
        #      (2) if the search fails to satisfy Armijo cond, but decreases
        #          the function at min stepsize, accept the step
        # (fd) I think change (1) is hit reasonably often in R3
        #      (and while usually bad is not necessarily so, particularly on 1st step)
        # (fd) Change (2) probably is infrequent (and might slow things down?)
        if alpha1 < minstep:
            if phi_a1 >= phi0:
                finite_diff = (phi_a1 - phi0) / alpha1 if alpha1 != 0 else float("inf")
                print(
                    "Inaccurate G! Step=",
                    alpha1,
                    " Deriv=",
                    derphi0,
                    " Finite=",
                    finite_diff,
                )
                return 0.0, phi0, n_evals, "inaccurate_G"
            return alpha1, phi_a1, n_evals, "minstep"

        alpha1 *= factor * factor  # see note above, decrease by factor^2
        phi_a1 = func(alpha1)
        n_evals += 1

    return alpha1, phi_a1, n_evals, "armijo"


class LBFGS_Armijo(Optimizer):
    """
    Implements L-BFGS algorithm with Armijo line search.
    All scaling and parameters taken directly from Rosetta

    Parameters:
        lr (float): learning rate (default: 1)
        max_iter (int): maximal number of iterations (default: 200)
        rtol (float): relative tolerance (default: 1e-6)
        atol (float): absolute tolerance (default: 0)
        gradtol (float): an absolute tolerance on max_i df/dx_i (default: 1e-4)
        history_size (int): update history size (default: 128).
    """

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
        self._numel_cache = None
        self._minstep = minstep
        self.verbose = verbose

    # LBFGS (as implemented) treats parameter groups all equally
    # * the following wrapper functions package parameters as a single param
    # * this code is based off PyTorch default LBFGS implementation
    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(
                lambda total, p: total + p.numel(), self._params, 0
            )
        return self._numel_cache

    # pack gradients into a single flat tensor
    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    # pack the current location into a single flat tensor
    def _gather_flat_x(self):
        views = []
        for p in self._params:
            if p.data.is_sparse:
                view = p.data.to_dense().view(-1)
            else:
                view = p.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    # unpack a new location
    def _set_x_from_flat(self, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            if p.data.is_sparse:
                p.data.copy_(
                    update[offset : offset + numel].view_as(p.data).to_sparse()
                )
            else:
                # view as to avoid deprecated pointwise semantics
                p.data.copy_(update[offset : offset + numel].view_as(p.data))
            offset += numel
        assert offset == self._numel()

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
        state["func_evals"] += 1

        x = param.data.view(-1)
        flat_grad = param.grad.data.view(-1)

        # Preallocate on first call. `d` and `x_backup` must have stable
        # tensor identity so linefn (defined once in step) can capture them.
        L = x.numel()
        if state.get("d") is None:
            state["d"] = torch.empty_like(x)
            state["x_backup"] = torch.empty_like(x)
            state["old_dirs_mat"] = torch.empty(
                (history_size, L), device=x.device, dtype=x.dtype
            )
            state["old_stps_mat"] = torch.empty(
                (history_size, L), device=x.device, dtype=x.dtype
            )
            state["history_start"] = 0  # Circular buffer start index
            state["history_count"] = 0  # Number of items in history
            state["x_ref"] = x.clone()  # reference position for s computation

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
            # history
            old_dirs_mat=state["old_dirs_mat"],
            old_stps_mat=state["old_stps_mat"],
            history_start=state["history_start"],
            history_count=state["history_count"],
            x_ref=state["x_ref"],
            # current eval
            orig_loss=orig_loss,
            loss=loss,
            # line-search accounting
            ls_evals=0,
            status=None,
        )

    def _compute_search_direction(self, ctx):
        """L-BFGS update + two-loop recursion."""
        from tmol.optimization.compiled import lbfgs_two_loop as _lbfgs_two_loop_op

        flat_grad = ctx.flat_grad
        d = ctx.d
        x = ctx.x

        if ctx.state["n_iter"] == 1:
            # initialize
            d.copy_(flat_grad).neg_()
            ctx.history_count = 0
            return

        # do lbfgs update (update memory)
        y = flat_grad.sub(ctx.prev_flat_grad)
        s = x.sub(ctx.x_ref)  # cumulative displacement since last good step
        ys = y.dot(s)  # y*s
        if ys.item() > 1e-6:
            # updating memory - write directly into circular buffer
            if ctx.history_count < ctx.history_size:
                # Still filling up the buffer
                idx = ctx.history_count
                ctx.history_count += 1
            else:
                # Buffer full, overwrite oldest entry
                idx = ctx.history_start
                ctx.history_start = (ctx.history_start + 1) % ctx.history_size

            ctx.old_dirs_mat[idx].copy_(y)
            ctx.old_stps_mat[idx].copy_(s)
            ctx.x_ref = x.clone()  # advance reference only on good steps

        # compute the approximate (L-BFGS) inverse Hessian
        if ctx.history_count == 0:
            # No history: use steepest descent direction
            d.copy_(flat_grad).neg_()
            return

        # Create views old -> new
        if ctx.history_count < ctx.history_size:
            old_dirs_view = ctx.old_dirs_mat[: ctx.history_count]
            old_stps_view = ctx.old_stps_mat[: ctx.history_count]
        else:
            # Buffer full, need to reorder: [start:end] + [0:start]
            indices = torch.cat(
                [
                    torch.arange(ctx.history_start, ctx.history_size, device=x.device),
                    torch.arange(0, ctx.history_start, device=x.device),
                ]
            )
            old_dirs_view = ctx.old_dirs_mat[indices]
            old_stps_view = ctx.old_stps_mat[indices]

        d.copy_(_lbfgs_two_loop_op(flat_grad, old_dirs_view, old_stps_view))

    def _rescue_failed_linesearch(self, ctx, linefn, n_iter):
        """Handle t==0.0 failure: reset L-BFGS history and retry with
        steepest descent at step 1/sqrt(|g|).  Returns True on fail."""
        ctx.history_count = 0
        ctx.history_start = 0
        ctx.x_ref = ctx.x.clone()  # reset reference position with history
        ctx.d.copy_(ctx.flat_grad).neg_()
        gtd_val = ctx.flat_grad.dot(ctx.d).item()
        if gtd_val > -1e-5:
            if self.verbose:
                print(f"  iter {n_iter:4d}  ls failed and gradient ~0, stopping")
            return True
        t_retry = 1.0 / ((-gtd_val) ** 0.5)
        ctx.prev_loss = ctx.loss
        start_t = max(min(t_retry / 0.5, 1.0), self._minstep)
        ctx.t, ctx.loss, ls_evals_retry, ctx.status = armijo_linesearch(
            linefn,
            gtd_val,
            ctx.prev_loss,
            alpha0=start_t,
            factor=0.5,
            sigma_decrease=0.1,
            sigma_increase=0.8,
            minstep=self._minstep,
        )
        ctx.ls_evals += ls_evals_retry
        if self.verbose:
            print(
                f"  iter {n_iter:4d}  [reset+retry] E={ctx.loss:.6f}"
                f"  ls_evals={ls_evals_retry}"
                f"  start_step={start_t:.6e}  accepted_step={ctx.t:.6e}"
            )
        if ctx.t == 0.0:
            if self.verbose:
                print(f"  iter {n_iter:4d}  ls failed again after reset, stopping")
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
        ctx = self._step_setup(closure)

        x = ctx.x
        x_backup = ctx.x_backup
        d = ctx.d

        def linefn(alpha_test):
            self.ls_func_evals += 1
            # Direct parameter update - eliminates _set_x_from_flat overhead
            x.copy_(x_backup).add_(d, alpha=alpha_test)
            E = closure()
            return E.item()

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

            # Armijo updates will track step length during optimization
            # thus, "learning rate" is only applied for the initial step
            if ctx.state["n_iter"] == 1:
                ctx.t = ctx.lr

            # directional derivative
            gtd_val = ctx.flat_grad.dot(d).item()

            # (fd) this is some hacky stuff I put in R3 that is not typically part
            # (fd)   of lbfgs because the bfgs update had us frequently searching
            # (fd)   in positive grad directions
            # check 1: if dir. deriv. is positive, flip signs of positive components
            if gtd_val > -1e-5:
                d.mul_(-torch.sign(ctx.flat_grad * d))
                gtd_val = ctx.flat_grad.dot(d).item()

            # check 2: if derivative is still positive, reset Hessian
            if gtd_val > -1e-5:
                d.copy_(ctx.flat_grad).neg_()
                gtd_val = ctx.flat_grad.dot(d).item()

            # define the line search function
            # we do not need to compute gradients in here
            self.ls_func_evals = 0

            # Optimization: save original position and work directly with param.data
            x_backup.copy_(x)

            # do the line search
            # match Rosetta: start at 2x prev accepted step, capped at 1.0
            start_t = min(ctx.t / 0.5, 1.0)
            ctx.t, ctx.loss, ctx.ls_evals, ctx.status = armijo_linesearch(
                linefn,  # callback for energy eval
                gtd_val,  # directional derivative
                ctx.prev_loss,  # current function value (at x)
                alpha0=start_t,  # stepsize
                factor=0.5,
                sigma_decrease=0.1,
                sigma_increase=0.8,
                minstep=self._minstep,
            )

            if ctx.t == 0.0:
                if self._rescue_failed_linesearch(ctx, linefn, n_iter):
                    break

            # update - direct modification
            x.copy_(x_backup).add_(d, alpha=ctx.t)

            # ONLY if the last step was 'step_back' then grads are out of date
            #   recompute the closure
            if ctx.status == "step_back":
                closure()

            ctx.flat_grad = ctx.param.grad.data.view(-1)  # Direct reference
            max_grad = ctx.flat_grad.max().item()

            # update func eval
            current_evals += self.ls_func_evals
            ctx.state["func_evals"] += self.ls_func_evals

            # converge check 1: gradient
            if max_grad <= ctx.gradtol:
                if self.verbose:
                    print(f"  converged: max_grad {max_grad:.4e} <= {ctx.gradtol}")
                break
            # converge check 2: abs tol
            if abs(ctx.loss - ctx.prev_loss) <= ctx.atol:
                if self.verbose:
                    print(
                        f"  converged: |dE| {abs(ctx.loss - ctx.prev_loss):.4e} <= atol {ctx.atol}"
                    )
                break
            # converge check 3: rel tol
            rdiff = (
                2
                * abs(ctx.loss - ctx.prev_loss)
                / (abs(ctx.loss) + abs(ctx.prev_loss) + 1e-10)
            )
            if rdiff <= ctx.rtol:
                if self.verbose:
                    print(f"  converged: rel_dE {rdiff:.4e} <= rtol {ctx.rtol}")
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
        ctx.state["x_ref"] = ctx.x_ref

        return ctx.orig_loss
