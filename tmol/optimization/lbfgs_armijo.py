import torch
from functools import reduce
from torch.optim import Optimizer


def armijo_linesearch(
    func,
    derphi0,
    old_fval,
    alpha0=1,
    factor=0.5,
    sigma_decrease=0.1,
    sigma_increase=0.8,
    minstep=1e-12,
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

        'factor' corresponds roughly to 'beta'
    """
    # evaluate phi(0) if not input
    if old_fval is None:
        phi0 = func(0.0)
    else:
        phi0 = old_fval

    # check armijo condition
    phi_a0 = func(alpha0)

    # first, we check if we can increase the stepsize
    #     (if the func is still behaving linearly)

    if phi_a0 <= phi0 + alpha0 * sigma_increase * derphi0:
        # attempt to increase stepsize
        alpha1 = alpha0 / factor
        phi_a1 = func(alpha1)
        if phi_a1 < phi_a0:
            return alpha1, phi_a1

        # step back
        return alpha0, phi_a0

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
                finite_diff = (phi_a1 - phi0) / alpha1
                print(
                    "Inaccurate G! Step=",
                    alpha1,
                    " Deriv=",
                    derphi0,
                    " Finite=",
                    finite_diff,
                )
                return 0.0, phi0
            return alpha1, phi_a1

        alpha1 *= factor * factor  # see note above, decrease by factor^2
        phi_a1 = func(alpha1)

    return alpha1, phi_a1


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
        rtol=1e-6,
        atol=0,
        gradtol=1e-4,
        history_size=128,
    ):
        defaults = dict(
            lr=lr,
            max_iter=max_iter,
            rtol=rtol,
            atol=atol,
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

    def step(self, closure):
        """
        The LBFGS minimization algorithm.

        Arguments:
            func (callable): a function that evaluates energy

        Returns:
            orig_loss: the energy (loss) following optimization

        Notes:
            Despite the name, this performs the full LBFGS minimization trajectory.
            Stores lots of information in self.state
        """
        # lbfgs only works w/ single parameter group
        assert len(self.param_groups) == 1

        group = self.param_groups[0]
        lr = group["lr"]
        max_iter = group["max_iter"]
        rtol = group["rtol"]
        atol = group["atol"]
        gradtol = group["gradtol"]
        history_size = group["history_size"]

        # NOTE: LBFGS has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        state = self.state[self._params[0]]
        state.setdefault("func_evals", 0)
        state.setdefault("n_iter", 0)

        # Optimization: work directly with the single parameter tensor
        assert len(self._params) == 1, "This optimized version requires single tensor"
        param = self._params[0]

        # evaluate initial f(x)
        orig_loss = closure()
        loss = float(orig_loss.detach().cpu().item())
        current_evals = 1
        state["func_evals"] += 1

        # ... and df/dx - direct reference instead of gather
        x = param.data.view(-1)
        flat_grad = param.grad.data.view(-1)
        max_grad = flat_grad.max()

        # tensors cached in state
        d = state.get("d")  # search direction
        t = state.get("t")  # stepsize

        prev_flat_grad = state.get("prev_flat_grad")  # previous grad
        prev_loss = state.get("prev_loss")  # previous energy

        # Pre-allocate stacked matrices for L-BFGS (reused each iteration)
        L = x.numel()
        if "old_dirs_mat" not in state:
            state["old_dirs_mat"] = torch.empty(
                (history_size, L), device=x.device, dtype=x.dtype
            )
            state["old_stps_mat"] = torch.empty(
                (history_size, L), device=x.device, dtype=x.dtype
            )
            state["history_start"] = 0  # Circular buffer start index
            state["history_count"] = 0  # Number of items in history

        old_dirs_mat = state["old_dirs_mat"]
        old_stps_mat = state["old_stps_mat"]
        history_start = state["history_start"]
        history_count = state["history_count"]

        n_iter = 0

        while n_iter < max_iter:
            n_iter += 1
            state["n_iter"] += 1

            ## LBFGS updates taken from torch LBFGS
            if state["n_iter"] == 1:
                # initialize
                d = flat_grad.neg()
                history_count = 0
            else:
                # do lbfgs update (update memory)
                y = flat_grad.sub(prev_flat_grad)
                s = d.mul(t)
                ys = y.dot(s)  # y*s
                if ys > 1e-10:
                    # updating memory - write directly into circular buffer
                    if history_count < history_size:
                        # Still filling up the buffer
                        idx = history_count
                        history_count += 1
                    else:
                        # Buffer full, overwrite oldest entry
                        idx = history_start
                        history_start = (history_start + 1) % history_size

                    old_dirs_mat[idx].copy_(y)
                    old_stps_mat[idx].copy_(s)

                # compute the approximate (L-BFGS) inverse Hessian
                if history_count == 0:
                    # No history: use steepest descent direction
                    d = r = flat_grad.neg()
                else:
                    # Create views old -> new
                    if history_count < history_size:
                        old_dirs_view = old_dirs_mat[:history_count]
                        old_stps_view = old_stps_mat[:history_count]
                    else:
                        # Buffer full, need to reorder: [start:end] + [0:start]
                        indices = torch.cat(
                            [
                                torch.arange(
                                    history_start, history_size, device=x.device
                                ),
                                torch.arange(0, history_start, device=x.device),
                            ]
                        )
                        old_dirs_view = old_dirs_mat[indices]
                        old_stps_view = old_stps_mat[indices]

                    # Compute all ro values in one batched operation
                    ro = 1.0 / torch.sum(old_dirs_view * old_stps_view, dim=1)

                    # First loop: backward pass - fully batched
                    q = flat_grad.neg()

                    # Compute all dot products: old_stps_mat @ q
                    stps_dot_q = torch.mv(old_stps_view, q)
                    al = stps_dot_q * ro

                    # Compute cumulative updates in reverse order
                    al_flipped = torch.flip(al, dims=[0])
                    old_dirs_flipped = torch.flip(old_dirs_view, dims=[0])

                    # Apply all updates at once: q -= old_dirs_mat.T @ al_flipped
                    q.add_(torch.mv(old_dirs_flipped.t(), al_flipped), alpha=-1.0)

                    # Second loop: forward pass - fully batched
                    r = q
                    be = (
                        torch.mv(old_dirs_view, r) * ro
                    )  # All dot products in one matmul

                    # Single batched update: r += old_stps_mat.T @ (al - be)
                    r.add_(torch.mv(old_stps_view.t(), al - be))

                    d = r

            if prev_flat_grad is None:
                prev_flat_grad = flat_grad.clone()
            else:
                prev_flat_grad.copy_(flat_grad)
            prev_loss = loss

            # Armijo updates will track step length during optimization
            # thus, "learning rate" is only applied for the initial step
            if state["n_iter"] == 1:
                t = lr

            # directional derivative
            gtd = flat_grad.dot(d)  # g * d

            # (fd) this is some hacky stuff I put in R3 that is not typically part
            # (fd)   of lbfgs because the bfgs update had us frequently searching
            # (fd)   in positive grad directions
            # check 1: if dir. deriv. is positive, flip signs of positive components
            if gtd > -1e-5:
                d *= -torch.sign(flat_grad * d)
                gtd = flat_grad.dot(d)

            # check 2: if derivative is still positive, reset Hessian
            if gtd > -1e-5:
                d = flat_grad.neg()
                old_dirs = []
                old_stps = []
                gtd = flat_grad.dot(d)

            # define the line search function
            # we do not need to compute gradients in here
            self.ls_func_evals = 0

            # Optimization: save original position and work directly with param.data
            x_backup = x.clone()

            def linefn(alpha_test):
                self.ls_func_evals += 1
                # Direct parameter update - eliminates _set_x_from_flat overhead
                x.copy_(x_backup).add_(d, alpha=alpha_test)
                E = closure()
                return E.to(dtype=gtd.dtype)

            # do the line search
            t, loss = armijo_linesearch(
                linefn,  # callback for energy eval
                gtd,  # directional derivative
                prev_loss,  # current function value (at x)
                alpha0=t,  # stepsize
                factor=0.5,
                sigma_decrease=0.1,
                sigma_increase=0.8,
                minstep=1e-12,
            )

            # update - direct modification
            x.copy_(x_backup).add_(d, alpha=t)

            closure()  # fd: needed for derivatives, but adds an extra func eval...

            flat_grad = param.grad.data.view(-1)  # Direct reference
            max_grad = flat_grad.max()

            # update func eval
            current_evals += self.ls_func_evals
            state["func_evals"] += self.ls_func_evals

            # converge check 1: gradient
            if max_grad <= gradtol:
                break

            # converge check 2: abs tol
            if abs(loss - prev_loss) <= atol:
                break

            # converge check 3: rel tol
            if 2 * abs(loss - prev_loss) <= rtol * (abs(loss) + abs(prev_loss) + 1e-10):
                break

        state["d"] = d
        state["t"] = t
        state["history_start"] = history_start
        state["history_count"] = history_count
        state["prev_flat_grad"] = prev_flat_grad
        state["prev_loss"] = prev_loss

        return orig_loss
