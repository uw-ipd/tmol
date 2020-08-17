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
        phi0 = func(0.)
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

        # evaluate initial f(x)
        orig_loss = closure()
        loss = float(orig_loss)
        current_evals = 1
        state["func_evals"] += 1

        # ... and df/dx
        x = self._gather_flat_x()
        flat_grad = self._gather_flat_grad()
        max_grad = flat_grad.max()

        # tensors cached in state
        d = state.get("d")  # search direction
        t = state.get("t")  # stepsize

        old_dirs = state.get("old_dirs")  # history of directions
        old_stps = state.get("old_stps")  # history of stepsizes

        prev_flat_grad = state.get("prev_flat_grad")  # previous grad
        prev_loss = state.get("prev_loss")  # previous energy

        n_iter = 0

        while n_iter < max_iter:
            n_iter += 1
            state["n_iter"] += 1

            ## LBFGS updates taken from torch LBFGS
            if state["n_iter"] == 1:
                # initialize
                d = flat_grad.neg()
                old_dirs = []
                old_stps = []
            else:
                # do lbfgs update (update memory)
                y = flat_grad.sub(prev_flat_grad)
                s = d.mul(t)
                ys = y.dot(s)  # y*s
                if ys > 1e-10:
                    # updating memory
                    if len(old_dirs) == history_size:
                        # shift history by one (limited-memory)
                        old_dirs.pop(0)
                        old_stps.pop(0)

                    # store new direction/step
                    old_dirs.append(y)
                    old_stps.append(s)

                # compute the approximate (L-BFGS) inverse Hessian
                # multiplied by the gradient
                num_old = len(old_dirs)

                if "ro" not in state:
                    state["ro"] = [None] * history_size
                    state["al"] = [None] * history_size
                ro = state["ro"]
                al = state["al"]

                for i in range(num_old):
                    ro[i] = 1. / old_dirs[i].dot(old_stps[i])

                # iteration in L-BFGS loop collapsed to use just one buffer
                q = flat_grad.neg()
                for i in range(num_old - 1, -1, -1):
                    al[i] = old_stps[i].dot(q) * ro[i]
                    q.add_(old_dirs[i], alpha=-al[i])

                # r/d is the final direction
                d = r = q
                for i in range(num_old):
                    be_i = old_dirs[i].dot(r) * ro[i]
                    r.add_(old_stps[i], alpha=al[i] - be_i)

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

            def linefn(alpha_test):
                self.ls_func_evals += 1
                self._set_x_from_flat(x + alpha_test * d)
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

            # update
            x = x + t * d
            self._set_x_from_flat(x)
            closure()  # fd: needed for derivatives, but adds an extra func eval...
            flat_grad = self._gather_flat_grad()
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

            # report if we have hit max cycles (mimicing R3)
            # if state['n_iter'] == max_iter - 1:
            #    print(
            #        "LBFGS_Armijo finished ", max_iter,
            #        " cycles without converging."
            #    )

        state["d"] = d
        state["t"] = t
        state["old_dirs"] = old_dirs
        state["old_stps"] = old_stps
        state["prev_flat_grad"] = prev_flat_grad
        state["prev_loss"] = prev_loss

        return orig_loss
