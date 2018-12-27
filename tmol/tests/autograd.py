import numpy
import pandas
import torch
import warnings

from torch.autograd.gradcheck import (
    _as_tuple,
    _differentiable_outputs,
    get_numerical_jacobian,
    get_analytical_jacobian,
    iter_tensors,
)

# No QA due to complexity error in gradcheck function
# flake8: noqa


def _gradcheck_summary_frame(analytic, numeric, atol, rtol):
    sframe = pandas.DataFrame.from_dict(
        dict(analytic=analytic.numpy().ravel(), numeric=numeric.numpy().ravel())
    )

    sframe.index = list(
        map(
            tuple,
            numpy.stack(
                numpy.unravel_index(numpy.arange(len(sframe)), numeric.shape)
            ).T,
        )
    )

    sframe["abs_error"] = (sframe["analytic"] - sframe["numeric"]).abs()
    sframe["rel_error"] = (
        numpy.clip((sframe["abs_error"] - atol), 0, None) / sframe["numeric"].abs()
    )

    sframe["failure"] = (
        numpy.clip((sframe["abs_error"] - atol), 0, None)
        > sframe["numeric"].abs() * rtol
    )

    return sframe


def gradcheck(
    func, inputs, eps=1e-6, atol=1e-5, rtol=1e-3, nfail=0, raise_exception=True
):
    r"""Direct-port of pytest.autograd.gradcheck with improved error reporting.

    Check gradients computed via small finite differences against analytical
    gradients w.r.t. tensors in :attr:`inputs` that are of floating point type
    and with ``requires_grad=True``.

    The check between numerical and analytical gradients has the same behaviour as
    `numpy.allclose <https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html>`_,
    i.e., it checks that

    .. math::

        \lvert a - n \rvert \leq \texttt{atol} + \texttt{rtol} \times \lvert n \rvert

    holds for all elements of analytical gradient :math:`a` and numerical
    gradient :math:`n`.

    .. note::
        The default values are designed for :attr:`input` of double precision.
        This check will likely fail if :attr:`input` is of less precision, e.g.,
        ``FloatTensor``.

    .. warning::
       If any checked tensor in :attr:`input` has overlapping memory, i.e.,
       different indices pointing to the same memory address (e.g., from
       :func:`torch.expand`), this check will likely fail because the numerical
       gradients computed by point perturbation at such indices will change
       values at all other indices that share the same memory address.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor or a tuple of Tensors
        inputs (tuple of Tensor): inputs to the function
        eps (float, optional): perturbation for finite differences
        atol (float, optional): absolute tolerance
        rtol (float, optional): relative tolerance
        nfail (int, optional): maximum number of allowed failures
        raise_exception (bool, optional): indicating whether to raise an exception if
            the check fails. The exception gives more information about the
            exact nature of the failure. This is helpful when debugging gradchecks.

    Returns:
        True if all differences satisfy allclose condition
    """
    tupled_inputs = _as_tuple(inputs)

    # Make sure that gradients are saved for all inputs
    any_input_requiring_grad = False
    for inp in tupled_inputs:
        if isinstance(inp, torch.Tensor):
            if inp.requires_grad:
                if inp.dtype != torch.float64:
                    warnings.warn(
                        "At least one of the inputs that requires gradient "
                        "is not of double precision floating point. "
                        "This check will likely fail if all the inputs are "
                        "not of double precision floating point. "
                    )
                any_input_requiring_grad = True
            inp.retain_grad()
    if not any_input_requiring_grad:
        raise ValueError(
            "gradcheck expects at least one input tensor to require gradient, "
            "but none of the them have requires_grad=True."
        )

    output = _differentiable_outputs(func(*inputs))

    def fail_test(msg):
        if raise_exception:
            raise RuntimeError(msg)
        return False

    for i, o in enumerate(output):
        if not o.requires_grad:
            continue

        def fn(input):
            return _as_tuple(func(*input))[i]

        analytical, reentrant, correct_grad_sizes = get_analytical_jacobian(
            tupled_inputs, o
        )
        numerical = get_numerical_jacobian(fn, inputs, eps=eps)

        if not correct_grad_sizes:
            return fail_test("Analytical gradient has incorrect size")

        for j, (a, n) in enumerate(zip(analytical, numerical)):
            if a.numel() != 0 or n.numel() != 0:
                summary_frame = _gradcheck_summary_frame(a, n, atol, rtol)

                assert summary_frame.failure.sum() <= nfail, (
                    f"Jacobian mismatch for output {i} with respect to input {j}:\n"
                    f"{summary_frame}\n\n"
                    f"failures:\n{summary_frame.query('failure')}\n\n"
                    f"summary:\n{summary_frame.describe()}\n\n"
                    f"atol: {atol} rtol: {rtol}\n"
                    f"failures: {summary_frame.failure.sum()} ({summary_frame.failure.mean() :.2f})\n"
                )

        if not reentrant:
            return fail_test(
                "Backward is not reentrant, i.e., running backward with same "
                "input and grad_output multiple times gives different values, "
                "although analytical gradient matches numerical gradient"
            )

    # check if the backward multiplies by grad_output
    output = _differentiable_outputs(func(*inputs))
    if any([o.requires_grad for o in output]):
        diff_input_list = list(iter_tensors(inputs, True))
        if not diff_input_list:
            raise RuntimeError("no Tensors requiring grad found in input")
        grads_input = torch.autograd.grad(
            output,
            diff_input_list,
            [torch.zeros_like(o) for o in output],
            allow_unused=True,
        )
        for gi, i in zip(grads_input, diff_input_list):
            if gi is None:
                continue
            if not gi.eq(0).all():
                return fail_test("backward not multiplied by grad_output")
            if gi.type() != i.type():
                return fail_test("grad is incorrect type")
            if gi.size() != i.size():
                return fail_test("grad is incorrect size")

    return True


class VectorizedOp:
    """Torch autograd wrapper for gufunc-vectorized "v and dv" functions.

    Test harness for evaluation of vector-valued functions of the form:
        f(*inputs_with_grad, *inputs_without_grad) -> (value, *d_value_d_inputs_with_grad)

    Functions are wrapped with a numpy.vectorize gufunc dispatcher, which performs vector dispatch
    over tensors of input values. An autograd 

    The [(input), ...] -> [(val), (d_val_d_input), ...] function is converted into:

    forward:
    1. Evaluates the function for val and d_val_d_input.
    2. Stores d_val_d_input for use in backward.
    2. Return sum(val).

    backward:
    1. Returns (d_d_val * d_val_d_input) for each input.
    """

    def __init__(self, f, signature=None):
        if isinstance(f, numpy.lib.function_base.vectorize):
            self.f = f
            signature = f.signature
        else:
            assert signature is not None
            self.f = numpy.vectorize(f, signature=signature)

        self.n_input_grad = len(signature.split("->")[1].split(",")) - 1
        self.n_input_non_grad = (
            len(signature.split("->")[0].split(",")) - self.n_input_grad
        )

        assert self.n_input_grad > 0

        super().__init__()

    def __call__(self, *args):
        return self._VectorizedFun(self)(*args)

    class _VectorizedFun(torch.autograd.Function):
        def __init__(self, op):
            self.op = op
            super().__init__()

        def forward(ctx, *args):
            args = tuple(args)
            tensor_args = tuple(map(torch.Tensor.detach, args[: ctx.op.n_input_grad]))
            other_args = tuple(args[ctx.op.n_input_grad :])
            result = ctx.op.f(*(tensor_args + other_args))

            v, *raw_dv_di = tuple(map(torch.from_numpy, result))
            assert len(raw_dv_di) == ctx.op.n_input_grad

            def _prep_grad(i, dv_di):
                if dv_di.shape != i.shape:
                    if dv_di.shape[1:] == i.shape:
                        # Broadcast along implicit leading dimension, sum off dimension
                        return dv_di.sum(dim=0)

                    # Broadcast along dim-1 leading dimension, sum into leading dimension
                    assert (1,) + dv_di.shape[
                        1:
                    ] == i.shape, "Unknown input broadcast pattern."
                    return dv_di.sum(dim=0, keepdim=True)

                else:
                    # No broadcast occured
                    return dv_di

            ctx.save_for_backward(
                *[_prep_grad(*v) for v in zip(tensor_args, raw_dv_di)]
            )

            assert v.dim() == 1

            return v.sum()

        def backward(ctx, d_d_v):
            d_d_i = tuple(d_d_v * d_v_d_i for d_v_d_i in ctx.saved_tensors)

            return d_d_i + (None,) * ctx.op.n_input_non_grad
