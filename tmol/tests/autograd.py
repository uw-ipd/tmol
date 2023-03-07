import numpy
import pandas
import torch
import warnings

from torch.autograd import gradcheck

# No QA due to complexity error in gradcheck function
# flake8: noqa


def gradcheck(
    func,
    inputs,
    eps=1e-6,
    atol=1e-5,
    rtol=1e-3,
    nfail=0,
    raise_exception=True,
    nondet_tol=0,
):
    torch.autograd.gradcheck(
        func, inputs, eps=eps, atol=atol, rtol=rtol, nondet_tol=nondet_tol
    )


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
        return self._VectorizedFun.apply(self, *args)

    class _VectorizedFun(torch.autograd.Function):
        @staticmethod
        def forward(ctx, op, *args):
            ctx.op = op

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

            assert v.dim() <= 1, "Expected scalar output."
            assert not torch.isnan(
                v.sum()
            ).any(), "Autograd test operation does not support nan-valued functions."

            return v.sum()

        @staticmethod
        def backward(ctx, d_d_v):
            d_d_i = tuple(d_d_v * d_v_d_i for d_v_d_i in ctx.saved_tensors)

            return (None,) + d_d_i + (None,) * ctx.op.n_input_non_grad
