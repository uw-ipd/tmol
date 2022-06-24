import numpy
import pandas
import torch
import warnings

import torch
from torch.types import _TensorOrTensors

# from torch._six import container_abcs, istuple
import collections.abc as container_abcs
import torch.testing
from itertools import product
import warnings
from typing import Callable, Union, Optional


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


def get_analytical_jacobian(input, output):
    diff_input_list = list(iter_tensors(input, True))
    jacobian = make_jacobian(input, output.numel())
    jacobian_reentrant = make_jacobian(input, output.numel())
    grad_output = torch.zeros_like(output)
    flat_grad_output = grad_output.view(-1)
    reentrant = True
    correct_grad_sizes = True

    for i in range(flat_grad_output.numel()):
        flat_grad_output.zero_()
        flat_grad_output[i] = 1
        for jacobian_c in (jacobian, jacobian_reentrant):
            grads_input = torch.autograd.grad(
                output,
                diff_input_list,
                grad_output,
                retain_graph=True,
                allow_unused=True,
            )
            for jacobian_x, d_x, x in zip(jacobian_c, grads_input, diff_input_list):
                if d_x is not None and d_x.size() != x.size():
                    correct_grad_sizes = False
                elif jacobian_x.numel() != 0:
                    if d_x is None:
                        jacobian_x[:, i].zero_()
                    else:
                        d_x_dense = d_x.to_dense() if d_x.is_sparse else d_x
                        assert jacobian_x[:, i].numel() == d_x_dense.numel()
                        jacobian_x[:, i] = d_x_dense.contiguous().view(-1)

    for jacobian_x, jacobian_reentrant_x in zip(jacobian, jacobian_reentrant):
        reentrant_delta = (jacobian_x - jacobian_reentrant_x).abs().max()

        if (
            jacobian_x.numel() != 0
            and (jacobian_x - jacobian_reentrant_x).abs().max() != 0
        ):
            warnings.warn(
                f"Analytical jacobian is not reentrant, max delta: {reentrant_delta}"
            )
            reentrant = False

    return jacobian, reentrant, correct_grad_sizes


#######################################
### Stolen from torch1.6 gradcheck.py #
#######################################


def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, container_abcs.Iterable):
        for elem in x:
            zero_gradients(elem)


def make_jacobian(input, num_out):
    if isinstance(input, torch.Tensor):
        if not input.is_floating_point() and not input.is_complex():
            return None
        if not input.requires_grad:
            return None
        return torch.zeros(input.nelement(), num_out, dtype=input.dtype)
    elif isinstance(input, container_abcs.Iterable) and not isinstance(input, str):
        jacobians = list(
            filter(
                lambda x: x is not None,
                (make_jacobian(elem, num_out) for elem in input),
            )
        )
        if not jacobians:
            return None
        return type(input)(jacobians)
    else:
        return None


def iter_tensors(x, only_requiring_grad=False):
    if isinstance(x, torch.Tensor):
        if x.requires_grad or not only_requiring_grad:
            yield x
    elif isinstance(x, container_abcs.Iterable) and not isinstance(x, str):
        for elem in x:
            for result in iter_tensors(elem, only_requiring_grad):
                yield result


def get_numerical_jacobian(fn, input, target=None, eps=1e-3):
    """
    input: input to `fn`
    target: the Tensors wrt whom Jacobians are calculated (default=`input`)

    Note that `target` may not even be part of `input` to `fn`, so please be
    **very careful** in this to not clone `target`.
    """
    if target is None:
        target = input
    output_size = fn(input).numel()
    jacobian = make_jacobian(target, output_size)

    # It's much easier to iterate over flattened lists of tensors.
    # These are reference to the same objects in jacobian, so any changes
    # will be reflected in it as well.
    x_tensors = iter_tensors(target, True)
    j_tensors = iter_tensors(jacobian)

    # TODO: compare structure
    for x_tensor, d_tensor in zip(x_tensors, j_tensors):
        is_complex = x_tensor.dtype.is_complex
        if is_complex:
            eps *= 1 + 1j
        if x_tensor.is_sparse:

            def get_stride(size):
                dim = len(size)
                tmp = 1
                stride = [0] * dim
                for i in reversed(range(dim)):
                    stride[i] = tmp
                    tmp *= size[i]
                return stride

            x_nnz = x_tensor._nnz()
            x_size = list(x_tensor.size())
            x_indices = x_tensor._indices().t()
            x_values = x_tensor._values()
            x_stride = get_stride(x_size)

            # Use .data here to get around the version check
            x_values = x_values.data

            for i in range(x_nnz):
                x_value = x_values[i]
                for x_idx in product(*[range(m) for m in x_values.size()[1:]]):
                    indices = x_indices[i].tolist() + list(x_idx)
                    d_idx = sum(indices[k] * x_stride[k] for k in range(len(x_size)))
                    orig = x_value[x_idx].item()
                    x_value[x_idx] = orig - eps
                    outa = fn(input).clone()
                    x_value[x_idx] = orig + eps
                    outb = fn(input).clone()
                    x_value[x_idx] = orig
                    r = (outb - outa) / (2 * eps)
                    d_tensor[d_idx] = r.detach().reshape(-1)
        elif x_tensor.layout == torch._mkldnn:
            # Use .data here to get around the version check
            x_tensor = x_tensor.data
            if len(input) != 1:
                raise ValueError(
                    "gradcheck currently only supports functions with 1 input, but got: ",
                    len(input),
                )
            for d_idx, x_idx in enumerate(
                product(*[range(m) for m in x_tensor.size()])
            ):
                # this is really inefficient, but without indexing implemented, there's
                # not really a better way than converting back and forth
                x_tensor_dense = x_tensor.to_dense()
                orig = x_tensor_dense[x_idx].item()

                x_tensor_dense[x_idx] = orig - eps
                x_tensor_mkl = x_tensor_dense.to_mkldnn()
                outa = fn([x_tensor_mkl])

                x_tensor_dense[x_idx] = orig + eps
                x_tensor_mkl = x_tensor_dense.to_mkldnn()
                outb = fn([x_tensor_mkl])

                r = (outb - outa) / (2 * eps)
                d_tensor[d_idx] = r.detach().reshape(-1)
        else:
            # Use .data here to get around the version check
            x_tensor = x_tensor.data
            for d_idx, x_idx in enumerate(
                product(*[range(m) for m in x_tensor.size()])
            ):
                orig = x_tensor[x_idx].item()
                x_tensor[x_idx] = orig - eps
                outa = fn(input).clone()
                x_tensor[x_idx] = orig + eps
                outb = fn(input).clone()
                x_tensor[x_idx] = orig
                r = (outb - outa) / (2 * eps)
                d_tensor[d_idx] = r.detach().reshape(-1)

    return jacobian


# def get_analytical_jacobian(input, output, nondet_tol=0.0):
#     # it is easier to call to_dense() on the sparse output than
#     # to modify analytical jacobian
#     if output.is_sparse:
#         raise ValueError('Sparse output is not supported at gradcheck yet. '
#                          'Please call to_dense() on the output of fn for gradcheck.')
#     if output.layout == torch._mkldnn:
#         raise ValueError('MKLDNN output is not supported at gradcheck yet. '
#                          'Please call to_dense() on the output of fn for gradcheck.')
#     diff_input_list = list(iter_tensors(input, True))
#     jacobian = make_jacobian(input, output.numel())
#     jacobian_reentrant = make_jacobian(input, output.numel())
#     grad_output = torch.zeros_like(output, memory_format=torch.legacy_contiguous_format)
#     flat_grad_output = grad_output.view(-1)
#     reentrant = True
#     correct_grad_sizes = True
#
#     for i in range(flat_grad_output.numel()):
#         flat_grad_output.zero_()
#         flat_grad_output[i] = 1
#         for jacobian_c in (jacobian, jacobian_reentrant):
#             grads_input = torch.autograd.grad(output, diff_input_list, grad_output,
#                                               retain_graph=True, allow_unused=True)
#             for jacobian_x, d_x, x in zip(jacobian_c, grads_input, diff_input_list):
#                 if d_x is not None and d_x.size() != x.size():
#                     correct_grad_sizes = False
#                 elif jacobian_x.numel() != 0:
#                     if d_x is None:
#                         jacobian_x[:, i].zero_()
#                     else:
#                         d_x_dense = d_x.to_dense() if not d_x.layout == torch.strided else d_x
#                         assert jacobian_x[:, i].numel() == d_x_dense.numel()
#                         jacobian_x[:, i] = d_x_dense.contiguous().view(-1)
#
#     for jacobian_x, jacobian_reentrant_x in zip(jacobian, jacobian_reentrant):
#         if jacobian_x.numel() != 0 and (jacobian_x - jacobian_reentrant_x).abs().max() > nondet_tol:
#             reentrant = False
#
#     return jacobian, reentrant, correct_grad_sizes


def istuple(obj):
    # license: pytorch
    #
    # Usually instances of PyStructSequence is also an instance of tuple
    # but in some py2 environment it is not, so we have to manually check
    # the name of the type to determine if it is a namedtupled returned
    # by a pytorch operator.
    t = type(obj)
    return isinstance(obj, tuple) or t.__module__ == "torch.return_types"


def _as_tuple(x):
    if istuple(x):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return (x,)


def _differentiable_outputs(x):
    return tuple(o for o in _as_tuple(x) if o.requires_grad)


# Note [VarArg of Tensors]
# ~~~~~~~~~~~~~~~~~~~~~~~~
# 'func' accepts a vararg of tensors, which isn't expressable in the type system at the moment.
# If https://mypy.readthedocs.io/en/latest/additional_features.html?highlight=callable#extended-callable-types is accepted,
# the '...' first argument of Callable can be replaced with VarArg(Tensor).
# For now, we permit any input.
# the '...' first argument of Callable can be replaced with VarArg(Tensor).
# For now, we permit any input.

# def gradcheck(
#     func: Callable[..., Union[_TensorOrTensors]],  # See Note [VarArg of Tensors]
#     inputs: _TensorOrTensors,
#     eps: float = 1e-6,
#     atol: float = 1e-5,
#     rtol: float = 1e-3,
#     raise_exception: bool = True,
#     check_sparse_nnz: bool = False,
#     nondet_tol: float = 0.0,
#     check_undefined_grad: bool = True
# ) -> bool:
#     r"""Check gradients computed via small finite differences against analytical
#     gradients w.r.t. tensors in :attr:`inputs` that are of floating point or complex type
#     and with ``requires_grad=True``.
#
#     The check between numerical and analytical gradients uses :func:`~torch.allclose`.
#
#     .. note::
#         The default values are designed for :attr:`input` of double precision.
#         This check will likely fail if :attr:`input` is of less precision, e.g.,
#         ``FloatTensor``.
#
#     .. warning::
#        If any checked tensor in :attr:`input` has overlapping memory, i.e.,
#        different indices pointing to the same memory address (e.g., from
#        :func:`torch.expand`), this check will likely fail because the numerical
#        gradients computed by point perturbation at such indices will change
#        values at all other indices that share the same memory address.
#
#     Args:
#         func (function): a Python function that takes Tensor inputs and returns
#             a Tensor or a tuple of Tensors
#         inputs (tuple of Tensor or Tensor): inputs to the function
#         eps (float, optional): perturbation for finite differences
#         atol (float, optional): absolute tolerance
#         rtol (float, optional): relative tolerance
#         raise_exception (bool, optional): indicating whether to raise an exception if
#             the check fails. The exception gives more information about the
#             exact nature of the failure. This is helpful when debugging gradchecks.
#         check_sparse_nnz (bool, optional): if True, gradcheck allows for SparseTensor input,
#             and for any SparseTensor at input, gradcheck will perform check at nnz positions only.
#         nondet_tol (float, optional): tolerance for non-determinism. When running
#             identical inputs through the differentiation, the results must either match
#             exactly (default, 0.0) or be within this tolerance.
#         check_undefined_grad (bool, options): if True, check if undefined output grads
#             are supported and treated as zeros
#
#     Returns:
#         True if all differences satisfy allclose condition
#     """
#     def fail_test(msg):
#         if raise_exception:
#             raise RuntimeError(msg)
#         return False
#
#     tupled_inputs = _as_tuple(inputs)
#     if any(t.is_sparse for t in tupled_inputs if isinstance(t, torch.Tensor)) and not check_sparse_nnz:
#         return fail_test('gradcheck expects all tensor inputs are dense when check_sparse_nnz is set to False.')
#
#     # Make sure that gradients are saved for at least one input
#     any_input_requiring_grad = False
#     for idx, inp in enumerate(tupled_inputs):
#         if isinstance(inp, torch.Tensor) and inp.requires_grad:
#             if not (inp.dtype == torch.float64 or inp.dtype == torch.complex128):
#                 warnings.warn(
#                     'The {}th input requires gradient and '
#                     'is not a double precision floating point or complex. '
#                     'This check will likely fail if all the inputs are '
#                     'not of double precision floating point or complex. ')
#             content = inp._values() if inp.is_sparse else inp
#             # TODO: To cover more problematic cases, replace stride = 0 check with
#             # "any overlap in memory" once we have a proper function to check it.
#             if content.layout is not torch._mkldnn and \
#                not all(st > 0 or sz <= 1 for st, sz in zip(content.stride(), content.size())):
#                 raise RuntimeError(
#                     'The {}th input has a dimension with stride 0. gradcheck only '
#                     'supports inputs that are non-overlapping to be able to '
#                     'compute the numerical gradients correctly. You should call '
#                     '.contiguous on the input before passing it to gradcheck.')
#             any_input_requiring_grad = True
#             inp.retain_grad()
#     if not any_input_requiring_grad:
#         raise ValueError(
#             'gradcheck expects at least one input tensor to require gradient, '
#             'but none of the them have requires_grad=True.')
#
#     func_out = func(*tupled_inputs)
#     output = _differentiable_outputs(func_out)
#
#     if not output:
#         for i, o in enumerate(func_out):
#             def fn(input):
#                 return _as_tuple(func(*input))[i]
#             numerical = get_numerical_jacobian(fn, tupled_inputs, eps=eps)
#             for n in numerical:
#                 if torch.ne(n, 0).sum() > 0:
#                     return fail_test('Numerical gradient for function expected to be zero')
#         return True
#
#     for i, o in enumerate(output):
#         if not o.requires_grad:
#             continue
#
#         def fn(input):
#             return _as_tuple(func(*input))[i]
#
#         analytical, reentrant, correct_grad_sizes = get_analytical_jacobian(tupled_inputs, o, nondet_tol=nondet_tol)
#         numerical = get_numerical_jacobian(fn, tupled_inputs, eps=eps)
#
#         if not correct_grad_sizes:
#             return fail_test('Analytical gradient has incorrect size')
#
#         for j, (a, n) in enumerate(zip(analytical, numerical)):
#             if a.numel() != 0 or n.numel() != 0:
#                 if not torch.allclose(a, n, rtol, atol):
#                     return fail_test('Jacobian mismatch for output %d with respect to input %d,\n'
#                                      'numerical:%s\nanalytical:%s\n' % (i, j, n, a))
#
#         if not reentrant:
#             return fail_test('Backward is not reentrant, i.e., running backward with same '
#                              'input and grad_output multiple times gives different values, '
#                              'although analytical gradient matches numerical gradient. '
#                              'The tolerance for nondeterminism was {}.'.format(nondet_tol))
#
#     # check if the backward multiplies by grad_output
#     output = _differentiable_outputs(func(*tupled_inputs))
#     if any([o.requires_grad for o in output]):
#         diff_input_list = list(iter_tensors(tupled_inputs, True))
#         if not diff_input_list:
#             raise RuntimeError("no Tensors requiring grad found in input")
#         grads_input = torch.autograd.grad(output, diff_input_list,
#                                           [torch.zeros_like(o, memory_format=torch.legacy_contiguous_format) for o in output],
#                                           allow_unused=True)
#         for gi, i in zip(grads_input, diff_input_list):
#             if gi is None:
#                 continue
#             if isinstance(gi, torch.Tensor) and gi.layout != torch.strided:
#                 if gi.layout != i.layout:
#                     return fail_test('grad is incorrect layout (' + str(gi.layout) + ' is not ' + str(i.layout) + ')')
#                 if gi.layout == torch.sparse_coo:
#                     if gi.sparse_dim() != i.sparse_dim():
#                         return fail_test('grad is sparse tensor, but has incorrect sparse_dim')
#                     if gi.dense_dim() != i.dense_dim():
#                         return fail_test('grad is sparse tensor, but has incorrect dense_dim')
#                 gi = gi.to_dense()
#                 i = i.to_dense()
#             if not gi.eq(0).all():
#                 return fail_test('backward not multiplied by grad_output')
#             if gi.dtype != i.dtype or gi.device != i.device or gi.is_sparse != i.is_sparse:
#                 return fail_test("grad is incorrect type")
#             if gi.size() != i.size():
#                 return fail_test('grad is incorrect size')
#
#         if check_undefined_grad:
#             def warn_bc_breaking():
#                 warnings.warn((
#                     'Backwards compatibility: New undefined gradient support checking '
#                     'feature is enabled by default, but it may break existing callers '
#                     'of this function. If this is true for you, you can call this '
#                     'function with "check_undefined_grad=False" to disable the feature'))
#
#             def check_undefined_grad_support(output_to_check):
#                 grads_output = [torch.zeros_like(o, memory_format=torch.legacy_contiguous_format) for o in output_to_check]
#                 try:
#                     grads_input = torch.autograd.grad(output_to_check,
#                                                       diff_input_list,
#                                                       grads_output,
#                                                       allow_unused=True)
#                 except RuntimeError:
#                     warn_bc_breaking()
#                     return fail_test((
#                         'Expected backward function to handle undefined output grads. '
#                         'Please look at "Notes about undefined output gradients" in '
#                         '"tools/autograd/derivatives.yaml"'))
#
#                 for gi, i in zip(grads_input, diff_input_list):
#                     if (gi is not None) and (not gi.eq(0).all()):
#                         warn_bc_breaking()
#                         return fail_test((
#                             'Expected all input grads to be undefined or zero when all output grads are undefined '
#                             'or zero. Please look at "Notes about undefined output gradients" in '
#                             '"tools/autograd/derivatives.yaml"'))
#                 return True
#
#             # All backward functions must work properly if all output grads are undefined
#             outputs_to_check = [[torch._C._functions.UndefinedGrad()(o) for o in _differentiable_outputs(func(*tupled_inputs))]]
#
#             # If there are multiple output grads, we should be able to undef one at a time without error
#             if len(outputs_to_check[0]) > 1:
#                 for undef_grad_idx in range(len(output)):
#                     output_to_check = _differentiable_outputs(func(*tupled_inputs))
#                     outputs_to_check.append([
#                         torch._C._functions.UndefinedGrad()(o) if idx == undef_grad_idx else o
#                         for idx, o in enumerate(output_to_check)])
#
#             for output_to_check in outputs_to_check:
#                 if not check_undefined_grad_support(output_to_check):
#                     return False
#
#     return True


def gradgradcheck(
    func: Callable[..., _TensorOrTensors],  # See Note [VarArg of Tensors]
    inputs: _TensorOrTensors,
    grad_outputs: Optional[_TensorOrTensors] = None,
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    gen_non_contig_grad_outputs: bool = False,
    raise_exception: bool = True,
    nondet_tol: float = 0.0,
    check_undefined_grad: bool = True,
) -> bool:
    r"""Check gradients of gradients computed via small finite differences
    against analytical gradients w.r.t. tensors in :attr:`inputs` and
    :attr:`grad_outputs` that are of floating point or complex type and with
    ``requires_grad=True``.

    This function checks that backpropagating through the gradients computed
    to the given :attr:`grad_outputs` are correct.

    The check between numerical and analytical gradients uses :func:`~torch.allclose`.

    .. note::
        The default values are designed for :attr:`input` and
        :attr:`grad_outputs` of double precision. This check will likely fail if
        they are of less precision, e.g., ``FloatTensor``.

    .. warning::
       If any checked tensor in :attr:`input` and :attr:`grad_outputs` has
       overlapping memory, i.e., different indices pointing to the same memory
       address (e.g., from :func:`torch.expand`), this check will likely fail
       because the numerical gradients computed by point perturbation at such
       indices will change values at all other indices that share the same
       memory address.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor or a tuple of Tensors
        inputs (tuple of Tensor or Tensor): inputs to the function
        grad_outputs (tuple of Tensor or Tensor, optional): The gradients with
            respect to the function's outputs.
        eps (float, optional): perturbation for finite differences
        atol (float, optional): absolute tolerance
        rtol (float, optional): relative tolerance
        gen_non_contig_grad_outputs (bool, optional): if :attr:`grad_outputs` is
            ``None`` and :attr:`gen_non_contig_grad_outputs` is ``True``, the
            randomly generated gradient outputs are made to be noncontiguous
        raise_exception (bool, optional): indicating whether to raise an exception if
            the check fails. The exception gives more information about the
            exact nature of the failure. This is helpful when debugging gradchecks.
        nondet_tol (float, optional): tolerance for non-determinism. When running
            identical inputs through the differentiation, the results must either match
            exactly (default, 0.0) or be within this tolerance. Note that a small amount
            of nondeterminism in the gradient will lead to larger inaccuracies in
            the second derivative.
        check_undefined_grad (bool, options): if True, check if undefined output grads
            are supported and treated as zeros

    Returns:
        True if all differences satisfy allclose condition
    """
    tupled_inputs = _as_tuple(inputs)

    if grad_outputs is None:
        # If grad_outputs is not specified, create random Tensors of the same
        # shape, type, and device as the outputs
        def randn_like(x):
            y = torch.testing.randn_like(
                x if (x.is_floating_point() or x.is_complex()) else x.double(),
                memory_format=torch.legacy_contiguous_format,
            )
            if gen_non_contig_grad_outputs:
                y = torch.testing.make_non_contiguous(y)
            return y.requires_grad_()

        outputs = _as_tuple(func(*tupled_inputs))
        tupled_grad_outputs = tuple(randn_like(x) for x in outputs)
    else:
        tupled_grad_outputs = _as_tuple(grad_outputs)

    num_outputs = len(tupled_grad_outputs)

    def new_func(*args):
        input_args = args[:-num_outputs]
        grad_outputs = args[-num_outputs:]
        outputs = _differentiable_outputs(func(*input_args))
        input_args = tuple(
            x for x in input_args if isinstance(x, torch.Tensor) and x.requires_grad
        )
        grad_inputs = torch.autograd.grad(
            outputs, input_args, grad_outputs, create_graph=True
        )
        return grad_inputs

    return gradcheck(
        new_func,
        tupled_inputs + tupled_grad_outputs,
        eps,
        atol,
        rtol,
        raise_exception,
        nondet_tol=nondet_tol,
        check_undefined_grad=check_undefined_grad,
    )
