import functools
import inspect
import toolz

import ast
import astor
import types
from itertools import zip_longest
from inspect import Signature, Parameter

import re


@functools.singledispatch
def _signature(f):
    """Resolve inspect.Signature for callable."""
    return inspect.signature(f)


@functools.singledispatch
def _wraps(f):
    """Resolve functools.wraps for callable."""
    return functools.wraps(f)


def ignore_unused_kwargs(func):
    """Ignore kwargs not present in func signature.

    Decorate func with wrapper dropping any kwargs not present in the func
    signature.

    Example:
        Allows function invocation with kwargs bags that are a superset
        of required args::

            >>> @ignore_unused_kwargs(lambda a, b: a + b)(a=1, b=2, c=5)
            3
    """

    sig = _signature(func)

    @_wraps(func)
    def wrapper(*args, **src_kwargs):
        kwargs = toolz.keyfilter(sig.parameters.__contains__, src_kwargs)
        bound_args = sig.bind(*args, **kwargs)
        return func(*bound_args.args, **bound_args.kwargs)

    return wrapper


try:
    import numba

    @_signature.register(numba.npyufunc.dufunc.DUFunc)
    def _dufunc_signature(f):
        """Resolve inspect.Signature for @numba.vectorize."""
        return inspect.signature(f._dispatcher.py_func)

    @_wraps.register(numba.npyufunc.dufunc.DUFunc)
    def _dufunc_wraps(f):
        """Resolve functools.wraps for @numba.vectorize."""
        return functools.wraps(f._dispatcher.py_func)


except ImportError:
    pass

try:
    import numpy

    @_signature.register(numpy.lib.function_base.vectorize)
    def _numpy_vectorize_signature(f):
        return _signature(f.pyfunc)


except ImportError:
    pass


@_signature.register(types.BuiltinFunctionType)
def _builtin_signature(f):
    """Resolve inspect.Signature for builtin types."""

    if getattr(f, "__text_signature__", None):
        return inspect.signature(f)
    else:
        return _pybind11_signature(f)


def _pybind11_doc_signatures(f):
    """Load overload signatures from pybind11 docstring."""
    name = f.__name__

    overloaded = (
        re.search(r"^Overloaded function.", f.__doc__, re.MULTILINE) is not None
    )

    if overloaded:
        doc_signatures = [
            m.group(1)
            for m in re.finditer(fr"^\d+\. ({name}.*)$", f.__doc__, re.MULTILINE)
        ]
    else:
        doc_signatures = [
            m.group(1) for m in re.finditer(fr"^({name}.*)$", f.__doc__, re.MULTILINE)
        ]

    return [_parse_doc_signature(ds) for ds in doc_signatures]


def _sanitize_pybind11_signature(sig):
    """ Sanitize a typeid signature into type annotation.

    Hack attempt to sanitize c++ type signatures into syntactically valid
    python type annotations. Split off the return value annotation then
    coerce c++ namespace and template markers into compatible syntax.
    """

    sub = toolz.curry(re.sub)
    sanitize = toolz.compose(
        *(
            sub(c, p)
            for c, p in (
                (r"::", "."),  # namespace separators
                (r"<", "["),  # template type parameters
                (r">", "]"),
                (r"(\d+)ul", r"\1"),  # Eigen ulong dimension parameters (eg 2ul)
            )
        )
    )

    return "->".join(map(sanitize, sig.split("->")))


def _parse_doc_signature(sig):
    fdef, = ast.parse(f"def {_sanitize_pybind11_signature(sig)}: pass").body

    # Just count the number of arguments w/ default values, no attempt to parse
    # the values.
    args_no_default = len(fdef.args.args) - len(fdef.args.defaults)
    kwargs_no_default = len(fdef.args.kwonlyargs) - len(fdef.args.kw_defaults)

    return Signature(
        [
            Parameter(
                a.arg,
                Parameter.POSITIONAL_OR_KEYWORD,
                default=Parameter.empty if i < args_no_default else True,
                annotation=astor.to_source(a.annotation).strip(),
            )
            for i, a in enumerate(fdef.args.args)
        ]
        + (
            [Parameter(fdef.args.kwarg, Parameter.VAR_KEYWORD)]
            if fdef.args.kwarg
            else []
        )
        + (
            [Parameter(fdef.args.vararg, Parameter.VAR_POSITIONAL)]
            if fdef.args.vararg
            else []
        )
        + [
            Parameter(
                a.arg,
                Parameter.KEYWORD_ONLY,
                default=Parameter.empty if i < kwargs_no_default else True,
                annotation=astor.to_source(a.annotation).strip(),
            )
            for i, a in enumerate(fdef.args.kwonlyargs)
        ]
    )


def _aligned_signature(sigs):
    """Align overload signatures into a single meta-sig, or raise error."""
    combined_params = []
    param_sets = zip_longest(
        *(
            (p.replace(annotation=Parameter.empty) for p in s.parameters.values())
            for s in sigs
        )
    )

    for i, ps in enumerate(param_sets):
        p = set(filter(None, ps))

        if len(p) != 1:
            raise ValueError(
                f"Incompatible params: {ps} index: {i} in signatures:\n{sigs}"
            )

        param = p.pop()

        combined_params.append(
            Parameter(
                param.name,
                param.kind,
                # If the parameter is not present in all signatures mark as
                # having a default value, otherwise use existing default.
                default=param.default if None not in ps else True,
            )
        )

    return Signature(combined_params)


def _pybind11_signature(pybind11_f):
    return _aligned_signature(_pybind11_doc_signatures(pybind11_f))
