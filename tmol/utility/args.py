import functools
import inspect
import toolz


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
    def wrapper(*args, **kwargs):
        kwargs = toolz.keyfilter(sig.parameters.__contains__, kwargs)
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
