import pytest

from tmol.utility.args import ignore_unused_kwargs

import numba
import numpy


def test_ignore_unused_kwargs_func():
    """ignore_unused_kwargs allows extra kwargs, not args."""

    @ignore_unused_kwargs
    def foo(a):
        return a

    assert foo(1) == 1
    assert foo(a=1) == 1

    assert foo(a=1, b=2) == 1
    assert foo(1, b=2) == 1

    with pytest.raises(TypeError):
        foo(1, 2)


def test_ignore_unused_kwargs_numba():
    """ignore_unused_kwargs support numba jit & vectorize functions."""

    @ignore_unused_kwargs
    @numba.jit
    def jit_foo(a):
        return a

    assert jit_foo(1) == 1
    assert jit_foo(a=1) == 1

    assert jit_foo(a=1, b=2) == 1
    assert jit_foo(1, b=2) == 1

    with pytest.raises(TypeError):
        jit_foo(1, 2)

    @ignore_unused_kwargs
    @numba.vectorize
    def vector_foo(a):
        return a

    v = numpy.arange(10)

    assert (vector_foo(v) == v).all()
    assert (vector_foo(a=v) == v).all()

    assert (vector_foo(a=v, b=2) == v).all()
    assert (vector_foo(v, b=2) == v).all()

    with pytest.raises(TypeError):
        vector_foo(v, 2)
