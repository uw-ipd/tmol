import pytest

from tmol.utility.args import _signature
from tmol.utility.args import ignore_unused_kwargs
from tmol.utility.cpp_extension import load_inline

import inspect

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


def test_ignore_unused_kwargs_numpy():
    """ignore_unused_kwargs support for numpy @vectorize functions"""

    @ignore_unused_kwargs
    @numpy.vectorize
    def vector_foo(a):
        return a

    v = numpy.arange(10)

    assert (vector_foo(v) == v).all()
    assert (vector_foo(a=v) == v).all()

    assert (vector_foo(a=v, b=2) == v).all()
    assert (vector_foo(v, b=2) == v).all()

    with pytest.raises(TypeError):
        vector_foo(v, 2)


def test_ignore_unused_kwargs_pybind11():
    test_source = """
#include <deque>

void template_param(std::deque<int> x) {
  return;
}

at::Tensor tensor_param(at::Tensor x) {
  return x;
}


template<typename Real>
Real overloaded(Real x) {
  return x;
}

template<typename Real>
Real defaults(Real x, Real y = 1) {
  return x;
}

int add_args(int x) {
  return x;
}

int add_args(int x, int y) {
  return x;
}

int invalid_overload(int x, int y_int) {
  return x;
}

int invalid_overload(int x, float y_float) {
  return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace pybind11::literals;
  m.def("tensor_param", &tensor_param,
    "A tensor parameter, includes '::' namespace separator.", "x"_a);

  m.def("template_param", &template_param,
    "A template parameter, includes '<', '>'.", "x"_a);

  m.def("overloaded", &overloaded<float>,
    "Multiple valid overloads.", "x"_a);
  m.def("overloaded", &overloaded<double>,
    "Multiple valid overloads.", "x"_a);

  m.def("defaults", &defaults<float>,
    "Has default values", "x"_a, "y"_a=1);

  m.def("add_args", static_cast<int(*)(int)>(add_args),
    "Additional arguments.", "x"_a);
  m.def("add_args", static_cast<int(*)(int, int)>(add_args),
    "Additional arguments.", "x"_a, "y"_a);

  m.def("invalid_overload", static_cast<int(*)(int, float)>(invalid_overload),
    "Overload with varying arg names.", "x"_a, "y_int"_a);
  m.def("invalid_overload", static_cast<int(*)(int, float)>(invalid_overload),
    "Overload with varying arg names.", "x"_a, "y_float"_a);
}
"""

    c = load_inline("test_ignore_unused_kwargs_pybind11", test_source)

    # Test signature extraction for various combinations of overloads, default
    # values, and type signatures.
    assert _signature(c.template_param) == inspect.signature(lambda x: None)
    assert _signature(c.tensor_param) == inspect.signature(lambda x: None)
    assert _signature(c.overloaded) == inspect.signature(lambda x: None)
    assert _signature(c.defaults) == inspect.signature(lambda x, y=True: None)
    assert _signature(c.add_args) == inspect.signature(lambda x, y=True: None)

    with pytest.raises(ValueError):
        _signature(c.invalid_overload)

    # Test ignore_unused_kwargs for a pybind11-defined function
    overloaded = ignore_unused_kwargs(c.overloaded)

    assert overloaded(1) == 1
    assert overloaded(x=1) == 1

    assert overloaded(x=1, b=2) == 1
    assert overloaded(1, b=2) == 1

    with pytest.raises(TypeError):
        overloaded(1, 2)

    with pytest.raises(TypeError):
        overloaded("expects int or float")
