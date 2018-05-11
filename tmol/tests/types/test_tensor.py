import pytest

import numpy
import torch

import attr

from tmol.types.tensor import TensorGroup, TensorType
import tmol.types.tensor as tensor
from tmol.types.array import NDArray
from tmol.types.torch import Tensor


def test_tensortype_instancecheck():
    assert isinstance(NDArray(float)[:], TensorType)
    assert isinstance(Tensor(float)[:], TensorType)


def test_attr_checking():
    @attr.s(auto_attribs=True, frozen=True, slots=True)
    class NoBroadcastShape(TensorGroup):
        no_broadcast: NDArray(int)[:]
        broadcast: NDArray(int)[..., 2]

    @attr.s(auto_attribs=True, frozen=True, slots=True)
    class NonFixedShape(TensorGroup):
        non_fixed: NDArray(int)[..., :]
        broadcast: NDArray(int)[..., 2]

    # Safely initialize with manual input
    inv = NoBroadcastShape(
        numpy.arange(10),
        numpy.arange(20).reshape(10, 2),
    )

    r = inv[:5]
    numpy.testing.assert_allclose(r.no_broadcast, numpy.arange(5))
    numpy.testing.assert_allclose(
        r.broadcast,
        numpy.arange(20).reshape(10, 2)[:5]
    )

    # Require broadcast shape for constructors
    with pytest.raises(TypeError):
        NoBroadcastShape.empty(5)

    with pytest.raises(TypeError):
        NonFixedShape.empty(5)

    @attr.s(auto_attribs=True, frozen=True, slots=True)
    class InvalidAttrType(TensorGroup):
        array: NDArray(int)[...]
        other: str

    # Detect attribute type errors on indexing
    v = InvalidAttrType(numpy.arange(10), "arange")

    with pytest.raises(TypeError):

        v[:5]


def test_nested_group():
    @attr.s(auto_attribs=True, frozen=True, slots=True)
    class SubGroup(TensorGroup):
        a: Tensor(float)[...]
        b: Tensor(float)[..., 5]

    @attr.s(auto_attribs=True, frozen=True, slots=True)
    class MultiGroup(TensorGroup):
        g1: SubGroup
        g2: SubGroup
        idx: Tensor(int)[..., 2]

    s = MultiGroup.ones((3, 5))
    assert s.g1.a.shape == (3, 5)
    assert s.g1.b.shape == (3, 5, 5)
    assert s.g2.a.shape == (3, 5)
    assert s.g2.b.shape == (3, 5, 5)
    assert s.idx.shape == (3, 5, 2)

    assert s[0].g1.a.shape == (5, )
    assert s[0].g1.b.shape == (5, 5)
    assert s[0].g2.a.shape == (5, )
    assert s[0].g2.b.shape == (5, 5)
    assert s[0].idx.shape == (5, 2)

    sg = SubGroup.full((5), numpy.pi)
    numpy.testing.assert_allclose(sg.a, torch.full((5, ), numpy.pi))
    numpy.testing.assert_allclose(sg.b, torch.full((5, 5), numpy.pi))

    # FrozenInstance prevents direct assignment to members
    with pytest.raises(attr.exceptions.FrozenInstanceError):
        s[0].g1 = sg

    # Slice assignment updates member tensors
    s[0].g1[:] = sg

    numpy.testing.assert_allclose(s.g1.a[0], torch.full((5, ), numpy.pi))
    numpy.testing.assert_allclose(s.g1.b[0], torch.full((5, 5), numpy.pi))
    numpy.testing.assert_allclose(s.g1.a[1], torch.full((5, ), 1.0))
    numpy.testing.assert_allclose(s.g1.b[1], torch.full((5, 5), 1.0))


def test_tensorgroup_smoke():
    @attr.s(auto_attribs=True, frozen=True, slots=True)
    class NumpyTensorGroup(TensorGroup):
        a: NDArray(float)[...]
        coord: NDArray(float)[..., 3]

    val = NumpyTensorGroup.zeros((10, ))
    numpy.testing.assert_allclose(val.a, numpy.zeros(10))
    numpy.testing.assert_allclose(val.coord, numpy.zeros((10, 3)))

    val = NumpyTensorGroup.ones(10)
    numpy.testing.assert_allclose(val.a, numpy.ones(10))
    numpy.testing.assert_allclose(val.coord, numpy.ones((10, 3)))

    val = NumpyTensorGroup.full(10, numpy.pi)
    numpy.testing.assert_allclose(val.a, numpy.full(10, numpy.pi))
    numpy.testing.assert_allclose(val.coord, numpy.full((10, 3), numpy.pi))

    val = NumpyTensorGroup.empty(10)
    assert val.a.shape == (10, )
    assert val.coord.shape == (10, 3)

    val = NumpyTensorGroup.empty((10, 100))
    assert val.a.shape == (10, 100)
    assert val.coord.shape == (10, 100, 3)

    @attr.s(auto_attribs=True, frozen=True, slots=True)
    class TorchTensorGroup(TensorGroup):
        a: Tensor(float)[...]
        coord: Tensor(float)[..., 3]

    val = TorchTensorGroup.zeros(10)
    numpy.testing.assert_allclose(val.a, torch.zeros((10, )))
    numpy.testing.assert_allclose(val.coord, torch.zeros((10, 3)))

    val = TorchTensorGroup.ones(10)
    numpy.testing.assert_allclose(val.a, torch.ones((10, )))
    numpy.testing.assert_allclose(val.coord, torch.ones((10, 3)))

    val = TorchTensorGroup.full(10, numpy.pi)
    numpy.testing.assert_allclose(val.a, torch.full((10, ), numpy.pi))
    numpy.testing.assert_allclose(val.coord, torch.full((10, 3), numpy.pi))

    val = TorchTensorGroup.empty(10)
    assert val.a.shape == (10, )
    assert val.coord.shape == (10, 3)

    val = TorchTensorGroup.empty((10, 100))
    assert val.a.shape == (10, 100)
    assert val.coord.shape == (10, 100, 3)


def test_tensorgroup_cat():
    @attr.s(auto_attribs=True, frozen=True, slots=True)
    class S(TensorGroup):
        a: Tensor(float)[...]
        b: NDArray(float)[..., 5]

    @attr.s(auto_attribs=True, frozen=True, slots=True)
    class M(TensorGroup):
        s: S
        foo: Tensor(int)[..., 2]

    m1 = M.full((3, 3), 1)
    m2 = M.full((3, 7), 2)

    # Simple valid cat
    m3 = tensor.cat((m1, m2), dim=1)

    assert len(m3) == 3

    assert m3.foo.shape == (3, 10, 2)
    numpy.testing.assert_array_equal(
        m3.foo, numpy.concatenate((m1.foo, m2.foo), axis=1)
    )

    assert m3.s.shape == (3, 10)
    numpy.testing.assert_array_equal(
        m3.s.a, numpy.concatenate((m1.s.a, m2.s.a), axis=1)
    )
    numpy.testing.assert_array_equal(
        m3.s.b, numpy.concatenate((m1.s.b, m2.s.b), axis=1)
    )

    # Negative dimension spec
    m3 = tensor.cat((m1, m2), dim=-1)

    assert m3.foo.shape == (3, 10, 2)
    numpy.testing.assert_array_equal(
        m3.foo, numpy.concatenate((m1.foo, m2.foo), axis=1)
    )

    assert m3.s.shape == (3, 10)
    numpy.testing.assert_array_equal(
        m3.s.a, numpy.concatenate((m1.s.a, m2.s.a), axis=1)
    )
    numpy.testing.assert_array_equal(
        m3.s.b, numpy.concatenate((m1.s.b, m2.s.b), axis=1)
    )

    # Invalid dimension, mismatch shape
    with pytest.raises(RuntimeError):
        tensor.cat((m1, m2))

    # Invalid dimension, exeeds bounds
    with pytest.raises(RuntimeError):
        tensor.cat((m1, m2), dim=3)

    with pytest.raises(ValueError):
        tensor.cat((m1, m2), dim=-5)
