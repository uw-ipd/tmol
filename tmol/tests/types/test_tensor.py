import pytest

import numpy
import torch

import attr

from tmol.types.tensor import TensorGroup, TensorType
import tmol.types.tensor as tensor
from tmol.types.array import NDArray
from tmol.types.torch import Tensor, like_kwargs

from tmol.tests.torch import requires_cuda


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


@attr.s(auto_attribs=True, frozen=True, slots=True)
class SubGroup(TensorGroup):
    a: Tensor(float)[...]
    b: Tensor(float)[..., 5]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class MultiGroup(TensorGroup):
    g1: SubGroup
    g2: SubGroup
    idx: Tensor(int)[..., 2]


def test_nested_group():
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


def test_tensor_group_reshape():
    # Test reshape operations over nested groups
    start = MultiGroup.full(10, numpy.pi)

    assert start.g1.a.shape == (10, )
    assert start.g1.b.shape == (10, 5)

    assert start.g2.a.shape == (10, )
    assert start.g2.b.shape == (10, 5)

    assert start.idx.shape == (10, 2)

    # Standard reshape, but with a list
    reshape = start.reshape((2, 5))

    assert reshape.g1.a.shape == (2, 5)
    assert reshape.g1.b.shape == (2, 5, 5)

    assert reshape.g2.a.shape == (2, 5)
    assert reshape.g2.b.shape == (2, 5, 5)

    assert reshape.idx.shape == (2, 5, 2)

    # Test args shape and implied dimension
    rereshape = start.reshape(5, -1)
    assert rereshape.g1.a.shape == (5, 2)
    assert rereshape.g1.b.shape == (5, 2, 5)

    assert rereshape.g2.a.shape == (5, 2)
    assert rereshape.g2.b.shape == (5, 2, 5)

    assert rereshape.idx.shape == (5, 2, 2)

    # Test flatten to implied dimension
    restore = reshape.reshape(-1)
    assert restore.g1.a.shape == (10, )
    assert restore.g1.b.shape == (10, 5)

    assert restore.g2.a.shape == (10, )
    assert restore.g2.b.shape == (10, 5)

    assert restore.idx.shape == (10, 2)


def test_tensor_group_invalid_reshape():
    @attr.s(auto_attribs=True)
    class InvalidType(TensorGroup):
        a: int  # Invalid non-tensor member
        b: Tensor(float)[...]
        c: Tensor(float)[..., 2]

    inv = InvalidType(a=1, b=torch.empty(10), c=torch.empty((10, 2)))

    with pytest.raises(TypeError):
        inv.reshape((5, 2))

    valid_tensor = SubGroup.empty(10)
    valid_tensor.reshape(5, 2)
    with pytest.raises(RuntimeError):
        valid_tensor.reshape(20)


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


def test_tensorgroup_to_dtypes():
    @attr.s(auto_attribs=True, frozen=True)
    class STG(TensorGroup):
        s: Tensor("f4")[..., 3, 3]

    @attr.s(auto_attribs=True, frozen=True)
    class TG(TensorGroup):
        s: Tensor("f4")[..., 3]
        d: Tensor("f8")[...]
        sub: STG

    cpu_float = dict(
        dtype=torch.float, layout=torch.strided, device=torch.device("cpu")
    )

    cpu_double = dict(
        dtype=torch.double, layout=torch.strided, device=torch.device("cpu")
    )

    cpu_het_group = TG.full(10, numpy.pi)
    assert like_kwargs(cpu_het_group.s) == cpu_float
    assert like_kwargs(cpu_het_group.d) == cpu_double
    assert like_kwargs(cpu_het_group.sub.s) == cpu_float

    cpu_float_group = cpu_het_group.to(torch.float)
    assert like_kwargs(cpu_float_group.s) == cpu_float
    assert like_kwargs(cpu_float_group.d) == cpu_float
    assert like_kwargs(cpu_float_group.sub.s) == cpu_float

    cpu_double_group = cpu_het_group.to(torch.double)
    assert like_kwargs(cpu_double_group.s) == cpu_double
    assert like_kwargs(cpu_double_group.d) == cpu_double
    assert like_kwargs(cpu_double_group.sub.s) == cpu_double

    # Assert noop if not changes needed
    assert cpu_het_group.to(torch.float) is not cpu_het_group
    assert cpu_float_group.to(torch.float) is cpu_float_group


@requires_cuda
def test_tensorgroup_to_device():
    @attr.s(auto_attribs=True, frozen=True)
    class STG(TensorGroup):
        s: Tensor("f4")[..., 3, 3]

    @attr.s(auto_attribs=True, frozen=True)
    class TG(TensorGroup):
        s: Tensor("f4")[..., 3]
        d: Tensor("f8")[...]
        sub: STG

    cpu_float = dict(
        dtype=torch.float,
        layout=torch.strided,
        device=torch.device("cpu"),
    )
    cpu_double = dict(
        dtype=torch.double,
        layout=torch.strided,
        device=torch.device("cpu"),
    )
    cuda_float = dict(
        dtype=torch.float,
        layout=torch.strided,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    cuda_double = dict(
        dtype=torch.double,
        layout=torch.strided,
        device=torch.device("cuda", torch.cuda.current_device()),
    )

    cpu_het_group = TG.full(10, numpy.pi)
    assert like_kwargs(cpu_het_group.s) == cpu_float
    assert like_kwargs(cpu_het_group.d) == cpu_double
    assert like_kwargs(cpu_het_group.sub.s) == cpu_float

    cuda_het_group = cpu_het_group.to(torch.device("cuda"))
    assert like_kwargs(cuda_het_group.s) == cuda_float
    assert like_kwargs(cuda_het_group.d) == cuda_double
    assert like_kwargs(cuda_het_group.sub.s) == cuda_float

    cuda_float_group = cpu_het_group.to(torch.device("cuda"), torch.float)
    assert like_kwargs(cuda_float_group.s) == cuda_float
    assert like_kwargs(cuda_float_group.d) == cuda_float
    assert like_kwargs(cuda_float_group.sub.s) == cuda_float

    cuda_double_group = cuda_float_group.to(torch.double)
    assert like_kwargs(cuda_double_group.s) == cuda_double
    assert like_kwargs(cuda_double_group.d) == cuda_double
    assert like_kwargs(cuda_double_group.sub.s) == cuda_double
