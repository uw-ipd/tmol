"""Type annotations for multidimensional tensors."""

import inspect
import collections
import functools
import attr
import typing

from .shape import Shape

from .converters import register_converter
from .validators import register_validator

from typish import SubscriptableType

_NOTHING = object()


class _TensorTypeMeta(SubscriptableType):
    def __instancecheck__(self, instance):
        try:
            self.validate(instance)
            return True
        except (TypeError, ValueError):
            return False

    def __repr__(self):
        return f"{self.__name__}[{self.dtype!r}]{self.shape!s}"


class _TensorType(metaclass=_TensorTypeMeta):

    _module: typing.ClassVar
    _tensortype: typing.ClassVar

    _dtype_arg: typing.ClassVar = _NOTHING
    _shape_arg: typing.ClassVar = _NOTHING

    dtype: typing.ClassVar = None
    shape: typing.ClassVar = Shape.spec[...]

    @classmethod
    def _after_subscription(cls, item):
        if cls._dtype_arg is _NOTHING:
            cls._dtype_arg = item
            cls.dtype = cls._convert_dtype(item)
        elif cls._shape_arg is _NOTHING:
            cls._shape_arg = item
            if not isinstance(item, tuple):
                item = (item,)

            cls.shape = Shape(item)
        else:
            raise TypeError(f"{cls} is not a generic class, fully specified.")

        return cls

    @classmethod
    def _convert_dtype(cls, dtype):
        raise NotImplementedError(f"{cls}._convert_dtype")

    @classmethod
    def validate(cls, value):
        if not isinstance(value, cls._tensortype):
            raise TypeError(f"expected {cls._tensortype!r}, received {type(value)!r}")
        if not value.dtype == cls.dtype:
            raise TypeError(f"expected {cls.dtype!r}, received {value.dtype!r}")
        try:
            cls.shape.validate(value.shape)
        except ValueError as err:
            raise TypeError(f"Tensor shape validation failed {str(err)}")

        return True

    @classmethod
    def _expanded_shape(cls, shape):
        if isinstance(shape, int):
            shape = (shape,)
        elif not isinstance(shape, tuple):
            shape = tuple(shape)

        broadcast, *subshape = [d.size for d in cls.shape.dims]
        if broadcast is not Ellipsis:
            raise TypeError(f"Tensor must be of broadcast shape: {cls}")
        if not all(isinstance(i, int) for i in subshape):
            raise TypeError(f"Tensor must be fixed size in non-broadcast dims: {cls}")

        return shape + tuple(subshape)

    @classmethod
    def zeros(cls, shape, **kwargs):
        return cls._module.zeros(cls._expanded_shape(shape), dtype=cls.dtype, **kwargs)

    @classmethod
    def ones(cls, shape, **kwargs):
        return cls._module.ones(cls._expanded_shape(shape), dtype=cls.dtype, **kwargs)

    @classmethod
    def empty(cls, shape, **kwargs):
        return cls._module.empty(cls._expanded_shape(shape), dtype=cls.dtype, **kwargs)

    @classmethod
    def full(cls, shape, fill_value, **kwargs):
        return cls._module.full(
            cls._expanded_shape(shape), fill_value, dtype=cls.dtype, **kwargs
        )

    @classmethod
    def _broadcasted_shape(cls, instance):
        """Get the "broadcast" dimensions of a given instance."""
        broadcast, *subshape = [d.size for d in cls.shape.dims]
        if broadcast is not Ellipsis:
            raise TypeError(f"Tensor must be of broadcast shape: {cls}")

        instance_shape = instance.shape
        if not subshape:
            return instance.shape
        else:
            assert len(instance_shape) >= len(subshape)
            return instance_shape[: -len(subshape)]


class TensorGroup:
    @property
    def _pure_tensor(self):
        return all(
            issubclass(a.type, (_TensorType, TensorGroup))
            for a in attr.fields(type(self))
        )

    def _check_pure_tensor(self):
        if not self._pure_tensor:
            raise TypeError("All fields must be TensorType or TensorGroup.")

    def __getitem__(self, idx):
        self._check_pure_tensor()

        return attr.evolve(
            self,
            **{a.name: getattr(self, a.name)[idx] for a in attr.fields(type(self))},
        )

    def __setitem__(self, idx, value):
        for a in self.__attrs_attrs__:
            getattr(self, a.name)[idx] = getattr(value, a.name)

    def reshape(self, *shape):
        self._check_pure_tensor()

        if len(shape) == 1 and isinstance(shape[0], collections.abc.Sequence):
            shape = tuple(shape[0])
        else:
            shape = tuple(shape)

        reshapes = {
            a.name: a.type._expanded_shape(shape) for a in attr.fields(type(self))
        }

        return attr.evolve(
            self,
            **{
                n: getattr(self, n).reshape(reshapes[n])
                for n in (f.name for f in attr.fields(type(self)))
            },
        )

    @property
    def shape(self):
        return self._broadcasted_shape(self)

    def __len__(self):
        return self.shape[0]

    @classmethod
    def full(cls, shape, fill_value, **kwargs):
        return cls(
            **{
                a.name: a.type.full(shape, fill_value, **kwargs)
                for a in attr.fields(cls)
            }
        )

    @classmethod
    def zeros(cls, shape, **kwargs):
        return cls(**{a.name: a.type.zeros(shape, **kwargs) for a in attr.fields(cls)})

    @classmethod
    def ones(cls, shape, **kwargs):
        return cls(**{a.name: a.type.ones(shape, **kwargs) for a in attr.fields(cls)})

    @classmethod
    def empty(cls, shape, **kwargs):
        return cls(**{a.name: a.type.empty(shape, **kwargs) for a in attr.fields(cls)})

    @classmethod
    def _broadcasted_shape(cls, instance):
        field_shapes = {
            a.type._broadcasted_shape(getattr(instance, a.name))
            for a in attr.fields(cls)
        }

        assert (
            len(field_shapes) == 1
        ), f"Group contained inconsistent shapes: {field_shapes}"

        return field_shapes.pop()

    @classmethod
    def _expanded_shape(cls, shape):
        return shape

    def to(self, *args, **kwargs):
        """Perform dtype/device conversion for all subtensors.

        Note that this may be an invalid operations if the TensorGroup contains
        heterogenous tensor dtypes.

        Performs Tensor dtype and/or device conversion. A :class:`torch.dtype`
        and :class:`torch.device` are inferred from the arguments of
        ``self.to(*args, **kwargs)``.

        If all subtensors already have the correct dtype and device then
        ``self`` is returned.
        """

        self._check_pure_tensor()

        components = attr.asdict(self, recurse=False)
        to_components = {k: v.to(*args, **kwargs) for k, v in components.items()}

        diff = set(n for n in components if components[n] is not to_components[n])
        if not diff:
            return self
        else:
            return attr.evolve(self, **to_components)


def cat(seq, dim=0, out=None):
    first, *rest = seq
    return _cat_internal(first, rest, dim=dim, out=out)


@functools.singledispatch
def _cat_internal(first_element, rest, dim=0, out=None):
    raise NotImplementedError(
        f"Unknown tensor type for cat, needs _cat_internal overload: {first_element}"
    )


@_cat_internal.register(TensorGroup)
def _cat_tensorgroup(first, rest, dim=0, out=None):
    if out is not None:
        raise NotImplementedError("TensorGroup cat does not support 'out' parameter.")
    cls = type(first)

    if dim < 0:
        component_ndims = {
            len(type(v)._broadcasted_shape(v)) for v in (first,) + tuple(rest)
        }
        if len(component_ndims) > 1:
            raise ValueError("Can not broadcast cat with negative dimension.")

        ndim = component_ndims.pop()
        real_dim = ndim + dim

        if real_dim < 0:
            raise ValueError(f"Specified dim: {dim} execeeding ndim: {ndim}")

        dim = real_dim

    return cls(
        **{
            a.name: cat(
                [getattr(first, a.name)] + [getattr(e, a.name) for e in rest], dim=dim
            )
            for a in attr.fields(cls)
        }
    )


register_validator(
    lambda t: inspect.isclass(t) and issubclass(t, _TensorType), lambda t: t.validate
)
register_converter(
    lambda t: inspect.isclass(t) and issubclass(t, _TensorType), lambda t: t.convert
)
