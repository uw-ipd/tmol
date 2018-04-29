import attr

from .shape import Shape

from tmol.extern.typeguard import CustomTypeGuard
from .converters import get_converter


class TensorType:
    def __getitem__(self, shape):
        if not isinstance(shape, tuple):
            shape = (shape, )

        shape = Shape(shape)

        return attr.evolve(self, shape=shape)

    def validate(self, value):
        if not isinstance(value, self._tensortype):
            raise TypeError(
                f"expected {self._tensortype!r}, received {type(value)!r}"
            )
        if not value.dtype == self.dtype:
            raise TypeError(
                f"expected {self.dtype!r}, received {value.dtype!r}"
            )
        self.shape.validate(value.shape)

        return True

    def __instancecheck__(self, obj):
        """Overloaded isinstance to check type and shape."""

        try:
            self.validate(obj)
            return True
        except (TypeError, ValueError):
            return False

    def __repr__(self):
        return f"{type(self).__name__}({self.dtype!r}){self.shape!s}"

    def __name__(self):
        return repr(self)

    def _expanded_shape(self, shape):
        if isinstance(shape, int):
            shape = (shape, )

        broadcast, *subshape = [d.size for d in self.shape.dims]
        if broadcast is not Ellipsis:
            raise TypeError(f"Tensor must be of broadcast shape: {self}")
        if not all(isinstance(i, int) for i in subshape):
            raise TypeError(
                f"Tensor must be fixed size in non-broadcast dims: {self}"
            )

        return shape + tuple(subshape)

    def zeros(self, shape, **kwargs):
        return self._module.zeros(
            self._expanded_shape(shape),
            dtype=self.dtype,
            **kwargs,
        )

    def ones(self, shape, **kwargs):
        return self._module.ones(
            self._expanded_shape(shape),
            dtype=self.dtype,
            **kwargs,
        )

    def empty(self, shape, **kwargs):
        return self._module.empty(
            self._expanded_shape(shape),
            dtype=self.dtype,
            **kwargs,
        )

    def full(self, shape, fill_value, **kwargs):
        return self._module.full(
            self._expanded_shape(shape),
            fill_value,
            dtype=self.dtype,
            **kwargs,
        )


class TensorGroup:
    def __getitem__(self, idx):
        if not all(isinstance(a.type, TensorType) or
                   issubclass(a.type, TensorGroup)
                   for a in attr.fields(type(self))): # yapf: disable
            raise TypeError("All fields must be TensorType or TensorGroup.")

        return attr.evolve(
            self, **{
                a.name: getattr(self, a.name)[idx]
                for a in attr.fields(type(self))
            }
        )

    def __setitem__(self, idx, value):
        for a in self.__attrs_attrs__:
            getattr(self, a.name)[idx] = getattr(value, a.name)[idx]

    @classmethod
    def full(cls, shape, fill_value, **kwargs):
        return cls(
            **{
                a.name: a.type.full(shape, fill_value, **kwargs)
                for a in attr.fields(cls)
            },
        )

    @classmethod
    def zeros(cls, shape, **kwargs):
        return cls(
            **{
                a.name: a.type.zeros(shape, **kwargs)
                for a in attr.fields(cls)
            },
        )

    @classmethod
    def ones(cls, shape, **kwargs):
        return cls(
            **{a.name: a.type.ones(shape, **kwargs)
               for a in attr.fields(cls)},
        )

    @classmethod
    def empty(cls, shape, **kwargs):
        return cls(
            **{
                a.name: a.type.empty(shape, **kwargs)
                for a in attr.fields(cls)
            },
        )


CustomTypeGuard.register(TensorType)


@get_converter.register(TensorType)
def convert_tensor(tensor_type):
    return tensor_type.convert
