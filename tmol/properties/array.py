import copy

import torch
import torch.autograd
import numpy

import properties.basic

from ..types.shape import Shape


class Array(properties.basic.Property):
    """Property for :class:`numpy arrays <numpy.ndarray>`

    **Available keywords** (in addition to those inherited from
    :ref:`Property <property>`):

    * **shape** - Tuple or Shape that describes the
      allowed shape of the array.
    * **dtype** - Allowed data type for the array.
    * **cast** - Cast data as if needed.
    """

    class_info = 'a numpy array'

    CAST_OPTIONS = {None, 'equiv', 'safe', 'same_kind', 'unsafe'}

    @property
    def cast(self):
        return getattr(self, '_cast', "safe")

    @cast.setter
    def cast(self, value):
        if value is True:
            value = "unsafe"
        if value is "no":
            value = None
        if not value:
            value = None

        if value not in self.CAST_OPTIONS:
            raise ValueError(
                "Invalid cast: {} options: {}".format(
                    value, self.CAST_OPTIONS
                )
            )

        self._cast = value

    @property
    def dtype(self):
        return getattr(self, '_dtype', (float, int))

    @dtype.setter
    def dtype(self, value):
        if isinstance(value, (list)):
            value = (value, )
        if not isinstance(value, (tuple)):
            value = (value, )

        self._dtype = tuple(map(numpy.dtype, value))

    @property
    def shape(self):
        return getattr(self, '_shape', Shape.spec[...])

    @shape.setter
    def shape(self, value):
        if not isinstance(value, Shape):
            value = Shape(value)

        self._shape = value

    def __getitem__(self, shape):
        if not isinstance(shape, tuple):
            shape = (shape, )

        shape = Shape(shape)

        new_prop = copy.copy(self)
        new_prop.shape = shape
        return new_prop

    @property
    def info(self):
        return '{class_info}(shape={shape}, dtype={dtype})'.format(
            class_info=self.class_info,
            dtype=', '.join([str(t) for t in self.dtype]),
            shape=str(self.shape),
        )

    def validate(self, instance, value):
        if self.cast and not isinstance(value, numpy.ndarray):
            value = numpy.array(value)

        for dt in self.dtype:
            if value.dtype == dt:
                break
        else:
            if self.cast:
                value = value.astype(
                    self.dtype[0], casting=self.cast if self.cast else "no"
                )
            else:
                raise ValueError(
                    "Invalid dtype: {} candidates: {}".format(
                        value.dtype, self.dtype
                    )
                )

        self.shape.validate(value.shape)

        return value

    def equal(self, a, b):
        return a is b

    @staticmethod
    def to_json(value, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def from_json(value, **kwargs):
        raise NotImplementedError()


class TensorT(properties.Instance):
    def __init__(self, doc, *args, **kwargs):
        super(TensorT, self).__init__(doc, instance_class=torch.Tensor)

    def equal(self, a, b):
        return a is b


class VariableT(properties.Instance):
    def __init__(self, doc, *args, **kwargs):
        super(VariableT, self).__init__(
            doc, instance_class=torch.autograd.Variable
        )

    def equal(self, a, b):
        return a is b
