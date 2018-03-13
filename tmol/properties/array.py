import numpy
import properties.basic

from .shape import spec, ShapeSpec


class Array(properties.basic.Property):
    """Property for :class:`numpy arrays <numpy.ndarray>`

    **Available keywords** (in addition to those inherited from
    :ref:`Property <property>`):

    * **shape** - Tuple or ShapeSpec that describes the
      allowed shape of the array.
    * **dtype** - Allowed data type for the array.
    * **cast** - Cast data as if needed.
    """

    class_info = 'a numpy array'

    CASTING_OPTIONS = {None, 'equiv', 'safe', 'same_kind', 'unsafe'}

    @property
    def casting(self):
        return getattr(self, '_casting', "safe")

    @casting.setter
    def casting(self, value):
        if value is True:
            value = "unsafe"
        if value is "no":
            value = None
        if not value:
            value = None

        if not value in self.CASTING_OPTIONS:
            raise ValueError(
                "Invalid casting: {} options: {}".format(
                value, self.CASTING_OPTIONS
            ))

        self._casting = value

    @property
    def dtype(self):
        return getattr(self, '_dtype', (float, int))

    @dtype.setter
    def dtype(self, value):
        if isinstance(value, (list)):
            value = (value,)
        if not isinstance(value, (tuple)):
            value = (value,)

        self._dtype = tuple(map(numpy.dtype, value))

    @property
    def shape(self):
        return getattr(self, '_shape', spec[...])

    @shape.setter
    def shape(self, value):
        if not isinstance(value, ShapeSpec):
            value = ShapeSpec(value)

        self._shape = value

    @property
    def info(self):
        if self.shape is None:
            shape_info = '[...]'
        else:
            shape_info = str(self.shape)

        return '{info} of {type} with shape {shp}'.format(
            info=self.class_info,
            type=', '.join([str(t) for t in self.dtype]),
            shp=shape_info,
        )

    def validate(self, instance, value):
        if self.casting and not isinstance(value, numpy.ndarray):
            value = numpy.array(value)

        for dt in self.dtype:
            if value.dtype == dt:
                break
        else:
            if self.casting:
                value = value.astype(self.dtype[0], casting=self.casting)
            else:
                raise ValueError(
                    "Invalid dtype: {} candidates: {}".format(
                    value.dtype, self.dtype)
                )

        self.shape.validate(value.shape)

        return value

    def equal(self, value_a, value_b):
        try:
            if value_a.__class__ is not value_b.__class__:
                return False
            nan_mask = ~np.isnan(value_a)
            if not np.array_equal(nan_mask, ~np.isnan(value_b)):
                return False
            return np.allclose(value_a[nan_mask], value_b[nan_mask], atol=TOL)
        except TypeError:
            return False


    @staticmethod
    def to_json(value, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def from_json(value, **kwargs):
        raise NotImplementedError()
