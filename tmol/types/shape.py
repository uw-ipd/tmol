"""
Shape Specifications
--------------------

Shape specifications are intended to allow a reasonable description of common
array shapes, with an emphasis on functionally relevant shape cases. This
includes dimensionality, shape of specific dimensions, implied broadcastable
dimensions, contiguous ordering (ie. C vs F ordering), and density.

``ndim`` and ``shape``
~~~~~~~~~~~~~~~~~~~~~~

Basic dimensionality and shape requirements are specified via slices.
Dimensions may be unconstrained or constrainted to a fixed shape.

- ``[:]`` - ndim 1, any shape
- ``[3]`` or ``[:3]`` - ndim 1, shape (3,)
- ``[:,3]`` - ndim 2, shape (n,3)
- ``[3,3]`` - ndim 2, shape (3,3)

Broadcastable Dimensions
~~~~~~~~~~~~~~~~~~~~~~~~

Optional dimensions are represented by an elipsis. This should generally be
limited to *only* [implicitly
broadcastable](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
upper dimensions.

- ``[...,:3]`` - ndim 1+, shape ([any]*, 3)
- ``[...,:,:3]`` - ndim 2+, shape ([any]+, 3, 3)
- ``[...,:3,:3]`` - ndim 2+, shape ([any]+, 3, 3)

Stride and Contiguous Dimensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Memory layout constraints can be used to specify contiguous dimensions and
their ordering. Dense dimensions are specified in the standard
(inner|c|numpy|row-major) order or or in (outer|fortran|col-major) order.
Any number of dimensions, starting from either ordering, can be specified as
dense. Elements of dense dimensions are contiguous support a raveled view.

The exact syntax for this dimension specification is unclear. The "inline",
utilizing the slide step component:


- ``[::1]`` - ndim 1, any shape, contiguous
- ``[:,::1]`` - ndim 2, any shape, c-contiguous
- ``[::1,:3]`` - ndim 2, shape (any, 3), f-contiguous
- ``[:4:1j,:4:1]`` - ndim 2, shape (4,4), fully dense, c-contiguous
- ``[:,:4:1j,:4:1]`` - ndim 3, shape (n, 4,4), 2-dense, c-contiguous

Or a standard, ordering/density:

- ``t[:].dense()`` - ndim 1, any shape, contiguous
- ``t[:,:].order('c')`` - ndim 2, any shape, c-contiguous
- ``t[:,:].dense().order('c')`` - ndim 2, any shape, fully dense, c-contiguous
- ``t[:,3].dense(1).order('f')`` - ndim 2, shape (any, 3), f-contiguous
- ``[:,4,4].dense(2).order('c')`` - ndim 3, shape (n, 4,4), 2-dense, c-contiguous
"""

import attr


@attr.s(frozen=True, slots=True)
class Dim:
    @staticmethod
    def _to_size(size):
        if size in (None, Ellipsis):
            return size
        elif isinstance(size, slice):
            if size.start is not None:
                raise ValueError("Invalid slice.", size)
            if size.step is not None:
                raise ValueError("Invalid slice.", size)
            return size.stop
        else:
            return int(size)

    size = attr.ib(converter=_to_size.__func__)

    @size.validator
    def _valid_size(self, _, size):
        if size is None:
            return
        elif size is Ellipsis:
            return
        else:
            if not isinstance(size, int) or size < 1:
                raise ValueError("size must be None, Ellipsis, or >1", size)

    def __str__(self):
        if self.size is Ellipsis:
            return "..."
        elif self.size is None:
            return ":"
        else:
            return str(self.size)


@attr.s(slots=True, frozen=True)
class Shape:
    class Factory(object):
        @staticmethod
        def __getitem__(args):
            if not isinstance(args, tuple):
                args = (args,)

            return Shape(list(args))

    spec = Factory()

    @staticmethod
    def _to_dims(dims):
        return tuple(map(Dim, dims))

    dims = attr.ib(converter=_to_dims.__func__)

    @dims.validator
    def _valid_dims(self, _, dims):
        if len(dims) < 1:
            raise ValueError("Must have at least one dim.")
        if any(e.size is Ellipsis for e in dims[1:]):
            raise ValueError("Invalid dims", dims)

    @classmethod
    def create(cls, dims):
        cls(dims=list(map(Dim, dims)))

    def __init__(self, dims):
        dims = list(map(Dim, dims))

    def validate(self, shape):
        dims = list(self.dims)
        adims = list(shape)

        if len(dims) < len(adims):
            if dims[0].size is not Ellipsis:
                raise ValueError(
                    f"No implied broadcast to shape. "
                    f"expected: {self!s} received: {adims}"
                )

            dims = [Dim(Ellipsis)] * (len(adims) - len(dims)) + dims
        elif len(dims) > len(adims):
            if not len(dims) - len(adims) == 1:
                raise ValueError(
                    f"Fewer than expected dims in shape. "
                    f"expected: {self!s} received: {adims}"
                )
            elif not dims[0].size is Ellipsis:
                raise ValueError(
                    f"No implied broadcast to shape. "
                    f"expected: {self!s} received: {adims}"
                )
            dims = dims[1:]

        assert len(dims) == len(adims)

        for d, a in zip(dims, adims):
            if d.size and d.size is not Ellipsis and d.size is not a:
                raise ValueError(
                    f"Invalid dimension size. "
                    f"expected: {self!s} received: {adims} dim: {d} size: {a}"
                )

        return True

    def __call__(self, trait, value):
        """Validate shape for given array."""
        try:
            self.validate(value.shape)
            return value
        except ValueError as vex:
            raise ValueError(f"Invalid shape: {value.shape} expected: {self}") from vex

    def __str__(self):
        return "[{}]".format(",".join(map(str, self.dims)))

    def _repr_pretty_(self, p, cycle):
        assert not cycle

        p.text(str(self))
