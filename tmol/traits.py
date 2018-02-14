"""
# Shape Specifications

Shape specifications are intended to allow a reasonable description of common array shapes, with an emphasis on functionally relevant shape cases. This includes dimensionality, shape of specific dimensions, implied broadcastable dimensions, contiguous ordering (ie. C vs F ordering), and density.


## Examples

### `ndim` and `shape`

Basic dimensionality and shape requirements are specified via slices. Dimensions may be unconstrained or constrainted to a fixed shape.

- `[:]` - ndim 1, any shape
- `[3]` or `[:3]` - ndim 1, shape (3,)
- `[:,3]` - ndim 2, shape (n,3)
- `[3,3]` - ndim 2, shape (3,3)

### Broadcastable Dimensions

Optional dimensions are represented by an elipsis. This should generally be limited to *only* [implicitly broadcastable](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html) upper dimensions.

- `[...,:3]` - ndim 1+, shape ([any]*, 3)
- `[...,:,:3]` - ndim 2+, shape ([any]+, 3, 3)
- `[...,:3,:3]` - ndim 2+, shape ([any]+, 3, 3)

### Stride and Contiguous Dimensions

Memory layout constraints can be used to specify contiguous dimensions and their ordering. Dense dimensions are specified in the standard \[inner|c|numpy|row-major\] order or or in \[outer|fortran|col-major\] order. Any number of dimensions, starting from either ordering, can be specified as dense. Elements of dense dimensions are contiguous support a raveled view.

The exact syntax for this dimension specification is unclear. The "inline", utilizing the slide step component:


- `[::1]` - ndim 1, any shape, contiguous
- `[:,::1]` - ndim 2, any shape, c-contiguous
- `[::1,:3]` - ndim 2, shape (any, 3), f-contiguous
- `[:4:1j,:4:1]` - ndim 2, shape (4,4), fully dense, c-contiguous
- `[:,:4:1j,:4:1]` - ndim 3, shape (n, 4,4), 2-dense, c-contiguous

Or a standard, ordering/density:

- `t[:].dense()` - ndim 1, any shape, contiguous
- `t[:,:].order('c')` - ndim 2, any shape, c-contiguous
- `t[:,:].dense().order('c')` - ndim 2, any shape, fully dense, c-contiguous
- `t[:,3].dense(1).order('f')` - ndim 2, shape (any, 3), f-contiguous
- `[:,4,4].dense(2).order('c')` - ndim 3, shape (n, 4,4), 2-dense, c-contiguous
"""

import numpy
import traitlets
from traitlets import Bool, Int, List, Instance, HasTraits, validate, TraitError


class SpecGenerator:
    def __getitem__(cls, args):
        if not isinstance(args, tuple):
            args = (args,)

        return ShapeSpec(list(args))

class Dim(HasTraits):
    size = Int(None, allow_none=True)
    @validate("size")
    def _valid_size(self, proposal):
        size = proposal['value']
        if size is not None and size <= 0:
            raise TraitError('size must be >= 1', size)

        return size

    implied = Bool(False)

    def __init__(self, size=None, implied=False):
        if size is Ellipsis:
            size = None
            implied = True
        elif isinstance(size, slice):
            if size.start is not None:
                raise TraitError("Invalid slice.", size)
            if size.step is not None:
                raise TraitError("Invalid slice.", size)
            size = size.stop
        elif isinstance(size, Dim):
            size = v.size
            implied = v.implied

        HasTraits.__init__(self, size=size, implied=implied)

    def __repr__(self):
        return "Dim(size={size}, implied={implied})".format(**self._trait_values)

    def _repr_pretty_(self, p, cycle):
        assert not cycle

        if self.size is not None:
            p.pretty(size)
            if self.implied:
                p.text("*")
        elif not self.implied:
            p.text(":")
        else:
            p.text("...")

class ShapeSpec(HasTraits):
    dims = List(Instance(Dim), minlen=1)

    def __init__(self, dims):
        dims = list(map(Dim, dims))

        if any(e.implied for e in dims[1:]):
            raise TraitError("Invalid dims", dims)
        HasTraits.__init__(self, dims=dims)

    def validate(self, array):
        dims = list(self.dims)
        adims = list(array.shape)

        if len(dims) < len(adims):
            if dims[0].implied is not True:
                raise TraitError("No implied broadcast in dims", dims, adims)

            dims = [Dim(implied=True)] * (len(adims) - len(dims)) + dims
        elif len(dims) > len(adims):
            if not len(dims) - len(adims) == 1:
                raise TraitError("Not enough array dims", dims, adims)
            elif not dims[0].implied:
                raise TraitError("No implied broadcast in dims", dims, adims)
            dims = dims[1:]

        assert len(dims) == len(adims)

        for d, a in zip(dims, adims):
            if d.size and a is not d.size:
                raise TraitError("Invalid dimension size.", d, a)

        return True

    def __repr__(self):
        return "[%s]" % ",".join(map(repr, self.dims))

    def _repr_pretty_(self, p, cycle):
        assert not cycle

        p.text("[")
        for i, d in enumerate(self.dims):
            if idx:
                p.text(',')
            p.pretty(d)
        p.text("]")

import unittest

class testShapeSpec(unittest.TestCase):
    def test(self):
        s = SpecGenerator()
        a = numpy.empty

        class AssertInvalidSpec:
            def __init__(self, case):
                self.case = case

            def __getitem__(self, v):
                with self.case.assertRaises(
                        TraitError, msg=repr(v)):
                    v = s[v]
                    self.case.fail(repr(v))

        inv = AssertInvalidSpec(self)

        examples = [
                # ndim and shape
                {
                    "spec" : s[:],
                    "valid" : [a(1), a(10)],
                    "invalid" : [a(()), a((2,2)), a((10, 10, 10))],
                    },
                {
                    "spec" : s[:,:],
                    "valid" : [a((10,3)), a((2,2)), a((1, 10))],
                    "invalid" : [a((1,)), a((3,)), a((10, 10, 3))],
                    },
                {
                    "spec" : s[3],
                    "valid" : [a(3)],
                    "invalid" : [a(1), a((1,3)), a((3, 3))]
                    },
                {
                    "spec" : s[:,3],
                    "valid" : [a((1,3)), a((3,3)), a((10, 3))],
                    "invalid" : [
                        a(1), a(3), a((3,1)), a((1, 1, 1))
                        ]
                    },
                {
                    "spec" : s[1,3],
                    "valid" : [a((1,3))],
                    "invalid" : [
                        a(3), a((3, 3)), a((3,1)), a((1, 3, 3))
                        ]
                    },
                {
                    "spec" : s[...,3],
                    "valid" : [a(3), a((100, 1, 3)), a((1,3))],
                    "invalid" : [
                        a((3, 1)), a((1))
                        ]
                    },
                {
                    "spec" : s[...,:,3],
                    "valid" : [a((100, 1, 3)), a((1,3))],
                    "invalid" : [
                        a((3, 1)), a((1)), a(3),
                        ]
                    },

                ]

        invalid_specs = [
                inv[3:],
                inv["test"],
                inv[1, "test"],
                inv[::1, ::1],
                inv[3, ...],
                inv[..., ...],
                ]

        for e in examples:
            spec = e["spec"]

            for v in e["valid"]:
                self.assertValid(spec, v)

            for v in e["invalid"]:
                self.assertInvalid(spec, v)


    def assertValid(self, spec, array, msg = None):
        spec.validate(array)

    def assertInvalid(self, spec, array, msg = None):
        with self.assertRaises(ValueError):
            spec.validate(array)
            self.fail("spec: %r matched invalid array shape: %s" % (spec, array.shape))

import toolz
if __name__ == "__main__":
    unittest.main()
    # test_cases =[
        # case(name)
        # for name, case in globals().items()
        # if isinstance(case, type) and issubclass(case, unittest.TestCase) and name.startswith("test")
        # for name in unittest.getTestCaseNames(case, "test")
    # ]


    # unittest.runner.TextTestRunner().run(unittest.TestSuite(test_cases))
