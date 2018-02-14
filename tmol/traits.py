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


class SpecGenerator:
    def __getitem__(cls, args):
        if not isinstance(args, tuple):
            args = (args,)

        return ShapeSpec(list(args))

class Dim(traitlets.HasTraits):
    size = traitlets.Int(None, allow_none=True)
    contig = traitlets.Bool(False)
    dense  = traitlets.Bool(False)
    implied = traitlets.Bool(False)

    def __init__(self, size=None, contig=False, dense=False, implied=False):
        if isinstance(size, int):
            if not size >= 1:
                raise ValueError("Invalid size.", size)
            v = None
        else:
            v = size
            size = None

        if isinstance(v, slice):
            if v.start is not None:
                raise ValueError("Invalid slice.", v)

            if v.stop is None:
                size = None
            else:
                if not v.stop >= 1:
                    raise ValueError("Invalid slice.", v)

                size = v.stop

            if v.step is None:
                contig = False
            else:
                if v.step == 1:
                    contig = True
                    dense = True
                elif v.step == 1j:
                    contig = False
                    dense = True
                else:
                    raise ValueError("Invalid slice.", v)
        elif isinstance(v, type(Ellipsis)):
            size = None
            contig = False
            dense = False
            implied = True
        elif isinstance(v, Dim):
            size = v.size
            contig = v.contig
            implied = v.implied
        elif v is None:
            pass
        else:
            raise ValueError("Unknown value type.", v)

        traitlets.HasTraits.__init__(self, size=size, contig=contig, dense=dense, implied=implied)

    def __repr__(self):
        return "Dim(size={size}, contig={contig}, implied={implied})".format(**self._trait_values)

class ShapeSpec(traitlets.HasTraits):
    dims = traitlets.List(traitlets.Instance(Dim),minlen=1)

    def __init__(self, dims):
        dims = list(map(Dim, dims))

        if any(e.implied for e in dims[1:]):
            raise ValueError("Invalid dims", dims)
        if dims[0].contig and any(d.contig for d in dims[1:]):
            raise ValueError("Invalid extra contiguous dims", dims)
        if dims[-1].contig and any(d.contig for d in dims[:-1]):
            raise ValueError("Invalid extra contiguous dims", dims)

        traitlets.HasTraits.__init__(self, dims=dims)

    def validate(self, array):
        dims = list(self.dims)
        adims = list(array.shape)

        if len(dims) < len(adims):
            if dims[0].implied is not True:
                raise ValueError("No implied broadcast in dims", dims, adims)

            dims = [Dim(implied=True)] * (len(adims) - len(dims)) + dims
        elif len(dims) > len(adims):
            if not len(dims) - len(adims) == 1:
                raise ValueError("Not enough array dims", dims, adims)
            elif not dims[0].implied:
                raise ValueError("No implied broadcast in dims", dims, adims)
            dims = dims[1:]

        assert len(dims) == len(adims)

        for d, a in zip(dims, adims):
            if d.size and a is not d.size:
                raise ValueError("Invalid dimension size.", d, a)

        if any(d.contig or d.dense for d in dims):
            raise NotImplementedError

        return True

    def matches(self, array):
        try:
            self.validate(array)
        except ValueError:
            return False

    def __repr__(self):
        return "[%s]" % ",".join(map(repr, self.dims))

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
                        ValueError, msg=repr(v)):
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
