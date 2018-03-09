import unittest
import traitlets
from traitlets import TraitError, HasTraits

import numpy

import tmol.traits
import tmol.traits.shape_traits

class testShapeSpec(unittest.TestCase):
    def test(self):
        s = tmol.traits.shape_traits.SpecGenerator()

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
                    "valid" : [(1,), (10,)],
                    "invalid" : [(), (2,2), (10, 10, 10)],
                    },
                {
                    "spec" : s[:,:],
                    "valid" : [(10,3), (2,2), (1, 10)],
                    "invalid" : [(1,), (3,), (10, 10, 3)],
                    },
                {
                    "spec" : s[3],
                    "valid" : [(3,)],
                    "invalid" : [(1,), (1,3), (3, 3)]
                    },
                {
                    "spec" : s[:,3],
                    "valid" : [(1,3), (3,3), (10, 3)],
                    "invalid" : [ (1,), (3,), (3,1), (1, 1, 1) ]
                    },
                {
                    "spec" : s[1,3],
                    "valid" : [(1,3)],
                    "invalid" : [ (3,), (3, 3), (3,1), (1, 3, 3) ]
                    },
                {
                    "spec" : s[...,3],
                    "valid" : [(3,), (100, 1, 3), (1,3)],
                    "invalid" : [ (3, 1), (1,) ]
                    },
                {
                    "spec" : s[...,:,3],
                    "valid" : [(100, 1, 3), (1,3)],
                    "invalid" : [ (3, 1), (1,), (3,)]
                    },

                ]

        invalid_specs = [
                inv[::1],
                inv[:,::1],
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
        with self.assertRaises(TraitError):
            spec.validate(array)
            self.fail("spec: %r matched invalid array shape: %s" % (spec, array.shape))

    def test_traitlets(self):
        class TType(HasTraits):
            coord = tmol.traits.Array(dtype=float).valid(tmol.traits.shape[3])
            coords = tmol.traits.Array(dtype=float).valid(tmol.traits.shape[:,3])

        t = TType()
        t.coord = list(range(3))
        numpy.testing.assert_allclose(t.coord, numpy.arange(3))
        self.assertEqual(t.coord.dtype, numpy.float)

        t.coords = [list(range(3))]
        numpy.testing.assert_allclose(t.coords, numpy.arange(3).reshape((1, 3)))
        self.assertEqual(t.coords.dtype, numpy.float)

        with self.assertRaises(TraitError):
            t.coords = list(range(3))

        class InvalidTType(HasTraits):
            coord = tmol.traits.Array(numpy.empty(10), dtype=float).valid(tmol.traits.shape[3])

        it = InvalidTType()
        with self.assertRaises(TraitError):
            it.coord

if __name__ == "__main__":
    unittest.main()


