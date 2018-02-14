import unittest
import traitlets
from traitlets import TraitError

import numpy

import tmol.traits

class testShapeSpec(unittest.TestCase):
    def test(self):
        s = tmol.traits.SpecGenerator()
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

if __name__ == "__main__":
    unittest.main()
