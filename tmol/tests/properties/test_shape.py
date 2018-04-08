import unittest

import numpy

import tmol.properties
import tmol.properties.shape

class testShapeSpec(unittest.TestCase):
    def test(self):
        s = tmol.properties.shape.SpecGenerator()

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
        with self.assertRaises(ValueError):
            spec.validate(array)
            self.fail("spec: %r matched invalid array shape: %s" % (spec, array.shape))

