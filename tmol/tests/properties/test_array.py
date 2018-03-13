import unittest
import numpy

import properties

from tmol.properties.array import Array
from tmol.properties.shape import spec

class testArrayProperty(unittest.TestCase):
    def test_prop(self):
        class TType(properties.HasProperties):
            coord = Array("a single coord", dtype=float, shape=spec[3])
            coords = Array("coords", dtype=float, shape=spec[:,3])
            dcoord = Array(
                    "a coord with default",
                    default=list(range(10)),
                    dtype=int,
                    shape=spec[10])

        t = TType()

        numpy.testing.assert_allclose(t.dcoord, numpy.arange(10))
        self.assertEqual(t.dcoord.dtype, numpy.int)

        t.coord = list(range(3))
        numpy.testing.assert_allclose(t.coord, numpy.arange(3))
        self.assertEqual(t.coord.dtype, numpy.float)

        t.coords = [list(range(3))]
        numpy.testing.assert_allclose(t.coords, numpy.arange(3).reshape((1, 3)))
        self.assertEqual(t.coords.dtype, numpy.float)

        with self.assertRaises(ValueError):
            t.coords = list(range(3))

        with self.assertRaises(ValueError):
            class InvalidTType(properties.HasProperties):
                coord = Array("coord",
                        default = numpy.empty(10),
                        dtype = float,
                        shape = spec[3])

if __name__ == "__main__":
    unittest.main()
