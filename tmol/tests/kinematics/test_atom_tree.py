import unittest
from collections import Counter
import tmol.kinematics.AtomTree as atree

class TestAtomTree(unittest.TestCase):
    def test_homogeneous_transform_default_ctor(self):
        ht = atree.HomogeneousTransform()
        self.assertEqual( ht.frame[0,0], 1 )
        self.assertEqual( ht.frame[1,1], 1 )
        self.assertEqual( ht.frame[2,2], 1 )
        self.assertEqual( ht.frame[3,3], 1 )
        self.assertEqual( ht.frame[0,1], 0 )
        self.assertEqual( ht.frame[0,2], 0 )
        self.assertEqual( ht.frame[0,3], 0 )
        self.assertEqual( ht.frame[1,0], 0 )
        self.assertEqual( ht.frame[1,2], 0 )
        self.assertEqual( ht.frame[1,3], 0 )
        self.assertEqual( ht.frame[2,0], 0 )
        self.assertEqual( ht.frame[2,1], 0 )
        self.assertEqual( ht.frame[2,3], 0 )
        self.assertEqual( ht.frame[3,0], 0 )
        self.assertEqual( ht.frame[3,1], 0 )
        self.assertEqual( ht.frame[3,2], 0 )

if __name__ == "__main__" :
    unittest.main()
