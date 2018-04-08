import unittest

from tmol.tests.data.pdb import data as test_pdbs
from tmol.system.residue.io import read_pdb

class testResidueSystem(unittest.TestCase):
    def test_smoke_io(self):
        test_names = ["1QYS", "1UBQ"]

        for tname in test_names:
            pdb = test_pdbs[tname]
            system = read_pdb(pdb)
