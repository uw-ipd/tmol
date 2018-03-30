import unittest
import numpy

from tmol.tests.data.pdb import data as test_pdbs

from tmol.system.residue import ResidueReader, PackedResidueSystem

class testResidueSystem(unittest.TestCase):
    def test_smoke_io(self):
        test_names = ["1QYS", "1UBQ"]

        for tname in test_names:
            pdb = test_pdbs[tname]
            residues = ResidueReader().parse_pdb(pdb)
            system = PackedResidueSystem().from_residues(residues)

if __name__ == "__main__":
    unittest.main()
