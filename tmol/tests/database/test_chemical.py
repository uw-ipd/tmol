import unittest
from collections import Counter

class TestChemicalDatabase(unittest.TestCase):
    def test_residue_defs(self):
        from tmol.database import default

        atom_types = set(default.chemical.atom_types)
        assert len(default.chemical.atom_types) == len(atom_types), "Duplicate atom types."

        atypenames = [ x[0] for x in atom_types ]
        for r in default.chemical.residues:
            for a in r.atoms:
                assert a.atom_type in atypenames, f"Invalid atom type. res: {r.name3} atom: {a}"
