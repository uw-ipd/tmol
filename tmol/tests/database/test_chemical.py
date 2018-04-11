import unittest
from collections import Counter

class TestChemicalDatabase(unittest.TestCase):
    def test_residue_defs(self):
        from tmol.database import default

        atom_types = set(default.chemical.atom_types)
        assert len(default.chemical.atom_types) == len(atom_types), "Duplicate atom types."

        for r in default.chemical.residues:
            for a in r.atoms:
                assert a.atom_type in atom_types, f"Invalid atom type. res: {r.name3} atom: {a}"

