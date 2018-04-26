import unittest
from collections import Counter


class TestLKLJScoringDatabase(unittest.TestCase):
    def test_ljlk_defs(self):
        from tmol.database import default

        db = default.scoring.ljlk
        atom_type_counts = Counter(n.name for n in db.atom_type_parameters)

        for at in atom_type_counts:
            assert atom_type_counts[at] == 1, \
                f"Duplicate ljlk type parameter: {at}"

        for at in db.atom_type_parameters:
            assert at.name in default.chemical.atom_types
