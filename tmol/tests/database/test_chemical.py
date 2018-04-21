import unittest


class TestChemicalDatabase(unittest.TestCase):
    def test_residue_defs(self):
        from tmol.database import default

        atom_types = set(default.chemical.atom_types)
        assert len(default.chemical.atom_types) == len(atom_types), \
            "Duplicate atom types."

        for r in default.chemical.residues:
            for a in r.atoms:
                assert a.atom_type in atom_types, f"Invalid atom type. res: {r.name3} atom: {a}"

            valid_icoor_names = (
                set(a.name for a in r.atoms).union({"UPPER", "LOWER"})
            )

            for icoor in r.icoors:
                assert icoor.name in valid_icoor_names
                assert icoor.parent in valid_icoor_names
                assert icoor.grand_parent in valid_icoor_names
                assert icoor.great_grand_parent in valid_icoor_names
