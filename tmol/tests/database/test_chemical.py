import unittest


class TestChemicalDatabase(unittest.TestCase):
    def test_residue_defs(self):
        from tmol.database import default

        atom_types = set(default.chemical.atom_types)
        assert len(default.chemical.atom_types) == len(atom_types), \
            "Duplicate atom types."

        for r in default.chemical.residues:
            for a in r.atoms:
                assert a.atom_type in atom_types, \
                    f"Invalid atom type. res: {r.name3} atom: {a}"

            atom_names = {a.name for a in r.atoms}
            assert len(atom_names) == len(r.atoms), "atom names not unique."

            connection_names = {c.name for c in r.connections}
            assert len(connection_names) == len(r.connections), \
                "connection names not unique."

            torsion_names = {t.name for t in r.torsions}
            assert len(torsion_names) == len(r.torsions), \
                "torsion names not unique"

            for t in r.torsions:
                for catom in (t.a, t.b, t.c, t.d):
                    assert catom.atom in atom_names
                    assert (
                        catom.connection is None or
                        catom.connection in connection_names
                    ) # yapf: disable
