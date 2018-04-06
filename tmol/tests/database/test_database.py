import unittest

class TestDatabase(unittest.TestCase):
    def test_residue_defs(self):
        from tmol.database import default

        atom_types = set(default.chemical.atom_types)
        assert len(default.chemical.atom_types) == len(atom_types), "Duplicate atom types."

        for r in default.chemical.residues:
            for a in r.atoms:
                assert a.atom_type in atom_types, f"Invalid atom type. res: {r.name3} atom: {a}"

    def test_hbond_defs(self):
        from tmol.database import default

        db = default.scoring.hbond

        for g in db.atom_groups.donors:
            assert g.donor_type in db.chemical_types.donors

        for g in db.atom_groups.sp2_acceptors:
            assert g.acceptor_type in db.chemical_types.sp2_acceptors

        for g in db.atom_groups.sp3_acceptors:
            assert g.acceptor_type in db.chemical_types.sp3_acceptors

        for g in db.atom_groups.ring_acceptors:
            assert g.acceptor_type in db.chemical_types.ring_acceptors
