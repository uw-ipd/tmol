import unittest
import itertools


class TestHbondScoringDatabase(unittest.TestCase):
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

        poly_params = set(p.name for p in db.polynomial_parameters)
        assert len(poly_params) == len(db.polynomial_parameters)

        for pair in db.pair_parameters:
            assert pair.AHdist in poly_params
            assert pair.cosAHD in poly_params
            assert pair.cosBAH in poly_params

        pair_params = set((p.don_chem_type, p.acc_chem_type)
                          for p in db.pair_parameters)
        assert len(pair_params) == len(db.pair_parameters)

        donor_types = list(db.chemical_types.donors)
        acceptor_types = (
            list(db.chemical_types.sp2_acceptors) +
            list(db.chemical_types.sp3_acceptors) +
            list(db.chemical_types.ring_acceptors)
        )

        donor_weights = set(p.name for p in db.don_weights)
        for d in donor_types:
            assert d in donor_weights

        acceptor_weights = set(p.name for p in db.acc_weights)
        for a in acceptor_types:
            assert a in acceptor_weights

        for d, a in itertools.product(donor_types, acceptor_types):
            assert (d, a) in pair_params
