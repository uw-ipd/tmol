import toolz
import itertools


def test_hbond_defs():
    from tmol.database import default

    db = default.scoring.hbond

    poly_params = set(p.name for p in db.polynomial_parameters)
    assert len(poly_params) == len(db.polynomial_parameters)

    for pair in db.pair_parameters:
        assert pair.AHdist in poly_params
        assert pair.cosAHD in poly_params
        assert pair.cosBAH in poly_params

    pair_params = set((p.don_chem_type, p.acc_chem_type)
                      for p in db.pair_parameters)
    assert len(pair_params) == len(db.pair_parameters)

    donor_types = set(g.donor_type for g in db.atom_groups.donors)
    acceptor_types = toolz.reduce(
        set.union, (
            set(g.acceptor_type for g in db.atom_groups.sp2_acceptors),
            set(g.acceptor_type for g in db.atom_groups.sp3_acceptors),
            set(g.acceptor_type for g in db.atom_groups.ring_acceptors),
        )
    )

    donor_weights = set(p.name for p in db.don_weights)
    for d in donor_types:
        assert d in donor_weights

    acceptor_weights = set(p.name for p in db.acc_weights)
    for a in acceptor_types:
        assert a in acceptor_weights

    for d, a in itertools.product(donor_types, acceptor_types):
        assert (d, a) in pair_params
