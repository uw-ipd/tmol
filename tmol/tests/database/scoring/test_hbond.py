import itertools
import tmol.database


def test_hbond_defs(default_database: tmol.database.ParameterDatabase):
    db = default_database.scoring.hbond

    atom_types = {t.name: t for t in default_database.chemical.atom_types}

    # All donor types have unique names, every defined atom has a valid donor
    # type, an atom type defined in the chemical database, and that atom
    # type is flagged as a donor.
    donor_types = {t.name: t for t in db.donor_type_params}
    assert len(donor_types) == len(db.donor_type_params), "donor types not unique."

    for dt in db.donor_type_params:
        assert dt.weight

    for da in db.donor_atom_types:
        assert da.donor_type in donor_types
        assert da.d in atom_types
        assert atom_types[da.d].is_donor

    # All acceptor types have unique names, every defined atom has a valid
    # acceptor type, non-none hybridization, an atom type defined in the
    # chemical database, and the atom is flagged as is_acceptor with a
    # hybridization state.
    acceptor_types = {t.name: t for t in db.acceptor_type_params}
    assert len(acceptor_types) == len(
        db.acceptor_type_params
    ), "acceptor types not unique."
    for at in db.acceptor_type_params:
        assert at.weight

    for da in db.acceptor_atom_types:
        assert da.acceptor_type in acceptor_types
        assert da.a in atom_types
        assert atom_types[da.a].is_acceptor
        assert atom_types[da.a].acceptor_hybridization is not None

    # All polynomial parameters have unique names.
    poly_params = {p.name: p for p in db.polynomial_parameters}
    assert len(poly_params) == len(db.polynomial_parameters)

    # Every set of pair parameters defines references defined donor/acceptor
    # types and polynomal params
    for pair in db.pair_parameters:
        assert pair.donor_type in donor_types
        assert pair.acceptor_type in acceptor_types

        assert pair.AHdist in poly_params
        assert pair.cosAHD in poly_params
        assert pair.cosBAH in poly_params

    # Pair params are defined for every donor/acceptor pair.
    pair_params = {(p.donor_type, p.acceptor_type): p for p in db.pair_parameters}
    assert len(pair_params) == len(db.pair_parameters)
    for d, a in itertools.product(donor_types, acceptor_types):
        assert (d, a) in pair_params
