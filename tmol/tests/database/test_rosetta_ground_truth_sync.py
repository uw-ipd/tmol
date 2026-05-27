import math

import tmol.database


# semantics from Rosetta (fa_standard + fa_standard_genpot)

EXPECTED_LJLK = {
    "COO": (1.916661, 0.141799, -2.508760, 3.5, 14.653000),
    "CH0": (2.011760, 0.062642, 1.248680, 3.5, 8.998000),
    "CH1": (2.011760, 0.062642, -6.492180, 3.5, 10.686000),
    "CH2": (2.011760, 0.062642, -2.551840, 3.5, 18.331000),
    "CH3": (2.011760, 0.062642, 7.727160, 3.5, 25.855000),
    "NtrR": (1.802452, 0.161725, -4.928020, 3.5, 9.779200),
    "NH2O": (1.802452, 0.161725, -7.666710, 3.5, 15.689000),
    "Narg": (1.802452, 0.161725, -8.696020, 3.5, 15.717000),
    "Npro": (1.802452, 0.161725, -1.511110, 3.5, 3.718100),
    "Owat": (1.550000, 0.159100, -5.460600, 3.5, 10.800000),
    "ONH2": (1.548662, 0.182924, -5.035010, 3.5, 10.102000),
    "OOC": (1.492871, 0.099873, -10.208220, 3.5, 9.995600),
    "S": (1.975967, 0.455970, -4.898020, 3.5, 17.640000),
    "SH1": (1.975967, 0.455970, 2.079450, 3.5, 23.240000),
    "Nbb": (1.802452, 0.161725, -12.846650, 3.5, 15.992000),
    "CAbb": (2.011760, 0.062642, 4.449450, 3.5, 12.137000),
    "CObb": (1.916661, 0.141799, 3.578990, 3.5, 13.221000),
    "OCbb": (1.540580, 0.142417, -9.529210, 3.5, 12.196000),
    # Halogen values follow Rosetta fa_standard_genpot LJ/LK parameters.
    "F": (1.694100, 0.075000, 2.500000, 3.5, 11.500000),
    "Cl": (2.049600, 0.239900, 1.744500, 3.5, 24.400000),
    "Br": (2.197100, 0.325500, -0.057400, 3.5, 35.500000),
    "I": (2.360000, 0.424000, -2.658800, 3.5, 44.600000),
}

EXPECTED_CHEMICAL = {
    "F": {"is_donor": False, "is_acceptor": False, "hybridization": "sp3"},
    "Cl": {"is_donor": False, "is_acceptor": False, "hybridization": "sp3"},
    "Br": {"is_donor": False, "is_acceptor": False, "hybridization": "sp3"},
    "I": {"is_donor": False, "is_acceptor": False, "hybridization": "sp3"},
    "Nam": {"is_donor": True, "is_acceptor": False, "hybridization": "sp3"},
    "Nam2": {"is_donor": True, "is_acceptor": False, "hybridization": "sp3"},
    "Nad": {"is_donor": True, "is_acceptor": True, "hybridization": "ring"},
    "Nad3": {"is_donor": False, "is_acceptor": True, "hybridization": "sp3"},
    "Nim": {"is_donor": False, "is_acceptor": True, "hybridization": "ring"},
    "Nin": {"is_donor": True, "is_acceptor": True, "hybridization": "ring"},
    "Ohx": {"is_donor": True, "is_acceptor": True, "hybridization": "sp3"},
    "NG3": {"is_donor": True, "is_acceptor": False, "hybridization": None},
    "Sth": {"is_donor": True, "is_acceptor": False, "hybridization": None},
    "Npro": {"is_donor": False, "is_acceptor": False, "hybridization": None},
}


# Frank generic list semantics (unioned with Rosetta where applicable).
EXPECTED_GENERIC_UNION = {
    "Nad": {"is_donor": True, "is_acceptor": True, "hybridization": "ring"},
    "Nad3": {"is_donor": False, "is_acceptor": True, "hybridization": "sp3"},
    "Nam": {"is_donor": True, "is_acceptor": False, "hybridization": "sp3"},
    "Nam2": {"is_donor": True, "is_acceptor": False, "hybridization": "sp3"},
    "Ngu1": {"is_donor": True, "is_acceptor": False, "hybridization": None},
    "Ngu2": {"is_donor": True, "is_acceptor": False, "hybridization": None},
    "Nim": {"is_donor": False, "is_acceptor": True, "hybridization": "ring"},
    "Nin": {"is_donor": True, "is_acceptor": True, "hybridization": "ring"},
    "NG1": {"is_donor": False, "is_acceptor": True, "hybridization": "sp2"},
    "NG2": {"is_donor": False, "is_acceptor": True, "hybridization": "ring"},
    "NG21": {"is_donor": True, "is_acceptor": False, "hybridization": None},
    "NG22": {"is_donor": True, "is_acceptor": False, "hybridization": None},
    "NG3": {"is_donor": True, "is_acceptor": False, "hybridization": None},
    "Oad": {"is_donor": False, "is_acceptor": True, "hybridization": "sp2"},
    "Oal": {"is_donor": False, "is_acceptor": True, "hybridization": "sp2"},
    "Oat": {"is_donor": False, "is_acceptor": True, "hybridization": "sp2"},
    "Oet": {"is_donor": False, "is_acceptor": True, "hybridization": "sp3"},
    "Ofu": {"is_donor": False, "is_acceptor": True, "hybridization": "ring"},
    "Ohx": {"is_donor": True, "is_acceptor": True, "hybridization": "sp3"},
    "Ont": {"is_donor": False, "is_acceptor": True, "hybridization": "sp2"},
    "OG2": {"is_donor": False, "is_acceptor": True, "hybridization": "sp2"},
    "OG3": {"is_donor": False, "is_acceptor": True, "hybridization": "sp3"},
    "OG31": {"is_donor": False, "is_acceptor": True, "hybridization": "sp3"},
    "Sth": {"is_donor": True, "is_acceptor": False, "hybridization": None},
}


def test_ljlk_rosetta_hybrid_ground_truth(
    default_database: tmol.database.ParameterDatabase,
):
    params = {p.name: p for p in default_database.scoring.ljlk.atom_type_parameters}
    for atom_type, expected in EXPECTED_LJLK.items():
        assert atom_type in params, f"Missing LJLK parameter for {atom_type}"
        actual = params[atom_type]
        got = (
            actual.lj_radius,
            actual.lj_wdepth,
            actual.lk_dgfree,
            actual.lk_lambda,
            actual.lk_volume,
        )
        assert all(
            math.isclose(g, e, rel_tol=0.0, abs_tol=1e-6) for g, e in zip(got, expected)
        ), f"LJLK mismatch for {atom_type}: expected {expected}, got {got}"


def test_chemical_rosetta_hybrid_semantics(
    default_database: tmol.database.ParameterDatabase,
):
    atom_types = {t.name: t for t in default_database.chemical.atom_types}

    for atom_type, expected in EXPECTED_CHEMICAL.items():
        assert atom_type in atom_types, f"Missing atom type {atom_type}"
        atom = atom_types[atom_type]
        assert atom.is_donor == expected["is_donor"]
        assert atom.is_acceptor == expected["is_acceptor"]
        assert atom.acceptor_hybridization == expected["hybridization"]


def test_hbond_generic_donor_alignment(
    default_database: tmol.database.ParameterDatabase,
):
    donor_atoms = {
        d.d
        for d in default_database.scoring.hbond.donor_atom_types
        if d.d
        in {
            "Nad",
            "Nam",
            "Nam2",
            "Nin",
            "NG21",
            "NG22",
            "NG3",
            "Ngu1",
            "Ngu2",
            "Ohx",
            "Sth",
        }
    }
    assert "Nad" in donor_atoms
    assert "NG3" in donor_atoms
    assert "Nin" in donor_atoms
    assert "Sth" in donor_atoms


def test_hbond_generic_acceptor_alignment(
    default_database: tmol.database.ParameterDatabase,
):
    acceptor_atoms = {
        a.a
        for a in default_database.scoring.hbond.acceptor_atom_types
        if a.a in {"Nad", "Nad3", "Nim", "Nin", "Ohx"}
    }
    assert "Nad" in acceptor_atoms
    assert "Nad3" in acceptor_atoms
    assert "Nim" in acceptor_atoms
    assert "Nin" in acceptor_atoms
    assert "Ohx" in acceptor_atoms


def test_chemical_generic_union_semantics(
    default_database: tmol.database.ParameterDatabase,
):
    atom_types = {t.name: t for t in default_database.chemical.atom_types}
    for atom_type, expected in EXPECTED_GENERIC_UNION.items():
        assert atom_type in atom_types, f"Missing generic atom type {atom_type}"
        atom = atom_types[atom_type]
        assert atom.is_donor == expected["is_donor"], atom_type
        assert atom.is_acceptor == expected["is_acceptor"], atom_type
        assert atom.acceptor_hybridization == expected["hybridization"], atom_type


def test_hbond_generic_union_lists_match_chemical(
    default_database: tmol.database.ParameterDatabase,
):
    expected_generic_donors = {
        name for name, vals in EXPECTED_GENERIC_UNION.items() if vals["is_donor"]
    }
    expected_generic_acceptors = {
        name for name, vals in EXPECTED_GENERIC_UNION.items() if vals["is_acceptor"]
    }

    hbond = default_database.scoring.hbond
    observed_generic_donors = {
        d.d for d in hbond.donor_atom_types if d.d in EXPECTED_GENERIC_UNION
    }
    observed_generic_acceptors = {
        a.a for a in hbond.acceptor_atom_types if a.a in EXPECTED_GENERIC_UNION
    }

    assert observed_generic_donors == expected_generic_donors
    assert observed_generic_acceptors == expected_generic_acceptors
