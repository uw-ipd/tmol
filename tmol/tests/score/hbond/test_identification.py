import cattr

import numpy
import pandas

import tmol.score

import tmol.system.restypes as restypes
from tmol.system.packed import PackedResidueSystem
from tmol.system.score_support import bonded_atoms_for_system

import tmol.database
from tmol.score.hbond.identification import HBondElementAnalysis


def test_ambig_identification(
    default_database: tmol.database.ParameterDatabase,
    water_box_system: PackedResidueSystem,
):
    """Tests identification in cases with 'ambiguous' acceptor bases."""

    atom_frame = pandas.DataFrame.from_records(water_box_system.atom_metadata)[
        ["atom_type", "atom_index", "residue_index"]
    ]

    expected_donors = (
        pandas.merge(
            atom_frame.query("atom_type == 'Owat'"),
            atom_frame.query("atom_type == 'Hwat'"),
            on="residue_index",
            suffixes=("_d", "_h"),
        )
        .rename(columns={"atom_index_d": "d", "atom_index_h": "h"})
        .sort_values(by=["d", "h"])
        .reset_index(drop=True)
    )

    assert len(expected_donors) == len(water_box_system.residues) * 2

    expected_acceptors = (
        atom_frame.query("atom_type == 'Owat'")
        .rename(columns={"atom_index": "a"})
        .sort_values(by="a")
        .reset_index(drop=True)
    )
    assert len(expected_acceptors) == len(water_box_system.residues)

    element_analysis = HBondElementAnalysis.setup_from_database(
        chemical_database=default_database.chemical,
        hbond_database=default_database.scoring.hbond,
        atom_types=water_box_system.atom_metadata["atom_type"],
        bonds=water_box_system.bonds,
    )

    identified_donors = (
        pandas.DataFrame.from_records(element_analysis.donors)
        .sort_values(by=["d", "h"])
        .reset_index(drop=True)
    )

    pandas.testing.assert_frame_equal(
        expected_donors[["d", "h"]].astype(int), identified_donors[["d", "h"]]
    )

    identified_acceptors = (
        pandas.DataFrame.from_records(element_analysis.acceptors)
        .sort_values(by="a")
        .reset_index(drop=True)
    )

    pandas.testing.assert_frame_equal(
        expected_acceptors[["a"]].astype(int), identified_acceptors[["a"]]
    )


def test_bb_identification(default_database, bb_hbond_database, ubq_system):
    tsys = ubq_system

    donors = []
    acceptors = []

    for ri, r in zip(tsys.res_start_ind, tsys.residues):
        if r.residue_type.name3 != "PRO":
            donors.append(
                {
                    "d": r.residue_type.atom_to_idx["N"] + ri,
                    "h": r.residue_type.atom_to_idx["H"] + ri,
                    "donor_type": "hbdon_PBA",
                }
            )

        acceptors.append(
            {
                "a": r.residue_type.atom_to_idx["O"] + ri,
                "b": r.residue_type.atom_to_idx["C"] + ri,
                "b0": r.residue_type.atom_to_idx["CA"] + ri,
                "acceptor_type": "hbacc_PBA",
            }
        )

    test_params = bonded_atoms_for_system(tsys)

    hbe = HBondElementAnalysis.setup_from_database(
        chemical_database=default_database.chemical,
        hbond_database=bb_hbond_database,
        atom_types=test_params["atom_types"][0],
        bonds=test_params["bonds"][:, 1:],
    )

    def _t(d):
        return tuple(tuple(r.items()) for r in d)

    hbe_donors = pandas.DataFrame.from_records(hbe.donors).to_dict(orient="records")
    assert len(_t(hbe_donors)) == len(set(_t(hbe_donors)))
    assert {(r["d"], r["h"]): r for r in hbe_donors} == {
        (r["d"], r["h"]): r for r in donors
    }

    hbe_acceptors = pandas.DataFrame.from_records(hbe.acceptors).to_dict(
        orient="records"
    )
    assert len(_t(hbe_acceptors)) == len(set(_t(hbe_acceptors)))
    assert {r["a"]: r for r in hbe_acceptors} == {r["a"]: r for r in acceptors}


def test_identification_by_chemical_types(
    default_database: tmol.database.ParameterDatabase,
):
    """Hbond donor/acceptor identification covers all donors and accceptor atom
    types in the chemical database."""
    db_res = default_database.chemical.residues
    residue_types = [
        cattr.structure(cattr.unstructure(r), restypes.ResidueType) for r in db_res
    ]

    atom_types = {t.name: t for t in default_database.chemical.atom_types}

    for rt in residue_types:
        res_atom_types = numpy.array([a.atom_type for a in rt.atoms])
        bonds = rt.bond_indicies

        hbe = HBondElementAnalysis.setup_from_database(
            chemical_database=default_database.chemical,
            hbond_database=default_database.scoring.hbond,
            atom_types=res_atom_types.astype(object),
            bonds=bonds,
        )
        identified_donors = set(hbe.donors["d"])
        identified_acceptors = set(hbe.acceptors["a"])

        for ai, at in enumerate(res_atom_types):
            if atom_types[at].is_donor:
                assert (
                    ai in identified_donors
                ), f"Unidentified donor. res: {rt.name} atom:{rt.atoms[ai]}"
            if atom_types[at].is_acceptor:
                assert (
                    ai in identified_acceptors
                ), f"Unidentified acceptor. res: {rt.name} atom:{rt.atoms[ai]}"
