import cattr

import numpy
import pandas

import tmol.score

import tmol.system.restypes as restypes
from tmol.system.packed import PackedResidueSystem, PackedResidueSystemStack
from tmol.system.score_support import bonded_atoms_for_system, stacked_bonded_atoms_for_system

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

    bonds = numpy.full((water_box_system.bonds.shape[0], 3), 0, dtype=numpy.int64)
    bonds[:, 1:] = water_box_system.bonds

    element_analysis = HBondElementAnalysis.setup_from_database(
        chemical_database=default_database.chemical,
        hbond_database=default_database.scoring.hbond,
        atom_types=water_box_system.atom_metadata["atom_type"][None, :],
        bonds=bonds,
    )

    identified_donors = (
        pandas.DataFrame.from_records(element_analysis.donors[0])
        .sort_values(by=["d", "h"])
        .reset_index(drop=True)
    )

    pandas.testing.assert_frame_equal(
        expected_donors[["d", "h"]].astype(int), identified_donors[["d", "h"]]
    )

    identified_acceptors = (
        pandas.DataFrame.from_records(element_analysis.acceptors[0])
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
        atom_types=test_params["atom_types"],
        bonds=test_params["bonds"],
    )

    def _t(d):
        return tuple(tuple(r.items()) for r in d)

    # numpy.set_printoptions(threshold=10000)
    # print("hbe.donors")
    # print(hbe.donors[0])
    # print("donors expected")
    # print(donors)
    
    hbe_donors = pandas.DataFrame.from_records(hbe.donors[0]).to_dict(orient="records")
    assert len(_t(hbe_donors)) == len(set(_t(hbe_donors)))
    assert {(r["d"], r["h"]): r for r in hbe_donors} == {
        (r["d"], r["h"]): r for r in donors
    }

    hbe_acceptors = pandas.DataFrame.from_records(hbe.acceptors[0]).to_dict(
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
        bonds = numpy.zeros([rt.bond_indicies.shape[0], 3], dtype=numpy.int64)
        bonds[:, 1:] = rt.bond_indicies

        hbe = HBondElementAnalysis.setup_from_database(
            chemical_database=default_database.chemical,
            hbond_database=default_database.scoring.hbond,
            atom_types=res_atom_types.astype(object)[None, :],
            bonds=bonds,
        )
        identified_donors = set(hbe.donors["d"][0])
        identified_acceptors = set(hbe.acceptors["a"][0])

        for ai, at in enumerate(res_atom_types):
            if atom_types[at].is_donor:
                assert (
                    ai in identified_donors
                ), f"Unidentified donor. res: {rt.name} atom:{rt.atoms[ai]}"
            if atom_types[at].is_acceptor:
                assert (
                    ai in identified_acceptors
                ), f"Unidentified acceptor. res: {rt.name} atom:{rt.atoms[ai]}"

def test_jagged_identification(ubq_res, default_database):
    ubq4 = PackedResidueSystem.from_residues(ubq_res[:4])
    ubq6 = PackedResidueSystem.from_residues(ubq_res[:6])
    twoubq = PackedResidueSystemStack((ubq4, ubq6))

    params4 = bonded_atoms_for_system(ubq4)
    params6 = bonded_atoms_for_system(ubq6)
    params_both = stacked_bonded_atoms_for_system(
        twoubq,
        stack_depth=2,
        system_size=int(ubq6.system_size))
    
    hbe4 = HBondElementAnalysis.setup_from_database(
        chemical_database=default_database.chemical,
        hbond_database=default_database.scoring.hbond,
        atom_types=params4["atom_types"],
        bonds=params4["bonds"])

    hbe6 = HBondElementAnalysis.setup_from_database(
        chemical_database=default_database.chemical,
        hbond_database=default_database.scoring.hbond,
        atom_types=params6["atom_types"],
        bonds=params6["bonds"])

    hbe_both = HBondElementAnalysis.setup_from_database(
        chemical_database=default_database.chemical,
        hbond_database=default_database.scoring.hbond,
        atom_types=params_both["atom_types"],
        bonds=params_both["bonds"])        

    assert hbe_both.donors.shape == (2, hbe6.donors.shape[1])
    assert hbe_both.acceptors.shape == (2, hbe6.acceptors.shape[1])
    
    numpy.testing.assert_equal(hbe4.donors[0], hbe_both.donors[0,:hbe4.donors.shape[1]])
    numpy.testing.assert_equal(hbe6.donors[0], hbe_both.donors[1,:hbe6.donors.shape[1]])
