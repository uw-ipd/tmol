import cattr

import numpy
import pandas

import tmol.score

import tmol.system.restypes as restypes
from tmol.system.packed import PackedResidueSystem
from tmol.system.score import system_bond_graph_inputs

import tmol.database
from tmol.score.hbond.identification import HBondElementAnalysis


def test_ambig_identification(water_box_system: PackedResidueSystem):
    """Tests identification in cases with 'ambiguous' acceptor bases."""

    atom_frame = pandas.DataFrame.from_records(
        water_box_system.atom_metadata
    )[["atom_type", "atom_index", "residue_index"]]

    expected_donors = (
        pandas.merge(
            atom_frame.query("atom_type == 'Owat'"),
            atom_frame.query("atom_type == 'Hwat'"),
            on="residue_index",
            suffixes=("_d", "_h"),
        ).rename(columns={
            "atom_index_d": "d",
            "atom_index_h": "h"
        })
        .sort_values(by=["d", "h"])
        .reset_index(drop=True)
    ) # yapf: disable

    assert len(expected_donors) == len(water_box_system.residues) * 2

    expected_acceptors = (
        atom_frame.query("atom_type == 'Owat'")
        .rename(columns={
            "atom_index": "a",
        })
        .sort_values(by="a")
        .reset_index(drop=True)
    ) # yapf: disable
    assert len(expected_acceptors) == len(water_box_system.residues)

    element_analysis: HBondElementAnalysis = HBondElementAnalysis.setup(
        hbond_database=tmol.database.default.scoring.hbond,
        atom_types=water_box_system.atom_metadata["atom_type"],
        bonds=water_box_system.bonds,
    )

    identified_donors = (
        pandas.DataFrame.from_records(element_analysis.donors)
        .sort_values(by=["d", "h"]).reset_index(drop=True)
    )

    pandas.testing.assert_frame_equal(
        expected_donors[["d", "h"]].astype(int),
        identified_donors[["d", "h"]],
    )

    identified_acceptors = (
        pandas.DataFrame.from_records(element_analysis.sp3_acceptors)
        .sort_values(by="a").reset_index(drop=True)
    )

    pandas.testing.assert_frame_equal(
        expected_acceptors[["a"]].astype(int),
        identified_acceptors[["a"]],
    )

    assert len(element_analysis.sp2_acceptors) == 0
    assert len(element_analysis.ring_acceptors) == 0


def test_bb_identification(bb_hbond_database, ubq_system):
    tsys = ubq_system

    donors = []
    acceptors = []

    for ri, r in zip(tsys.res_start_ind, tsys.residues):
        if r.residue_type.name3 != "PRO":
            donors.append({
                "d": r.residue_type.atom_to_idx["N"] + ri,
                "h": r.residue_type.atom_to_idx["H"] + ri,
                "donor_type": "hbdon_PBA",
            })

        acceptors.append({
            "a": r.residue_type.atom_to_idx["O"] + ri,
            "b": r.residue_type.atom_to_idx["C"] + ri,
            "b0": r.residue_type.atom_to_idx["CA"] + ri,
            "acceptor_type": "hbacc_PBA",
        })

    test_params = system_bond_graph_inputs(tsys)

    hbe = HBondElementAnalysis.setup(
        hbond_database=bb_hbond_database,
        atom_types=test_params["atom_types"],
        bonds=test_params["bonds"],
    )

    pandas.testing.assert_frame_equal(
        pandas.DataFrame.from_records(donors, columns=hbe.donors.dtype.names
                                      ).sort_values("d"),
        pandas.DataFrame.from_records(hbe.donors).sort_values("d")
    )

    pandas.testing.assert_frame_equal(
        pandas.DataFrame.from_records(
            acceptors, columns=hbe.sp2_acceptors.dtype.names
        ).sort_values("a"),
        pandas.DataFrame.from_records(hbe.sp2_acceptors).sort_values("a")
    )


def test_identification_by_ljlk_types():
    db_res = tmol.database.default.chemical.residues
    types = [
        cattr.structure(cattr.unstructure(r), restypes.ResidueType)
        for r in db_res
    ]

    lj_types = {
        t.name: t
        for t in tmol.database.default.scoring.ljlk.atom_type_parameters
    }

    for t in types:
        atom_types = numpy.array([a.atom_type for a in t.atoms])
        bonds = t.bond_indicies

        hbe = HBondElementAnalysis.setup(
            hbond_database=tmol.database.default.scoring.hbond,
            atom_types=atom_types.astype(object),
            bonds=bonds
        )
        identified_donors = set(hbe.donors["d"])
        identified_acceptors = set(
            list(hbe.sp2_acceptors["a"]) + list(hbe.sp3_acceptors["a"]) +
            list(hbe.ring_acceptors["a"])
        )

        for ai, at in enumerate(atom_types):
            if lj_types[at].is_donor:
                assert ai in identified_donors, \
                    f"Unidentified donor. res: {t.name} atom:{t.atoms[ai]}"
            if lj_types[at].is_acceptor:
                assert ai in identified_acceptors, \
                    f"Unidentified acceptor. res: {t.name} atom:{t.atoms[ai]}"
