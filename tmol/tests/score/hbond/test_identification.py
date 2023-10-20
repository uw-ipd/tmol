import cattr

import numpy
import pandas

import tmol.score

import tmol.chemical.restypes as restypes

from tmol.system.packed import PackedResidueSystem, PackedResidueSystemStack
from tmol.score.modules.bases import ScoreSystem
from tmol.score.modules.hbond import HBondScore
from tmol.score.modules.bonded_atom import (
    bonded_atoms_for_system,
    stacked_bonded_atoms_for_system,
)

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
