import toolz
import yaml
import cattr
import pandas
import numpy
import scipy.spatial
import pytest

import tmol.system.residue.restypes as restypes
import tmol.database

from tmol.score.interatomic_distance import NaiveInteratomicDistanceGraph

from tmol.system.residue.io import read_pdb
from tmol.score.hbond import HBondElementAnalysis, HBondScoreGraph, hbond_donor_sp2_score
import tmol.tests.data.rosetta_baseline as rosetta_baseline
import tmol.tests.data.pdb as test_pdbs

import torch


def test_bb_identification(bb_hbond_database):
    tsys = read_pdb(test_pdbs.data["1ubq"])

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

    test_params = tmol.score.system_graph_params(tsys, requires_grad=False)

    hbond_graph = HBondScoreGraph(
        hbond_database=bb_hbond_database, **test_params
    )

    hbe = hbond_graph.hbond_elements

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


def test_bb_dummy_score(bb_hbond_database):
    tsys = read_pdb(test_pdbs.data["1ubq"])
    test_params = tmol.score.system_graph_params(tsys, requires_grad=False)

    atom_pair_distances = scipy.spatial.distance.squareform(
        NaiveInteratomicDistanceGraph(**test_params).atom_pair_dist
    )

    hbond_graph = HBondScoreGraph(
        hbond_database=bb_hbond_database,
        **test_params,
    )
    hbond_elements = hbond_graph.hbond_elements

    h_i = hbond_elements.donors["h"].reshape((-1, 1))
    sp2_i = hbond_elements.sp2_acceptors["a"].reshape((1, -1))
    sp3_i = hbond_elements.sp3_acceptors["a"].reshape((1, -1))
    ring_i = hbond_elements.ring_acceptors["a"].reshape((1, -1))

    max_dis = hbond_graph.hbond_database.global_parameters.max_dis

    total_count = (
        numpy.count_nonzero(atom_pair_distances[h_i, sp2_i] <= max_dis) +
        numpy.count_nonzero(atom_pair_distances[h_i, sp3_i] <= max_dis) +
        numpy.count_nonzero(atom_pair_distances[h_i, ring_i] <= max_dis)
    )

    assert total_count == hbond_graph.total_hbond


def test_dummy_score():
    tsys = read_pdb(test_pdbs.data["1ubq"])
    test_params = tmol.score.system_graph_params(tsys, requires_grad=False)

    atom_pair_distances = scipy.spatial.distance.squareform(
        NaiveInteratomicDistanceGraph(**test_params).atom_pair_dist
    )

    hbond_graph = HBondScoreGraph(**test_params)
    hbond_elements = hbond_graph.hbond_elements

    h_i = hbond_elements.donors["h"].reshape((-1, 1))
    sp2_i = hbond_elements.sp2_acceptors["a"].reshape((1, -1))
    sp3_i = hbond_elements.sp3_acceptors["a"].reshape((1, -1))
    ring_i = hbond_elements.ring_acceptors["a"].reshape((1, -1))

    max_dis = hbond_graph.hbond_database.global_parameters.max_dis

    total_count = (
        numpy.count_nonzero(atom_pair_distances[h_i, sp2_i] <= max_dis) +
        numpy.count_nonzero(atom_pair_distances[h_i, sp3_i] <= max_dis) +
        numpy.count_nonzero(atom_pair_distances[h_i, ring_i] <= max_dis)
    )

    assert total_count == hbond_graph.total_hbond


def test_identification_by_ljlk_types():
    db_res = tmol.database.default.chemical.residues
    types = [
        cattr.structure(cattr.unstructure(r), restypes.ResidueType)
        for r in db_res
    ]
    assert len(types) == 21

    lj_types = {
        t.name: t
        for t in tmol.database.default.scoring.ljlk.atom_type_parameters
    }

    for t in types:
        atom_types = numpy.array([a.atom_type for a in t.atoms])
        bonds = t.bond_indicies

        hbe = HBondElementAnalysis(atom_types=atom_types, bonds=bonds).setup()
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


def test_bb_pyrosetta_comparison(bb_hbond_database, pyrosetta):
    rosetta_system = rosetta_baseline.data["1ubq"]

    test_system = (
        tmol.system.residue.packed.PackedResidueSystem()
        .from_residues(rosetta_system.tmol_residues)
    )  # yapf: disable
    hbond_graph = HBondScoreGraph(
        hbond_database=bb_hbond_database,
        **tmol.score.system_graph_params(test_system, requires_grad=False)
    )

    # Extract list of hbonds from packed system into summary table
    # via atom metadata
    h_i = hbond_graph.hbond_h_ind
    a_i = hbond_graph.hbond_acceptor_ind
    tmol_hbonds = pandas.DataFrame.from_dict({
        "h": h_i,
        "a": a_i,
        "h_res": test_system.atom_metadata["residue_index"][h_i],
        "h_atom": test_system.atom_metadata["atom_name"][h_i],
        "a_res": test_system.atom_metadata["residue_index"][a_i],
        "a_atom": test_system.atom_metadata["atom_name"][a_i],
        "score": hbond_graph.hbond_scores,
    }).query("score != 0").set_index(["a", "h"])
    del h_i, a_i

    # Merge with named atom index to get atom indicies in packed system
    # hbonds columns: ["a_atom", "a_res", "h_atom", "h_res", "energy"]
    named_atom_index = (
        pandas.DataFrame(test_system.atom_metadata)
        .set_index(["residue_index", "atom_name"])["atom_index"]
    )
    rosetta_hbonds = toolz.curried.reduce(pandas.merge)((
        (
            rosetta_system.hbonds.set_index(["a_atom", "h_atom"])
            .loc["O", "H"].reset_index()
        ),
        (
            named_atom_index.rename_axis(["a_res", "a_atom"])
            .to_frame("a").reset_index()
        ),
        (
            named_atom_index.rename_axis(["h_res", "h_atom"])
            .to_frame("h").reset_index()
        ),
    )).set_index(["a", "h"])

    # Extract subsets via index operations.
    rosetta_not_tmol = rosetta_hbonds.loc[
        (rosetta_hbonds.index.difference(tmol_hbonds.index))
    ]
    tmol_not_rosetta = tmol_hbonds.loc[
        (tmol_hbonds.index.difference(rosetta_hbonds.index))
    ]

    # Report difference via set operator.
    assert set(rosetta_hbonds.index.tolist()
               ) == set(tmol_hbonds.index.tolist()), (
                   f"Mismatched bb hbond identification:\n"
                   f"unidentified:\n{rosetta_not_tmol}\n"
                   f"extra:\n{tmol_not_rosetta}"
               )


bb_hbond_config = yaml.load(
    """
    global_parameters:
        max_dis : 4.2
        hb_sp2_range_span: 1.6
        hb_sp2_BAH180_rise: 0.75
        hb_sp2_outer_width: 0.357
    atom_groups:
        donors:
            - { d: Nbb, h: HNbb, donor_type: hbdon_PBA }
        sp2_acceptors:
            - { a: OCbb, b: CObb, b0: CAbb, acceptor_type: hbacc_PBA }
        sp3_acceptors: []
        ring_acceptors: []
    chemical_types:
        donors:
            - hbdon_PBA
        sp2_acceptors:
            - hbacc_PBA
        sp3_acceptors: []
        ring_acceptors: []
    pair_parameters:
      - don_chem_type: hbdon_PBA
        acc_chem_type: hbacc_PBA
        AHdist: hbpoly_ahdist_aGLY_dGLY_9gt3_hesmooth_min1p6
        cosBAH: poly_cosBAH_off
        cosAHD: poly_AHD_1j
    don_weights:
      - name: hbdon_PBA
        weight: 1.41
    acc_weights:
      - name: hbacc_PBA
        weight: 1.08
    polynomial_parameters:
      - name: hbpoly_ahdist_aGLY_dGLY_9gt3_hesmooth_min1p6
        dimension: hbgd_AHdist
        xmin: 1.38403812683
        xmax: 2.9981039433
        min_val: 1.1
        max_val: 1.1
        degree: 10
        c_a: 0.0 
        c_b: -0.5307601
        c_c: 6.47949946
        c_d: -22.39522814
        c_e: -55.14303544
        c_f: 708.30945242
        c_g: -2619.49318162
        c_h: 5227.8805795
        c_i: -6043.31211632
        c_j: 3806.04676175
        c_k: -1007.66024144
      - name: poly_cosBAH_off
        dimension: hbgd_cosBAH
        xmin: -1234.0
        xmax: 1.1
        min_val: 1.1
        max_val: 1.1
        degree: 1
        c_a: 0.0
        c_b: 0.0
        c_c: 0.0
        c_d: 0.0
        c_e: 0.0
        c_f: 0.0
        c_g: 0.0
        c_h: 0.0
        c_i: 0.0
        c_j: 0.0
        c_k: 0.0
      - name: poly_AHD_1j
        dimension: hbgd_AHD
        xmin: 1.1435646388
        xmax: 3.1416
        min_val: 1.1
        max_val: 1.1
        degree: 10
        c_a: 0.0
        c_b: 0.47683259
        c_c: -9.54524724
        c_d: 83.62557693
        c_e: -420.55867774
        c_f: 1337.19354878
        c_g: -2786.26265686
        c_h: 3803.178227
        c_i: -3278.62879901
        c_j: 1619.04116204
        c_k: -347.50157909
"""
)


def test_bb_single_hbond():
    hbpoly_ahdist_aGLY_dGLY_9gt3_hesmooth_min1p6 = torch.tensor([[
        0.0, -0.5307601, 6.47949946, -22.39522814, 55.14303544, 708.30945242,
        -2619.49318162, 5227.8805795, -6043.31211632, 3806.04676175,
        -1007.66024144
    ]])
    poly_cosBAH_off = torch.tensor([[
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]])
    poly_AHD_1j = torch.tensor([[
        0.0, 0.47683259, -9.54524724, 83.62557693, 420.55867774,
        1337.19354878 - 2786.26265686, 3803.178227, -3278.62879901,
        1619.04116204, -347.50157909
    ]])

    atomD = torch.tensor([[0.409, 3.986, -1.316]])
    atomH = torch.tensor([[0.913, 3.226, -0.880]])
    atomA = torch.tensor([[1.383, 2.339, -0.529]])
    atomB = torch.tensor([[2.009, 1.420, 0.000]])
    atomB0 = torch.tensor([[1.458, 0.000, 0.000]])

    energy = tmol.score.hbond.hbond_donor_sp2_score(
        atomD, atomH, atomA, atomB, atomB0,
        hbpoly_ahdist_aGLY_dGLY_9gt3_hesmooth_min1p6, poly_cosBAH_off,
        poly_AHD_1j, 4.2
    )

    assert (False)


@pytest.fixture
def bb_hbond_database():
    return cattr.structure(
        bb_hbond_config, tmol.database.scoring.HBondDatabase
    )
