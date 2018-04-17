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
from tmol.score.hbond import HBondElementAnalysis, HBondScoreGraph
from tmol.tests.data.pdb import data as test_pdbs

from ..support.rosetta import requires_pyrosetta


def test_bb_identification(bb_hbond_database):
    tsys = read_pdb(test_pdbs["1ubq"])

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
    tsys = read_pdb(test_pdbs["1ubq"])
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
    tsys = read_pdb(test_pdbs["1ubq"])
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


@requires_pyrosetta
def test_bb_pyrosetta_comparison(bb_hbond_database, pyrosetta):
    from tmol.support.scoring.rosetta import PoseScoreWrapper

    rosetta_system = (PoseScoreWrapper.from_pdbstring(test_pdbs["1ubq"]))
    test_system = (
        tmol.system.residue.packed.PackedResidueSystem()
        .from_residues(rosetta_system.tmol_residues)
    )  # yapf: disable

    hbond_graph = HBondScoreGraph(
        hbond_database=bb_hbond_database,
        **tmol.score.system_graph_params(test_system, requires_grad=False)
    )

    rosetta_bb_hbonds = rosetta_system.hbonds.set_index(["a_atom", "h_atom"]
                                                        ).loc[["O", "H"]]
    rosetta_bb_pairs = set(
        map(tuple, rosetta_bb_hbonds[["h_res", "a_res"]].values)
    )

    bb_hbonds = pandas.DataFrame.from_dict(
        dict(
            h_res=test_system.atom_to_res_ind(hbond_graph.hbond_h_ind),
            a_res=test_system.atom_to_res_ind(hbond_graph.hbond_acceptor_ind),
            score=hbond_graph.hbond_scores,
        )
    ).query("score != 0")
    bb_pairs = set(map(tuple, bb_hbonds[["h_res", "a_res"]].values))

    assert rosetta_bb_pairs == bb_pairs


bb_hbond_config = yaml.load(
    """
    global_parameters:
        max_dis : 3.2
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
        cosBAH_short: poly_cosBAH_off
        cosBAH_long: poly_cosBAH_off
        cosBAH2_long: poly_cosBAH_flat_neg0p5
        cosAHD_short: poly_AHD_1j
        cosAHD_long: poly_AHD_1j
    polynomial_parameters:
      - name: poly_cosBAH_off
        dimension: hbgd_cosBAH
        xmin: -1234.0
        xmax: 1.1
        min_val: 1.1
        max_val: 1.1
        root1: 0.0
        root2: 1.01
        degree: 1
        c_a: 0.0
        c_b: nan
        c_c: nan
        c_d: nan
        c_e: nan
        c_f: nan
        c_g: nan
        c_h: nan
        c_i: nan
        c_j: nan
        c_k: nan
      - name: poly_cosBAH_flat_neg0p5
        dimension: hbgd_cosBAH
        xmin: -1234.0
        xmax: 1.1
        min_val: 1.1
        max_val: 1.1
        root1: 0.0
        root2: 1.01
        degree: 1
        c_a: -0.5
        c_b: nan
        c_c: nan
        c_d: nan
        c_e: nan
        c_f: nan
        c_g: nan
        c_h: nan
        c_i: nan
        c_j: nan
        c_k: nan
      - name: poly_AHD_1j
        dimension: hbgd_AHD
        xmin: 1.1435646388
        xmax: 3.1416
        min_val: 1.1
        max_val: 1.1
        root1: 0.7256433000000001
        root2: nan
        degree: 10
        c_a: 0.47683259
        c_b: -9.54524724
        c_c: 83.62557693
        c_d: -420.55867774
        c_e: 1337.19354878
        c_f: -2786.26265686
        c_g: 3803.178227
        c_h: -3278.62879901
        c_i: 1619.04116204
        c_j: -347.50157909
        c_k: nan
"""
)


@pytest.fixture
def bb_hbond_database():
    return cattr.structure(
        bb_hbond_config, tmol.database.scoring.HBondDatabase
    )
