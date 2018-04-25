import toolz
import yaml
import cattr
import pandas
import numpy
import pytest

import tmol.system.residue.restypes as restypes
import tmol.database

from tmol.system.residue.io import read_pdb
from tmol.score.hbond import HBondElementAnalysis, HBondScoreGraph
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
            .sort_index().loc["O", "H"].reset_index()
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
        hb_sp3_softmax_fade: 2.5
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


def test_sp2_single_hbond():
    hbpoly_ahdist_aGLY_dGLY_9gt3_hesmooth_min1p6 = torch.tensor([[
        0.0, -0.5307601, 6.47949946, -22.39522814, -55.14303544, 708.30945242,
        -2619.49318162, 5227.8805795, -6043.31211632, 3806.04676175,
        -1007.66024144
    ]])
    hbpoly_ahdist_aGLY_dGLY_9gt3_hesmooth_min1p6_range = torch.tensor([[
        1.38403812683, 2.9981039433
    ]])
    hbpoly_ahdist_aGLY_dGLY_9gt3_hesmooth_min1p6_bounds = torch.tensor([[
        1.1, 1.1
    ]])

    poly_cosBAH_off = torch.tensor([[
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]])
    poly_cosBAH_off_range = torch.tensor([[-1234.0, 1.1]])
    poly_cosBAH_off_bounds = torch.tensor([[1.1, 1.1]])

    poly_AHD_1j = torch.tensor([[
        0.0, 0.47683259, -9.54524724, 83.62557693, -420.55867774,
        1337.19354878, -2786.26265686, 3803.178227, -3278.62879901,
        1619.04116204, -347.50157909
    ]])

    poly_AHD_1j_range = torch.tensor([[1.1435646388, 3.1416]])
    poly_AHD_1j_bounds = torch.tensor([[1.1, 1.1]])

    atomD = torch.tensor([[-0.337, 3.640, -1.365]])
    atomH = torch.tensor([[-0.045, 3.220, -0.496]])
    atomA = torch.tensor([[0.929, 2.820, 1.149]])
    atomB = torch.tensor([[1.369, 1.690, 1.360]])
    atomB0 = torch.tensor([[1.060, 0.538, 0.412]])

    donwt = torch.tensor([[1.45]])
    accwt = torch.tensor([[1.19]])

    energy = tmol.score.hbond.hbond_donor_sp2_score(
        atomD, atomH, atomA, atomB, atomB0, accwt, donwt,
        hbpoly_ahdist_aGLY_dGLY_9gt3_hesmooth_min1p6,
        hbpoly_ahdist_aGLY_dGLY_9gt3_hesmooth_min1p6_range,
        hbpoly_ahdist_aGLY_dGLY_9gt3_hesmooth_min1p6_bounds, poly_cosBAH_off,
        poly_cosBAH_off_range, poly_cosBAH_off_bounds, poly_AHD_1j,
        poly_AHD_1j_range, poly_AHD_1j_bounds, 1.6, 0.75, 0.357, 4.2
    )

    assert (float(energy.data[0]) == pytest.approx(-2.41, 0.01))


def test_sp3_single_hbond():
    hbpoly_ahdist_aSER_dGLY_9gt3_hesmooth_min1p6 = torch.tensor([[
        0.0, -1.32847415, 22.67528654, -172.53450064, 770.79034865,
        -2233.48829652, 4354.38807288, -5697.35144236, 4803.38686157,
        -2361.48028857, 518.28202382
    ]])

    hbpoly_ahdist_aSER_dGLY_9gt3_hesmooth_min1p6_range = torch.tensor([[
        1.38565621563, 2.74160605537
    ]])
    hbpoly_ahdist_aSER_dGLY_9gt3_hesmooth_min1p6_bounds = torch.tensor([[
        1.1, 1.1
    ]])

    poly_cosBAH_6i = torch.tensor([[
        0.0, -0.82209300, -3.75364636, 46.88852157, -129.54405640,
        146.69151428, -67.60598792, 2.91683129, 9.26673173, -3.84488178,
        0.05706659
    ]])
    poly_cosBAH_6i_range = torch.tensor([[-0.0193738506669, 1.1]])
    poly_cosBAH_6i_bounds = torch.tensor([[1.1, 1.1]])

    poly_AHD_1i = torch.tensor([[
        0.0,
        -0.18888801,
        3.48241679,
        -25.65508662,
        89.57085435,
        -95.91708218,
        -367.93452341,
        1589.69047020,
        -2662.35821350,
        2184.40194483,
        -723.28383545,
    ]])
    poly_AHD_1i_range = torch.tensor([[1.59914724347, 3.1416]])
    poly_AHD_1i_bounds = torch.tensor([[1.1, 1.1]])

    atomD = torch.tensor([[-1.447, 4.942, -3.149]])
    atomH = torch.tensor([[-1.756, 4.013, -2.912]])
    atomA = torch.tensor([[-2.196, 2.211, -2.339]])
    atomB = torch.tensor([[-3.156, 2.109, -1.327]])
    atomB0 = torch.tensor([[-1.436, 1.709, -2.035]])

    donwt = torch.tensor([[1.45]])
    accwt = torch.tensor([[1.15]])

    energy = tmol.score.hbond.hbond_donor_sp3_score(
        atomD, atomH, atomA, atomB, atomB0, accwt, donwt,
        hbpoly_ahdist_aSER_dGLY_9gt3_hesmooth_min1p6,
        hbpoly_ahdist_aSER_dGLY_9gt3_hesmooth_min1p6_range,
        hbpoly_ahdist_aSER_dGLY_9gt3_hesmooth_min1p6_bounds, poly_cosBAH_6i,
        poly_cosBAH_6i_range, poly_cosBAH_6i_bounds, poly_AHD_1i,
        poly_AHD_1i_range, poly_AHD_1i_bounds, 2.5, 4.2
    )

    assert (float(energy.data[0]) == pytest.approx(-2.00, 0.01))


def test_ring_single_hbond():

    hbpoly_ahdist_aHIS_dGLY_9gt3_hesmooth_min1p6 = torch.tensor([[
        0.0, -1.68095217, 21.31894078, -107.72203494, 251.81021758,
        -134.07465831, -707.64527046, 1894.62827430, -2156.85951846,
        1216.83585872, -275.48078944
    ]])
    hbpoly_ahdist_aHIS_dGLY_9gt3_hesmooth_min1p6_range = torch.tensor([[
        1.01629363411, 2.58523052904
    ]])
    hbpoly_ahdist_aHIS_dGLY_9gt3_hesmooth_min1p6_bounds = torch.tensor([[
        1.1, 1.1
    ]])

    poly_cosBAH_7 = torch.tensor([[
        0.0, 0.0, -27.942923450028001, 136.039920253367995,
        -268.069590567470016, 275.400462507918974, -153.502076215948989,
        39.741591385461000, 0.693861510121000, -3.885952320499000,
        1.024765090788892
    ]])
    poly_cosBAH_7_range = torch.tensor([[-0.0193738506669, 1.1]])
    poly_cosBAH_7_bounds = torch.tensor([[1.1, 1.1]])

    poly_AHD_1i = torch.tensor([[
        0.0,
        -0.18888801,
        3.48241679,
        -25.65508662,
        89.57085435,
        -95.91708218,
        -367.93452341,
        1589.69047020,
        -2662.35821350,
        2184.40194483,
        -723.28383545,
    ]])
    poly_AHD_1i_range = torch.tensor([[1.59914724347, 3.1416]])
    poly_AHD_1i_bounds = torch.tensor([[1.1, 1.1]])

    atomD = torch.tensor([[-0.624, 5.526, -2.146]])
    atomH = torch.tensor([[-1.023, 4.664, -2.481]])
    atomA = torch.tensor([[-1.579, 2.834, -2.817]])
    atomB = torch.tensor([[-0.774, 1.927, -3.337]])
    atomB0 = torch.tensor([[-2.327, 2.261, -1.817]])

    donwt = torch.tensor([[1.45]])
    accwt = torch.tensor([[1.13]])

    energy = tmol.score.hbond.hbond_donor_ring_score(
        atomD, atomH, atomA, atomB, atomB0, accwt, donwt,
        hbpoly_ahdist_aHIS_dGLY_9gt3_hesmooth_min1p6,
        hbpoly_ahdist_aHIS_dGLY_9gt3_hesmooth_min1p6_range,
        hbpoly_ahdist_aHIS_dGLY_9gt3_hesmooth_min1p6_bounds, poly_cosBAH_7,
        poly_cosBAH_7_range, poly_cosBAH_7_bounds, poly_AHD_1i,
        poly_AHD_1i_range, poly_AHD_1i_bounds, 4.2
    )

    assert (float(energy.data[0]) == pytest.approx(-2.17, 0.01))


@pytest.fixture
def bb_hbond_database():
    return cattr.structure(
        bb_hbond_config, tmol.database.scoring.HBondDatabase
    )
