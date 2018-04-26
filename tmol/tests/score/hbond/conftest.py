import pytest

import cattr
import yaml

import tmol.database.scoring

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


@pytest.fixture
def bb_hbond_database():
    return cattr.structure(
        bb_hbond_config, tmol.database.scoring.HBondDatabase
    )
