import numpy
from tmol.io.canonical_ordering import (
    ordered_canonical_aa_types,
    ordered_canonical_aa_atoms,
    max_n_canonical_atoms,
)
from tmol.io.details.his_taut_resolution import (
    HisTautomerResolution,
    resolve_his_tautomerization,
)


def test_resolve_his_HD1_provided():
    # from 5t0y with hydrogens added by Reduce
    # ATOM   1460  N   HIS L 209    -100.575   9.344   3.020  1.00 59.29           N
    # ATOM   1461  CA  HIS L 209    -100.536   9.519   1.572  1.00 58.91           C
    # ATOM   1462  C   HIS L 209    -101.554   8.594   0.912  1.00 58.19           C
    # ATOM   1463  O   HIS L 209    -102.310   7.913   1.602  1.00 58.34           O
    # ATOM   1464  CB  HIS L 209     -99.131   9.245   1.036  1.00 59.27           C
    # ATOM   1465  CG  HIS L 209     -98.079  10.138   1.617  1.00 60.32           C
    # ATOM   1466  ND1 HIS L 209     -97.637  11.279   0.982  1.00 61.12           N
    # ATOM   1467  CD2 HIS L 209     -97.384  10.061   2.776  1.00 60.83           C
    # ATOM   1468  CE1 HIS L 209     -96.714  11.864   1.724  1.00 62.34           C
    # ATOM   1469  NE2 HIS L 209     -96.541  11.145   2.818  1.00 62.20           N
    # ATOM      0  H   HIS L 209    -100.653   8.526   3.272  1.00 59.29           H   new
    # ATOM      0  HA  HIS L 209    -100.764  10.438   1.360  1.00 58.91           H   new
    # ATOM      0  HB2 HIS L 209     -98.898   8.322   1.220  1.00 59.27           H   new
    # ATOM      0  HB3 HIS L 209     -99.136   9.349   0.072  1.00 59.27           H   new
    # ATOM      0  HD1 HIS L 209     -97.918  11.565   0.221  1.00 61.12           H   new
    # ATOM      0  HD2 HIS L 209     -97.463   9.398   3.423  1.00 60.83           H   new
    # ATOM      0  HE1 HIS L 209     -96.263  12.649   1.512  1.00 62.34           H   new
    # ATOM      0  HE2 HIS L 209     -95.990  11.326   3.453  1.00 62.20           H   new <-- skip this one

    coords = numpy.zeros((1, 1, max_n_canonical_atoms, 3), dtype=numpy.float32)
    atom_is_present = numpy.zeros((1, 1, max_n_canonical_atoms), dtype=numpy.int32)
    res_types = numpy.full(
        (1, 1), ordered_canonical_aa_types.index("HIS"), dtype=numpy.int32
    )

    def atind(atname):
        return ordered_canonical_aa_atoms["HIS"].index(atname.strip())

    coords[0, 0, atind(" N  ")] = (-100.575, 9.344, 3.020)
    coords[0, 0, atind(" CA ")] = (-100.536, 9.519, 1.572)
    coords[0, 0, atind(" C  ")] = (-101.554, 8.594, 0.912)
    coords[0, 0, atind(" O  ")] = (-102.310, 7.913, 1.602)
    coords[0, 0, atind(" CB ")] = (-99.131, 9.245, 1.036)
    coords[0, 0, atind(" CG ")] = (-98.079, 10.138, 1.617)
    coords[0, 0, atind(" ND1")] = (-97.637, 11.279, 0.982)
    coords[0, 0, atind(" CD2")] = (-97.384, 10.061, 2.776)
    coords[0, 0, atind(" CE1")] = (-96.714, 11.864, 1.724)
    coords[0, 0, atind(" NE2")] = (-96.541, 11.145, 2.818)
    coords[0, 0, atind(" H  ")] = (-100.653, 8.526, 3.272)
    coords[0, 0, atind(" HA ")] = (-100.764, 10.438, 1.360)
    coords[0, 0, atind(" HB2")] = (-98.898, 8.322, 1.220)
    coords[0, 0, atind(" HB3")] = (-99.136, 9.349, 0.072)
    coords[0, 0, atind(" HD1")] = (-97.918, 11.565, 0.221)
    coords[0, 0, atind(" HD2")] = (-97.463, 9.398, 3.423)
    coords[0, 0, atind(" HE1")] = (-96.263, 12.649, 1.512)
    # coords[0, 0, atind(" HE2")] = ( -95.990,  11.326,   3.453)
    atom_is_present[0, 0, atind(" N  ")] = 1
    atom_is_present[0, 0, atind(" CA ")] = 1
    atom_is_present[0, 0, atind(" C  ")] = 1
    atom_is_present[0, 0, atind(" O  ")] = 1
    atom_is_present[0, 0, atind(" CB ")] = 1
    atom_is_present[0, 0, atind(" CG ")] = 1
    atom_is_present[0, 0, atind(" ND1")] = 1
    atom_is_present[0, 0, atind(" CD2")] = 1
    atom_is_present[0, 0, atind(" CE1")] = 1
    atom_is_present[0, 0, atind(" NE2")] = 1
    atom_is_present[0, 0, atind(" H  ")] = 1
    atom_is_present[0, 0, atind(" HA ")] = 1
    atom_is_present[0, 0, atind(" HB2")] = 1
    atom_is_present[0, 0, atind(" HB3")] = 1
    atom_is_present[0, 0, atind(" HD1")] = 1
    atom_is_present[0, 0, atind(" HD2")] = 1
    atom_is_present[0, 0, atind(" HE1")] = 1
    # atom_is_present[0, 0, atind(" HE2")] = 1

    his_taut_res = resolve_his_tautomerization(res_types, coords, atom_is_present)
    his_taut_res_gold = numpy.array(
        [[HisTautomerResolution.his_taut_HD1.value]], dtype=numpy.int32
    )
    numpy.testing.assert_equal(his_taut_res, his_taut_res_gold)


def test_resolve_his_HE2_provided():
    # from 5t0y with hydrogens added by Reduce
    # ATOM   1460  N   HIS L 209    -100.575   9.344   3.020  1.00 59.29           N
    # ATOM   1461  CA  HIS L 209    -100.536   9.519   1.572  1.00 58.91           C
    # ATOM   1462  C   HIS L 209    -101.554   8.594   0.912  1.00 58.19           C
    # ATOM   1463  O   HIS L 209    -102.310   7.913   1.602  1.00 58.34           O
    # ATOM   1464  CB  HIS L 209     -99.131   9.245   1.036  1.00 59.27           C
    # ATOM   1465  CG  HIS L 209     -98.079  10.138   1.617  1.00 60.32           C
    # ATOM   1466  ND1 HIS L 209     -97.637  11.279   0.982  1.00 61.12           N
    # ATOM   1467  CD2 HIS L 209     -97.384  10.061   2.776  1.00 60.83           C
    # ATOM   1468  CE1 HIS L 209     -96.714  11.864   1.724  1.00 62.34           C
    # ATOM   1469  NE2 HIS L 209     -96.541  11.145   2.818  1.00 62.20           N
    # ATOM      0  H   HIS L 209    -100.653   8.526   3.272  1.00 59.29           H   new
    # ATOM      0  HA  HIS L 209    -100.764  10.438   1.360  1.00 58.91           H   new
    # ATOM      0  HB2 HIS L 209     -98.898   8.322   1.220  1.00 59.27           H   new
    # ATOM      0  HB3 HIS L 209     -99.136   9.349   0.072  1.00 59.27           H   new
    # ATOM      0  HD1 HIS L 209     -97.918  11.565   0.221  1.00 61.12           H   new <-- skip this one
    # ATOM      0  HD2 HIS L 209     -97.463   9.398   3.423  1.00 60.83           H   new
    # ATOM      0  HE1 HIS L 209     -96.263  12.649   1.512  1.00 62.34           H   new
    # ATOM      0  HE2 HIS L 209     -95.990  11.326   3.453  1.00 62.20           H   new

    coords = numpy.zeros((1, 1, max_n_canonical_atoms, 3), dtype=numpy.float32)
    atom_is_present = numpy.zeros((1, 1, max_n_canonical_atoms), dtype=numpy.int32)
    res_types = numpy.full(
        (1, 1), ordered_canonical_aa_types.index("HIS"), dtype=numpy.int32
    )

    def atind(atname):
        return ordered_canonical_aa_atoms["HIS"].index(atname.strip())

    coords[0, 0, atind(" N  ")] = (-100.575, 9.344, 3.020)
    coords[0, 0, atind(" CA ")] = (-100.536, 9.519, 1.572)
    coords[0, 0, atind(" C  ")] = (-101.554, 8.594, 0.912)
    coords[0, 0, atind(" O  ")] = (-102.310, 7.913, 1.602)
    coords[0, 0, atind(" CB ")] = (-99.131, 9.245, 1.036)
    coords[0, 0, atind(" CG ")] = (-98.079, 10.138, 1.617)
    coords[0, 0, atind(" ND1")] = (-97.637, 11.279, 0.982)
    coords[0, 0, atind(" CD2")] = (-97.384, 10.061, 2.776)
    coords[0, 0, atind(" CE1")] = (-96.714, 11.864, 1.724)
    coords[0, 0, atind(" NE2")] = (-96.541, 11.145, 2.818)
    coords[0, 0, atind(" H  ")] = (-100.653, 8.526, 3.272)
    coords[0, 0, atind(" HA ")] = (-100.764, 10.438, 1.360)
    coords[0, 0, atind(" HB2")] = (-98.898, 8.322, 1.220)
    coords[0, 0, atind(" HB3")] = (-99.136, 9.349, 0.072)
    # coords[0, 0, atind(" HD1")] = ( -97.918,  11.565,   0.221)
    coords[0, 0, atind(" HD2")] = (-97.463, 9.398, 3.423)
    coords[0, 0, atind(" HE1")] = (-96.263, 12.649, 1.512)
    coords[0, 0, atind(" HE2")] = (-95.990, 11.326, 3.453)
    atom_is_present[0, 0, atind(" N  ")] = 1
    atom_is_present[0, 0, atind(" CA ")] = 1
    atom_is_present[0, 0, atind(" C  ")] = 1
    atom_is_present[0, 0, atind(" O  ")] = 1
    atom_is_present[0, 0, atind(" CB ")] = 1
    atom_is_present[0, 0, atind(" CG ")] = 1
    atom_is_present[0, 0, atind(" ND1")] = 1
    atom_is_present[0, 0, atind(" CD2")] = 1
    atom_is_present[0, 0, atind(" CE1")] = 1
    atom_is_present[0, 0, atind(" NE2")] = 1
    atom_is_present[0, 0, atind(" H  ")] = 1
    atom_is_present[0, 0, atind(" HA ")] = 1
    atom_is_present[0, 0, atind(" HB2")] = 1
    atom_is_present[0, 0, atind(" HB3")] = 1
    # atom_is_present[0, 0, atind(" HD1")] = 1
    atom_is_present[0, 0, atind(" HD2")] = 1
    atom_is_present[0, 0, atind(" HE1")] = 1
    atom_is_present[0, 0, atind(" HE2")] = 1

    his_taut_res = resolve_his_tautomerization(res_types, coords, atom_is_present)
    his_taut_res_gold = numpy.array(
        [[HisTautomerResolution.his_taut_HE2.value]], dtype=numpy.int32
    )
    numpy.testing.assert_equal(his_taut_res, his_taut_res_gold)


def test_resolve_his_HD1_provided_as_HN():

    coords = numpy.zeros((1, 1, max_n_canonical_atoms, 3), dtype=numpy.float32)
    atom_is_present = numpy.zeros((1, 1, max_n_canonical_atoms), dtype=numpy.int32)
    res_types = numpy.full(
        (1, 1), ordered_canonical_aa_types.index("HIS"), dtype=numpy.int32
    )

    def atind(atname):
        return ordered_canonical_aa_atoms["HIS"].index(atname.strip())

    coords[0, 0, atind(" N  ")] = (-100.575, 9.344, 3.020)
    coords[0, 0, atind(" CA ")] = (-100.536, 9.519, 1.572)
    coords[0, 0, atind(" C  ")] = (-101.554, 8.594, 0.912)
    coords[0, 0, atind(" O  ")] = (-102.310, 7.913, 1.602)
    coords[0, 0, atind(" CB ")] = (-99.131, 9.245, 1.036)
    coords[0, 0, atind(" CG ")] = (-98.079, 10.138, 1.617)
    coords[0, 0, atind(" ND1")] = (-97.637, 11.279, 0.982)
    coords[0, 0, atind(" CD2")] = (-97.384, 10.061, 2.776)
    coords[0, 0, atind(" CE1")] = (-96.714, 11.864, 1.724)
    coords[0, 0, atind(" NE2")] = (-96.541, 11.145, 2.818)
    coords[0, 0, atind(" H  ")] = (-100.653, 8.526, 3.272)
    coords[0, 0, atind(" HA ")] = (-100.764, 10.438, 1.360)
    coords[0, 0, atind(" HB2")] = (-98.898, 8.322, 1.220)
    coords[0, 0, atind(" HB3")] = (-99.136, 9.349, 0.072)
    coords[0, 0, atind(" HN ")] = (-97.918, 11.565, 0.221)
    coords[0, 0, atind(" HD2")] = (-97.463, 9.398, 3.423)
    coords[0, 0, atind(" HE1")] = (-96.263, 12.649, 1.512)
    # coords[0, 0, atind(" HE2")] = ( -95.990,  11.326,   3.453)
    atom_is_present[0, 0, atind(" N  ")] = 1
    atom_is_present[0, 0, atind(" CA ")] = 1
    atom_is_present[0, 0, atind(" C  ")] = 1
    atom_is_present[0, 0, atind(" O  ")] = 1
    atom_is_present[0, 0, atind(" CB ")] = 1
    atom_is_present[0, 0, atind(" CG ")] = 1
    atom_is_present[0, 0, atind(" ND1")] = 1
    atom_is_present[0, 0, atind(" CD2")] = 1
    atom_is_present[0, 0, atind(" CE1")] = 1
    atom_is_present[0, 0, atind(" NE2")] = 1
    atom_is_present[0, 0, atind(" H  ")] = 1
    atom_is_present[0, 0, atind(" HA ")] = 1
    atom_is_present[0, 0, atind(" HB2")] = 1
    atom_is_present[0, 0, atind(" HB3")] = 1
    atom_is_present[0, 0, atind(" HN ")] = 1
    atom_is_present[0, 0, atind(" HD2")] = 1
    atom_is_present[0, 0, atind(" HE1")] = 1
    # atom_is_present[0, 0, atind(" HE2")] = 1

    his_taut_res = resolve_his_tautomerization(res_types, coords, atom_is_present)
    his_taut_res_gold = numpy.array(
        [[HisTautomerResolution.his_taut_HD1.value]], dtype=numpy.int32
    )
    numpy.testing.assert_equal(his_taut_res, his_taut_res_gold)


def test_resolve_his_HE2_provided_as_HN():
    coords = numpy.zeros((1, 1, max_n_canonical_atoms, 3), dtype=numpy.float32)
    atom_is_present = numpy.zeros((1, 1, max_n_canonical_atoms), dtype=numpy.int32)
    res_types = numpy.full(
        (1, 1), ordered_canonical_aa_types.index("HIS"), dtype=numpy.int32
    )

    def atind(atname):
        return ordered_canonical_aa_atoms["HIS"].index(atname.strip())

    coords[0, 0, atind(" N  ")] = (-100.575, 9.344, 3.020)
    coords[0, 0, atind(" CA ")] = (-100.536, 9.519, 1.572)
    coords[0, 0, atind(" C  ")] = (-101.554, 8.594, 0.912)
    coords[0, 0, atind(" O  ")] = (-102.310, 7.913, 1.602)
    coords[0, 0, atind(" CB ")] = (-99.131, 9.245, 1.036)
    coords[0, 0, atind(" CG ")] = (-98.079, 10.138, 1.617)
    coords[0, 0, atind(" ND1")] = (-97.637, 11.279, 0.982)
    coords[0, 0, atind(" CD2")] = (-97.384, 10.061, 2.776)
    coords[0, 0, atind(" CE1")] = (-96.714, 11.864, 1.724)
    coords[0, 0, atind(" NE2")] = (-96.541, 11.145, 2.818)
    coords[0, 0, atind(" H  ")] = (-100.653, 8.526, 3.272)
    coords[0, 0, atind(" HA ")] = (-100.764, 10.438, 1.360)
    coords[0, 0, atind(" HB2")] = (-98.898, 8.322, 1.220)
    coords[0, 0, atind(" HB3")] = (-99.136, 9.349, 0.072)
    # coords[0, 0, atind(" HD1")] = ( -97.918,  11.565,   0.221)
    coords[0, 0, atind(" HD2")] = (-97.463, 9.398, 3.423)
    coords[0, 0, atind(" HE1")] = (-96.263, 12.649, 1.512)
    coords[0, 0, atind(" HN ")] = (-95.990, 11.326, 3.453)
    atom_is_present[0, 0, atind(" N  ")] = 1
    atom_is_present[0, 0, atind(" CA ")] = 1
    atom_is_present[0, 0, atind(" C  ")] = 1
    atom_is_present[0, 0, atind(" O  ")] = 1
    atom_is_present[0, 0, atind(" CB ")] = 1
    atom_is_present[0, 0, atind(" CG ")] = 1
    atom_is_present[0, 0, atind(" ND1")] = 1
    atom_is_present[0, 0, atind(" CD2")] = 1
    atom_is_present[0, 0, atind(" CE1")] = 1
    atom_is_present[0, 0, atind(" NE2")] = 1
    atom_is_present[0, 0, atind(" H  ")] = 1
    atom_is_present[0, 0, atind(" HA ")] = 1
    atom_is_present[0, 0, atind(" HB2")] = 1
    atom_is_present[0, 0, atind(" HB3")] = 1
    # atom_is_present[0, 0, atind(" HD1")] = 1
    atom_is_present[0, 0, atind(" HD2")] = 1
    atom_is_present[0, 0, atind(" HE1")] = 1
    atom_is_present[0, 0, atind(" HN ")] = 1

    his_taut_res = resolve_his_tautomerization(res_types, coords, atom_is_present)
    his_taut_res_gold = numpy.array(
        [[HisTautomerResolution.his_taut_HE2.value]], dtype=numpy.int32
    )
    numpy.testing.assert_equal(his_taut_res, his_taut_res_gold)


def test_resolve_his_ND1_provided_as_NH():
    coords = numpy.zeros((1, 1, max_n_canonical_atoms, 3), dtype=numpy.float32)
    atom_is_present = numpy.zeros((1, 1, max_n_canonical_atoms), dtype=numpy.int32)
    res_types = numpy.full(
        (1, 1), ordered_canonical_aa_types.index("HIS"), dtype=numpy.int32
    )

    def atind(atname):
        return ordered_canonical_aa_atoms["HIS"].index(atname.strip())

    coords[0, 0, atind(" N  ")] = (-100.575, 9.344, 3.020)
    coords[0, 0, atind(" CA ")] = (-100.536, 9.519, 1.572)
    coords[0, 0, atind(" C  ")] = (-101.554, 8.594, 0.912)
    coords[0, 0, atind(" O  ")] = (-102.310, 7.913, 1.602)
    coords[0, 0, atind(" CB ")] = (-99.131, 9.245, 1.036)
    coords[0, 0, atind(" CG ")] = (-98.079, 10.138, 1.617)
    coords[0, 0, atind(" NH ")] = (-97.637, 11.279, 0.982)
    coords[0, 0, atind(" CD2")] = (-97.384, 10.061, 2.776)
    coords[0, 0, atind(" CE1")] = (-96.714, 11.864, 1.724)
    coords[0, 0, atind(" NN ")] = (-96.541, 11.145, 2.818)
    coords[0, 0, atind(" H  ")] = (-100.653, 8.526, 3.272)
    coords[0, 0, atind(" HA ")] = (-100.764, 10.438, 1.360)
    coords[0, 0, atind(" HB2")] = (-98.898, 8.322, 1.220)
    coords[0, 0, atind(" HB3")] = (-99.136, 9.349, 0.072)
    coords[0, 0, atind(" HN ")] = (-97.918, 11.565, 0.221)
    coords[0, 0, atind(" HD2")] = (-97.463, 9.398, 3.423)
    coords[0, 0, atind(" HE1")] = (-96.263, 12.649, 1.512)
    # coords[0, 0, atind(" HE2")] = ( -95.990,  11.326,   3.453)
    atom_is_present[0, 0, atind(" N  ")] = 1
    atom_is_present[0, 0, atind(" CA ")] = 1
    atom_is_present[0, 0, atind(" C  ")] = 1
    atom_is_present[0, 0, atind(" O  ")] = 1
    atom_is_present[0, 0, atind(" CB ")] = 1
    atom_is_present[0, 0, atind(" CG ")] = 1
    atom_is_present[0, 0, atind(" NH ")] = 1
    atom_is_present[0, 0, atind(" CD2")] = 1
    atom_is_present[0, 0, atind(" CE1")] = 1
    atom_is_present[0, 0, atind(" NN ")] = 1
    atom_is_present[0, 0, atind(" H  ")] = 1
    atom_is_present[0, 0, atind(" HA ")] = 1
    atom_is_present[0, 0, atind(" HB2")] = 1
    atom_is_present[0, 0, atind(" HB3")] = 1
    atom_is_present[0, 0, atind(" HN ")] = 1
    atom_is_present[0, 0, atind(" HD2")] = 1
    atom_is_present[0, 0, atind(" HE1")] = 1
    # atom_is_present[0, 0, atind(" HE2")] = 1

    his_taut_res = resolve_his_tautomerization(res_types, coords, atom_is_present)
    his_taut_res_gold = numpy.array(
        [[HisTautomerResolution.his_taut_NH_is_ND1.value]], dtype=numpy.int32
    )
    numpy.testing.assert_equal(his_taut_res, his_taut_res_gold)


def test_resolve_his_NE2_provided_as_NH():
    coords = numpy.zeros((1, 1, max_n_canonical_atoms, 3), dtype=numpy.float32)
    atom_is_present = numpy.zeros((1, 1, max_n_canonical_atoms), dtype=numpy.int32)
    res_types = numpy.full(
        (1, 1), ordered_canonical_aa_types.index("HIS"), dtype=numpy.int32
    )

    def atind(atname):
        return ordered_canonical_aa_atoms["HIS"].index(atname.strip())

    coords[0, 0, atind(" N  ")] = (-100.575, 9.344, 3.020)
    coords[0, 0, atind(" CA ")] = (-100.536, 9.519, 1.572)
    coords[0, 0, atind(" C  ")] = (-101.554, 8.594, 0.912)
    coords[0, 0, atind(" O  ")] = (-102.310, 7.913, 1.602)
    coords[0, 0, atind(" CB ")] = (-99.131, 9.245, 1.036)
    coords[0, 0, atind(" CG ")] = (-98.079, 10.138, 1.617)
    coords[0, 0, atind(" NN ")] = (-97.637, 11.279, 0.982)
    coords[0, 0, atind(" CD2")] = (-97.384, 10.061, 2.776)
    coords[0, 0, atind(" CE1")] = (-96.714, 11.864, 1.724)
    coords[0, 0, atind(" NH ")] = (-96.541, 11.145, 2.818)
    coords[0, 0, atind(" H  ")] = (-100.653, 8.526, 3.272)
    coords[0, 0, atind(" HA ")] = (-100.764, 10.438, 1.360)
    coords[0, 0, atind(" HB2")] = (-98.898, 8.322, 1.220)
    coords[0, 0, atind(" HB3")] = (-99.136, 9.349, 0.072)
    # coords[0, 0, atind(" HD1")] = ( -97.918,  11.565,   0.221)
    coords[0, 0, atind(" HD2")] = (-97.463, 9.398, 3.423)
    coords[0, 0, atind(" HE1")] = (-96.263, 12.649, 1.512)
    coords[0, 0, atind(" HN ")] = (-95.990, 11.326, 3.453)
    atom_is_present[0, 0, atind(" N  ")] = 1
    atom_is_present[0, 0, atind(" CA ")] = 1
    atom_is_present[0, 0, atind(" C  ")] = 1
    atom_is_present[0, 0, atind(" O  ")] = 1
    atom_is_present[0, 0, atind(" CB ")] = 1
    atom_is_present[0, 0, atind(" CG ")] = 1
    atom_is_present[0, 0, atind(" NN ")] = 1
    atom_is_present[0, 0, atind(" CD2")] = 1
    atom_is_present[0, 0, atind(" CE1")] = 1
    atom_is_present[0, 0, atind(" NH ")] = 1
    atom_is_present[0, 0, atind(" H  ")] = 1
    atom_is_present[0, 0, atind(" HA ")] = 1
    atom_is_present[0, 0, atind(" HB2")] = 1
    atom_is_present[0, 0, atind(" HB3")] = 1
    # atom_is_present[0, 0, atind(" HD1")] = 1
    atom_is_present[0, 0, atind(" HD2")] = 1
    atom_is_present[0, 0, atind(" HE1")] = 1
    atom_is_present[0, 0, atind(" HN ")] = 1

    his_taut_res = resolve_his_tautomerization(res_types, coords, atom_is_present)
    his_taut_res_gold = numpy.array(
        [[HisTautomerResolution.his_taut_NN_is_ND1.value]], dtype=numpy.int32
    )
    numpy.testing.assert_equal(his_taut_res, his_taut_res_gold)
