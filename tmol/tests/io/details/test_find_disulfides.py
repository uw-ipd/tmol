import numpy
from tmol.io.canonical_ordering import (
    ordered_canonical_aa_types,
    ordered_canonical_aa_atoms,
    max_n_canonical_atoms,
)
from tmol.io.details.disulfide_search import find_disulfides

# ATOM    160  N   CYS L  23     -60.489 -22.492  -9.505  1.00 63.92           N
# ATOM    161  CA  CYS L  23     -60.175 -22.416  -8.082  1.00 64.85           C
# ATOM    162  C   CYS L  23     -58.707 -22.056  -7.882  1.00 66.35           C
# ATOM    163  O   CYS L  23     -57.818 -22.854  -8.180  1.00 64.65           O
# ATOM    164  CB  CYS L  23     -60.494 -23.741  -7.387  1.00 66.78           C
# ATOM    165  SG  CYS L  23     -60.229 -23.740  -5.597  1.00 67.32           S
# ATOM      0  H   CYS L  23     -59.879 -22.870  -9.979  1.00 63.92           H   new
# ATOM      0  HA  CYS L  23     -60.724 -21.722  -7.685  1.00 64.85           H   new
# ATOM      0  HB2 CYS L  23     -61.420 -23.970  -7.564  1.00 66.78           H   new
# ATOM      0  HB3 CYS L  23     -59.948 -24.439  -7.782  1.00 66.78           H   new
# ATOM    665  N   CYS L 104     -64.451 -25.696  -3.612  1.00 55.18           N
# ATOM    666  CA  CYS L 104     -63.153 -26.354  -3.688  1.00 55.61           C
# ATOM    667  C   CYS L 104     -62.485 -26.357  -2.317  1.00 55.02           C
# ATOM    668  O   CYS L 104     -62.596 -25.392  -1.561  1.00 54.69           O
# ATOM    669  CB  CYS L 104     -62.254 -25.670  -4.721  1.00 58.68           C
# ATOM    670  SG  CYS L 104     -61.805 -23.962  -4.330  1.00 59.51           S
# ATOM      0  H   CYS L 104     -64.416 -24.837  -3.620  1.00 55.18           H   new
# ATOM      0  HA  CYS L 104     -63.291 -27.272  -3.971  1.00 55.61           H   new
# ATOM      0  HB2 CYS L 104     -61.441 -26.190  -4.817  1.00 58.68           H   new
# ATOM      0  HB3 CYS L 104     -62.703 -25.684  -5.581  1.00 58.68           H   new
# ATOM   1823  N   CYS H  23     -73.572 -35.750  16.917  1.00 57.95           N
# ATOM   1824  CA  CYS H  23     -73.371 -36.324  15.590  1.00 59.65           C
# ATOM   1825  C   CYS H  23     -73.263 -37.844  15.674  1.00 64.29           C
# ATOM   1826  O   CYS H  23     -72.168 -38.391  15.810  1.00 63.51           O
# ATOM   1827  CB  CYS H  23     -72.119 -35.735  14.933  1.00 59.52           C
# ATOM   1828  SG  CYS H  23     -71.884 -36.185  13.193  1.00 60.65           S
# ATOM      0  H   CYS H  23     -73.206 -36.198  17.554  1.00 57.95           H   new
# ATOM      0  HA  CYS H  23     -74.140 -36.101  15.042  1.00 59.65           H   new
# ATOM      0  HB2 CYS H  23     -72.159 -34.768  15.001  1.00 59.52           H   new
# ATOM      0  HB3 CYS H  23     -71.341 -36.022  15.435  1.00 59.52           H   new
# ATOM   2415  N   CYS H 104     -71.413 -31.922  10.130  1.00 50.69           N
# ATOM   2416  CA  CYS H 104     -70.357 -32.833  10.549  1.00 51.59           C
# ATOM   2417  C   CYS H 104     -70.083 -33.823   9.420  1.00 54.14           C
# ATOM   2418  O   CYS H 104     -71.011 -34.345   8.800  1.00 55.52           O
# ATOM   2419  CB  CYS H 104     -70.729 -33.562  11.846  1.00 55.82           C
# ATOM   2420  SG  CYS H 104     -72.042 -34.800  11.712  1.00 58.54           S
# ATOM      0  H   CYS H 104     -72.142 -32.309   9.887  1.00 50.69           H   new
# ATOM      0  HA  CYS H 104     -69.552 -32.324  10.734  1.00 51.59           H   new
# ATOM      0  HB2 CYS H 104     -69.933 -33.996  12.191  1.00 55.82           H   new
# ATOM      0  HB3 CYS H 104     -70.999 -32.900  12.502  1.00 55.82           H   new


def test_find_disulfide_pairs():
    coords = numpy.zeros((1, 4, max_n_canonical_atoms, 3), dtype=numpy.float32)
    atom_is_present = numpy.zeros((1, 4, max_n_canonical_atoms), dtype=numpy.int32)
    res_types = numpy.full(
        (1, 4), ordered_canonical_aa_types.index("CYS"), dtype=numpy.int32
    )

    def set_coord(res_ind, at_name, xyz):
        at_ind = ordered_canonical_aa_atoms["CYS"].index(at_name.strip())
        coords[0, res_ind, at_ind] = xyz
        atom_is_present[0, res_ind, at_ind] = 1

    set_coord(0, " N  ", (-60.489, -22.492, -9.505))
    set_coord(0, " CA ", (-60.175, -22.416, -8.082))
    set_coord(0, " C  ", (-58.707, -22.056, -7.882))
    set_coord(0, " O  ", (-57.818, -22.854, -8.180))
    set_coord(0, " CB ", (-60.494, -23.741, -7.387))
    set_coord(0, " SG ", (-60.229, -23.740, -5.597))
    set_coord(0, " H  ", (-59.879, -22.870, -9.979))
    set_coord(0, " HA ", (-60.724, -21.722, -7.685))
    set_coord(0, " HB2", (-61.420, -23.970, -7.564))
    set_coord(0, " HB3", (-59.948, -24.439, -7.782))
    set_coord(1, " N  ", (-64.451, -25.696, -3.612))
    set_coord(1, " CA ", (-63.153, -26.354, -3.688))
    set_coord(1, " C  ", (-62.485, -26.357, -2.317))
    set_coord(1, " O  ", (-62.596, -25.392, -1.561))
    set_coord(1, " CB ", (-62.254, -25.670, -4.721))
    set_coord(1, " SG ", (-61.805, -23.962, -4.330))
    set_coord(1, " H  ", (-64.416, -24.837, -3.620))
    set_coord(1, " HA ", (-63.291, -27.272, -3.971))
    set_coord(1, " HB2", (-61.441, -26.190, -4.817))
    set_coord(1, " HB3", (-62.703, -25.684, -5.581))
    set_coord(2, " N  ", (-73.572, -35.750, 16.917))
    set_coord(2, " CA ", (-73.371, -36.324, 15.590))
    set_coord(2, " C  ", (-73.263, -37.844, 15.674))
    set_coord(2, " O  ", (-72.168, -38.391, 15.810))
    set_coord(2, " CB ", (-72.119, -35.735, 14.933))
    set_coord(2, " SG ", (-71.884, -36.185, 13.193))
    set_coord(2, " H  ", (-73.206, -36.198, 17.554))
    set_coord(2, " HA ", (-74.140, -36.101, 15.042))
    set_coord(2, " HB2", (-72.159, -34.768, 15.001))
    set_coord(2, " HB3", (-71.341, -36.022, 15.435))
    set_coord(3, " N  ", (-71.413, -31.922, 10.130))
    set_coord(3, " CA ", (-70.357, -32.833, 10.549))
    set_coord(3, " C  ", (-70.083, -33.823, 9.420))
    set_coord(3, " O  ", (-71.011, -34.345, 8.800))
    set_coord(3, " CB ", (-70.729, -33.562, 11.846))
    set_coord(3, " SG ", (-72.042, -34.800, 11.712))
    set_coord(3, " H  ", (-72.142, -32.309, 9.887))
    set_coord(3, " HA ", (-69.552, -32.324, 10.734))
    set_coord(3, " HB2", (-69.933, -33.996, 12.191))
    set_coord(3, " HB3", (-70.999, -32.900, 12.502))

    found_dslf = find_disulfides(res_types, coords, atom_is_present)
    found_dslf_gold = numpy.array([[0, 0, 1], [0, 2, 3]], dtype=numpy.int32)
    numpy.testing.assert_equal(found_dslf, found_dslf_gold)
