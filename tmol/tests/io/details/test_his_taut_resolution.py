import numpy
import torch
from tmol.io.canonical_ordering import (
    ordered_canonical_aa_types,
    ordered_canonical_aa_atoms,
    max_n_canonical_atoms,
)
from tmol.io.details.his_taut_resolution import (
    HisTautomerResolution,
    resolve_his_tautomerization,
    his_taut_variant_NE2_protonated,
    his_taut_variant_ND1_protonated,
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

    coords = torch.zeros((1, 1, max_n_canonical_atoms, 3), dtype=torch.float32)
    atom_is_present = torch.zeros((1, 1, max_n_canonical_atoms), dtype=torch.int32)
    res_types = torch.full(
        (1, 1), ordered_canonical_aa_types.index("HIS"), dtype=torch.int32
    )
    res_type_variants = torch.zeros_like(res_types)

    def atind(atname):
        return ordered_canonical_aa_atoms["HIS"].index(atname.strip())

    def xyz(x, y, z):
        return torch.tensor((x, y, z), dtype=torch.float32)

    coords[0, 0, atind(" N  ")] = xyz(-100.575, 9.344, 3.020)
    coords[0, 0, atind(" CA ")] = xyz(-100.536, 9.519, 1.572)
    coords[0, 0, atind(" C  ")] = xyz(-101.554, 8.594, 0.912)
    coords[0, 0, atind(" O  ")] = xyz(-102.310, 7.913, 1.602)
    coords[0, 0, atind(" CB ")] = xyz(-99.131, 9.245, 1.036)
    coords[0, 0, atind(" CG ")] = xyz(-98.079, 10.138, 1.617)
    coords[0, 0, atind(" ND1")] = xyz(-97.637, 11.279, 0.982)
    coords[0, 0, atind(" CD2")] = xyz(-97.384, 10.061, 2.776)
    coords[0, 0, atind(" CE1")] = xyz(-96.714, 11.864, 1.724)
    coords[0, 0, atind(" NE2")] = xyz(-96.541, 11.145, 2.818)
    coords[0, 0, atind(" H  ")] = xyz(-100.653, 8.526, 3.272)
    coords[0, 0, atind(" HA ")] = xyz(-100.764, 10.438, 1.360)
    coords[0, 0, atind(" HB2")] = xyz(-98.898, 8.322, 1.220)
    coords[0, 0, atind(" HB3")] = xyz(-99.136, 9.349, 0.072)
    coords[0, 0, atind(" HD1")] = xyz(-97.918, 11.565, 0.221)
    coords[0, 0, atind(" HD2")] = xyz(-97.463, 9.398, 3.423)
    coords[0, 0, atind(" HE1")] = xyz(-96.263, 12.649, 1.512)
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

    (
        his_taut_res,
        res_type_variants,
        resolved_coords,
        resolved_atom_is_present,
    ) = resolve_his_tautomerization(
        res_types, res_type_variants, coords, atom_is_present
    )

    his_taut_res = his_taut_res.cpu().numpy()
    his_taut_res_gold = numpy.array(
        [[HisTautomerResolution.his_taut_HD1.value]], dtype=numpy.int32
    )
    numpy.testing.assert_equal(his_taut_res, his_taut_res_gold)

    res_type_variants = res_type_variants.cpu().numpy()
    res_type_variants_gold = numpy.array(
        [[his_taut_variant_ND1_protonated]], dtype=numpy.int32
    )
    numpy.testing.assert_equal(res_type_variants, res_type_variants_gold)

    resolved_coords = resolved_coords.cpu().numpy()
    resolved_coords_gold = coords.cpu().numpy()
    numpy.testing.assert_equal(resolved_coords, resolved_coords_gold)

    resolved_atom_is_present = resolved_atom_is_present.cpu().numpy()
    resolved_atom_is_present_gold = atom_is_present.cpu().numpy()
    numpy.testing.assert_equal(resolved_atom_is_present, resolved_atom_is_present_gold)


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

    coords = torch.zeros((1, 1, max_n_canonical_atoms, 3), dtype=torch.float32)
    atom_is_present = torch.zeros((1, 1, max_n_canonical_atoms), dtype=torch.int32)
    res_types = torch.full(
        (1, 1), ordered_canonical_aa_types.index("HIS"), dtype=torch.int32
    )
    res_type_variants = torch.zeros_like(res_types)

    def atind(atname):
        return ordered_canonical_aa_atoms["HIS"].index(atname.strip())

    def xyz(x, y, z):
        return torch.tensor((x, y, z), dtype=torch.float32)

    coords[0, 0, atind(" N  ")] = xyz(-100.575, 9.344, 3.020)
    coords[0, 0, atind(" CA ")] = xyz(-100.536, 9.519, 1.572)
    coords[0, 0, atind(" C  ")] = xyz(-101.554, 8.594, 0.912)
    coords[0, 0, atind(" O  ")] = xyz(-102.310, 7.913, 1.602)
    coords[0, 0, atind(" CB ")] = xyz(-99.131, 9.245, 1.036)
    coords[0, 0, atind(" CG ")] = xyz(-98.079, 10.138, 1.617)
    coords[0, 0, atind(" ND1")] = xyz(-97.637, 11.279, 0.982)
    coords[0, 0, atind(" CD2")] = xyz(-97.384, 10.061, 2.776)
    coords[0, 0, atind(" CE1")] = xyz(-96.714, 11.864, 1.724)
    coords[0, 0, atind(" NE2")] = xyz(-96.541, 11.145, 2.818)
    coords[0, 0, atind(" H  ")] = xyz(-100.653, 8.526, 3.272)
    coords[0, 0, atind(" HA ")] = xyz(-100.764, 10.438, 1.360)
    coords[0, 0, atind(" HB2")] = xyz(-98.898, 8.322, 1.220)
    coords[0, 0, atind(" HB3")] = xyz(-99.136, 9.349, 0.072)
    # coords[0, 0, atind(" HD1")] = xyz( -97.918,  11.565,   0.221)
    coords[0, 0, atind(" HD2")] = xyz(-97.463, 9.398, 3.423)
    coords[0, 0, atind(" HE1")] = xyz(-96.263, 12.649, 1.512)
    coords[0, 0, atind(" HE2")] = xyz(-95.990, 11.326, 3.453)
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

    (
        his_taut_res,
        res_type_variants,
        resolved_coords,
        resolved_atom_is_present,
    ) = resolve_his_tautomerization(
        res_types, res_type_variants, coords, atom_is_present
    )

    his_taut_res = his_taut_res.cpu().numpy()
    his_taut_res_gold = numpy.array(
        [[HisTautomerResolution.his_taut_HE2.value]], dtype=numpy.int32
    )
    numpy.testing.assert_equal(his_taut_res, his_taut_res_gold)

    res_type_variants = res_type_variants.cpu().numpy()
    res_type_variants_gold = numpy.array(
        [[his_taut_variant_NE2_protonated]], dtype=numpy.int32
    )
    numpy.testing.assert_equal(res_type_variants, res_type_variants_gold)

    resolved_coords = resolved_coords.cpu().numpy()
    resolved_coords_gold = coords.cpu().numpy()
    numpy.testing.assert_equal(resolved_coords, resolved_coords_gold)

    resolved_atom_is_present = resolved_atom_is_present.cpu().numpy()
    resolved_atom_is_present_gold = atom_is_present.cpu().numpy()
    numpy.testing.assert_equal(resolved_atom_is_present, resolved_atom_is_present_gold)


def test_resolve_his_HD1_provided_as_HN():
    coords = torch.zeros((1, 1, max_n_canonical_atoms, 3), dtype=torch.float32)
    atom_is_present = torch.zeros((1, 1, max_n_canonical_atoms), dtype=torch.int32)
    res_types = torch.full(
        (1, 1), ordered_canonical_aa_types.index("HIS"), dtype=torch.int32
    )
    res_type_variants = torch.zeros_like(res_types)

    def atind(atname):
        return ordered_canonical_aa_atoms["HIS"].index(atname.strip())

    def xyz(x, y, z):
        return torch.tensor((x, y, z), dtype=torch.float32)

    coords[0, 0, atind(" N  ")] = xyz(-100.575, 9.344, 3.020)
    coords[0, 0, atind(" CA ")] = xyz(-100.536, 9.519, 1.572)
    coords[0, 0, atind(" C  ")] = xyz(-101.554, 8.594, 0.912)
    coords[0, 0, atind(" O  ")] = xyz(-102.310, 7.913, 1.602)
    coords[0, 0, atind(" CB ")] = xyz(-99.131, 9.245, 1.036)
    coords[0, 0, atind(" CG ")] = xyz(-98.079, 10.138, 1.617)
    coords[0, 0, atind(" ND1")] = xyz(-97.637, 11.279, 0.982)
    coords[0, 0, atind(" CD2")] = xyz(-97.384, 10.061, 2.776)
    coords[0, 0, atind(" CE1")] = xyz(-96.714, 11.864, 1.724)
    coords[0, 0, atind(" NE2")] = xyz(-96.541, 11.145, 2.818)
    coords[0, 0, atind(" H  ")] = xyz(-100.653, 8.526, 3.272)
    coords[0, 0, atind(" HA ")] = xyz(-100.764, 10.438, 1.360)
    coords[0, 0, atind(" HB2")] = xyz(-98.898, 8.322, 1.220)
    coords[0, 0, atind(" HB3")] = xyz(-99.136, 9.349, 0.072)
    coords[0, 0, atind(" HN ")] = xyz(-97.918, 11.565, 0.221)
    coords[0, 0, atind(" HD2")] = xyz(-97.463, 9.398, 3.423)
    coords[0, 0, atind(" HE1")] = xyz(-96.263, 12.649, 1.512)
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

    (
        his_taut_res,
        res_type_variants,
        resolved_coords,
        resolved_atom_is_present,
    ) = resolve_his_tautomerization(
        res_types, res_type_variants, coords, atom_is_present
    )

    his_taut_res = his_taut_res.cpu().numpy()
    his_taut_res_gold = numpy.array(
        [[HisTautomerResolution.his_taut_HD1.value]], dtype=numpy.int32
    )
    numpy.testing.assert_equal(his_taut_res, his_taut_res_gold)

    res_type_variants = res_type_variants.cpu().numpy()
    res_type_variants_gold = numpy.array(
        [[his_taut_variant_ND1_protonated]], dtype=numpy.int32
    )
    numpy.testing.assert_equal(res_type_variants, res_type_variants_gold)

    resolved_coords = resolved_coords.cpu().numpy()
    resolved_coords_gold = coords.cpu().numpy()
    resolved_coords_gold[0, 0, atind("HD1")] = resolved_coords_gold[0, 0, atind("HN")]
    numpy.testing.assert_equal(resolved_coords, resolved_coords_gold)

    resolved_atom_is_present = resolved_atom_is_present.cpu().numpy()
    resolved_atom_is_present_gold = atom_is_present.cpu().numpy()
    resolved_atom_is_present_gold[0, 0, atind("HD1")] = 1
    numpy.testing.assert_equal(resolved_atom_is_present, resolved_atom_is_present_gold)


def test_resolve_his_HE2_provided_as_HN():
    coords = torch.zeros((1, 1, max_n_canonical_atoms, 3), dtype=torch.float32)
    atom_is_present = torch.zeros((1, 1, max_n_canonical_atoms), dtype=torch.int32)
    res_types = torch.full(
        (1, 1), ordered_canonical_aa_types.index("HIS"), dtype=torch.int32
    )
    res_type_variants = torch.zeros_like(res_types)

    def atind(atname):
        return ordered_canonical_aa_atoms["HIS"].index(atname.strip())

    def xyz(x, y, z):
        return torch.tensor((x, y, z), dtype=torch.float32)

    coords[0, 0, atind(" N  ")] = xyz(-100.575, 9.344, 3.020)
    coords[0, 0, atind(" CA ")] = xyz(-100.536, 9.519, 1.572)
    coords[0, 0, atind(" C  ")] = xyz(-101.554, 8.594, 0.912)
    coords[0, 0, atind(" O  ")] = xyz(-102.310, 7.913, 1.602)
    coords[0, 0, atind(" CB ")] = xyz(-99.131, 9.245, 1.036)
    coords[0, 0, atind(" CG ")] = xyz(-98.079, 10.138, 1.617)
    coords[0, 0, atind(" ND1")] = xyz(-97.637, 11.279, 0.982)
    coords[0, 0, atind(" CD2")] = xyz(-97.384, 10.061, 2.776)
    coords[0, 0, atind(" CE1")] = xyz(-96.714, 11.864, 1.724)
    coords[0, 0, atind(" NE2")] = xyz(-96.541, 11.145, 2.818)
    coords[0, 0, atind(" H  ")] = xyz(-100.653, 8.526, 3.272)
    coords[0, 0, atind(" HA ")] = xyz(-100.764, 10.438, 1.360)
    coords[0, 0, atind(" HB2")] = xyz(-98.898, 8.322, 1.220)
    coords[0, 0, atind(" HB3")] = xyz(-99.136, 9.349, 0.072)
    # coords[0, 0, atind(" HD1")] = xyz( -97.918,  11.565,   0.221)
    coords[0, 0, atind(" HD2")] = xyz(-97.463, 9.398, 3.423)
    coords[0, 0, atind(" HE1")] = xyz(-96.263, 12.649, 1.512)
    coords[0, 0, atind(" HN ")] = xyz(-95.990, 11.326, 3.453)
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

    (
        his_taut_res,
        res_type_variants,
        resolved_coords,
        resolved_atom_is_present,
    ) = resolve_his_tautomerization(
        res_types, res_type_variants, coords, atom_is_present
    )

    his_taut_res = his_taut_res.cpu().numpy()
    his_taut_res_gold = numpy.array(
        [[HisTautomerResolution.his_taut_HE2.value]], dtype=numpy.int32
    )
    numpy.testing.assert_equal(his_taut_res, his_taut_res_gold)

    res_type_variants = res_type_variants.cpu().numpy()
    res_type_variants_gold = numpy.array(
        [[his_taut_variant_NE2_protonated]], dtype=numpy.int32
    )
    numpy.testing.assert_equal(res_type_variants, res_type_variants_gold)

    resolved_coords = resolved_coords.cpu().numpy()
    resolved_coords_gold = coords.cpu().numpy()
    resolved_coords_gold[0, 0, atind("HE2")] = resolved_coords_gold[0, 0, atind("HN")]
    numpy.testing.assert_equal(resolved_coords, resolved_coords_gold)

    resolved_atom_is_present = resolved_atom_is_present.cpu().numpy()
    resolved_atom_is_present_gold = atom_is_present.cpu().numpy()
    resolved_atom_is_present_gold[0, 0, atind("HE2")] = 1
    numpy.testing.assert_equal(resolved_atom_is_present, resolved_atom_is_present_gold)


def test_resolve_his_ND1_provided_as_NH():
    coords = torch.zeros((1, 1, max_n_canonical_atoms, 3), dtype=torch.float32)
    atom_is_present = torch.zeros((1, 1, max_n_canonical_atoms), dtype=torch.int32)
    res_types = torch.full(
        (1, 1), ordered_canonical_aa_types.index("HIS"), dtype=torch.int32
    )
    res_type_variants = torch.zeros_like(res_types)

    def atind(atname):
        return ordered_canonical_aa_atoms["HIS"].index(atname.strip())

    def xyz(x, y, z):
        return torch.tensor((x, y, z), dtype=torch.float32)

    coords[0, 0, atind(" N  ")] = xyz(-100.575, 9.344, 3.020)
    coords[0, 0, atind(" CA ")] = xyz(-100.536, 9.519, 1.572)
    coords[0, 0, atind(" C  ")] = xyz(-101.554, 8.594, 0.912)
    coords[0, 0, atind(" O  ")] = xyz(-102.310, 7.913, 1.602)
    coords[0, 0, atind(" CB ")] = xyz(-99.131, 9.245, 1.036)
    coords[0, 0, atind(" CG ")] = xyz(-98.079, 10.138, 1.617)
    coords[0, 0, atind(" NH ")] = xyz(-97.637, 11.279, 0.982)
    coords[0, 0, atind(" CD2")] = xyz(-97.384, 10.061, 2.776)
    coords[0, 0, atind(" CE1")] = xyz(-96.714, 11.864, 1.724)
    coords[0, 0, atind(" NN ")] = xyz(-96.541, 11.145, 2.818)
    coords[0, 0, atind(" H  ")] = xyz(-100.653, 8.526, 3.272)
    coords[0, 0, atind(" HA ")] = xyz(-100.764, 10.438, 1.360)
    coords[0, 0, atind(" HB2")] = xyz(-98.898, 8.322, 1.220)
    coords[0, 0, atind(" HB3")] = xyz(-99.136, 9.349, 0.072)
    coords[0, 0, atind(" HN ")] = xyz(-97.918, 11.565, 0.221)
    coords[0, 0, atind(" HD2")] = xyz(-97.463, 9.398, 3.423)
    coords[0, 0, atind(" HE1")] = xyz(-96.263, 12.649, 1.512)
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

    (
        his_taut_res,
        res_type_variants,
        resolved_coords,
        resolved_atom_is_present,
    ) = resolve_his_tautomerization(
        res_types, res_type_variants, coords, atom_is_present
    )

    his_taut_res = his_taut_res.cpu().numpy()
    his_taut_res_gold = numpy.array(
        [[HisTautomerResolution.his_taut_NH_is_ND1.value]], dtype=numpy.int32
    )
    numpy.testing.assert_equal(his_taut_res, his_taut_res_gold)

    res_type_variants = res_type_variants.cpu().numpy()
    res_type_variants_gold = numpy.array(
        [[his_taut_variant_ND1_protonated]], dtype=numpy.int32
    )
    numpy.testing.assert_equal(res_type_variants, res_type_variants_gold)

    resolved_coords = resolved_coords.cpu().numpy()
    resolved_coords_gold = coords.cpu().numpy()
    resolved_coords_gold[0, 0, atind("HD1")] = resolved_coords_gold[0, 0, atind("HN")]
    resolved_coords_gold[0, 0, atind("ND1")] = resolved_coords_gold[0, 0, atind("NH")]
    resolved_coords_gold[0, 0, atind("NE2")] = resolved_coords_gold[0, 0, atind("NN")]
    numpy.testing.assert_equal(resolved_coords, resolved_coords_gold)

    resolved_atom_is_present = resolved_atom_is_present.cpu().numpy()
    resolved_atom_is_present_gold = atom_is_present.cpu().numpy()
    resolved_atom_is_present_gold[0, 0, atind("HD1")] = 1
    resolved_atom_is_present_gold[0, 0, atind("ND1")] = 1
    resolved_atom_is_present_gold[0, 0, atind("NE2")] = 1
    numpy.testing.assert_equal(resolved_atom_is_present, resolved_atom_is_present_gold)


def test_resolve_his_NE2_provided_as_NH():
    coords = torch.zeros((1, 1, max_n_canonical_atoms, 3), dtype=torch.float32)
    atom_is_present = torch.zeros((1, 1, max_n_canonical_atoms), dtype=torch.int32)
    res_types = torch.full(
        (1, 1), ordered_canonical_aa_types.index("HIS"), dtype=torch.int32
    )
    res_type_variants = torch.zeros_like(res_types)

    def atind(atname):
        return ordered_canonical_aa_atoms["HIS"].index(atname.strip())

    def xyz(x, y, z):
        return torch.tensor((x, y, z), dtype=torch.float32)

    coords[0, 0, atind(" N  ")] = xyz(-100.575, 9.344, 3.020)
    coords[0, 0, atind(" CA ")] = xyz(-100.536, 9.519, 1.572)
    coords[0, 0, atind(" C  ")] = xyz(-101.554, 8.594, 0.912)
    coords[0, 0, atind(" O  ")] = xyz(-102.310, 7.913, 1.602)
    coords[0, 0, atind(" CB ")] = xyz(-99.131, 9.245, 1.036)
    coords[0, 0, atind(" CG ")] = xyz(-98.079, 10.138, 1.617)
    coords[0, 0, atind(" NN ")] = xyz(-97.637, 11.279, 0.982)
    coords[0, 0, atind(" CD2")] = xyz(-97.384, 10.061, 2.776)
    coords[0, 0, atind(" CE1")] = xyz(-96.714, 11.864, 1.724)
    coords[0, 0, atind(" NH ")] = xyz(-96.541, 11.145, 2.818)
    coords[0, 0, atind(" H  ")] = xyz(-100.653, 8.526, 3.272)
    coords[0, 0, atind(" HA ")] = xyz(-100.764, 10.438, 1.360)
    coords[0, 0, atind(" HB2")] = xyz(-98.898, 8.322, 1.220)
    coords[0, 0, atind(" HB3")] = xyz(-99.136, 9.349, 0.072)
    # coords[0, 0, atind(" HD1")] = xyz( -97.918,  11.565,   0.221)
    coords[0, 0, atind(" HD2")] = xyz(-97.463, 9.398, 3.423)
    coords[0, 0, atind(" HE1")] = xyz(-96.263, 12.649, 1.512)
    coords[0, 0, atind(" HN ")] = xyz(-95.990, 11.326, 3.453)
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

    (
        his_taut_res,
        res_type_variants,
        resolved_coords,
        resolved_atom_is_present,
    ) = resolve_his_tautomerization(
        res_types, res_type_variants, coords, atom_is_present
    )

    his_taut_res = his_taut_res.cpu().numpy()
    his_taut_res_gold = numpy.array(
        [[HisTautomerResolution.his_taut_NN_is_ND1.value]], dtype=numpy.int32
    )
    numpy.testing.assert_equal(his_taut_res, his_taut_res_gold)

    res_type_variants = res_type_variants.cpu().numpy()
    res_type_variants_gold = numpy.array(
        [[his_taut_variant_NE2_protonated]], dtype=numpy.int32
    )
    numpy.testing.assert_equal(res_type_variants, res_type_variants_gold)

    resolved_coords = resolved_coords.cpu().numpy()
    resolved_coords_gold = coords.cpu().numpy()
    resolved_coords_gold[0, 0, atind("HE2")] = resolved_coords_gold[0, 0, atind("HN")]
    resolved_coords_gold[0, 0, atind("ND1")] = resolved_coords_gold[0, 0, atind("NN")]
    resolved_coords_gold[0, 0, atind("NE2")] = resolved_coords_gold[0, 0, atind("NH")]
    numpy.testing.assert_equal(resolved_coords, resolved_coords_gold)

    resolved_atom_is_present = resolved_atom_is_present.cpu().numpy()
    resolved_atom_is_present_gold = atom_is_present.cpu().numpy()
    resolved_atom_is_present_gold[0, 0, atind("HE2")] = 1
    resolved_atom_is_present_gold[0, 0, atind("ND1")] = 1
    resolved_atom_is_present_gold[0, 0, atind("NE2")] = 1
    numpy.testing.assert_equal(resolved_atom_is_present, resolved_atom_is_present_gold)

    # his_taut_res = resolve_his_tautomerization(res_types, coords, atom_is_present)
    # his_taut_res_gold = numpy.array(
    #     [[HisTautomerResolution.his_taut_HE2.value]], dtype=numpy.int32
    # )
    # numpy.testing.assert_equal(his_taut_res, his_taut_res_gold)
