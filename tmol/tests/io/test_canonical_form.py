import numpy
from tmol.io.canonical_ordering import (
    default_canonical_ordering,
    CanonicalOrdering,
    canonical_form_from_pdb_lines,
)

# from tmol.io.details.canonical_packed_block_types import default_canonical_packed_block_types


def test_create_canonical_ordering_smoke(default_database):
    chemdb = default_database.chemical
    co = CanonicalOrdering.from_chemdb(chemdb)

    n_name3s = len(set([x.name3 for x in chemdb.residues]))
    assert co.n_restype_io_equiv_classes == n_name3s
    for name3 in co.restype_io_equiv_classes:
        assert name3 in co.restypes_ordered_atom_names
        assert name3 in co.restypes_atom_index_mapping

    assert "nterm" in co.down_termini_patches
    assert "cterm" in co.up_termini_patches
    for x in ["H1", "H2", "H3"]:
        assert x in co.termini_patch_added_atoms["nterm"]
    assert "OXT" in co.termini_patch_added_atoms["cterm"]
    assert co.max_n_canonical_atoms >= 28


def test_default_canonical_ordering():
    co1 = default_canonical_ordering()
    co2 = default_canonical_ordering()
    assert co2 is co1


def test_default_canonical_form_from_pdb_lines(pertuzumab_pdb, torch_device):
    can_ord = default_canonical_ordering()
    # can_pbt = default_canonical_packed_block_types(torch_device)
    (
        chain_id,
        res_types,
        coords,
        atom_is_present,
    ) = canonical_form_from_pdb_lines(can_ord, pertuzumab_pdb, torch_device)
    def_co = default_canonical_ordering()
    assert chain_id.device == torch_device
    assert res_types.device == torch_device
    assert coords.device == torch_device
    assert atom_is_present.device == torch_device
    assert chain_id.shape[0] == res_types.shape[0]
    assert chain_id.shape[0] == coords.shape[0]
    assert chain_id.shape[0] == atom_is_present.shape[0]
    assert chain_id.shape[1] == res_types.shape[1]
    assert chain_id.shape[1] == coords.shape[1]
    assert chain_id.shape[1] == atom_is_present.shape[1]
    assert atom_is_present.shape[2] == def_co.max_n_canonical_atoms
    assert coords.shape[2] == def_co.max_n_canonical_atoms
    assert coords.shape[3] == 3
    chain_id_gold = numpy.zeros(res_types.shape, dtype=numpy.int32)
    chain_id_gold[0, 214:] = 1

    numpy.testing.assert_equal(chain_id_gold, chain_id.cpu().numpy())


def test_canonical_form_w_unk(torch_device):
    sam_pdb_lines = [
        "ATOM      1  N   MET B   1     -31.268  39.117  48.475  1.00 77.09           N\n",
        "ATOM      2  CA  MET B   1     -31.028  38.597  49.816  1.00 67.70           C\n",
        "ATOM      3  C   MET B   1     -29.543  38.428  50.141  1.00 73.35           C\n",
        "ATOM      4  O   MET B   1     -28.694  38.339  49.250  1.00 72.27           O\n",
        "ATOM      5  CB  MET B   1     -31.735  37.257  50.001  1.00 63.36           C\n",
        "ATOM      6  CG  MET B   1     -33.079  37.365  50.691  1.00 77.42           C\n",
        "ATOM      7  SD  MET B   1     -33.100  36.341  52.169  1.00 96.87           S\n",
        "ATOM      8  CE  MET B   1     -31.711  37.023  53.081  1.00 65.46           C\n",
        "ATOM      9  N   GLU B   2     -29.240  38.380  51.436  1.00 69.51           N\n",
        "ATOM     10  CA  GLU B   2     -27.881  38.092  51.867  1.00 62.90           C\n",
        "ATOM     11  C   GLU B   2     -27.547  36.614  51.710  1.00 58.48           C\n",
        "ATOM     12  O   GLU B   2     -26.386  36.265  51.469  1.00 56.03           O\n",
        "ATOM     13  CB  GLU B   2     -27.697  38.532  53.318  1.00 67.66           C\n",
        "ATOM     14  CG  GLU B   2     -26.255  38.725  53.729  1.00 85.07           C\n",
        "ATOM     15  CD  GLU B   2     -26.124  39.192  55.168  1.00102.46           C\n",
        "ATOM     16  OE1 GLU B   2     -25.106  38.856  55.814  1.00103.40           O\n",
        "ATOM     17  OE2 GLU B   2     -27.044  39.888  55.654  1.00110.47           O\n",
        "ATOM     18  N   THR B   3     -28.546  35.740  51.830  1.00 52.42           N\n",
        "ATOM     19  CA  THR B   3     -28.306  34.311  51.663  1.00 53.32           C\n",
        "ATOM     20  C   THR B   3     -27.868  33.979  50.238  1.00 56.91           C\n",
        "ATOM     21  O   THR B   3     -26.918  33.212  50.040  1.00 57.45           O\n",
        "ATOM     22  CB  THR B   3     -29.564  33.527  52.049  1.00 61.79           C\n",
        "ATOM     23  OG1 THR B   3     -29.752  33.594  53.468  1.00 65.60           O\n",
        "ATOM     24  CG2 THR B   3     -29.439  32.078  51.644  1.00 57.63           C\n",
        "HETATM 6157  N1  BYO B 401     -22.332  15.716  15.427  0.96 68.18           N\n",
        "HETATM 6158  O1  BYO B 401     -24.679  19.014  15.247  0.96 77.46           O\n",
        "HETATM 6159  O2  BYO B 401     -22.168  16.008  17.686  0.96 89.97           O\n",
        "HETATM 6160  N   SAM B 402     -11.973  14.831  12.839  1.00 43.01           N\n",
        "HETATM 6161  CA  SAM B 402     -12.317  16.240  12.957  1.00 49.99           C\n",
        "HETATM 6162  C   SAM B 402     -11.843  16.832  14.280  1.00 57.71           C\n",
        "HETATM 6163  O   SAM B 402     -11.215  17.890  14.322  1.00 69.08           O\n",
        "HETATM 6164  OXT SAM B 402     -12.068  16.268  15.351  1.00 50.47           O\n",
        "HETATM 6165  CB  SAM B 402     -13.826  16.408  12.825  1.00 59.35           C\n",
        "HETATM 6166  CG  SAM B 402     -14.273  17.046  11.515  1.00 62.73           C\n",
        "HETATM 6167  SD  SAM B 402     -16.057  16.955  11.204  1.00 72.80           S\n",
        "HETATM 6168  CE  SAM B 402     -16.756  16.414  12.789  1.00 65.06           C\n",
        "HETATM 6169  C5' SAM B 402     -16.141  15.460  10.178  1.00 50.86           C\n",
        "HETATM 6170  C4' SAM B 402     -15.126  15.534   9.037  1.00 54.58           C\n",
        "HETATM 6171  O4' SAM B 402     -15.212  14.390   8.207  1.00 51.23           O\n",
        "HETATM 6172  C3' SAM B 402     -15.399  16.749   8.160  1.00 56.20           C\n",
        "HETATM 6173  O3' SAM B 402     -14.222  17.499   8.048  1.00 53.00           O\n",
        "HETATM 6174  C2' SAM B 402     -15.777  16.207   6.801  1.00 53.40           C\n",
        "HETATM 6175  O2' SAM B 402     -15.172  16.912   5.751  1.00 49.57           O\n",
        "HETATM 6176  C1' SAM B 402     -15.190  14.819   6.855  1.00 52.80           C\n",
        "HETATM 6177  N9  SAM B 402     -15.895  13.904   5.967  1.00 50.26           N\n",
        "HETATM 6178  C8  SAM B 402     -17.243  13.755   5.774  1.00 52.22           C\n",
        "HETATM 6179  N7  SAM B 402     -17.412  12.780   4.854  1.00 51.55           N\n",
        "HETATM 6180  C5  SAM B 402     -16.191  12.321   4.470  1.00 53.98           C\n",
        "HETATM 6181  C6  SAM B 402     -15.776  11.347   3.575  1.00 50.97           C\n",
        "HETATM 6182  N6  SAM B 402     -16.670  10.646   2.891  1.00 49.66           N\n",
        "HETATM 6183  N1  SAM B 402     -14.420  11.117   3.412  1.00 51.75           N\n",
        "HETATM 6184  C2  SAM B 402     -13.481  11.837   4.123  1.00 48.76           C\n",
        "HETATM 6185  N3  SAM B 402     -13.902  12.803   5.006  1.00 46.34           N\n",
        "HETATM 6186  C4  SAM B 402     -15.231  13.034   5.166  1.00 53.49           C\n",
    ]
    can_ord = default_canonical_ordering()
    (
        chain_id,
        res_types,
        coords,
        atom_is_present,
    ) = canonical_form_from_pdb_lines(can_ord, sam_pdb_lines, torch_device)
    def_co = default_canonical_ordering()

    assert chain_id.device == torch_device
    assert res_types.device == torch_device
    assert coords.device == torch_device
    assert atom_is_present.device == torch_device

    # three and not four residues because the SAM is ignored
    assert chain_id.shape == (1, 3)
    assert res_types.shape == (1, 3)
    assert coords.shape == (1, 3, def_co.max_n_canonical_atoms, 3)
    assert atom_is_present.shape == (1, 3, def_co.max_n_canonical_atoms)
