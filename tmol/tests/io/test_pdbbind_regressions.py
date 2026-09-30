"""Regressions found reading PDBbind v2013-core complexes (PDB, MOL2 and SDF files)."""

import zstandard

from tmol.io import atom_array_from_file, atom_array_from_mol2
from tmol.tests.data import data_path


def _unpacked(tmp_path, name):
    """A fixture decompressed under its own name, for readers that go by suffix."""
    path = tmp_path / name.removesuffix(".zst")
    fixture = data_path("pdbbind_regressions", name)
    path.write_bytes(zstandard.decompress(fixture.read_bytes()))
    return path


def test_ring_delocalized_cations_localize(tmp_path):
    """Tripos ``ar`` bonds of a C.cat inside a ring give one double bond and a cation."""
    for name, net in (
        ("cyclic_acylguanidinium_3ge7.mol2.zst", 2),
        ("iminohydantoin_4djv.mol2.zst", 1),
    ):
        ligand = atom_array_from_mol2(_unpacked(tmp_path, name))
        assert int(ligand.charge.sum()) == net
        assert set(ligand.element[ligand.charge != 0]) == {"N"}


def test_sdf_ligand_reads_like_its_mol2(tmp_path):
    """An SDF ligand has its MOL2's chemistry: the charge the SDF writes on the
    C.cat carbon sits on a nitrogen, and its stray aromatic bond is single."""
    mol2, sdf = (
        atom_array_from_file(
            _unpacked(tmp_path, f"cyclic_acylguanidinium_3ge7.{ext}.zst")
        )
        for ext in ("mol2", "sdf")
    )
    assert list(sdf.element) == list(mol2.element)
    assert int(sdf.charge.sum()) == int(mol2.charge.sum()) == 2
    assert set(sdf.element[sdf.charge != 0]) == {"N"}
