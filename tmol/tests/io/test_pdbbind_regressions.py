"""Regressions found reading PDBbind v2013-core complexes (PDB, MOL2 and SDF files)."""

import torch
import zstandard

from tmol.io import atom_array_from_file, atom_array_from_mol2, pose_stack_from_biotite
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


def _bonded(pose, first, second):
    labels = pose.pdb_info.residue_labels[0].tolist()
    partners = pose.inter_residue_connections[0, labels.index(first), :, 0]
    return labels.index(second) in partners.tolist()


def test_numbered_bond_across_a_gap_is_not_kept():
    """Residues numbered 37 and 38 but 5.6 A apart are bonded only when asked."""
    structure = atom_array_from_file(
        data_path("pdbbind_regressions", "numbered_gap_3kgp.pdb.zst")
    )
    for threshold, bonded in ((2.4, False), (0.0, True)):
        pose = pose_stack_from_biotite(
            structure,
            torch.device("cpu"),
            missing_density_distance_threshold=threshold,
        )
        assert _bonded(pose, 37, 38) == bonded
