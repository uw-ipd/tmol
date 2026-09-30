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


def test_sdf_ligand_writes_params_like_its_mol2(tmp_path):
    """Parameters are written from 3GE7's SDF as from its MOL2: one residue with
    the MOL2's heavy atoms and net charge."""
    from tmol.ligand._detect import nonstandard_residue_info_from_file
    from tmol.ligand import write_params_from_mol2

    infos = []
    for ext in ("mol2", "sdf"):
        path = _unpacked(tmp_path, f"cyclic_acylguanidinium_3ge7.{ext}.zst")
        write_params_from_mol2(path, tmp_path / f"{ext}.tmol", res_name="AFQ")
        infos.append(nonstandard_residue_info_from_file(path, res_name="AFQ"))
        assert "AFQ" in (tmp_path / f"{ext}.tmol").read_text()
    mol2, sdf = (info.atom_array for info in infos)
    assert sorted(sdf.element[sdf.element != "H"]) == sorted(
        mol2.element[mol2.element != "H"]
    )
    assert int(sdf.charge.sum()) == int(mol2.charge.sum())


def test_a_carboxyl_written_with_single_bonds_reads_as_a_carboxylate(tmp_path):
    """2XEJ's PrepWizard MOL2 types its ligand's C-terminal carbon C.3 with single
    bonds to O and OXT; the planar group is a carboxylate, not a geminal diol."""
    from rdkit import Chem

    from tmol.ligand._structure_to_smiles import ligand_smiles_from_atom_array

    ligand = atom_array_from_mol2(
        _unpacked(tmp_path, "single_bonded_carboxyl_2xej.mol2.zst")
    )
    carbon = int(list(ligand.atom_name).index("C")) + 1
    mol = Chem.MolFromSmiles(ligand_smiles_from_atom_array(ligand, with_atom_map=True))
    (atom,) = [a for a in mol.GetAtoms() if a.GetAtomMapNum() == carbon]
    assert sorted(b.GetBondTypeAsDouble() for b in atom.GetBonds()) == [1, 1, 2]


def test_repeated_pdb_atom_names_take_the_mol2_names(tmp_path):
    """PDBbind writes 10gs' glutathione conjugate as one MOL residue whose atom
    names repeat; they read as the MOL2 reader names them, so its parameters fit."""
    pdb = atom_array_from_file(
        data_path("pdbbind_regressions", "repeated_names_ligand_10gs.pdb.zst")
    )
    mol2 = atom_array_from_mol2(
        _unpacked(tmp_path, "repeated_names_ligand_10gs.mol2.zst")
    )
    assert list(pdb.atom_name) == list(mol2.atom_name)


def test_pdb_charge_column_states_an_ion_charge():
    """PoseBusters' 6TW5 writes its chloride as Cl1- in the PDB charge column, so
    the lone ion is prepared as the anion the file states."""
    ion = atom_array_from_file(
        data_path("pdbbind_regressions", "chloride_charge_column_6tw5.pdb.zst")
    )
    pose = pose_stack_from_biotite(ion, torch.device("cpu"), prepare_ligands=True)
    assert int((pose.block_type_ind >= 0).sum()) == 1


def test_heavy_atom_sdf_ligands_are_prepared(tmp_path):
    """PoseBusters SDFs have no hydrogens: their Kekule bonds say which ring N
    carries one (6T88 imidazole), and the aromatic bonds AtomWorks' protonation
    returns keep their atoms aromatic (6TW5 indazole)."""
    for name in (
        "heavy_atom_imidazole_6t88.sdf.zst",
        "heavy_atom_indazole_6tw5.sdf.zst",
    ):
        ligand = atom_array_from_file(_unpacked(tmp_path, name))
        pose = pose_stack_from_biotite(
            ligand, torch.device("cpu"), prepare_ligands=True
        )
        assert int((pose.block_type_ind >= 0).sum()) == 1


def test_a_capped_chain_break_is_a_terminus():
    """PrepWizard caps 1ERR's break after A 380 with H1 and H2 on ALA 382, which
    no mid-chain ALA holds, so ALA 382 is an N-terminus."""
    protein = atom_array_from_file(
        data_path("pdbbind_regressions", "capped_break_1err.pdb.zst")
    )
    pose = pose_stack_from_biotite(protein, torch.device("cpu"))
    bts = pose.packed_block_types.active_block_types
    labels = pose.pdb_info.residue_labels[0].tolist()
    block = pose.block_type_ind[0, labels.index(382)]
    assert bts[block].name == "ALA:nterm"


def _bonded(pose, first, second):
    labels = pose.pdb_info.residue_labels[0].tolist()
    partners = pose.inter_residue_connections[0, labels.index(first), :, 0]
    return labels.index(second) in partners.tolist()


def test_pocket_is_one_chain_broken_at_its_gaps():
    """A pocket PDB with a blank chain ID reads as one chain, and GLY 119 and
    TYR 121, 3 A apart across the missing residue 120, are not bonded."""
    pocket = atom_array_from_file(
        data_path("pdbbind_regressions", "pocket_1gpk.pdb.zst")
    )
    assert set(pocket.chain_id) == {"A"}
    pose = pose_stack_from_biotite(pocket, torch.device("cpu"))
    assert _bonded(pose, 118, 119) and not _bonded(pose, 119, 121)


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
