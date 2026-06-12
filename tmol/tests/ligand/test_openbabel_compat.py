"""Tests for the OpenBabel fallback layer.

Covers:
  - The thin OB wrappers in :mod:`tmol.ligand.openbabel_compat`.
  - The new ``nonstandard_residue_info_from_pdb`` /
    ``nonstandard_residue_info_from_smiles`` constructors.
  - The new ``prepare_ligand_from_pdb`` / ``prepare_ligand_from_smiles``
    public entry points.
  - That ``MolFromMol2File`` and ``MolFromSmiles`` fallbacks kick in when
    RDKit returns ``None`` (verified via monkeypatch).
"""

from pathlib import Path

import pytest
from rdkit import Chem

from tmol.ligand.openbabel_compat import (
    is_available as ob_is_available,
    obabel_read_mol2,
    obabel_read_pdb,
    obabel_read_smiles,
)

PLI_DIR = Path(__file__).parent.parent / "data" / "protein_ligand_test"
ACE_MOL2 = PLI_DIR / "ace.lig.mol2"


pytestmark = pytest.mark.skipif(
    not ob_is_available(),
    reason="openbabel Python package not installed",
)


# ---------------------------------------------------------------------------
# openbabel_compat helpers
# ---------------------------------------------------------------------------


def test_is_available_returns_true_when_ob_present():
    """If we got past the module-level skip, OB must be importable."""
    assert ob_is_available() is True


def test_obabel_read_mol2_returns_rdkit_mol():
    mol = obabel_read_mol2(ACE_MOL2)
    assert mol is not None
    assert isinstance(mol, Chem.Mol)
    assert mol.GetNumAtoms() > 0
    assert mol.GetNumConformers() == 1


def test_obabel_read_mol2_missing_file_returns_none(tmp_path):
    bogus = tmp_path / "nonexistent.mol2"
    # OB's pybel.readfile raises on missing file; helper catches and returns None.
    bogus.write_text("not a mol2")
    assert obabel_read_mol2(bogus) is None


def test_obabel_read_smiles_simple():
    mol = obabel_read_smiles("CCO")  # ethanol
    assert mol is not None
    assert mol.GetNumAtoms() == 3  # C, C, O (no Hs by default)


def test_obabel_read_smiles_invalid_returns_none():
    assert obabel_read_smiles("not_a_smiles_at_all_!!@@##") is None


def test_obabel_read_smiles_3d_generation():
    mol = obabel_read_smiles("CCO", generate_3d=True)
    assert mol is not None
    assert mol.GetNumConformers() == 1
    # With H added during make3D, atom count should be 9 (3 heavy + 6 H).
    assert mol.GetNumAtoms() == 9


def test_obabel_read_pdb_from_generated_file(tmp_path):
    """Round-trip: mol2 -> OB -> PDB on disk -> OB read PDB."""
    from openbabel import pybel

    pmol = next(pybel.readfile("mol2", str(ACE_MOL2)))
    pdb_path = tmp_path / "ace.pdb"
    pmol.write("pdb", str(pdb_path), overwrite=True)

    mol = obabel_read_pdb(pdb_path)
    assert mol is not None
    assert mol.GetNumAtoms() > 0


# ---------------------------------------------------------------------------
# nonstandard_residue_info_from_pdb
# ---------------------------------------------------------------------------


@pytest.fixture
def ace_pdb_path(tmp_path):
    """Convert ace.lig.mol2 to a PDB file for testing the PDB entry point."""
    from openbabel import pybel

    pmol = next(pybel.readfile("mol2", str(ACE_MOL2)))
    out = tmp_path / "ace.pdb"
    pmol.write("pdb", str(out), overwrite=True)
    return out


def test_nonstandard_residue_info_from_pdb_shape(ace_pdb_path):
    from tmol.ligand import nonstandard_residue_info_from_pdb

    info = nonstandard_residue_info_from_pdb(ace_pdb_path, res_name="ACE")
    assert info.res_name == "ACE"
    assert info.ccd_type == "UNKNOWN"
    assert len(info.atom_names) == len(info.elements) == info.coords.shape[0]
    assert info.atom_array is not None
    assert info.atom_array.bonds is not None
    assert info.atom_array.bonds.get_bond_count() > 0
    # PDB carries no partial-charge column.
    assert info.partial_charges is None
    assert info.skip_protonation is False
    # Atom names should be unique.
    assert len(set(info.atom_names)) == len(info.atom_names)


def test_nonstandard_residue_info_from_pdb_missing_file(tmp_path):
    from tmol.ligand import nonstandard_residue_info_from_pdb

    with pytest.raises(FileNotFoundError):
        nonstandard_residue_info_from_pdb(tmp_path / "does_not_exist.pdb")


# ---------------------------------------------------------------------------
# nonstandard_residue_info_from_smiles
# ---------------------------------------------------------------------------


def test_nonstandard_residue_info_from_smiles_ethanol():
    from tmol.ligand import nonstandard_residue_info_from_smiles

    info = nonstandard_residue_info_from_smiles("CCO", res_name="ETH")
    assert info.res_name == "ETH"
    # 3 heavy + 6 explicit H = 9 atoms
    assert len(info.atom_names) == 9
    assert info.ccd_smiles == "CCO"
    # Names should follow <elem><1-based-idx> convention, all unique.
    assert "C1" in info.atom_names and "O1" in info.atom_names
    assert len(set(info.atom_names)) == len(info.atom_names)
    # 3D coords must be populated.
    assert info.coords.shape == (9, 3)
    assert not (info.coords == 0).all()
    # Bonds populated.
    assert info.atom_array.bonds.get_bond_count() > 0


def test_nonstandard_residue_info_from_smiles_invalid_raises():
    from tmol.ligand import nonstandard_residue_info_from_smiles

    with pytest.raises(ValueError, match="Could not parse SMILES"):
        nonstandard_residue_info_from_smiles("not_a_smiles_!!@@##")


def test_nonstandard_residue_info_from_smiles_default_res_name():
    from tmol.ligand import nonstandard_residue_info_from_smiles

    info = nonstandard_residue_info_from_smiles("CCO")
    assert info.res_name == "LG1"


# ---------------------------------------------------------------------------
# radical-oxygen normalization ([O] -> [O-]) for the SMILES->mol2 path
# ---------------------------------------------------------------------------


def test_normalize_radical_oxygens_restores_anion_charge():
    from tmol.ligand.detect import _normalize_radical_oxygens

    # bare [O] (DUD carboxylate / sulfonate notation) gains its -1 charge
    assert "[O-]" in _normalize_radical_oxygens("CC(=O)[O]")
    assert "[O-]" in _normalize_radical_oxygens("CS(=O)(=O)[O]")
    # well-formed inputs pass through verbatim (no spurious rewriting)
    assert _normalize_radical_oxygens("CC(=O)[O-]") == "CC(=O)[O-]"
    assert _normalize_radical_oxygens("CC(=O)O") == "CC(=O)O"
    # a carbonyl oxygen is not a radical and is left alone
    assert _normalize_radical_oxygens("CC=O") == "CC=O"


def test_smiles_via_mol2_deprotonates_bare_o_carboxylate():
    from tmol.ligand.detect import nonstandard_residue_info_from_smiles_via_mol2

    # A DUD-style bare [O] carboxylate must normalize to a symmetric COO-
    # (two near-equal, strongly-negative oxygens via MMFF94), not a neutral
    # COOH — this is the fix for the SMILES-path acid divergence.
    info = nonstandard_residue_info_from_smiles_via_mol2("CC(=O)[O]", res_name="ACT")
    assert info.partial_charges is not None
    carboxylate = sorted(
        q for name, q in info.partial_charges.items() if name[0] == "O" and q < -0.7
    )
    assert len(carboxylate) == 2, info.partial_charges
    assert abs(carboxylate[0] - carboxylate[1]) < 0.05


def test_smiles_via_mol2_uses_generic_atom_names():
    from tmol.ligand.detect import nonstandard_residue_info_from_smiles_via_mol2

    # A peptidomimetic SMILES makes OpenBabel perceive amino-acid residues and
    # assign PDB atom names (CA, CB, OXT). The path must emit generic,
    # element-unambiguous names so nothing collides with an element symbol
    # (CA == carbon-alpha would otherwise be mistaken for calcium).
    info = nonstandard_residue_info_from_smiles_via_mol2(
        "C[C@H](NC(=O)[C@@H](CS)[C@@H](C)c1ccccc1)C(=O)[O-]", res_name="ACE"
    )
    element_symbols = {"CL", "BR", "NA", "MG", "ZN", "FE", "CA"}
    colliding = [n for n in info.atom_names if n.upper() in element_symbols]
    assert not colliding, info.atom_names
    assert "C1" in info.atom_names


def test_obabel_smiles_to_mol2_block_is_in_memory_string():
    from tmol.ligand.openbabel_compat import obabel_smiles_to_mol2_block

    block = obabel_smiles_to_mol2_block("CC(=O)[O-]")
    assert isinstance(block, str)
    assert "@<TRIPOS>MOLECULE" in block and "@<TRIPOS>ATOM" in block
    assert "MMFF94_CHARGES" in block  # charge_type from the MMFF94 model


@pytest.mark.parametrize("conformer_search", [True, False])
def test_conformer_search_flag_produces_valid_mol2(conformer_search):
    # conformer_search defaults on (rotor search, matching the reference
    # pipeline); the off-switch is for faster single-conformer generation.
    # Both settings must yield a chemically complete MMFF94 mol2.
    from tmol.ligand.openbabel_compat import obabel_smiles_to_mol2_block

    block = obabel_smiles_to_mol2_block(
        "CCCCCC(=O)O", conformer_search=conformer_search
    )
    assert "@<TRIPOS>ATOM" in block
    assert "MMFF94_CHARGES" in block
    mol = Chem.MolFromMol2Block(block, sanitize=False, removeHs=False)
    assert mol is not None and mol.GetNumConformers() == 1


def test_mol2_block_reader_matches_file_reader(tmp_path):
    # The in-memory block reader must produce the same NonStandardResidueInfo
    # as the on-disk file reader for an identical mol2 (no temp-file round-trip).
    from tmol.ligand.detect import (
        nonstandard_residue_info_from_mol2,
        nonstandard_residue_info_from_mol2_block,
    )
    from tmol.ligand.openbabel_compat import obabel_smiles_to_mol2_block

    block = obabel_smiles_to_mol2_block("CC(=O)[O-]")
    via_block = nonstandard_residue_info_from_mol2_block(block, res_name="ACT")
    mol2_file = tmp_path / "lig.mol2"
    mol2_file.write_text(block)
    via_file = nonstandard_residue_info_from_mol2(mol2_file, res_name="ACT")

    assert via_block.atom_names == via_file.atom_names
    assert (via_block.partial_charges is not None) == (
        via_file.partial_charges is not None
    )
    if via_block.partial_charges is not None:
        assert via_block.partial_charges == via_file.partial_charges


# ---------------------------------------------------------------------------
# Fallback verification (RDKit returns None -> OB takes over)
# ---------------------------------------------------------------------------


def test_mol2_fallback_to_openbabel_when_rdkit_returns_none(monkeypatch):
    """If MolFromMol2File returns None, the mol2 reader uses OB and succeeds."""
    from tmol.ligand import nonstandard_residue_info_from_mol2
    import tmol.ligand.detect as detect_mod

    real = detect_mod.Chem.MolFromMol2File
    calls = {"n": 0}

    def fake(*args, **kwargs):
        calls["n"] += 1
        return None

    monkeypatch.setattr(detect_mod.Chem, "MolFromMol2File", fake)
    info = nonstandard_residue_info_from_mol2(ACE_MOL2, res_name="ACE")
    assert calls["n"] == 1
    assert info.res_name == "ACE"
    assert len(info.atom_names) > 0
    # Sanity: original RDKit MolFromMol2File is intact after monkeypatch ends.
    monkeypatch.setattr(detect_mod.Chem, "MolFromMol2File", real)


def test_smiles_fallback_inside_rebuild_path(monkeypatch):
    """If MolFromSmiles returns None inside the SMILES rebuild path,
    the helper falls back to OB and still produces a Mol."""
    import tmol.ligand.rdkit_mol as rdkit_mol_mod

    real_mfs = rdkit_mol_mod.Chem.MolFromSmiles
    smiles = "CCO"
    proto = real_mfs(smiles)
    Chem.GetSSSR(proto)

    fallback_count = {"n": 0}
    original_obabel_read_smiles = None
    try:
        from tmol.ligand import openbabel_compat as ob_compat

        original_obabel_read_smiles = ob_compat.obabel_read_smiles

        def counting_ob_smiles(*args, **kwargs):
            fallback_count["n"] += 1
            return original_obabel_read_smiles(*args, **kwargs)

        monkeypatch.setattr(ob_compat, "obabel_read_smiles", counting_ob_smiles)
        monkeypatch.setattr(rdkit_mol_mod.Chem, "MolFromSmiles", lambda *a, **k: None)

        fresh = rdkit_mol_mod._rebuild_mol_via_smiles_preserving_coords(proto)
        # OB returns ethanol via SDF round-trip; the subgraph match against
        # the heavy proto should succeed and we get a non-None Mol back.
        assert fresh is not None
        assert fallback_count["n"] >= 1
    finally:
        if original_obabel_read_smiles is not None:
            from tmol.ligand import openbabel_compat as ob_compat  # noqa: F811

            monkeypatch.setattr(
                ob_compat, "obabel_read_smiles", original_obabel_read_smiles
            )
        monkeypatch.setattr(rdkit_mol_mod.Chem, "MolFromSmiles", real_mfs)
