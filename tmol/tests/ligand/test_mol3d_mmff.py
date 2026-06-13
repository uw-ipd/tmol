import pytest
from rdkit import Chem

import tmol.ligand.mol3d as mol3d
import tmol.ligand.rdkit_mol as rdkit_mol


class _FakeMMFFProps:
    def __init__(self, atom_count: int):
        self.atom_count = atom_count

    def GetMMFFPartialCharge(self, atom_idx: int) -> float:
        return float(atom_idx) / 10.0


def _simple_mol() -> Chem.Mol:
    mol = Chem.MolFromSmiles("CC")
    assert mol is not None
    return Chem.AddHs(mol)


def test_compute_mmff94_charges_retries_after_initial_failure(monkeypatch):
    mol = _simple_mol()
    calls = {"mmff": 0, "canonicalize": 0}

    def _fake_mmff_get_props(mmff_mol, mmffVariant):
        assert mmffVariant == "MMFF94"
        calls["mmff"] += 1
        if calls["mmff"] == 1:
            raise RuntimeError("initial-mmff-failure")
        return _FakeMMFFProps(mmff_mol.GetNumAtoms())

    orig_canonicalize = mol3d._canonicalize_mol_for_mmff

    def _track_canonicalize(mmff_mol):
        calls["canonicalize"] += 1
        orig_canonicalize(mmff_mol)

    monkeypatch.setattr(
        mol3d.AllChem, "MMFFGetMoleculeProperties", _fake_mmff_get_props
    )
    monkeypatch.setattr(mol3d, "_canonicalize_mol_for_mmff", _track_canonicalize)

    charges = mol3d.compute_mmff94_charges(mol)

    assert len(charges) == mol.GetNumAtoms()
    assert calls["mmff"] == 2
    assert calls["canonicalize"] == 1
    assert charges[0] == 0.0


def test_canonicalize_mol_for_mmff_clears_source_props():
    mol = _simple_mol()
    mol.SetProp(rdkit_mol._SOURCE_AROMATIC_PROP, "1")
    mol.SetProp(rdkit_mol._SOURCE_KEKULE_PROP, "1")

    mol3d._canonicalize_mol_for_mmff(mol)

    assert not mol.HasProp(rdkit_mol._SOURCE_AROMATIC_PROP)
    assert not mol.HasProp(rdkit_mol._SOURCE_KEKULE_PROP)


def test_strip_all_aromaticity_allows_sanitize_after_inconsistent_flags():
    mol = Chem.MolFromSmiles("c1ccncc1")
    assert mol is not None
    Chem.SanitizeMol(mol)
    mol = Chem.AddHs(mol)

    # Force an inconsistent aromatic-perception state: ring atoms/bonds are
    # flagged aromatic and carry the AROMATIC bond-order placeholder.
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in {"C", "N"}:
            atom.SetIsAromatic(True)
    for bond in mol.GetBonds():
        bond.SetIsAromatic(True)
        bond.SetBondType(Chem.BondType.AROMATIC)

    # clear_aromatic_perception_flags drops the stale aromatic flags and the
    # AROMATIC bond-order placeholders so RDKit can re-sanitize cleanly.
    mol3d.clear_aromatic_perception_flags(mol)

    assert not any(atom.GetIsAromatic() for atom in mol.GetAtoms())
    assert not any(bond.GetIsAromatic() for bond in mol.GetBonds())
    assert all(bond.GetBondType() != Chem.BondType.AROMATIC for bond in mol.GetBonds())

    # Sanitization now succeeds without raising.
    Chem.SanitizeMol(mol)


def test_compute_mmff94_charges_reports_attempt_diagnostics(monkeypatch):
    mol = _simple_mol()

    def _always_fail(*_args, **_kwargs):
        raise RuntimeError("persistent-mmff-failure")

    monkeypatch.setattr(mol3d.AllChem, "MMFFGetMoleculeProperties", _always_fail)

    with pytest.raises(
        RuntimeError,
        match="MMFF94 parameterization failed after canonicalization retry",
    ) as exc:
        mol3d.compute_mmff94_charges(mol)

    msg = str(exc.value)
    assert "attempts=[" in msg
    assert "input: RuntimeError: persistent-mmff-failure" in msg
    assert "mmff-canonicalized: RuntimeError: persistent-mmff-failure" in msg
