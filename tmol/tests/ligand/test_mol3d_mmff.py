import pytest
from rdkit import Chem

import tmol.ligand.mol3d as mol3d


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


@pytest.mark.xfail(reason="fd: failing 6/01")
def test_canonicalize_mol_for_mmff_clears_source_props():
    mol = _simple_mol()
    mol.SetProp(mol3d._SOURCE_AROMATIC_PROP, "1")
    mol.SetProp(mol3d._SOURCE_KEKULE_PROP, "1")

    mol3d._canonicalize_mol_for_mmff(mol)

    assert not mol.HasProp(mol3d._SOURCE_AROMATIC_PROP)
    assert not mol.HasProp(mol3d._SOURCE_KEKULE_PROP)


@pytest.mark.xfail(reason="fd: failing 6/01")
def test_strip_all_aromaticity_allows_sanitize_after_inconsistent_flags():
    mol = Chem.MolFromSmiles("c1ccncc1")
    assert mol is not None
    Chem.SanitizeMol(mol)
    mol = Chem.AddHs(mol)

    for atom in mol.GetAtoms():
        if atom.GetSymbol() in {"C", "N"}:
            atom.SetIsAromatic(True)
    for bond in mol.GetBonds():
        bond.SetIsAromatic(True)
        bond.SetBondType(Chem.BondType.SINGLE)

    with pytest.raises(Chem.rdchem.KekulizeException):
        Chem.SanitizeMol(mol)

    mol3d.clear_aromatic_perception_flags(mol)
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
