"""Tests for tmol.ligand.bond_order_assignment.

The tool takes a 3D structure (whose bond orders are missing or wrong) plus a
SMILES, and produces a Mol with the SMILES bond orders but the structure's atom
names + coordinates. Tests synthesize structures from SMILES with all bonds
forced to single (the "extracted from PDB/CIF" situation), then assert the tool
recovers the correct chemistry without disturbing names or coordinates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from tmol.ligand.bond_order_assignment import (
    _ensure_atom_names,
    _infer_format,
    assign_bond_orders_from_smiles,
    main,
    read_structure,
    write_structure,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "ligand_ground_truth"


# --- helpers ----------------------------------------------------------------


def _embed(smiles: str, *, with_hs: bool = True) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    if with_hs:
        mol = Chem.AddHs(mol)
    assert (
        AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE) == 0
    ), f"embed failed: {smiles}"
    return mol


def _single_bond_copy(mol: Chem.Mol) -> Chem.Mol:
    """Return a copy with every bond demoted to a plain SINGLE bond."""
    rw = Chem.RWMol(mol)
    for bond in rw.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    for atom in rw.GetAtoms():
        atom.SetIsAromatic(False)
        atom.SetFormalCharge(0)
    return rw.GetMol()


def _write_single_bond_sdf(mol: Chem.Mol, path: Path) -> None:
    Chem.MolToMolFile(_single_bond_copy(mol), str(path), kekulize=False)


def _write_single_bond_mol2(mol: Chem.Mol, path: Path) -> list[str]:
    """Write a minimal TRIPOS mol2 with all-single bonds; return atom names."""
    conf = mol.GetConformer()
    sybyl = {"C": "C.3", "N": "N.3", "O": "O.3", "S": "S.3", "P": "P.3", "H": "H"}
    names: list[str] = []
    counts: dict[str, int] = {}
    atom_lines = []
    for i, atom in enumerate(mol.GetAtoms()):
        sym = atom.GetSymbol()
        counts[sym] = counts.get(sym, 0) + 1
        name = f"{sym}{counts[sym]}"
        names.append(name)
        pos = conf.GetAtomPosition(i)
        atom_lines.append(
            f"{i + 1} {name} {pos.x:.4f} {pos.y:.4f} {pos.z:.4f} "
            f"{sybyl.get(sym, sym)} 1 LIG 0.0000"
        )
    bond_lines = [
        f"{j + 1} {b.GetBeginAtomIdx() + 1} {b.GetEndAtomIdx() + 1} 1"
        for j, b in enumerate(mol.GetBonds())
    ]
    text = "\n".join(
        [
            "@<TRIPOS>MOLECULE",
            "LIG",
            f"{mol.GetNumAtoms()} {mol.GetNumBonds()} 0 0 0",
            "SMALL",
            "NO_CHARGES",
            "",
            "@<TRIPOS>ATOM",
            *atom_lines,
            "@<TRIPOS>BOND",
            *bond_lines,
            "",
        ]
    )
    path.write_text(text)
    return names


def _bond_type_counts(mol: Chem.Mol) -> dict[str, int]:
    counts: dict[str, int] = {}
    for bond in mol.GetBonds():
        key = str(bond.GetBondType())
        counts[key] = counts.get(key, 0) + 1
    return counts


# --- format inference -------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("x.cif", "cif"),
        ("x.mmcif", "cif"),
        ("x.mol2", "mol2"),
        ("x.sdf", "sdf"),
        ("x.mol", "mol"),
        ("x.pdb", "pdb"),
        ("x.ent", "pdb"),
    ],
)
def test_infer_format(name, expected):
    assert _infer_format(Path(name)) == expected


def test_infer_format_unknown_raises():
    with pytest.raises(ValueError, match="infer format"):
        _infer_format(Path("ligand.xyz"))


def test_ensure_atom_names_synthesizes_unique_names():
    # Two carbons named "C1"/"" : the synthesized fallback for the unnamed
    # carbon must not collide with the existing "C1".
    rw = Chem.RWMol()
    a0 = Chem.Atom(6)
    a0.SetProp("_TriposAtomName", "C1")
    rw.AddAtom(a0)
    rw.AddAtom(Chem.Atom(6))  # unnamed
    rw.AddAtom(Chem.Atom(8))  # unnamed oxygen
    mol = rw.GetMol()

    _ensure_atom_names(mol)
    names = [a.GetProp("_TriposAtomName") for a in mol.GetAtoms()]
    assert names[0] == "C1"
    assert names[1] != "C1"  # collision avoided
    assert len(set(names)) == len(names)  # all unique


# --- core bond-order transfer ----------------------------------------------


def test_pdb_recovers_aromatic_benzene(tmp_path):
    pdb = tmp_path / "bnz.pdb"
    Chem.MolToPDBFile(_embed("c1ccccc1"), str(pdb))  # PDB carries no bond orders

    fixed = assign_bond_orders_from_smiles(pdb, "c1ccccc1")

    counts = _bond_type_counts(fixed)
    assert counts.get("AROMATIC", 0) == 6  # the 6 ring bonds are aromatic again


def test_sdf_corrects_wrong_single_bonds(tmp_path):
    sdf = tmp_path / "phenol.sdf"
    _write_single_bond_sdf(_embed("Oc1ccccc1"), sdf)

    # Before: everything single. After: aromatic ring recovered.
    raw = read_structure(sdf)
    assert _bond_type_counts(raw).get("AROMATIC", 0) == 0

    fixed = assign_bond_orders_from_smiles(sdf, "Oc1ccccc1")
    assert _bond_type_counts(fixed).get("AROMATIC", 0) == 6


def test_double_bond_recovered(tmp_path):
    sdf = tmp_path / "allyl.sdf"
    _write_single_bond_sdf(_embed("OCC=C"), sdf)

    fixed = assign_bond_orders_from_smiles(sdf, "OCC=C")
    assert _bond_type_counts(fixed).get("DOUBLE", 0) == 1


def test_coordinates_preserved(tmp_path):
    mol = _embed("Oc1ccccc1")
    sdf = tmp_path / "phenol.sdf"
    _write_single_bond_sdf(mol, sdf)

    fixed = assign_bond_orders_from_smiles(sdf, "Oc1ccccc1")

    src = mol.GetConformer()
    dst = fixed.GetConformer()
    assert fixed.GetNumAtoms() == mol.GetNumAtoms()
    for i in range(mol.GetNumAtoms()):
        a, b = src.GetAtomPosition(i), dst.GetAtomPosition(i)
        assert np.allclose([a.x, a.y, a.z], [b.x, b.y, b.z], atol=1e-3)


def test_explicit_hydrogens_preserved(tmp_path):
    mol = _embed("OCC=C", with_hs=True)
    n_h = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "H")
    assert n_h > 0
    sdf = tmp_path / "allyl.sdf"
    _write_single_bond_sdf(mol, sdf)

    fixed = assign_bond_orders_from_smiles(sdf, "OCC=C")
    assert sum(1 for a in fixed.GetAtoms() if a.GetSymbol() == "H") == n_h


def test_formal_charge_transferred(tmp_path):
    # Acetate: the SMILES carries a -1 on a carboxylate oxygen.
    sdf = tmp_path / "acetate.sdf"
    _write_single_bond_sdf(_embed("CC(=O)[O-]"), sdf)

    fixed = assign_bond_orders_from_smiles(sdf, "CC(=O)[O-]")
    assert sum(a.GetFormalCharge() for a in fixed.GetAtoms()) == -1


def test_atom_names_preserved_through_mol2(tmp_path):
    names = _write_single_bond_mol2(_embed("OCC=C"), tmp_path / "allyl.mol2")
    mol = read_structure(tmp_path / "allyl.mol2")
    read_names = [a.GetProp("_TriposAtomName") for a in mol.GetAtoms()]
    assert read_names == names

    fixed = assign_bond_orders_from_smiles(tmp_path / "allyl.mol2", "OCC=C")
    fixed_names = [a.GetProp("_TriposAtomName") for a in fixed.GetAtoms()]
    assert fixed_names == names
    assert _bond_type_counts(fixed).get("DOUBLE", 0) == 1


def test_real_mol2_fixture_reads_with_names_and_coords():
    mol = read_structure(DATA / "ref1.mol2")
    assert mol.GetNumAtoms() == 51
    assert mol.GetNumConformers() == 1
    names = [a.GetProp("_TriposAtomName") for a in mol.GetAtoms()]
    assert all(names)  # every atom got a non-empty name
    assert names[0] == "N1"


# --- error handling ---------------------------------------------------------


def test_mismatched_smiles_raises(tmp_path):
    sdf = tmp_path / "bnz.sdf"
    _write_single_bond_sdf(_embed("c1ccccc1"), sdf)
    with pytest.raises(ValueError, match="substructure"):
        assign_bond_orders_from_smiles(sdf, "CCO")  # ethanol != benzene


def test_unparseable_smiles_raises(tmp_path):
    sdf = tmp_path / "bnz.sdf"
    _write_single_bond_sdf(_embed("c1ccccc1"), sdf)
    with pytest.raises(ValueError, match="parse SMILES"):
        assign_bond_orders_from_smiles(sdf, "this is not a smiles ((")


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        read_structure("/nonexistent/ligand.sdf")


# --- output -----------------------------------------------------------------


def test_write_rejects_lossy_pdb(tmp_path):
    mol = assign_bond_orders_from_smiles(_make_sdf("c1ccccc1", tmp_path), "c1ccccc1")
    with pytest.raises(ValueError, match="Unsupported output format"):
        write_structure(mol, tmp_path / "out.pdb")


def _make_sdf(smiles: str, tmp_path: Path) -> Path:
    sdf = tmp_path / "in.sdf"
    _write_single_bond_sdf(_embed(smiles), sdf)
    return sdf


def test_sdf_output_roundtrip_preserves_orders_and_names(tmp_path):
    src = _make_sdf("Oc1ccccc1", tmp_path)
    mol = assign_bond_orders_from_smiles(src, "Oc1ccccc1")
    out = tmp_path / "fixed.sdf"
    write_structure(mol, out, title="PHEN")

    reread = Chem.MolFromMolFile(str(out), sanitize=True, removeHs=False)
    assert _bond_type_counts(reread).get("AROMATIC", 0) == 6
    # atom names survive as an SD data field
    suppl = Chem.SDMolSupplier(str(out), sanitize=False, removeHs=False)
    rec = next(iter(suppl))
    assert rec.HasProp("atom_names")
    assert len(rec.GetProp("atom_names").split()) == mol.GetNumAtoms()
    # names also survive as MOL "A" alias lines
    assert any(a.HasProp("molFileAlias") for a in rec.GetAtoms())


# --- CLI --------------------------------------------------------------------


def test_cli_writes_sdf(tmp_path, capsys):
    pdb = tmp_path / "bnz.pdb"
    Chem.MolToPDBFile(_embed("c1ccccc1"), str(pdb))
    out = tmp_path / "bnz.sdf"

    rc = main([str(pdb), "c1ccccc1", "-o", str(out)])
    assert rc == 0
    assert out.is_file()
    captured = capsys.readouterr()
    assert "AROMATIC=6" in captured.out

    reread = Chem.MolFromMolFile(str(out))
    assert _bond_type_counts(reread).get("AROMATIC", 0) == 6


def test_cli_default_output_path(tmp_path):
    sdf_in = _make_sdf("OCC=C", tmp_path)
    rc = main([str(sdf_in), "OCC=C"])
    assert rc == 0
    assert (tmp_path / "in.bondfix.sdf").is_file()


def test_cli_mismatch_returns_error(tmp_path, capsys):
    sdf_in = _make_sdf("c1ccccc1", tmp_path)
    rc = main([str(sdf_in), "CCO", "-o", str(tmp_path / "out.sdf")])
    assert rc == 1
    assert "error:" in capsys.readouterr().err
