"""CIF normalization and audit utilities for ligand inputs.

This module centralizes the MOL2->CIF rendering policy used by large-scale
equivalence runs and provides an audit/repair helper for paired CIF/MOL2
inputs (for example PLI fixtures).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import biotite.structure.io.pdbx as pdbx
from rdkit import Chem

from tmol.ligand.atom_typing import sanitize_tolerant

_BOND_ORDER_TO_CIF = {
    "SINGLE": "SING",
    "DOUBLE": "DOUB",
    "TRIPLE": "TRIP",
    "AROMATIC": "AROM",
}


@dataclass(frozen=True)
class CifBondAuditResult:
    """Structured result for CIF-vs-MOL2 bond-table audits."""

    consistent: bool
    atom_count_cif: int
    atom_count_mol2: int
    missing_in_cif: tuple[tuple[str, str, str, str], ...]
    extra_in_cif: tuple[tuple[str, str, str, str], ...]
    note: str = ""


def infer_paired_mol2_path(cif_path: str | Path) -> Path | None:
    """Infer a sibling MOL2 for a PLI-style ``*.ligand.cif`` path."""
    path = Path(cif_path)
    if not path.name.endswith(".ligand.cif"):
        return None
    target = path.name[: -len(".ligand.cif")]
    candidate = path.parent.parent / f"{target}.lig.mol2"
    return candidate if candidate.is_file() else None


def _load_mol2(path: Path) -> Chem.Mol:
    mol = Chem.MolFromMol2File(
        str(path),
        sanitize=False,
        removeHs=False,
        cleanupSubstructures=False,
    )
    if mol is None:
        raise ValueError(f"Could not parse MOL2: {path}")
    sanitize_tolerant(mol)
    if mol.GetNumConformers() == 0:
        raise ValueError(f"MOL2 has no 3D coordinates: {path}")
    return mol


def _canonical_atom_name(atom: Chem.Atom) -> str:
    if atom.HasProp("_TriposAtomName"):
        name = atom.GetProp("_TriposAtomName").strip()
        if name:
            return name
    return f"{atom.GetSymbol()}{atom.GetIdx() + 1}"


def _source_subtype_from_mol2_type(mol2_type: str) -> str:
    if not mol2_type:
        return "?"
    parts = mol2_type.split(".")
    if len(parts) < 2:
        return "?"
    return parts[1] or "?"


def _canonical_cif_bond_tuple(
    atom_name_1: str,
    atom_name_2: str,
    value_order: str,
    aromatic_flag: str,
) -> tuple[str, str, str, str]:
    a1, a2 = sorted((atom_name_1.strip(), atom_name_2.strip()))
    order = str(value_order).strip().upper() or "SING"
    aromatic = "Y" if str(aromatic_flag).strip().upper() == "Y" else "N"
    return (a1, a2, order, aromatic)


def _expected_bonds_from_mol2(mol: Chem.Mol) -> set[tuple[str, str, str, str]]:
    expected: set[tuple[str, str, str, str]] = set()
    for bond in mol.GetBonds():
        a = bond.GetBeginAtom()
        b = bond.GetEndAtom()
        a_name = _canonical_atom_name(a)
        b_name = _canonical_atom_name(b)
        # Keep the DUD-scale normalization policy: aromatic non-ring bonds
        # are represented as plain single bonds in CIF chemistry tables.
        if bond.GetIsAromatic() and not bond.IsInRing():
            order = "SING"
            aromatic_flag = "N"
        else:
            btype = "AROMATIC" if bond.GetIsAromatic() else str(bond.GetBondType())
            order = _BOND_ORDER_TO_CIF.get(btype, "SING")
            aromatic_flag = "Y" if (bond.GetIsAromatic() and bond.IsInRing()) else "N"
        expected.add(_canonical_cif_bond_tuple(a_name, b_name, order, aromatic_flag))
    return expected


def _actual_bonds_from_cif(
    cif_path: Path,
) -> tuple[set[tuple[str, str, str, str]], int]:
    cif = pdbx.CIFFile.read(str(cif_path))
    block = cif.block
    atom_site = block["atom_site"]
    atom_count = len(atom_site["label_atom_id"].as_array())
    if "chem_comp_bond" not in block:
        return set(), atom_count
    bond_site = block["chem_comp_bond"]
    atom_id_1 = [str(v) for v in bond_site["atom_id_1"].as_array()]
    atom_id_2 = [str(v) for v in bond_site["atom_id_2"].as_array()]
    value_order = [str(v) for v in bond_site["value_order"].as_array()]
    if "pdbx_aromatic_flag" in bond_site:
        aromatic_flags = [str(v) for v in bond_site["pdbx_aromatic_flag"].as_array()]
    else:
        aromatic_flags = ["N"] * len(value_order)
    records: set[tuple[str, str, str, str]] = set()
    for a1, a2, order, aromatic in zip(
        atom_id_1, atom_id_2, value_order, aromatic_flags, strict=False
    ):
        records.add(_canonical_cif_bond_tuple(a1, a2, order, aromatic))
    return records, atom_count


def audit_cif_bonds_vs_mol2(
    cif_path: str | Path,
    mol2_path: str | Path,
) -> CifBondAuditResult:
    """Compare CIF bond-table records against MOL2-derived canonical records."""
    cif_path = Path(cif_path)
    mol2_path = Path(mol2_path)
    mol = _load_mol2(mol2_path)
    expected = _expected_bonds_from_mol2(mol)
    actual, atom_count_cif = _actual_bonds_from_cif(cif_path)
    atom_count_mol2 = mol.GetNumAtoms()

    missing = tuple(sorted(expected - actual))
    extra = tuple(sorted(actual - expected))
    note = ""
    if atom_count_cif != atom_count_mol2:
        note = (
            f"Atom count mismatch (cif={atom_count_cif}, mol2={atom_count_mol2}); "
            "regeneration recommended."
        )
    return CifBondAuditResult(
        consistent=(
            len(missing) == 0 and len(extra) == 0 and atom_count_cif == atom_count_mol2
        ),
        atom_count_cif=atom_count_cif,
        atom_count_mol2=atom_count_mol2,
        missing_in_cif=missing,
        extra_in_cif=extra,
        note=note,
    )


def render_cif_from_mol2(mol2_path: str | Path, res_name: str) -> str:
    """Render a CIF text block from a MOL2 file with tmol annotations."""
    mol = _load_mol2(Path(mol2_path))
    conf = mol.GetConformer()

    lines: list[str] = ["data_structure", "#", "loop_"]
    lines.extend(
        [
            "_chem_comp_bond.pdbx_ordinal ",
            "_chem_comp_bond.comp_id ",
            "_chem_comp_bond.atom_id_1 ",
            "_chem_comp_bond.atom_id_2 ",
            "_chem_comp_bond.value_order ",
            "_chem_comp_bond.pdbx_aromatic_flag ",
            "_chem_comp_bond.pdbx_stereo_config ",
        ]
    )

    for idx, bond in enumerate(mol.GetBonds(), start=1):
        a = bond.GetBeginAtom()
        b = bond.GetEndAtom()
        a_name = _canonical_atom_name(a)
        b_name = _canonical_atom_name(b)
        if bond.GetIsAromatic() and not bond.IsInRing():
            order = "SING"
            aromatic_flag = "N"
        else:
            btype = "AROMATIC" if bond.GetIsAromatic() else str(bond.GetBondType())
            order = _BOND_ORDER_TO_CIF.get(btype, "SING")
            aromatic_flag = "Y" if (bond.GetIsAromatic() and bond.IsInRing()) else "N"
        lines.append(
            f"{idx:<2} {res_name:<3} {a_name:<4} {b_name:<4} {order:<4} {aromatic_flag} ?"
        )

    lines.extend(
        [
            "#",
            "loop_",
            "_atom_site.group_PDB ",
            "_atom_site.type_symbol ",
            "_atom_site.label_atom_id ",
            "_atom_site.label_alt_id ",
            "_atom_site.label_comp_id ",
            "_atom_site.label_asym_id ",
            "_atom_site.label_entity_id ",
            "_atom_site.label_seq_id ",
            "_atom_site.pdbx_PDB_ins_code ",
            "_atom_site.auth_seq_id ",
            "_atom_site.auth_comp_id ",
            "_atom_site.auth_asym_id ",
            "_atom_site.auth_atom_id ",
            "_atom_site.B_iso_or_equiv ",
            "_atom_site.occupancy ",
            "_atom_site.pdbx_formal_charge ",
            "_atom_site.Cartn_x ",
            "_atom_site.Cartn_y ",
            "_atom_site.Cartn_z ",
            "_atom_site.pdbx_PDB_model_num ",
            "_atom_site.id ",
            "_atom_site.partial_charge ",
            "_atom_site.tmol_aromatic ",
            "_atom_site.tmol_source_subtype ",
        ]
    )

    for idx, atom in enumerate(mol.GetAtoms(), start=1):
        atom_name = _canonical_atom_name(atom)
        mol2_type = (
            atom.GetProp("_TriposAtomType") if atom.HasProp("_TriposAtomType") else ""
        )
        subtype = _source_subtype_from_mol2_type(mol2_type)
        charge = (
            float(atom.GetProp("_TriposPartialCharge"))
            if atom.HasProp("_TriposPartialCharge")
            else float(atom.GetFormalCharge())
        )
        p = conf.GetAtomPosition(atom.GetIdx())
        aromatic_flag = "Y" if atom.GetIsAromatic() else "N"
        lines.append(
            "HETATM "
            f"{atom.GetSymbol():<2} "
            f"{atom_name:<4} "
            ". "
            f"{res_name:<3} "
            "A 1 1 . "
            f"1 {res_name:<3} A {atom_name:<4} "
            "nan 0.0 ? "
            f"{float(p.x):.4f} {float(p.y):.4f} {float(p.z):.4f} "
            f"1 {idx} {charge:+.6f} {aromatic_flag} {subtype}"
        )

    lines.append("#")
    lines.append("")
    return "\n".join(lines)


def repaired_cif_path_from_mol2(
    cif_path: str | Path,
    mol2_path: str | Path,
    res_name: str,
) -> tuple[Path, CifBondAuditResult, bool]:
    """Return a CIF path repaired from MOL2 when bond tables are inconsistent.

    Returns ``(path_to_use, audit_result, regenerated)``.
    """
    cif_path = Path(cif_path)
    audit = audit_cif_bonds_vs_mol2(cif_path, mol2_path)
    if audit.consistent:
        return cif_path, audit, False

    regenerated_text = render_cif_from_mol2(mol2_path, res_name=res_name)
    with NamedTemporaryFile("w", suffix=".ligand.regenerated.cif", delete=False) as tmp:
        tmp.write(regenerated_text)
        tmp.flush()
        regen_path = Path(tmp.name)
    return regen_path, audit, True
