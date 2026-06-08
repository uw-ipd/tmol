"""Non-destructive mol2 -> tmol preparation entry points.

Mirrors Rosetta ``mol2genparams.py`` on crystallographic mol2 inputs: preserve
atom names, coordinates, bond orders, and partial charges; run only the tmol
atom-type classifier and residue builder on the OpenBabel molecule graph.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from openbabel import openbabel, pybel

from tmol.ligand.mol2_names import disambiguate_mol2_atom_name
from tmol.ligand.mol3d import compute_mmff94_charges_obmol, get_partial_charges_by_index
from tmol.ligand.ob_atom_typing import assign_obmol_atom_types
from tmol.ligand.ob_residue_builder import build_residue_type_from_obmol
from tmol.ligand.params_io import write_params_file
from tmol.ligand.registry import LigandPreparation, _build_cartbonded_params

logger = logging.getLogger(__name__)


def read_mol2(path: str | Path) -> pybel.Molecule:
    """Read the first molecule from a TRIPOS mol2 file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"mol2 file not found: {p}")
    reader = pybel.readfile("mol2", str(p))
    try:
        first = next(reader)
    except StopIteration:
        raise ValueError(f"no molecules in mol2 file: {p}") from None
    try:
        next(reader)
        logger.warning("mol2 %s contains multiple molecules; using only the first", p)
    except StopIteration:
        pass
    return first


def parse_mol2_atom_names(path: str | Path) -> list[str]:
    """Parse TRIPOS atom-name column directly from a mol2 file."""
    names: list[str] = []
    in_atom = False
    seen_molecule = False
    with open(path) as fh:
        for line in fh:
            stripped = line.strip()
            if stripped.startswith("@<TRIPOS>"):
                section = stripped
                if section == "@<TRIPOS>MOLECULE":
                    if seen_molecule:
                        break
                    seen_molecule = True
                    in_atom = False
                    continue
                in_atom = section == "@<TRIPOS>ATOM"
                continue
            if not in_atom or not stripped:
                continue
            cols = stripped.split()
            if len(cols) >= 2:
                names.append(cols[1])
    return names


def parse_mol2_elements(path: str | Path) -> list[str]:
    """Parse element symbols from mol2 Tripos atom-type column."""
    elements: list[str] = []
    in_atom = False
    seen_molecule = False
    with open(path) as fh:
        for line in fh:
            stripped = line.strip()
            if stripped.startswith("@<TRIPOS>"):
                section = stripped
                if section == "@<TRIPOS>MOLECULE":
                    if seen_molecule:
                        break
                    seen_molecule = True
                    in_atom = False
                    continue
                in_atom = section == "@<TRIPOS>ATOM"
                continue
            if not in_atom or not stripped:
                continue
            cols = stripped.split()
            if len(cols) >= 6:
                tripos = cols[5]
                if len(tripos) >= 2 and tripos[1].islower():
                    elements.append(tripos[:2])
                else:
                    elements.append(tripos[0])
    return elements


def _apply_mol2_elements_to_obmol(obmol: openbabel.OBMol, mol2_path: Path) -> None:
    """Restore element identity when OpenBabel mis-reads mol2 atom types (e.g. F)."""
    from openbabel import openbabel as ob

    elements = parse_mol2_elements(mol2_path)
    for obatom in ob.OBMolAtomIter(obmol):
        idx = obatom.GetIndex()
        if idx >= len(elements):
            continue
        z = ob.GetAtomicNum(elements[idx])
        if z > 0:
            obatom.SetAtomicNum(z)


def parse_mol2_sybyl_types(path: str | Path) -> list[str]:
    """Parse TRIPOS atom-type column from a mol2 file."""
    types: list[str] = []
    in_atom = False
    seen_molecule = False
    with open(path) as fh:
        for line in fh:
            stripped = line.strip()
            if stripped.startswith("@<TRIPOS>"):
                section = stripped
                if section == "@<TRIPOS>MOLECULE":
                    if seen_molecule:
                        break
                    seen_molecule = True
                    in_atom = False
                    continue
                in_atom = section == "@<TRIPOS>ATOM"
                continue
            if not in_atom or not stripped:
                continue
            cols = stripped.split()
            if len(cols) >= 6:
                types.append(cols[5])
    return types


def get_disambiguated_mol2_atom_names(mol2_path: str | Path) -> list[str]:
    """Parse mol2 atom names with Rosetta-style duplicate disambiguation."""
    raw = parse_mol2_atom_names(mol2_path)
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in raw:
        seen[name] = seen.get(name, 0) + 1
        out.append(disambiguate_mol2_atom_name(name, seen[name]))
    return out


def get_original_atom_names(
    obmol: openbabel.OBMol,
    mol2_path: str | Path | None = None,
) -> dict[int, str]:
    """Map OBMol atom indices to original mol2 atom names."""
    if mol2_path is not None:
        parsed = get_disambiguated_mol2_atom_names(mol2_path)
        if len(parsed) == obmol.NumAtoms():
            return {i: name for i, name in enumerate(parsed)}
        logger.warning(
            "mol2 %s has %d atom-name rows but OBMol has %d atoms",
            mol2_path,
            len(parsed),
            obmol.NumAtoms(),
        )
    names: dict[int, str] = {}
    for obatom in openbabel.OBMolAtomIter(obmol):
        elem = openbabel.GetSymbol(obatom.GetAtomicNum()) or "X"
        names[obatom.GetIndex()] = f"{elem}{obatom.GetIdx()}"
    return names


def _default_res_name(mol: pybel.Molecule, fallback: str = "LG1") -> str:
    title = (mol.title or "").strip()
    if title:
        return title.split()[0][:3].upper() or fallback
    return fallback


def _obmol_coords_by_name(
    obmol: openbabel.OBMol, name_by_idx: dict[int, str]
) -> dict[str, tuple[float, float, float]]:
    coords: dict[str, tuple[float, float, float]] = {}
    for obatom in openbabel.OBMolAtomIter(obmol):
        idx = obatom.GetIndex()
        name = name_by_idx.get(idx)
        if name is None:
            continue
        coords[name] = (
            float(obatom.GetX()),
            float(obatom.GetY()),
            float(obatom.GetZ()),
        )
    return coords


def prepare_ligand_from_mol2_passthrough(
    mol2_path: str | Path,
    res_name: Optional[str] = None,
    *,
    rename_atoms: bool = False,
    charge_mode: str = "input",
) -> LigandPreparation:
    """Build a ``LigandPreparation`` from mol2 without rebuilding the ligand."""
    path = Path(mol2_path)
    mol = read_mol2(path)
    obmol = mol.OBMol
    _apply_mol2_elements_to_obmol(obmol, path)
    original_names = get_original_atom_names(obmol, mol2_path=path)
    original_charges_by_idx = get_partial_charges_by_index(mol)

    from tmol.ligand.ob_atom_typing import hyb_from_mol2_sybyl

    sybyl = parse_mol2_sybyl_types(path)
    mol2_hyb_by_idx = {
        i: hyb_from_mol2_sybyl(s) for i, s in enumerate(sybyl) if i < obmol.NumAtoms()
    }
    assignments = assign_obmol_atom_types(obmol, mol2_hyb_by_idx=mol2_hyb_by_idx)
    if not rename_atoms:
        assignments = [
            (
                a._replace(atom_name=original_names[a.index])
                if a.index in original_names
                else a
            )
            for a in assignments
        ]

    final_res_name = res_name if res_name is not None else _default_res_name(mol)
    restype = build_residue_type_from_obmol(obmol, final_res_name, assignments)

    name_by_idx = {a.index: a.atom_name for a in assignments}
    input_charges = {
        name_by_idx[i]: q
        for i, q in original_charges_by_idx.items()
        if i in name_by_idx
    }

    if charge_mode == "mmff94":
        by_index = compute_mmff94_charges_obmol(obmol)
        charges = {name_by_idx[i]: by_index[i] for i in by_index if i in name_by_idx}
    else:
        charges = input_charges

    coords = _obmol_coords_by_name(obmol, name_by_idx)
    atom_type_elements = {at.atom_type: at.element for at in assignments}
    return LigandPreparation(
        residue_type=restype,
        partial_charges=charges,
        cartbonded_params=_build_cartbonded_params(restype, coords=coords),
        atom_type_elements=atom_type_elements,
    )


def write_params_from_mol2(
    mol2_path: str | Path,
    output_path: str | Path,
    res_name: Optional[str] = None,
    rename_atoms: bool = False,
) -> None:
    """Write a classic Rosetta ``.params`` file from a mol2 input."""
    prep = prepare_ligand_from_mol2_passthrough(
        mol2_path,
        res_name=res_name,
        rename_atoms=rename_atoms,
        charge_mode="input",
    )
    restype, charges = prep.residue_type, prep.partial_charges
    write_params_file(restype, output_path, partial_charges=charges)
