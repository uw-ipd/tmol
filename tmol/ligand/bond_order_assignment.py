"""Assign bond orders from a SMILES template onto a 3D ligand structure.

Small molecules extracted from a PDB or mmCIF rarely carry chemistry-level
bond orders: the coordinates are trustworthy, but distance-based bond
perception (e.g. OpenBabel's ``PerceiveBondOrders``) frequently guesses the
wrong double / aromatic bonds. When a correct SMILES is known, this module
transfers the SMILES bond orders and formal charges onto the experimental
structure while preserving the input's atom names and 3D coordinates.

The transfer is RDKit's :func:`AssignBondOrdersFromTemplate`: the heavy-atom
template (parsed from SMILES) is matched as a substructure of the structure's
connectivity graph, then every heavy-heavy bond order and formal charge is set
from the template. Atom names, coordinates, and any explicit hydrogens from the
input are kept unchanged -- only bond chemistry comes from the SMILES.

Usage::

    python -m tmol.ligand.bond_order_assignment ligand.pdb "c1ccccc1O" -o fixed.sdf

or as a library::

    from tmol.ligand.bond_order_assignment import assign_bond_orders_from_smiles
    mol = assign_bond_orders_from_smiles("ligand.cif", "c1ccccc1O")
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)

# Map a file extension to a canonical input/output format name.
_EXT_TO_FORMAT = {
    ".cif": "cif",
    ".mmcif": "cif",
    ".mol2": "mol2",
    ".sdf": "sdf",
    ".mol": "mol",
    ".pdb": "pdb",
    ".ent": "pdb",
}

_NAME_PROP = "_TriposAtomName"


def _infer_format(path: Path) -> str:
    """Return the canonical format name for ``path`` from its extension."""
    ext = path.suffix.lower()
    if ext not in _EXT_TO_FORMAT:
        raise ValueError(
            f"Cannot infer format from {path.name!r}; supported extensions: "
            f"{sorted(_EXT_TO_FORMAT)}. Pass an explicit format instead."
        )
    return _EXT_TO_FORMAT[ext]


def _normalize_element(element: str) -> str:
    """Normalize a biotite element symbol (``"CL"``) to RDKit casing (``"Cl"``)."""
    element = element.strip()
    if not element:
        return "*"
    return element[0].upper() + element[1:].lower()


def _ensure_atom_names(mol: Chem.Mol) -> None:
    """Ensure every atom carries a stable name in the ``_TriposAtomName`` prop.

    Names are taken (in priority order) from an existing ``_TriposAtomName``
    property (set by RDKit's mol2 reader), then a PDB residue-info name, and
    finally synthesized as ``<element><1-based-running-count>`` when absent.
    """
    # First pass: gather the names already present so synthesized fallback
    # names can avoid colliding with them (atom names must stay unique).
    existing: list[str] = []
    used: set[str] = set()
    for atom in mol.GetAtoms():
        name = ""
        if atom.HasProp(_NAME_PROP):
            name = atom.GetProp(_NAME_PROP).strip()
        if not name:
            info = atom.GetPDBResidueInfo()
            if info is not None:
                name = (info.GetName() or "").strip()
        existing.append(name)
        if name:
            used.add(name)

    counts: dict[str, int] = {}
    for atom, name in zip(mol.GetAtoms(), existing):
        if not name:
            symbol = atom.GetSymbol()
            while True:
                counts[symbol] = counts.get(symbol, 0) + 1
                candidate = f"{symbol}{counts[symbol]}"
                if candidate not in used:
                    name = candidate
                    used.add(name)
                    break
        atom.SetProp(_NAME_PROP, name)


def _mol_from_atom_array(atom_array) -> Chem.Mol:
    """Build a coordinate-bearing RDKit Mol from a biotite ``AtomArray``.

    Only connectivity is taken from the array (every bond is created as a
    SINGLE bond); bond orders are supplied later by the SMILES template. Atom
    names and 3D coordinates are preserved.
    """
    from rdkit.Geometry import Point3D

    n_atoms = len(atom_array)
    rw = Chem.RWMol()
    names = atom_array.atom_name
    elements = atom_array.element
    for i in range(n_atoms):
        atom = Chem.Atom(_normalize_element(str(elements[i])))
        atom.SetProp(_NAME_PROP, str(names[i]).strip())
        atom.SetNoImplicit(True)
        rw.AddAtom(atom)

    if atom_array.bonds is None or atom_array.bonds.as_array().shape[0] == 0:
        raise ValueError(
            "CIF AtomArray carries no bond table; cannot transfer bond orders. "
            "Ensure the CIF includes _chem_comp_bond connectivity."
        )
    for bond in atom_array.bonds.as_array():
        rw.AddBond(int(bond[0]), int(bond[1]), Chem.BondType.SINGLE)

    conf = Chem.Conformer(n_atoms)
    coords = atom_array.coord
    for i in range(n_atoms):
        conf.SetAtomPosition(
            i, Point3D(float(coords[i][0]), float(coords[i][1]), float(coords[i][2]))
        )
    rw.AddConformer(conf, assignId=True)
    return rw.GetMol()


def read_structure(structure_path: str | Path, *, fmt: Optional[str] = None) -> Chem.Mol:
    """Read a 3D ligand structure, preserving atom names and coordinates.

    Bond *orders* from the input are intentionally ignored downstream (only the
    connectivity graph is used), so files with missing or wrong bond orders are
    fine. CIF is read via tmol's biotite-based reader; mol2 / sdf / pdb are read
    natively by RDKit (no OpenBabel dependency on these paths).

    Args:
        structure_path: Path to a ``.cif`` / ``.mol2`` / ``.sdf`` / ``.mol`` /
            ``.pdb`` ligand file containing 3D coordinates.
        fmt: Override the format inferred from the file extension.

    Returns:
        An RDKit ``Mol`` with one 3D conformer and a ``_TriposAtomName`` property
        on every atom.
    """
    path = Path(structure_path)
    if not path.is_file():
        raise FileNotFoundError(f"Structure file not found: {path}")
    fmt = (fmt or _infer_format(path)).lower()

    if fmt in ("sdf", "mol"):
        mol = Chem.MolFromMolFile(str(path), sanitize=False, removeHs=False)
    elif fmt == "mol2":
        mol = Chem.MolFromMol2File(str(path), sanitize=False, removeHs=False)
    elif fmt == "pdb":
        mol = Chem.MolFromPDBFile(
            str(path), sanitize=False, removeHs=False, proximityBonding=True
        )
    elif fmt == "cif":
        from tmol.ligand.detect import nonstandard_residue_info_from_cif

        info = nonstandard_residue_info_from_cif(str(path))
        mol = _mol_from_atom_array(info.atom_array)
    else:
        raise ValueError(
            f"Unsupported input format {fmt!r}; expected one of "
            "cif, mol2, sdf, mol, pdb."
        )

    if mol is None:
        raise ValueError(
            f"Could not read a molecule from {path} (format={fmt}). "
            "The file may be malformed or unsupported by RDKit's parser."
        )
    if mol.GetNumConformers() == 0:
        raise ValueError(f"{path}: input has no 3D coordinates.")

    _ensure_atom_names(mol)
    return mol


def assign_bond_orders_from_smiles(
    structure_path: str | Path,
    smiles: str,
    *,
    fmt: Optional[str] = None,
) -> Chem.Mol:
    """Apply the bond orders of ``smiles`` onto the structure at ``structure_path``.

    Args:
        structure_path: Path to a 3D ligand structure (cif/mol2/sdf/mol/pdb).
        smiles: SMILES for the same molecule, supplying the trusted bond
            orders and formal charges.
        fmt: Override the input format inferred from the file extension.

    Returns:
        A sanitized RDKit ``Mol`` with the input coordinates and atom names but
        the SMILES bond orders / formal charges.

    Raises:
        ValueError: if the SMILES cannot be parsed, or if its heavy-atom graph
            is not a substructure of the input connectivity (i.e. the SMILES
            does not correspond to the structure).
    """
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        raise ValueError(f"Could not parse SMILES: {smiles!r}")

    mol = read_structure(structure_path, fmt=fmt)

    try:
        fixed = AllChem.AssignBondOrdersFromTemplate(template, mol)
    except ValueError as exc:
        raise ValueError(
            f"Could not transfer bond orders from SMILES {smiles!r} onto "
            f"{Path(structure_path).name}: {exc}. The SMILES heavy-atom graph "
            "must be a substructure of the input's connectivity (same heavy "
            "atoms and bonds). Verify the SMILES matches the structure, "
            "including protonation/charge state."
        ) from exc
    return fixed


def _atom_names(mol: Chem.Mol) -> list[str]:
    """Return per-atom names from the ``_TriposAtomName`` property (index order)."""
    return [
        atom.GetProp(_NAME_PROP) if atom.HasProp(_NAME_PROP) else ""
        for atom in mol.GetAtoms()
    ]


def write_structure(
    mol: Chem.Mol,
    output_path: str | Path,
    *,
    fmt: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """Write ``mol`` to ``output_path`` preserving bond orders, coords, and names.

    Only SDF / MOL output is supported, because they losslessly encode bond
    orders, 3D coordinates, and (via atom aliases + an ``atom_names`` data
    field) the atom names. PDB and mol2 output are rejected: PDB cannot encode
    bond orders, and a faithful mol2 writer would require OpenBabel.
    """
    path = Path(output_path)
    fmt = (fmt or _infer_format(path)).lower()
    if fmt not in ("sdf", "mol"):
        raise ValueError(
            f"Unsupported output format {fmt!r}. Use .sdf or .mol -- they "
            "preserve bond orders, coordinates, and atom names. (PDB cannot "
            "encode bond orders; mol2 writing needs OpenBabel.)"
        )

    out = Chem.Mol(mol)
    names = _atom_names(out)
    for atom, name in zip(out.GetAtoms(), names):
        if name:
            # Atom aliases ("A" lines) carry the names through MOL/SDF.
            atom.SetProp("molFileAlias", name)
    if title:
        out.SetProp("_Name", title)

    # Write to a sibling temp file and atomically replace, so a mid-write
    # failure never leaves a truncated output behind.
    tmp = path.with_name(path.name + ".tmp")
    try:
        if fmt == "mol":
            Chem.MolToMolFile(out, str(tmp))
        else:
            out.SetProp("atom_names", " ".join(names))
            writer = Chem.SDWriter(str(tmp))
            try:
                writer.write(out)
            finally:
                writer.close()
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _bond_order_summary(mol: Chem.Mol) -> str:
    """Return a compact ``type=count`` summary of a Mol's bond orders."""
    counts: dict[str, int] = {}
    for bond in mol.GetBonds():
        key = str(bond.GetBondType())
        counts[key] = counts.get(key, 0) + 1
    return ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m tmol.ligand.bond_order_assignment",
        description=(
            "Apply bond orders from a SMILES string onto a 3D ligand structure "
            "(cif/mol2/sdf/mol/pdb), keeping the input's atom names and "
            "coordinates. Writes an SDF/MOL with the corrected chemistry."
        ),
    )
    parser.add_argument("structure", help="Input structure file (cif/mol2/sdf/mol/pdb).")
    parser.add_argument("smiles", help="SMILES for the same molecule (bond orders).")
    parser.add_argument(
        "-o",
        "--output",
        help="Output path (.sdf or .mol). Default: <structure stem>.bondfix.sdf",
    )
    parser.add_argument(
        "--in-format",
        dest="in_format",
        help="Override input format (otherwise inferred from extension).",
    )
    parser.add_argument(
        "--out-format",
        dest="out_format",
        help="Override output format (otherwise inferred from extension).",
    )
    parser.add_argument(
        "--title",
        help="Molecule title written to the output (default: input stem).",
    )
    args = parser.parse_args(argv)

    structure_path = Path(args.structure)
    output_path = (
        Path(args.output)
        if args.output
        else structure_path.with_suffix("").with_suffix(".bondfix.sdf")
    )
    title = args.title or structure_path.stem

    try:
        mol = assign_bond_orders_from_smiles(
            structure_path, args.smiles, fmt=args.in_format
        )
        write_structure(mol, output_path, fmt=args.out_format, title=title)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Wrote {output_path} "
        f"({mol.GetNumAtoms()} atoms; bonds: {_bond_order_summary(mol)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
