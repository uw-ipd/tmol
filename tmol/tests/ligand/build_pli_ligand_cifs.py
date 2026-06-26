"""Generate protein_ligand_test ligand CIF fixtures from reference .tmol data.

This script builds ligand-only CIFs that carry explicit bond orders and
reference partial charges. Coordinates are taken from ``*_complex*.pdb`` when
atom names match the reference (Rosetta/LG1 naming); otherwise from the paired
``.lig.mol2`` (mol2gen uses the same names as the mol2 file, with duplicate
names disambiguated as ``C2'2``, etc.).
"""

from __future__ import annotations

from pathlib import Path

import biotite.structure as struc
import biotite.structure.io as struc_io

from tmol.ligand.mol2_names import disambiguate_mol2_atom_name
from tmol.ligand.params_file import load_params_file

PLI_DIR = Path(__file__).parent.parent / "data" / "protein_ligand_test"
CIF_OUT_DIR = PLI_DIR / "cif_inputs"

_BOND_ORDER_TO_CIF = {
    "SINGLE": "SING",
    "DOUBLE": "DOUB",
    "TRIPLE": "TRIP",
    "AROMATIC": "AROM",
}


def _targets() -> list[str]:
    """Return the sorted target stems with a reference ``.tmol`` in ``PLI_DIR``."""
    suffix = ".xtal-lig.mmff94.tmol"
    return sorted(p.name[: -len(suffix)] for p in PLI_DIR.glob(f"*{suffix}"))


def _find_complex_path(target: str) -> Path:
    """Return the single ``*_complex*.pdb`` for ``target``.

    Raises:
        ValueError: If zero or more than one matching complex PDB exists.
    """
    matches = sorted(PLI_DIR.glob(f"{target}_complex*.pdb"))
    if len(matches) != 1:
        raise ValueError(
            f"{target}: expected exactly one complex PDB, found {len(matches)}"
        )
    return matches[0]


def _load_ligand_coords_by_name(
    complex_pdb: Path, res_name: str
) -> dict[str, tuple[str, tuple[float, float, float]]]:
    """Map ligand atom name to ``(element, xyz)`` from a complex PDB.

    Args:
        complex_pdb: Path to the complex PDB containing the ligand.
        res_name: Residue name selecting the hetero ligand atoms.

    Returns:
        Mapping of atom name to its element symbol and Cartesian coordinate.

    Raises:
        ValueError: If no matching hetero atoms exist or a name is duplicated.
    """
    bt_struct = struc_io.load_structure(str(complex_pdb), include_bonds=False)
    if isinstance(bt_struct, struc.AtomArrayStack):
        bt_struct = bt_struct[0]

    mask = bt_struct.res_name == res_name
    if hasattr(bt_struct, "hetero"):
        mask &= bt_struct.hetero
    ligand = bt_struct[mask]
    if len(ligand) == 0:
        raise ValueError(
            f"{complex_pdb.name}: no hetero atoms found for residue {res_name}"
        )

    coords_by_name: dict[str, tuple[str, tuple[float, float, float]]] = {}
    for atom_name, element, coord in zip(
        ligand.atom_name, ligand.element, ligand.coord
    ):
        name = str(atom_name).strip()
        if name in coords_by_name:
            raise ValueError(f"{complex_pdb.name}: duplicate ligand atom name {name}")
        coords_by_name[name] = (
            str(element).strip(),
            (float(coord[0]), float(coord[1]), float(coord[2])),
        )
    return coords_by_name


def _element_from_tripos_type(tripos_type: str) -> str:
    """Extract the element symbol from a TRIPOS SYBYL atom type."""
    tripos_type = tripos_type.strip()
    if not tripos_type:
        return "X"
    if len(tripos_type) >= 2 and tripos_type[1].islower():
        return tripos_type[:2]
    return tripos_type[0]


def _load_ligand_coords_from_mol2(
    mol2_path: Path,
) -> dict[str, tuple[str, tuple[float, float, float]]]:
    """Map disambiguated mol2 atom name to ``(element, xyz)``.

    Raises:
        ValueError: If a disambiguated name is duplicated or no atoms are found.
    """
    coords_by_name: dict[str, tuple[str, tuple[float, float, float]]] = {}
    seen: dict[str, int] = {}
    in_atoms = False
    for line in mol2_path.read_text().splitlines():
        if line.startswith("@<TRIPOS>ATOM"):
            in_atoms = True
            continue
        if line.startswith("@<TRIPOS>"):
            in_atoms = False
            continue
        if not in_atoms or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        raw_name = parts[1]
        seen[raw_name] = seen.get(raw_name, 0) + 1
        name = disambiguate_mol2_atom_name(raw_name, seen[raw_name])
        if name in coords_by_name:
            raise ValueError(f"{mol2_path.name}: duplicate mol2 atom name {name}")
        element = _element_from_tripos_type(parts[5])
        coord = (float(parts[2]), float(parts[3]), float(parts[4]))
        coords_by_name[name] = (element, coord)
    if not coords_by_name:
        raise ValueError(f"{mol2_path.name}: no atoms found in @<TRIPOS>ATOM block")
    return coords_by_name


def _load_ligand_coords_for_target(
    target: str,
    restype,
    complex_path: Path,
) -> dict[str, tuple[str, tuple[float, float, float]]]:
    """Load coordinates for every atom in ``restype``."""
    atom_names = {str(atom.name) for atom in restype.atoms}
    res_name = str(restype.name)

    pdb_coords = _load_ligand_coords_by_name(complex_path, res_name)
    if atom_names <= set(pdb_coords):
        return {name: pdb_coords[name] for name in atom_names}

    mol2_path = PLI_DIR / f"{target}.lig.mol2"
    if not mol2_path.is_file():
        missing = sorted(atom_names - set(pdb_coords))
        raise ValueError(
            f"{target}: atoms {missing} missing from {complex_path.name} "
            f"and no {mol2_path.name} fallback"
        )

    mol2_coords = _load_ligand_coords_from_mol2(mol2_path)
    missing = sorted(atom_names - set(mol2_coords))
    if missing:
        raise ValueError(
            f"{target}: atoms {missing} missing from {mol2_path.name} "
            f"(PDB names also differ: examples pdb-only={sorted(set(pdb_coords) - atom_names)[:5]})"
        )
    return {name: mol2_coords[name] for name in atom_names}


def _aromatic_atom_set(restype) -> set[str]:
    """Return the atom names participating in aromatic ring chemistry.

    Combines atoms on ring AROMATIC bonds with atoms whose reference atom
    type encodes aromatic-like character (e.g. ``Nim``/``Ofu``).
    """
    aromatic_atoms: set[str] = set()
    for a, b, bond_type, *rest in restype.bonds:
        ring = bool(rest[0]) if rest else False
        if str(bond_type) == "AROMATIC" and ring:
            aromatic_atoms.add(str(a))
            aromatic_atoms.add(str(b))
    # Preserve aromatic-atom identity encoded in the reference atom types.
    # This keeps hetero-aromatic assignments (e.g. Nim/Ofu) aligned with the
    # existing .tmol references even after RDKit kekulization steps.
    aromatic_like_types = {
        "CR",
        "CRp",
        "Nim",
        "Nin",
        "Ofu",
        "SR",
    }
    for atom in restype.atoms:
        if str(atom.atom_type) in aromatic_like_types:
            aromatic_atoms.add(str(atom.name))
    return aromatic_atoms


def _source_subtype_from_atom_type(element: str, atom_type: str) -> str:
    """Best-effort source subtype hint for atom typing compatibility.

    The PLI reference `.tmol` files do not carry original mol2 subtype tags.
    We map a subset from the reference atom type to keep the CIF-driven
    classifier aligned with the reference chemistry.
    """
    e = element.upper()
    t = atom_type
    if e == "C":
        if t.startswith("CR"):
            return "ar"
        if t == "CSp":
            return "cat"
        if t.startswith("CD"):
            return "2"
        if t.startswith("CS"):
            return "3"
        if t.startswith("CT"):
            return "1"
    if e == "O":
        if t in {"Oad", "Oal", "Oat"}:
            return "2"
        if t in {"Oet", "Ofu", "Ohx", "OG3", "OG31"}:
            return "3"
    if e == "N":
        if t in {"Ngu1", "Ngu2"}:
            return "pl3"
        if t in {"Nad", "Nad3"}:
            return "am"
        if t in {"Nin", "Nim"}:
            return "ar"
    return "?"


def _render_ligand_cif(
    prep, coords_by_name: dict[str, tuple[str, tuple[float, float, float]]]
) -> str:
    """Render a ligand-only CIF with explicit bond orders and partial charges.

    Args:
        prep: The ligand preparation supplying topology and charges.
        coords_by_name: Atom-name -> ``(element, xyz)`` coordinate donor.

    Returns:
        The CIF document as a single string.

    Raises:
        ValueError: On an unsupported bond type or an atom missing coordinates
            or a reference charge.
    """
    restype = prep.residue_type
    aromatic_atoms = _aromatic_atom_set(restype)

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
    for idx, (a, b, bond_type, *rest) in enumerate(restype.bonds, start=1):
        btype = str(bond_type)
        ring = bool(rest[0]) if rest else False
        if btype == "AROMATIC" and not ring:
            # Non-ring aromatic bonds in .tmol represent delocalized/resonance
            # chemistry (e.g. carboxylate/amide-like patterns), not true ring
            # aromaticity. Writing these as AROM+N yields an unsupported bond
            # code in biotite's CIF parser; keep them explicit and non-aromatic.
            order = "SING"
            aromatic_flag = "N"
        else:
            order = _BOND_ORDER_TO_CIF.get(btype)
            if order is None:
                raise ValueError(f"{restype.name}: unsupported bond type {btype}")
            aromatic_flag = "Y" if (btype == "AROMATIC" and ring) else "N"
        lines.append(
            f"{idx:<2} {restype.name:<3} {str(a):<4} {str(b):<4} {order:<4} {aromatic_flag} ?"
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

    for idx, atom in enumerate(restype.atoms, start=1):
        name = str(atom.name)
        if name not in coords_by_name:
            raise ValueError(
                f"{restype.name}: atom {name} missing from coordinate donor structure"
            )
        if name not in prep.partial_charges:
            raise ValueError(f"{restype.name}: atom {name} missing reference charge")

        element, coord = coords_by_name[name]
        partial_charge = prep.partial_charges[name]
        aromatic_flag = "Y" if name in aromatic_atoms else "N"
        source_subtype = _source_subtype_from_atom_type(element, atom.atom_type)
        lines.append(
            "HETATM "
            f"{element:<2} "
            f"{name:<4} "
            ". "
            f"{restype.name:<3} "
            "A 1 1 . "
            f"1 {restype.name:<3} A {name:<4} "
            "nan 0.0 ? "
            f"{coord[0]:.4f} {coord[1]:.4f} {coord[2]:.4f} "
            f"1 {idx} {partial_charge:+.6f} {aromatic_flag} {source_subtype}"
        )
    lines.append("#")
    lines.append("")
    return "\n".join(lines)


def ensure_pli_ligand_cifs(output_dir: Path = CIF_OUT_DIR) -> list[Path]:
    """Build a ligand CIF for every PLI target and return the written paths.

    Raises:
        ValueError: If a reference ``.tmol`` does not contain exactly one residue.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for target in _targets():
        tmol_path = PLI_DIR / f"{target}.xtal-lig.mmff94.tmol"
        preps = load_params_file(tmol_path)
        if len(preps) != 1:
            raise ValueError(
                f"{tmol_path.name}: expected one residue, got {len(preps)}"
            )
        prep = preps[0]
        restype = prep.residue_type
        complex_path = _find_complex_path(target)
        coords_by_name = _load_ligand_coords_for_target(target, restype, complex_path)
        cif_text = _render_ligand_cif(prep, coords_by_name)

        out_path = output_dir / f"{target}.ligand.cif"
        out_path.write_text(cif_text)
        written.append(out_path)

    return written


def main() -> None:
    """Build the ligand CIF fixtures and print a summary of what was written."""
    written = ensure_pli_ligand_cifs()
    print(f"Wrote {len(written)} ligand CIF fixtures to {CIF_OUT_DIR}")


if __name__ == "__main__":
    main()
