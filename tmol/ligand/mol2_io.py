"""Non-destructive mol2 -> tmol params entry point.

Mirrors Rosetta's ``mol2genparams.py`` behavior: read a TRIPOS mol2 file
verbatim and produce a tmol RawResidueType (and matching per-atom partial
charges) without regenerating coordinates, hydrogens, protonation state,
atom names, or charges.

Use this when the caller has already prepared the ligand (docked pose,
externally typed charges, deliberate protonation) and wants the tmol
atom-type classifier applied without any rebuild. For ligands discovered
inside a CIF/PDB where rebuild is appropriate, use ``prepare_ligands``.
"""

import logging
from pathlib import Path
from typing import Optional

from openbabel import openbabel, pybel

from tmol.database.chemical import RawResidueType
from tmol.ligand.atom_typing import assign_tmol_atom_types
from tmol.ligand.mol3d import get_partial_charges_by_index
from tmol.ligand.params_io import write_params_file
from tmol.ligand.residue_builder import build_residue_type

logger = logging.getLogger(__name__)


def read_mol2(path: str | Path) -> pybel.Molecule:
    """Read a TRIPOS mol2 file and return the first molecule.

    Args:
        path: Path to a ``.mol2`` file.

    Returns:
        A ``pybel.Molecule`` with atom names, coordinates, partial charges,
        and bond orders populated from the mol2.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file contains no readable molecule.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"mol2 file not found: {p}")

    reader = pybel.readfile("mol2", str(p))
    try:
        first = next(reader)
    except StopIteration:
        raise ValueError(f"no molecules in mol2 file: {p}")

    try:
        next(reader)
        logger.warning(
            "mol2 %s contains multiple molecules; using only the first",
            p,
        )
    except StopIteration:
        pass

    return first


def parse_mol2_atom_names(path: str | Path) -> list[str]:
    """Read the TRIPOS atom-name column directly from a mol2 file.

    OpenBabel's mol2 reader does not preserve the atom-name column in any
    accessible OBAtom or OBResidue field, so we parse it ourselves. The
    returned list is in mol2 row order, which matches ``OBMol``'s
    ``OBAtom.GetIndex()`` (0-based) when the same file is loaded with
    :func:`read_mol2`.

    Args:
        path: Path to a TRIPOS mol2 file.

    Returns:
        Atom names in file order (one entry per atom in the first
        ``@<TRIPOS>MOLECULE`` block).
    """
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
                        break  # second molecule reached; stop
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


def get_original_atom_names(
    obmol: openbabel.OBMol,
    mol2_path: str | Path | None = None,
) -> dict[int, str]:
    """Map OBMol atom indices to their original mol2 atom names.

    If ``mol2_path`` is provided, names are read directly from the TRIPOS
    atom-name column (the reliable source — OpenBabel discards this column
    on read). Without a path, falls back to ``<element><1-based-index>``,
    which only happens to be correct for mol2 files whose atom names
    already follow that convention.

    Args:
        obmol: An OBMol read from a mol2 file.
        mol2_path: Optional path to the source mol2; required for true
            name preservation.

    Returns:
        Mapping ``{ob_index_0_based: atom_name}``.
    """
    if mol2_path is not None:
        parsed = parse_mol2_atom_names(mol2_path)
        if len(parsed) != obmol.NumAtoms():
            logger.warning(
                "mol2 %s has %d atom-name rows but OBMol has %d atoms; "
                "falling back to <elem><idx>",
                mol2_path,
                len(parsed),
                obmol.NumAtoms(),
            )
        else:
            return {i: name for i, name in enumerate(parsed)}

    names: dict[int, str] = {}
    for obatom in openbabel.OBMolAtomIter(obmol):
        elem = openbabel.GetSymbol(obatom.GetAtomicNum()) or "X"
        names[obatom.GetIndex()] = f"{elem}{obatom.GetIdx()}"
    return names


def _default_res_name(mol: pybel.Molecule, fallback: str = "LG1") -> str:
    """Derive a 3-letter residue name from the mol2 title, falling back."""
    title = (mol.title or "").strip()
    if title:
        return title.split()[0][:3].upper() or fallback
    return fallback


def prepare_ligand_from_mol2(
    mol2_path: str | Path,
    res_name: Optional[str] = None,
    rename_atoms: bool = False,
) -> tuple[RawResidueType, dict[str, float]]:
    """Build a RawResidueType from a mol2 file without rebuilding the ligand.

    Mirrors Rosetta ``mol2genparams.py``: reads the mol2 verbatim, runs the
    tmol Rosetta-port atom-type classifier on the OBMol, and constructs a
    ``RawResidueType`` whose atom names, 3D coordinates, partial charges,
    and bond orders all come straight from the mol2. No ``make3D``,
    ``localopt``, ``addh``, dimorphite protonation, or MMFF94 recomputation
    runs on this path.

    Args:
        mol2_path: Path to a TRIPOS mol2 file.
        res_name: Three-letter residue name to assign. If ``None``, derived
            from the mol2 title (first token, uppercased, truncated to 3
            chars); falls back to ``"LG1"``.
        rename_atoms: If ``False`` (default), atom names from the mol2 are
            preserved on the resulting residue type. If ``True``, atom names
            are regenerated using the Rosetta convention
            (``C1``, ``C2``, ..., ``HC1``, ...) as produced by
            :func:`assign_tmol_atom_types`. Mirrors Rosetta's
            ``--rename_atoms`` flag.

    Returns:
        A tuple ``(restype, charges)`` where ``restype`` is the populated
        :class:`RawResidueType` and ``charges`` is a ``{atom_name: charge}``
        mapping taken directly from the mol2 partial-charge column.
    """
    mol = read_mol2(mol2_path)

    original_names = get_original_atom_names(mol.OBMol, mol2_path=mol2_path)
    original_charges_by_idx = get_partial_charges_by_index(mol)

    assignments = assign_tmol_atom_types(mol.OBMol)

    if not rename_atoms:
        assignments = [
            a._replace(atom_name=original_names[a.index])
            if a.index in original_names
            else a
            for a in assignments
        ]

    final_res_name = res_name if res_name is not None else _default_res_name(mol)
    restype = build_residue_type(mol.OBMol, final_res_name, assignments)

    name_by_idx = {a.index: a.atom_name for a in assignments}
    charges = {
        name_by_idx[i]: q
        for i, q in original_charges_by_idx.items()
        if i in name_by_idx
    }

    return restype, charges


def atom_array_from_mol2(
    mol2_path: str | Path,
    res_name: str = "LG1",
    chain_id: str = "X",
    res_id: int = 1,
    rename_atoms: bool = False,
):
    """Build a Biotite ``AtomArray`` directly from a mol2 file.

    Preserves atom names, elements, 3D coordinates, and bond orders from
    the mol2 (no PDB intermediary). The returned array is ready to
    concatenate with a protein ``AtomArray`` and pass to
    ``pose_stack_from_biotite(..., prepare_ligands=False, param_db=...)``
    after the ligand has been registered via
    :func:`prepare_ligand_from_mol2` + ``register_ligand``.

    Args:
        mol2_path: Path to a TRIPOS mol2 file.
        res_name: Three-letter residue name written to every atom row.
            Must match the ``res_name`` used in ``prepare_ligand_from_mol2``.
        chain_id: Chain identifier written to every atom row.
        res_id: Residue id written to every atom row. Choose something
            outside the protein's residue range.
        rename_atoms: If ``True``, atom names are regenerated using the
            Rosetta convention (``C1``/``HC1``/...). Must match the
            ``rename_atoms`` value used in ``prepare_ligand_from_mol2`` so
            atom names line up between the AtomArray and the registered
            RawResidueType.

    Returns:
        A ``biotite.structure.AtomArray`` with a populated ``BondList``.
    """
    import biotite.structure as struc
    import numpy as np

    mol = read_mol2(mol2_path)
    obmol = mol.OBMol
    n = obmol.NumAtoms()

    if rename_atoms:
        from tmol.ligand.atom_typing import assign_tmol_atom_types

        assignments = assign_tmol_atom_types(obmol)
        idx_to_name = {a.index: a.atom_name for a in assignments}
        names = [idx_to_name[i] for i in range(n)]
    else:
        original = get_original_atom_names(obmol, mol2_path=mol2_path)
        names = [original[i] for i in range(n)]

    elements = []
    coords = np.zeros((n, 3), dtype=np.float32)
    for obatom in openbabel.OBMolAtomIter(obmol):
        i = obatom.GetIndex()
        elements.append(openbabel.GetSymbol(obatom.GetAtomicNum()) or "X")
        coords[i] = (obatom.GetX(), obatom.GetY(), obatom.GetZ())

    arr = struc.AtomArray(n)
    arr.atom_name = np.array(names, dtype="U6")
    arr.element = np.array(elements, dtype="U2")
    arr.res_name = np.array([res_name] * n, dtype="U3")
    arr.res_id = np.array([res_id] * n, dtype=np.int32)
    arr.chain_id = np.array([chain_id] * n, dtype="U4")
    arr.hetero = np.array([True] * n, dtype=bool)
    arr.coord = coords

    bond_rows = []
    for obbond in openbabel.OBMolBondIter(obmol):
        a = obbond.GetBeginAtomIdx() - 1
        b = obbond.GetEndAtomIdx() - 1
        if obbond.IsAromatic():
            t = struc.BondType.AROMATIC
        else:
            order = obbond.GetBondOrder()
            t = {
                1: struc.BondType.SINGLE,
                2: struc.BondType.DOUBLE,
                3: struc.BondType.TRIPLE,
            }.get(order, struc.BondType.SINGLE)
        bond_rows.append((a, b, t))

    if bond_rows:
        bond_array = np.array(bond_rows, dtype=np.int32)
        arr.bonds = struc.BondList(n, bond_array)

    return arr


def write_params_from_mol2(
    mol2_path: str | Path,
    output_path: str | Path,
    res_name: Optional[str] = None,
    rename_atoms: bool = False,
) -> None:
    """Convert a mol2 file to a Rosetta-format params file.

    Convenience wrapper over :func:`prepare_ligand_from_mol2` that writes
    the resulting residue type to disk via :func:`write_params_file`.

    See :func:`prepare_ligand_from_mol2` for argument semantics; this
    function adds only the ``output_path`` parameter.
    """
    restype, charges = prepare_ligand_from_mol2(
        mol2_path,
        res_name=res_name,
        rename_atoms=rename_atoms,
    )
    write_params_file(restype, output_path, partial_charges=charges)
