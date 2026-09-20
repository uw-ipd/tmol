"""Assemble one well-formed CIF from the separate files an input arrives in.

A structure prediction is often split across formats: the protein in a PDB, a
ligand in a Tripos MOL2. Neither format carries everything AtomWorks needs --
a PDB states connectivity but never bond order, and a MOL2 describes a molecule
no dictionary knows. Converting each part here, joining them, and writing one
CIF that carries its own component chemistry means AtomWorks is handed a
complete description rather than asked to guess at an incomplete one.
"""

from __future__ import annotations

from pathlib import Path

import biotite.structure as struc
import numpy as np
from atomworks.io.utils.atom_array_plus import (
    as_atom_array_plus,
    concatenate_atom_array_plus,
)
from atomworks.io.utils.io_utils import CIFWriteConfig, to_cif_string

# Chain IDs to draw from when a part collides with one already taken.
_CHAIN_ID_POOL = tuple("LMNOPQRSTUVWXYZABCDEFGHIJK")


def atom_array_from_mol2(
    mol2_path: str | Path,
    *,
    res_name: str | None = None,
    chain_id: str = "L",
    res_id: int = 1,
) -> struc.AtomArray:
    """Read a Tripos MOL2 ligand as an AtomArray with its chemistry intact.

    The returned array carries explicit bond orders, formal charges and
    aromatic flags from the file, which is what lets the ligand describe itself
    in a CIF rather than depend on a dictionary entry that does not exist.

    Args:
        mol2_path: Path to the MOL2 file.
        res_name: Component code to give the ligand. Taken from the file's
            substructure record when None.
        chain_id: Chain to place the ligand on.
        res_id: Residue number within that chain.

    Returns:
        A single-residue AtomArray with a populated bond list.

    Examples:
        >>> ligand = atom_array_from_mol2("ligand.mol2", res_name="LIG")
        >>> ligand.bonds.get_bond_count() > 0
        True
    """
    # Imported here because tmol.ligand imports tmol.io, as elsewhere in this package.
    from tmol.ligand._detect import nonstandard_residue_info_from_mol2

    info = nonstandard_residue_info_from_mol2(mol2_path, res_name=res_name)
    atom_array = as_atom_array_plus(info.atom_array.copy())
    atom_array.chain_id[:] = chain_id
    atom_array.res_id[:] = res_id
    atom_array.hetero[:] = True
    # A ligand code is a placeholder, and placeholders collide: "LG1" is also a real
    # dictionary entry. Registering this molecule as its own component template means the
    # file describes what was read, not whatever else happens to share the name.
    atom_array._custom_ccd_registry[str(atom_array.res_name[0]).upper()] = (
        _component_template(atom_array)
    )
    return atom_array


def _component_template(residue: struc.AtomArray) -> struc.AtomArray:
    """The residue as its own component template, in the annotations a CIF writer reads."""
    template = residue.copy()
    if "is_aromatic" not in template.get_annotation_categories():
        aromatic = (
            template.get_annotation("tmol_aromatic")
            if "tmol_aromatic" in template.get_annotation_categories()
            else np.zeros(template.array_length(), dtype=bool)
        )
        template.set_annotation("is_aromatic", np.asarray(aromatic, dtype=bool))
    if "is_leaving_atom" not in template.get_annotation_categories():
        template.set_annotation(
            "is_leaving_atom", np.zeros(template.array_length(), dtype=bool)
        )
    return template


def _harmonised(parts: list[struc.AtomArray]) -> list[struc.AtomArray]:
    """Give every part the union of the annotations, so none is dropped on join.

    Concatenation refuses categories that are missing from some input. Filling
    them keeps chemistry -- ``charge`` above all -- that dropping would discard.
    """
    union: dict[str, np.ndarray] = {}
    for part in parts:
        for category in part.get_annotation_categories():
            union.setdefault(category, part.get_annotation(category))

    filled = []
    for part in parts:
        part = part.copy()
        for category, sample in union.items():
            if category in part.get_annotation_categories():
                continue
            part.set_annotation(
                category, np.zeros(part.array_length(), dtype=sample.dtype)
            )
        filled.append(part)
    return filled


def _with_unique_chain_ids(parts: list[struc.AtomArray]) -> list[struc.AtomArray]:
    """Move a part onto free chain IDs when it collides with an earlier one."""
    taken: set[str] = set()
    result = []
    for part in parts:
        part = part.copy()
        collisions = {str(c) for c in np.unique(part.chain_id)} & taken
        for collision in sorted(collisions):
            replacement = next(
                (
                    c
                    for c in _CHAIN_ID_POOL
                    if c not in taken and c not in part.chain_id
                ),
                None,
            )
            if replacement is None:
                raise ValueError("Ran out of chain IDs while assembling the input")
            part.chain_id[part.chain_id == collision] = replacement
            taken.add(replacement)
        taken.update(str(c) for c in np.unique(part.chain_id))
        result.append(part)
    return result


def assemble_input(*parts: struc.AtomArray) -> struc.AtomArray:
    """Join separately-read parts into one structure.

    Chain IDs are kept as read and only moved where two parts claim the same
    one. Bonds, annotations and any component templates the parts carry all
    survive the join.

    Args:
        *parts: Structures to join, in the order they should appear.

    Returns:
        The joined structure.

    Raises:
        ValueError: If no parts are given, or chain IDs cannot be made unique.

    Examples:
        >>> complex = assemble_input(protein, ligand)
        >>> sorted(set(complex.chain_id))
        ['A', 'L']
    """
    if not parts:
        raise ValueError("assemble_input requires at least one structure")
    prepared = _harmonised(_with_unique_chain_ids(list(parts)))
    return concatenate_atom_array_plus(prepared)


def cif_from_atom_array(
    atom_array: struc.AtomArray,
    *,
    path: str | Path | None = None,
    entry_id: str = "assembled",
) -> str | Path:
    """Write a structure as a CIF that carries its own component chemistry.

    Components the dictionary knows are written from it; anything it does not
    know -- a ligand read from a MOL2, say -- has its ``chem_comp_atom`` and
    ``chem_comp_bond`` rows written from the structure itself. The result parses
    without needing a dictionary entry for the ligand.

    Args:
        atom_array: Structure to write.
        path: Where to write. Returns the CIF text instead when None.
        entry_id: Data block name.

    Returns:
        The written path, or the CIF text when ``path`` is None.

    Examples:
        >>> text = cif_from_atom_array(assemble_input(protein, ligand))
        >>> "_chem_comp_bond" in text
        True
    """
    config = CIFWriteConfig(
        id=entry_id,
        include_entity_categories=True,
        # "ccd" writes a known component from the dictionary and falls back to
        # the structure for one the dictionary lacks, which is exactly the split
        # between a standard residue and a ligand read from a file.
        chem_comp_source="ccd",
        warn_on_ccd_without_registry=False,
    )
    text = to_cif_string(atom_array, config=config)
    if path is None:
        return text
    path = Path(path)
    path.write_text(text)
    return path
