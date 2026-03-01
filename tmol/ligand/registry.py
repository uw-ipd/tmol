"""Registration of dynamically created ligand residue types.

Extends tmol's ChemicalDatabase and CanonicalOrdering with new residue
types built by the ligand preparation pipeline. Follows the pattern
established by the jflat06/atp_ligand_load branch.
"""

import logging
from typing import Optional

from tmol.database.chemical import (
    AtomType,
    ChemicalDatabase,
    RawResidueType,
)
from tmol.io.canonical_ordering import CanonicalOrdering

logger = logging.getLogger(__name__)

_ligand_cache: dict[str, RawResidueType] = {}


def get_cached_ligand(res_name: str) -> Optional[RawResidueType]:
    """Retrieve a previously prepared ligand from the cache.

    Args:
        res_name: Three-letter residue code.

    Returns:
        The cached RawResidueType, or None if not cached.
    """
    return _ligand_cache.get(res_name)


def cache_ligand(res_name: str, restype: RawResidueType) -> None:
    """Store a prepared ligand in the cache.

    Args:
        res_name: Three-letter residue code.
        restype: The prepared RawResidueType.
    """
    _ligand_cache[res_name] = restype


def clear_cache() -> None:
    """Clear the ligand cache."""
    _ligand_cache.clear()


def _collect_new_atom_types(
    chem_db: ChemicalDatabase,
    residue_type: RawResidueType,
) -> list[AtomType]:
    """Identify atom types used by the residue that aren't in the database.

    Args:
        chem_db: The current ChemicalDatabase.
        residue_type: The new residue type to register.

    Returns:
        A list of AtomType objects that need to be added.
    """
    existing = {at.name for at in chem_db.atom_types}
    needed: dict[str, str] = {}

    for atom in residue_type.atoms:
        if atom.atom_type not in existing and atom.atom_type not in needed:
            needed[atom.atom_type] = atom.atom_type

    return [AtomType(name=name, element="C") for name in needed]


def register_ligand(
    chem_db: ChemicalDatabase,
    residue_type: RawResidueType,
) -> ChemicalDatabase:
    """Register a new ligand residue type in the ChemicalDatabase.

    Creates a new ChemicalDatabase instance with the ligand added to the
    residue list. Also adds any atom types that are missing.

    Args:
        chem_db: The current (immutable) ChemicalDatabase.
        residue_type: The ligand RawResidueType to register.

    Returns:
        A new ChemicalDatabase with the ligand included.
    """
    existing_names = {r.name for r in chem_db.residues}
    if residue_type.name in existing_names:
        logger.info("Residue %s already registered, skipping", residue_type.name)
        return chem_db

    new_atom_types = _collect_new_atom_types(chem_db, residue_type)
    if new_atom_types:
        logger.info(
            "Adding %d new atom types: %s",
            len(new_atom_types),
            [at.name for at in new_atom_types],
        )

    cache_ligand(residue_type.name, residue_type)

    logger.info(
        "Registering ligand %s (%d atoms, %d bonds)",
        residue_type.name,
        len(residue_type.atoms),
        len(residue_type.bonds),
    )

    return ChemicalDatabase(
        element_types=chem_db.element_types,
        atom_types=chem_db.atom_types + tuple(new_atom_types),
        residues=chem_db.residues + (residue_type,),
        variants=chem_db.variants,
    )


def rebuild_canonical_ordering(
    chem_db: ChemicalDatabase,
) -> CanonicalOrdering:
    """Build a new CanonicalOrdering from a (possibly extended) ChemicalDatabase.

    This creates a fresh CanonicalOrdering that includes any newly
    registered ligand residue types.

    Args:
        chem_db: A ChemicalDatabase (with ligands registered).

    Returns:
        A new CanonicalOrdering covering all residue types.
    """
    from tmol.chemical.patched_chemdb import PatchedChemicalDatabase

    patched = PatchedChemicalDatabase.from_chem_db(chem_db)
    return CanonicalOrdering.from_chemdb(patched)
