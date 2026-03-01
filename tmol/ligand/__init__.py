"""Ligand preparation pipeline for tmol.

Detects non-standard residues in biotite AtomArrays, builds SMILES
representations, protonates them at a target pH, generates 3D structures
with MMFF94 partial charges, assigns Rosetta-compatible atom types, and
registers the resulting residue types in tmol's ChemicalDatabase for
transparent loading via PoseStack.

Typical usage::

    from tmol.ligand import prepare_ligands

    chem_db, co = prepare_ligands(atom_array)
    # Now use chem_db and co to build a PoseStack that includes ligands.
"""

import logging
from typing import Optional

import biotite.structure as struc

from tmol.database.chemical import ChemicalDatabase, RawResidueType
from tmol.io.canonical_ordering import CanonicalOrdering
from tmol.ligand.atom_typing import assign_tmol_atom_types
from tmol.ligand.detect import LigandInfo, detect_nonstandard_residues
from tmol.ligand.mol3d import get_partial_charges, smiles_to_obmol
from tmol.ligand.registry import (
    get_cached_ligand,
    rebuild_canonical_ordering,
    register_ligand,
)
from tmol.ligand.residue_builder import build_residue_type
from tmol.ligand.smiles import perceive_smiles, protonate_ligand_smiles

logger = logging.getLogger(__name__)


def prepare_single_ligand(
    ligand_info: LigandInfo,
    ph: float = 7.4,
) -> tuple[RawResidueType, dict[str, float]]:
    """Run the full preparation pipeline for a single ligand.

    Args:
        ligand_info: Detected ligand information.
        ph: Target pH for protonation.

    Returns:
        A (RawResidueType, partial_charges) tuple.
    """
    smiles = perceive_smiles(ligand_info)
    protonated = protonate_ligand_smiles(smiles, ph=ph)
    mol = smiles_to_obmol(protonated)
    atom_types = assign_tmol_atom_types(mol.OBMol)
    charges = get_partial_charges(mol)
    restype = build_residue_type(mol.OBMol, ligand_info.res_name, atom_types)
    return restype, charges


def prepare_ligands(
    atom_array: struc.AtomArray,
    chem_db: Optional[ChemicalDatabase] = None,
    canonical_ordering: Optional[CanonicalOrdering] = None,
    ph: float = 7.4,
) -> tuple[ChemicalDatabase, CanonicalOrdering]:
    """Detect, prepare, and register all non-standard residues.

    Scans the input AtomArray for residues not in the ChemicalDatabase,
    runs each through the ligand preparation pipeline (SMILES perception,
    protonation, 3D generation, atom typing, residue building), and
    registers them in the database.

    Args:
        atom_array: A biotite AtomArray from a CIF or PDB file.
        chem_db: The ChemicalDatabase to extend. If None, the default
            database is loaded.
        canonical_ordering: The current CanonicalOrdering. If None, a
            default ordering is built from chem_db.
        ph: Target pH for ligand protonation.

    Returns:
        A (ChemicalDatabase, CanonicalOrdering) tuple with all detected
        ligands registered. If no ligands are found, the inputs are
        returned unchanged.
    """
    if chem_db is None:
        chem_db = ChemicalDatabase.get_default()

    if canonical_ordering is None:
        canonical_ordering = rebuild_canonical_ordering(chem_db)

    ligands = detect_nonstandard_residues(atom_array, canonical_ordering)

    if not ligands:
        logger.info("No non-standard residues detected")
        return chem_db, canonical_ordering

    logger.info("Found %d non-standard residue type(s) to prepare", len(ligands))

    modified = False
    for lig in ligands:
        cached = get_cached_ligand(lig.res_name)
        if cached is not None:
            logger.info("Using cached preparation for %s", lig.res_name)
            chem_db = register_ligand(chem_db, cached)
            modified = True
            continue

        logger.info(
            "Preparing %s (CCD type: %s, is_ligand: %s)",
            lig.res_name,
            lig.ccd_type,
            lig.is_ligand,
        )

        try:
            restype, charges = prepare_single_ligand(lig, ph=ph)
            chem_db = register_ligand(chem_db, restype)
            modified = True
        except Exception:
            logger.error(
                "Failed to prepare ligand %s, skipping",
                lig.res_name,
                exc_info=True,
            )

    if modified:
        canonical_ordering = rebuild_canonical_ordering(chem_db)

    return chem_db, canonical_ordering
