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

from tmol.database.chemical import AtomAlias, ChemicalDatabase, RawResidueType
from tmol.io.canonical_ordering import CanonicalOrdering
from tmol.ligand.atom_typing import assign_tmol_atom_types
from tmol.ligand.detect import LigandInfo, detect_nonstandard_residues
from tmol.ligand.graph_match import match_heavy_atoms
from tmol.ligand.mol3d import get_partial_charges, smiles_to_obmol
from tmol.ligand.registry import (
    get_cached_ligand,
    rebuild_canonical_ordering,
    register_ligand,
)
from tmol.ligand.residue_builder import build_residue_type
from tmol.ligand.smiles import (
    _import_pybel,
    perceive_smiles,
    protonate_ligand_smiles,
)

logger = logging.getLogger(__name__)


def _build_cif_obmol(ligand_info: LigandInfo):
    """Build an OBMol from CIF atom names, elements, and coordinates."""
    openbabel, _ = _import_pybel()
    from tmol.ligand.smiles import _ELEMENT_TO_ATOMIC_NUM

    obmol = openbabel.OBMol()
    obmol.BeginModify()
    for elem, coord in zip(ligand_info.elements, ligand_info.coords):
        obatom = obmol.NewAtom()
        atomic_num = _ELEMENT_TO_ATOMIC_NUM.get(elem.strip(), 0)
        if atomic_num == 0:
            atomic_num = openbabel.GetAtomicNum(elem.strip())
        obatom.SetAtomicNum(atomic_num)
        obatom.SetVector(float(coord[0]), float(coord[1]), float(coord[2]))
    obmol.EndModify()
    obmol.ConnectTheDots()
    obmol.PerceiveBondOrders()
    return obmol


def _build_atom_aliases(
    pipeline_obmol,
    atom_types: list,
    ligand_info: LigandInfo,
) -> tuple[AtomAlias, ...]:
    """Build atom aliases mapping CIF atom names to pipeline atom names.

    Uses graph isomorphism to match heavy atoms between the pipeline
    OBMol and the CIF OBMol, then creates AtomAlias entries for each
    heavy atom whose CIF name differs from the pipeline name.
    """
    cif_obmol = _build_cif_obmol(ligand_info)

    try:
        idx_mapping = match_heavy_atoms(pipeline_obmol, cif_obmol)
    except ValueError:
        logger.warning(
            "Could not match CIF atoms for %s, skipping aliases",
            ligand_info.res_name,
        )
        return ()

    pipeline_idx_to_name = {a.index: a.atom_name for a in atom_types}

    # CIF OBMol atoms are in the same order as ligand_info.atom_names
    cif_idx_to_name = {}
    cif_i = 0
    from openbabel import openbabel

    for obatom in openbabel.OBMolAtomIter(cif_obmol):
        if cif_i < len(ligand_info.atom_names):
            cif_idx_to_name[obatom.GetIndex()] = ligand_info.atom_names[cif_i]
        cif_i += 1

    aliases = []
    for pipeline_idx, cif_idx in idx_mapping.items():
        pipeline_name = pipeline_idx_to_name.get(pipeline_idx)
        cif_name = cif_idx_to_name.get(cif_idx)
        if pipeline_name and cif_name and pipeline_name != cif_name:
            aliases.append(AtomAlias(name=pipeline_name, alt_name=cif_name))

    return tuple(aliases)


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

    aliases = _build_atom_aliases(mol.OBMol, atom_types, ligand_info)

    restype = build_residue_type(
        mol.OBMol,
        ligand_info.res_name,
        atom_types,
        atom_aliases=aliases,
    )
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

        restype, charges = prepare_single_ligand(lig, ph=ph)
        chem_db = register_ligand(chem_db, restype)
        modified = True

    if modified:
        canonical_ordering = rebuild_canonical_ordering(chem_db)

    return chem_db, canonical_ordering
