"""Ligand preparation pipeline for tmol.

Detects non-standard residues in biotite AtomArrays, builds RDKit molecules
directly from atom arrays, protonates them at a target pH, generates 3D
structures with MMFF94 partial charges, assigns Rosetta-compatible atom
types, and registers the resulting residue types in tmol's ParameterDatabase
for transparent loading via PoseStack.

Typical usage::

    from tmol.ligand import prepare_ligands

    param_db, co = prepare_ligands(atom_array)
    # param_db now contains chemical + scoring data for all ligands.
"""

import logging
import functools
from typing import Optional

import biotite.structure as struc
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds

from tmol.database import ParameterDatabase
from tmol.database.chemical import RawResidueType
from tmol.io.canonical_ordering import CanonicalOrdering
from tmol.ligand.atom_typing import AtomTypeAssignment, assign_tmol_atom_types
from tmol.ligand.detect import LigandInfo, detect_nonstandard_residues
from tmol.ligand.graph_match import match_heavy_atoms
from tmol.ligand.mol3d import compute_mmff94_charges
from tmol.ligand.registry import (
    LigandPreparationCache,
    cache_ligand,
    get_cached_charges_for_key,
    get_cached_ligand_for_key,
    get_default_cache,
    rebuild_canonical_ordering,
    register_ligand,
)
from tmol.ligand.residue_builder import build_residue_type
from tmol.ligand.rdkit_mol import (
    ELEMENT_TO_ATOMIC_NUM,
    ligand_atom_array_to_rdkit_mol,
    protonate_ligand_mol,
)

logger = logging.getLogger(__name__)


@functools.cache
def _default_detection_ordering() -> CanonicalOrdering:
    """Canonical ordering used to detect non-standard residues.

    Detection should be stable across repeated calls even when ``param_db`` has
    already been extended with ligands in earlier invocations.
    """
    return CanonicalOrdering.from_chemdb(ParameterDatabase.get_fresh_default().chemical)


def _build_cif_rdkit_mol(ligand_info: LigandInfo) -> Chem.Mol:
    """Build a Chem.Mol from CIF atom names, elements, and coordinates.

    Uses rdDetermineBonds.DetermineBonds to infer bonds from heavy-atom
    coordinates, if necessary.
    """
    rwmol = Chem.RWMol()
    n = len(ligand_info.elements)
    conf = Chem.Conformer(n)
    for i, (elem, coord) in enumerate(zip(ligand_info.elements, ligand_info.coords)):
        sym = elem.strip()
        if sym in ELEMENT_TO_ATOMIC_NUM:
            atom = Chem.Atom(sym)
        else:
            z = Chem.GetPeriodicTable().GetAtomicNumber(sym)
            atom = Chem.Atom(z)
        rwmol.AddAtom(atom)
        conf.SetAtomPosition(i, (float(coord[0]), float(coord[1]), float(coord[2])))
    rwmol.AddConformer(conf, assignId=True)
    if rwmol.GetNumAtoms() > 1:
        try:
            rdDetermineBonds.DetermineBonds(rwmol)
        except Exception:
            logger.debug(
                "DetermineBonds failed for CIF Mol of %s; using bondless Mol",
                ligand_info.res_name,
                exc_info=True,
            )

    return rwmol.GetMol()


def _rename_atoms_to_cif(
    pipeline_mol: Chem.Mol,
    atom_types: list[AtomTypeAssignment],
    ligand_info: LigandInfo,
) -> list[AtomTypeAssignment]:
    """Rename pipeline atoms to use CIF atom names via graph matching.

    Heavy atoms are matched between the pipeline Mol and the CIF Mol by
    molecular graph isomorphism, then renamed to the CIF names.
    Hydrogens keep their auto-generated names (no CIF equivalent).

    Returns a new list of AtomTypeAssignment with updated atom_name fields.
    """
    cif_mol = _build_cif_rdkit_mol(ligand_info)

    try:
        idx_mapping = match_heavy_atoms(pipeline_mol, cif_mol)
    except ValueError:
        logger.warning(
            "Could not match CIF atoms for %s, keeping pipeline names",
            ligand_info.res_name,
        )
        return atom_types

    cif_idx_to_name = {}
    for i, atom in enumerate(cif_mol.GetAtoms()):
        if i < len(ligand_info.atom_names):
            cif_idx_to_name[atom.GetIdx()] = ligand_info.atom_names[i]

    pipeline_to_cif = {}
    for pipeline_idx, cif_idx in idx_mapping.items():
        cif_name = cif_idx_to_name.get(cif_idx)
        if cif_name:
            pipeline_to_cif[pipeline_idx] = cif_name

    return [
        at._replace(atom_name=pipeline_to_cif.get(at.index, at.atom_name))
        for at in atom_types
    ]


def _rename_atoms_to_cif_by_index(
    atom_types: list[AtomTypeAssignment], ligand_info: LigandInfo
) -> list[AtomTypeAssignment] | None:
    """Rename heavy atoms from CIF names using direct index alignment.

    Returns None if index-based mapping is not safe, so callers can fall
    back to graph-based matching.
    """
    if len(ligand_info.atom_names) != len(ligand_info.elements):
        return None

    renamed: list[AtomTypeAssignment] = []
    seen_names: set[str] = set()
    for at in atom_types:
        new_name = at.atom_name
        if at.element != "H":
            if at.index >= len(ligand_info.atom_names):
                return None
            cif_elem = ligand_info.elements[at.index].strip().upper()
            if cif_elem != at.element.upper():
                return None
            new_name = ligand_info.atom_names[at.index]
        if new_name in seen_names:
            return None
        seen_names.add(new_name)
        renamed.append(at._replace(atom_name=new_name))
    return renamed


def prepare_single_ligand(
    ligand_info: LigandInfo,
    ph: float = 7.4,
) -> tuple[RawResidueType, dict[str, float], dict[str, str]]:
    """Run the full preparation pipeline for a single ligand.

    Args:
        ligand_info: Detected ligand information.
        ph: Target pH for protonation.

    Returns:
        A tuple of (RawResidueType, partial_charges, atom_type_elements).
    """
    rdkit_mol = ligand_atom_array_to_rdkit_mol(ligand_info)
    protonated = protonate_ligand_mol(rdkit_mol, ph=ph)
    try:
        protonated = AllChem.AssignBondOrdersFromTemplate(protonated, rdkit_mol)
    except Exception:
        logger.debug(
            "AssignBondOrdersFromTemplate failed for %s, using protonated mol directly",
            ligand_info.res_name,
        )
    protonated = Chem.AddHs(protonated, addCoords=True)
    charges_by_index = compute_mmff94_charges(protonated)
    atom_types = assign_tmol_atom_types(protonated)
    atom_types_by_index = _rename_atoms_to_cif_by_index(atom_types, ligand_info)
    if atom_types_by_index is not None:
        atom_types = atom_types_by_index
    else:
        atom_types = _rename_atoms_to_cif(protonated, atom_types, ligand_info)

    restype = build_residue_type(
        protonated,
        ligand_info.res_name,
        atom_types,
    )

    # Build final charge map only after names are finalized and residue built.
    # This guarantees alignment with scoring lookups keyed by (residue, atom_name).
    charges = {
        at.atom_name: charges_by_index[at.index]
        for at in atom_types
        if at.index in charges_by_index
    }
    atom_type_elements: dict[str, str] = {}
    for at in atom_types:
        prev = atom_type_elements.get(at.atom_type)
        if prev is not None and prev != at.element:
            raise RuntimeError(
                f"{ligand_info.res_name}: inconsistent element assignment for atom type "
                f"{at.atom_type} ({prev} vs {at.element})"
            )
        atom_type_elements[at.atom_type] = at.element
    restype_atom_names = {a.name for a in restype.atoms}
    charges = {name: q for name, q in charges.items() if name in restype_atom_names}
    missing_names = sorted(restype_atom_names - set(charges))
    if missing_names:
        raise RuntimeError(
            f"{ligand_info.res_name}: missing partial charges for atoms: {missing_names}"
        )

    return restype, charges, atom_type_elements


def prepare_ligands(
    atom_array: struc.AtomArray,
    param_db: Optional[ParameterDatabase] = None,
    ph: float = 7.4,
    strict_atom_types: bool = False,
    cache: LigandPreparationCache | None = None,
) -> tuple[ParameterDatabase, CanonicalOrdering]:
    """Detect, prepare, and register all non-standard residues.

    Scans the input AtomArray for residues not in the ParameterDatabase,
    runs each through the ligand preparation pipeline (direct RDKit molecule
    construction, protonation, 3D generation, atom typing, residue building),
    and registers them — extending both the chemical and scoring databases.

    Args:
        atom_array: A biotite AtomArray from a CIF or PDB file.
        param_db: The ParameterDatabase to extend. If None, the default
            database is loaded as a fresh instance. Mutated in place.
        ph: Target pH for ligand protonation.
        strict_atom_types: If True, fail when unknown atom-type element
            mappings are encountered during registration.
        cache: Optional cache object controlling ligand reuse behavior.
            If None, uses the process-global default cache.

    Returns:
        A (ParameterDatabase, CanonicalOrdering) tuple with all detected
        ligands registered.
    """
    if param_db is None:
        param_db = ParameterDatabase.get_fresh_default()
    if cache is None:
        cache = get_default_cache()

    canonical_ordering = rebuild_canonical_ordering(param_db)

    ligands = detect_nonstandard_residues(atom_array, _default_detection_ordering())

    if not ligands:
        logger.info("No non-standard residues detected")
        return param_db, canonical_ordering

    logger.info("Found %d non-standard residue type(s) to prepare", len(ligands))

    modified = False
    for lig in ligands:
        cache_key = (
            lig.res_name,
            round(ph, 3),
            tuple(lig.atom_names),
            tuple(lig.elements),
        )
        cached = get_cached_ligand_for_key(cache_key, cache=cache)
        if cached is not None:
            logger.info("Using cached preparation for %s", lig.res_name)
            inserted = register_ligand(
                param_db,
                cached,
                partial_charges=get_cached_charges_for_key(cache_key, cache=cache),
                strict_atom_types=strict_atom_types,
            )
            modified = modified or inserted
            continue

        logger.info(
            "Preparing %s (CCD type: %s, is_ligand: %s)",
            lig.res_name,
            lig.ccd_type,
            lig.is_ligand,
        )

        restype, charges, atom_type_elements = prepare_single_ligand(lig, ph=ph)
        inserted = register_ligand(
            param_db,
            restype,
            partial_charges=charges,
            atom_type_elements=atom_type_elements,
            strict_atom_types=strict_atom_types,
        )
        cache_ligand(
            lig.res_name,
            restype,
            charges=charges,
            cache_key=cache_key,
            cache=cache,
        )
        modified = modified or inserted

    if modified:
        canonical_ordering = rebuild_canonical_ordering(param_db)

    return param_db, canonical_ordering
