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
from typing import Optional

import biotite.structure as struc
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds

from tmol.database import ParameterDatabase
from tmol.database.chemical import RawResidueType  # noqa: F401  re-exported
from tmol.io.canonical_ordering import CanonicalOrdering
from tmol.ligand.atom_typing import AtomTypeAssignment, assign_tmol_atom_types
from tmol.ligand.detect import (
    NonStandardResidueInfo,
    _METAL_SYMBOLS,
    detect_nonstandard_residues,
)
from tmol.ligand.graph_match import match_heavy_atoms
from tmol.ligand.registry import (
    LigandPreparation,
    LigandPreparationCache,
    _build_cartbonded_params,
    cache_ligand,
    get_cached_charges_for_key,
    get_cached_ligand_for_key,
    get_default_cache,
    inject_ligand_preparations,
    rebuild_canonical_ordering,
)
from tmol.ligand.mol3d import build_partial_charges
from tmol.ligand.residue_builder import build_residue_type
from tmol.ligand.rdkit_mol import (
    ELEMENT_TO_ATOMIC_NUM,
    ligand_atom_array_to_rdkit_mol,
    protonate_ligand_mol,
)

logger = logging.getLogger(__name__)


def _build_cif_rdkit_mol(ligand_info: NonStandardResidueInfo) -> Chem.Mol:
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
    ligand_info: NonStandardResidueInfo,
) -> list[AtomTypeAssignment]:
    """Rename pipeline atoms to use the CIF atom names from ``ligand_info``.

    Tries the fast index-aligned path first (works whenever the pipeline
    didn't reorder heavy atoms relative to the input AtomArray, which is
    the common mol2 / CCD case) and falls back to graph isomorphism only
    if the index path can't be applied safely.

    Hydrogens keep their pipeline-assigned names regardless — the input
    rarely names them, and Frank's reference uses a sequential
    convention that doesn't survive a Chem.RemoveHs / Chem.AddHs
    round-trip anyway.
    """
    by_index = _rename_atoms_to_cif_by_index(atom_types, ligand_info)
    if by_index is not None:
        return by_index
    return _rename_atoms_to_cif_by_graph(pipeline_mol, atom_types, ligand_info)


def _rename_atoms_to_cif_by_index(
    atom_types: list[AtomTypeAssignment], ligand_info: NonStandardResidueInfo
) -> list[AtomTypeAssignment] | None:
    """Fast path: rename heavy atoms by direct RDKit-index alignment.

    Returns ``None`` when the index alignment isn't safe (atom count
    mismatch, element mismatch, or duplicate names produced) so the
    caller can try graph matching.
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


def _rename_atoms_to_cif_by_graph(
    pipeline_mol: Chem.Mol,
    atom_types: list[AtomTypeAssignment],
    ligand_info: NonStandardResidueInfo,
) -> list[AtomTypeAssignment]:
    """Slow-path fallback: VF2 heavy-atom isomorphism between the pipeline
    Mol and a Mol rebuilt from the CIF input. Used when the fast index
    path can't be applied. Returns the input atom types unchanged if no
    isomorphism is found.
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


def prepare_single_ligand(
    ligand_info: NonStandardResidueInfo,
    ph: float = 7.4,
) -> LigandPreparation:
    """Run the full RDKit preparation pipeline for a single ligand.

    Returns a :class:`LigandPreparation` — the same struct
    :func:`tmol.ligand.params_file.load_params_file` produces for each
    residue defined in a ``.tmol`` file, so the AtomArray-driven path
    and the params-file path converge on a single abstraction that
    :func:`inject_ligand_preparations` consumes.

    Args:
        ligand_info: Detected ligand information.
        ph: Target pH for protonation.
    """
    rdkit_mol = ligand_atom_array_to_rdkit_mol(ligand_info)
    # When the caller supplies authoritative per-atom partial charges
    # (e.g. AM1-BCC from a Tripos mol2 file) we treat that as evidence
    # the input already encodes the desired protonation state, and skip
    # Dimorphite-DL — otherwise it can flip ring nitrogens (imidazole-
    # type) to their protonated form at pH 7.4, adding a hydrogen that
    # the caller's reference deliberately omits.
    if ligand_info.partial_charges:
        protonated = rdkit_mol
    else:
        protonated = protonate_ligand_mol(rdkit_mol, ph=ph)
        try:
            protonated = AllChem.AssignBondOrdersFromTemplate(protonated, rdkit_mol)
        except Exception:
            logger.debug(
                "AssignBondOrdersFromTemplate failed for %s, using protonated mol directly",
                ligand_info.res_name,
            )
    # Use the smart sanitize so source-supplied aromatic flags
    # (CIF ``_atom_site.tmol_aromatic``) are not blown away by RDKit's
    # default aromaticity perception.
    from tmol.ligand.atom_typing import sanitize_tolerant

    sanitize_tolerant(protonated)
    protonated = Chem.AddHs(protonated, addCoords=True)

    atom_types = assign_tmol_atom_types(protonated)
    atom_types = _rename_atoms_to_cif(protonated, atom_types, ligand_info)

    restype = build_residue_type(
        protonated,
        ligand_info.res_name,
        atom_types,
    )

    charges = build_partial_charges(
        protonated,
        atom_types,
        input_charges=ligand_info.partial_charges,
    )

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

    coords: dict[str, tuple[float, float, float]] = {}
    if protonated.GetNumConformers() > 0:
        conf = protonated.GetConformer()
        for at in atom_types:
            if at.atom_name in restype_atom_names:
                p = conf.GetAtomPosition(at.index)
                coords[at.atom_name] = (float(p.x), float(p.y), float(p.z))

    return LigandPreparation(
        residue_type=restype,
        partial_charges=charges,
        cartbonded_params=_build_cartbonded_params(restype, coords=coords),
        atom_type_elements=atom_type_elements,
    )


def prepare_ligands(
    atom_array: struc.AtomArray,
    param_db: Optional[ParameterDatabase] = None,
    ph: float = 7.4,
    strict_atom_types: bool = False,
    cache: LigandPreparationCache | None = None,
    params_files: list[str] | None = None,
    params_output: str | None = None,
) -> tuple[ParameterDatabase, CanonicalOrdering]:
    """Detect, prepare, and register all non-standard residues.

    Scans the input AtomArray for residues not in the ParameterDatabase,
    runs each through the ligand preparation pipeline (direct RDKit molecule
    construction, protonation, 3D generation, atom typing, residue building),
    and returns a **new** ParameterDatabase with the ligand data injected.

    Args:
        atom_array: A biotite AtomArray from a CIF or PDB file.
        param_db: The base ParameterDatabase (not modified). If None, the
            default database is used.
        ph: Target pH for ligand protonation.
        strict_atom_types: If True, fail when unknown atom-type element
            mappings are encountered during registration.
        cache: Optional cache object controlling ligand reuse behavior.
            If None, uses the process-global default cache.
        params_files: Optional list of tmol YAML params file paths to
            inject before detection. Residues defined in these files
            skip the RDKit/OB preparation pipeline.
        params_output: Optional path to write all prepared ligand data
            to a tmol YAML params file for later reuse.

    Returns:
        A (ParameterDatabase, CanonicalOrdering) tuple. The returned
        ParameterDatabase is a new instance with all detected ligands
        injected; the input ``param_db`` is not modified.
    """
    if isinstance(atom_array, struc.AtomArrayStack):
        if len(atom_array) == 1:
            atom_array = atom_array[0]
        else:
            raise TypeError(
                "prepare_ligands expects a single AtomArray, not an "
                f"AtomArrayStack with {len(atom_array)} models. "
                "Select a single model first (e.g. stack[0])."
            )
    if param_db is None:
        param_db = ParameterDatabase.get_default()
    if cache is None:
        cache = get_default_cache()

    if params_files:
        from tmol.ligand.params_file import inject_params_files

        param_db = inject_params_files(
            param_db, params_files, strict_atom_types=strict_atom_types
        )

    canonical_ordering = rebuild_canonical_ordering(param_db)

    ligands = detect_nonstandard_residues(atom_array, canonical_ordering)

    if not ligands:
        logger.info("No non-standard residues detected")
        return param_db, canonical_ordering

    logger.info("Found %d non-standard residue type(s) to prepare", len(ligands))

    preparations: list[LigandPreparation] = []
    for lig in ligands:
        metals_present = sorted(
            {
                e.strip().capitalize()
                for e in lig.elements
                if e.strip().capitalize() in _METAL_SYMBOLS
            }
        )
        if metals_present:
            logger.warning(
                "Skipping %s: ligands containing metal atoms (%s) are not supported",
                lig.res_name,
                metals_present,
            )
            continue

        if lig.covalently_linked:
            logger.warning(
                "Skipping %s: ligand is covalently linked to another residue "
                "(e.g. glycan attached to protein) — not supported",
                lig.res_name,
            )
            continue

        cache_key = (
            lig.res_name,
            round(ph, 3),
            tuple(lig.atom_names),
            tuple(lig.elements),
        )
        cached_rt = get_cached_ligand_for_key(cache_key, cache=cache)
        cached_q = get_cached_charges_for_key(cache_key, cache=cache)
        if cached_rt is not None and cached_q is not None:
            logger.info("Using cached preparation for %s", lig.res_name)
            # Cache predates the unified struct, so it stores
            # (restype, charges) only. Rebuild the cartbonded slice
            # from the cached residue type's icoors here.
            preparations.append(
                LigandPreparation(
                    residue_type=cached_rt,
                    partial_charges=cached_q,
                    cartbonded_params=_build_cartbonded_params(cached_rt),
                    atom_type_elements=None,
                )
            )
            continue

        logger.info("Preparing %s (CCD type: %s)", lig.res_name, lig.ccd_type)
        prep = prepare_single_ligand(lig, ph=ph)
        cache_ligand(
            lig.res_name,
            prep.residue_type,
            charges=prep.partial_charges,
            cache_key=cache_key,
            cache=cache,
        )
        preparations.append(prep)

    if preparations:
        param_db = inject_ligand_preparations(
            param_db, preparations, strict_atom_types=strict_atom_types
        )
        canonical_ordering = rebuild_canonical_ordering(param_db)

        if params_output:
            from tmol.ligand.params_file import write_params_file

            all_residues = [p.residue_type for p in preparations]
            all_charges = {p.residue_type.name: p.partial_charges for p in preparations}
            all_cartbonded = {
                p.residue_type.name: p.cartbonded_params for p in preparations
            }
            write_params_file(params_output, all_residues, all_charges, all_cartbonded)
            logger.info("Wrote params to %s", params_output)

    return param_db, canonical_ordering
