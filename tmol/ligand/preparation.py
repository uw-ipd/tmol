"""Ligand preparation implementation for tmol.

This module contains the concrete preparation pipeline implementation.
`tmol.ligand.__init__` re-exports the public API from here.
"""

import logging
from collections.abc import Callable
from typing import Optional

import biotite.structure as struc
from biotite.interface.rdkit import to_mol
from rdkit import Chem
from rdkit.Chem import AllChem

from tmol.database import ParameterDatabase
from tmol.io.canonical_ordering import CanonicalOrdering
from tmol.ligand.atom_typing import AtomTypeAssignment, assign_tmol_atom_types
from tmol.ligand.detect import (
    NonStandardResidueInfo,
    _METAL_SYMBOLS,
    detect_nonstandard_residues,
    nonstandard_residue_info_from_cif,
    nonstandard_residue_info_from_mol2,
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
    ligand_atom_array_to_rdkit_mol,
    protonate_ligand_mol,
    reapply_ligand_source_annotations,
)

logger = logging.getLogger(__name__)


def _build_cif_rdkit_mol(ligand_info: NonStandardResidueInfo) -> Chem.Mol:
    """Build a Chem.Mol from CIF atom names, elements, and coordinates.

    This helper is only used for fallback atom-name graph matching and
    therefore follows the same explicit-bond contract as the main ligand
    pipeline: no bond inference.
    """
    atom_array = ligand_info.atom_array
    has_bonds = atom_array.bonds is not None and atom_array.bonds.get_bond_count() > 0
    if not has_bonds:
        raise ValueError(
            f"{ligand_info.res_name}: explicit ligand bonds required for graph "
            "matching; bond inference is unsupported."
        )
    return to_mol(atom_array)


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

    When the pipeline preserved input explicit hydrogens (``skip_protonation``),
    all atoms including H are renamed by index. Otherwise hydrogens keep
    pipeline-assigned names (Dimorphite / ``AddHs`` round-trip).
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

    preserve_input_atoms = len(atom_types) == len(ligand_info.atom_names)

    renamed: list[AtomTypeAssignment] = []
    seen_names: set[str] = set()
    for at in atom_types:
        new_name = at.atom_name
        if preserve_input_atoms or at.element != "H":
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


def prepare_single_ligand(  # noqa: C901
    ligand_info: NonStandardResidueInfo,
    ph: float = 7.4,
    charge_mode: str = "auto",
    *,
    prepare_mode: str = "rebuild",
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
        charge_mode: Partial-charge source policy passed to
            :func:`tmol.ligand.mol3d.build_partial_charges`.
        prepare_mode: ``"rebuild"`` runs Dimorphite protonation (default).
            ``"passthrough"`` preserves input H, coordinates, and bond orders
            (crystallographic / pre-prepared mol2).
    """
    if prepare_mode not in {"rebuild", "passthrough"}:
        raise ValueError(
            f"Unsupported prepare_mode={prepare_mode!r}; use 'rebuild' or 'passthrough'"
        )

    if (
        prepare_mode == "passthrough"
        and ligand_info.source_path is not None
        and str(ligand_info.source_path).lower().endswith(".mol2")
    ):
        from tmol.ligand.mol2_io import prepare_ligand_from_mol2_passthrough

        passthrough_charge_mode = charge_mode
        if passthrough_charge_mode == "auto":
            passthrough_charge_mode = (
                "input" if ligand_info.partial_charges is not None else "mmff94"
            )
        if passthrough_charge_mode not in {"input", "mmff94"}:
            passthrough_charge_mode = "mmff94"
        return prepare_ligand_from_mol2_passthrough(
            ligand_info.source_path,
            res_name=ligand_info.res_name,
            charge_mode=passthrough_charge_mode,
        )

    skip_protonation = (
        prepare_mode == "passthrough"
        or ligand_info.skip_protonation
        or ligand_info.partial_charges is not None
    )
    rdkit_mol = ligand_atom_array_to_rdkit_mol(
        ligand_info, keep_hydrogens=skip_protonation
    )
    # Skip protonation when:
    # - prepare_mode is passthrough,
    # - The caller explicitly sets skip_protonation=True (keeps input explicit
    #   H and protonation from mol2/CIF), OR
    # - The caller supplies authoritative partial charges (e.g. AM1-BCC
    #   from a Tripos mol2 file), which implies the input already encodes
    #   the desired protonation state (heavy-only RDKit mol; AddHs below).
    if skip_protonation:
        protonated = rdkit_mol
    else:
        protonated = protonate_ligand_mol(rdkit_mol, ph=ph)
        try:
            # Apply source bond orders onto the protonated topology.
            # RDKit API: AssignBondOrdersFromTemplate(template, mol)
            # where `template` has trusted bond orders.
            protonated = AllChem.AssignBondOrdersFromTemplate(rdkit_mol, protonated)
        except Exception:
            logger.debug(
                "AssignBondOrdersFromTemplate failed for %s, using protonated mol directly",
                ligand_info.res_name,
            )
    # Tolerant sanitize before typing so CIF/mol2 aromatic/subtype hints survive.
    from tmol.ligand.atom_typing import sanitize_tolerant

    sanitize_tolerant(protonated)
    # Authoritative-charge inputs already carry explicit H at the right state.
    if not skip_protonation:
        protonated = Chem.AddHs(protonated, addCoords=True)

    reapply_ligand_source_annotations(
        protonated,
        ligand_info,
        keep_hydrogens=skip_protonation,
    )

    # Atom typing rewrites bond orders in place; charge from a snapshot so MMFF94
    # sees the un-mutated graph (typing preserves indices).
    charge_mol = Chem.Mol(protonated)

    atom_types = assign_tmol_atom_types(protonated)
    atom_types = _rename_atoms_to_cif(protonated, atom_types, ligand_info)

    restype = build_residue_type(
        protonated,
        ligand_info.res_name,
        atom_types,
    )

    charges = build_partial_charges(
        charge_mol,
        atom_types,
        input_charges=ligand_info.partial_charges,
        ligand_name=ligand_info.res_name,
        charge_mode=charge_mode,
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

    # Register coords so pose construction can remap differently-named complex-PDB
    # ligand atoms onto this residue type by coordinate. Source AtomArray coords
    # are preferred (always present; the pipeline conformer can be lost).
    from tmol.ligand.coord_name_registry import register_ligand_coords

    registry_coords: dict[str, tuple[float, float, float]] = {}
    src_names = ligand_info.atom_names
    src_coords = ligand_info.coords
    if src_coords is not None and len(src_coords) == len(src_names):
        for name, xyz in zip(src_names, src_coords):
            if name in restype_atom_names:
                registry_coords[name] = (
                    float(xyz[0]),
                    float(xyz[1]),
                    float(xyz[2]),
                )
    # Fill atoms missing from source coords (e.g. renamed H) from the conformer.
    for name, xyz in coords.items():
        registry_coords.setdefault(name, xyz)

    register_ligand_coords(restype.name, registry_coords)

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
    charge_mode: str = "auto",
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
        charge_mode: Charge policy for prepared ligands. ``"auto"`` uses
            authoritative source charges when complete, else MMFF94;
            ``"mmff94"`` always computes MMFF94; ``"input"`` requires
            complete input charges.

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
            charge_mode,
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
        prep = prepare_single_ligand(lig, ph=ph, charge_mode=charge_mode)
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


def prepare_ligand_from_cif(
    cif_path: str,
    *,
    param_db: Optional[ParameterDatabase] = None,
    ph: float = 7.4,
    strict_atom_types: bool = False,
    res_name: str | None = None,
    charge_mode: str = "auto",
) -> tuple[ParameterDatabase, CanonicalOrdering]:
    """Prepare a single ligand from a CIF file and inject it into a database.

    This helper preserves ligand chemistry annotations from CIF fields such as
    ``partial_charge``, ``tmol_aromatic``, and ``tmol_source_subtype``.
    ``charge_mode`` controls whether those source charges are consumed or
    MMFF94 is recomputed.
    """
    return _prepare_ligand_from_file(
        cif_path,
        loader=nonstandard_residue_info_from_cif,
        param_db=param_db,
        ph=ph,
        strict_atom_types=strict_atom_types,
        res_name=res_name,
        charge_mode=charge_mode,
    )


def prepare_ligand_from_mol2(
    mol2_path: str,
    *,
    param_db: Optional[ParameterDatabase] = None,
    ph: float = 7.4,
    strict_atom_types: bool = False,
    res_name: str | None = None,
    charge_mode: str = "auto",
    prepare_mode: str = "rebuild",
) -> tuple[ParameterDatabase, CanonicalOrdering]:
    """Prepare a single ligand from a Mol2 file and inject it into a database.

      This helper preserves Tripos bond/aromaticity/subtype metadata and uses
      Tripos partial charges when they are present on all atoms unless
      ``charge_mode="mmff94"`` forces recomputation.

    ``prepare_mode="passthrough"`` skips Dimorphite protonation and preserves
      the mol2 hydrogen/bond state (Rosetta mol2genparams-style).
    """
    return _prepare_ligand_from_file(
        mol2_path,
        loader=nonstandard_residue_info_from_mol2,
        param_db=param_db,
        ph=ph,
        strict_atom_types=strict_atom_types,
        res_name=res_name,
        charge_mode=charge_mode,
        prepare_mode=prepare_mode,
    )


def _prepare_ligand_from_file(
    path: str,
    *,
    loader: Callable[[str, str | None], NonStandardResidueInfo],
    param_db: Optional[ParameterDatabase],
    ph: float,
    strict_atom_types: bool,
    res_name: str | None,
    charge_mode: str,
    prepare_mode: str = "rebuild",
) -> tuple[ParameterDatabase, CanonicalOrdering]:
    """Load one ligand from file, prepare it, and inject it.

    Both ``prepare_ligand_from_cif`` and ``prepare_ligand_from_mol2`` use this
    helper to keep single-ligand file ingestion behavior identical.
    """
    if param_db is None:
        param_db = ParameterDatabase.get_default()

    lig = loader(path, res_name)
    prep = prepare_single_ligand(
        lig, ph=ph, charge_mode=charge_mode, prepare_mode=prepare_mode
    )
    param_db = inject_ligand_preparations(
        param_db, [prep], strict_atom_types=strict_atom_types
    )
    return param_db, rebuild_canonical_ordering(param_db)
