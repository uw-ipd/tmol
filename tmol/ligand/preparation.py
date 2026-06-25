"""Ligand preparation implementation for tmol.

This module contains the concrete preparation pipeline implementation.
`tmol.ligand.__init__` re-exports the public API from here.
"""

import logging
from typing import Optional

import biotite.structure as struc
import numpy as np
from biotite.interface.rdkit import to_mol
from rdkit import Chem

from tmol.database import ParameterDatabase
from tmol.io.canonical_ordering import CanonicalOrdering
from tmol.ligand.atom_typing import AtomTypeAssignment, assign_tmol_atom_types
from tmol.ligand.detect import (
    NonStandardResidueInfo,
    _METAL_SYMBOLS,
    detect_nonstandard_residues,
    nonstandard_residue_info_from_smiles_via_mol2,
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
from tmol.ligand.mol3d import authoritative_charges_by_index
from tmol.ligand.residue_builder import build_residue_type
from tmol.ligand.external.atomworks_rdkit import atom_array_to_rdkit
from tmol.ligand.structure_to_smiles import (
    ligand_smiles_candidates_from_atom_array,
)
from tmol.ligand.rdkit_mol import ligand_atom_array_to_rdkit_mol

logger = logging.getLogger(__name__)


class LigandPreparationError(RuntimeError):
    """A detected ligand could not be prepared, registered, or retained.

    Raised by :func:`prepare_ligands` (and the ``prepare_ligands=True`` IO
    paths) when ``strict_ligands=True`` and a non-standard residue is skipped
    or fails preparation, instead of silently dropping it. Pass
    ``strict_ligands=False`` to downgrade these failures to warnings.
    """


def _skip_or_raise(strict_ligands: bool, message: str) -> None:
    """Raise :class:`LigandPreparationError` if strict, else log a warning.

    Centralizes the strict-versus-lenient handling for ligands that
    :func:`prepare_ligands` cannot register. The lenient branch appends a hint
    so the warning matches the strict error's guidance.
    """
    if strict_ligands:
        raise LigandPreparationError(
            f"{message}. Pass strict_ligands=False to skip it with a warning, "
            "or supply prebuilt params via ligand_params_files."
        )
    logger.warning("Skipping %s", message)


def _build_cif_rdkit_mol(ligand_info: NonStandardResidueInfo) -> Chem.Mol:
    """Build a Chem.Mol from the CIF ligand for heavy-atom graph matching.

    Atom order is preserved (atom ``i`` corresponds to
    ``ligand_info.atom_names[i]``) so the matched indices can be mapped back to
    CIF atom names. Uses the explicit CIF bond table when present and falls
    back to geometry-based bond perception (vendored atomworks
    :func:`atom_array_to_rdkit`) for bonds-absent CIFs.
    """
    atom_array = ligand_info.atom_array
    has_bonds = atom_array.bonds is not None and atom_array.bonds.get_bond_count() > 0
    if has_bonds:
        return to_mol(atom_array)

    if "charge" in atom_array.get_annotation_categories():
        system_charge = int(np.nansum(atom_array.charge))
    else:
        system_charge = 0
    return atom_array_to_rdkit(
        atom_array,
        infer_bonds=True,
        system_charge=system_charge,
        hydrogen_policy="keep",
    )


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


def prepare_single_ligand(
    ligand_info: NonStandardResidueInfo,
    sample_proton_chi: bool = True,
    name_source: Optional[NonStandardResidueInfo] = None,
) -> LigandPreparation:
    """Build a :class:`LigandPreparation` from a SMILES-derived ligand.

    This is the final, naming-and-typing step of the unified pipeline. Its input
    must already be fully resolved chemistry: explicit hydrogens at the desired
    protonation state and authoritative per-atom partial charges (the OpenBabel
    MMFF94 charges produced by the SMILES -> mol2 step). Protonation and charge
    generation are *not* done here -- they happen upstream in
    :func:`tmol.ligand.detect.nonstandard_residue_info_from_smiles_via_mol2`.

    Charges are mapped onto atoms by stable RDKit index (source atom order),
    so they are independent of the atom renaming below and never recomputed.

    Returns a :class:`LigandPreparation` -- the same struct
    :func:`tmol.ligand.params_file.load_params_file` produces for each residue
    defined in a ``.tmol`` file, so the AtomArray-driven path and the params-file
    path converge on a single abstraction that
    :func:`inject_ligand_preparations` consumes.

    Args:
        ligand_info: A SMILES-derived ligand (``skip_protonation=True`` with
            authoritative ``partial_charges``). Raw CIF/atom-array ligands must
            be routed through :func:`prepare_ligands` / :func:`prepare_ligand_from_cif`.
        sample_proton_chi: Whether to emit proton-chi samples.
        name_source: Optional ligand whose atom names the prepared residue should
            adopt (graph-matched to the prepared heavy atoms). On the unified CIF
            path this is the original CIF ligand, so pose-build can place CIF
            coordinates by ``(res_name, atom_name)``. Defaults to ``ligand_info``.

    Raises:
        ValueError: If ``ligand_info`` lacks explicit hydrogens / authoritative
            charges (there is no charge-generation fallback).
    """
    if not ligand_info.skip_protonation or not ligand_info.partial_charges:
        raise ValueError(
            f"{ligand_info.res_name}: prepare_single_ligand requires a ligand that "
            "already carries explicit hydrogens and authoritative partial charges "
            "(skip_protonation=True). Route raw CIF/atom-array ligands through the "
            "unified SMILES path (prepare_ligands / prepare_ligand_from_cif), which "
            "derives a SMILES and generates OpenBabel MMFF94 charges. No RDKit/"
            "Gasteiger charge fallback is used."
        )

    from tmol.ligand.atom_typing import sanitize_tolerant

    protonated = ligand_atom_array_to_rdkit_mol(ligand_info, keep_hydrogens=True)
    sanitize_tolerant(protonated)

    atom_types, typing_state = assign_tmol_atom_types(protonated, return_state=True)

    # Charges come straight from the SMILES -> OpenBabel MMFF94 step, carried on
    # ``ligand_info`` in source-atom order. Map them onto atoms by stable RDKit
    # index *before* renaming so they are wholly independent of atom naming --
    # no name-based bridging and no force-field recomputation.
    charge_by_index = authoritative_charges_by_index(
        ligand_info.atom_names,
        ligand_info.partial_charges,
        protonated,
        ligand_name=ligand_info.res_name,
    )

    atom_types = _rename_atoms_to_cif(
        protonated, atom_types, name_source if name_source is not None else ligand_info
    )

    restype = build_residue_type(
        protonated,
        ligand_info.res_name,
        atom_types,
        typing_state=typing_state,
        sample_proton_chi=sample_proton_chi,
        original_single_bonds=ligand_info.original_single_bonds,
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
    charges = {
        at.atom_name: charge_by_index[at.index]
        for at in atom_types
        if at.atom_name in restype_atom_names
    }
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


def _cif_heavy_atom_names(ligand_info: NonStandardResidueInfo) -> set[str]:
    """Heavy-atom names of the (CIF) ligand, used to verify name matching."""
    return {
        name
        for name, element in zip(ligand_info.atom_names, ligand_info.elements)
        if str(element).strip().upper() != "H"
    }


def _residue_covers_cif_heavy_atoms(
    prep: LigandPreparation, cif_heavy_names: set[str]
) -> bool:
    """Return True if the prepared residue carries every CIF heavy-atom name.

    When the SMILES-derived residue's heavy-atom names are a superset of the
    CIF ligand's heavy-atom names, pose-build can place every CIF heavy-atom
    coordinate by ``(res_name, atom_name)`` match.
    """
    if not cif_heavy_names:
        return True
    elements = prep.atom_type_elements or {}
    restype_heavy: set[str] = set()
    for atom in prep.residue_type.atoms:
        element = elements.get(atom.atom_type)
        if element is not None and element.upper() == "H":
            continue
        restype_heavy.add(atom.name)
    return cif_heavy_names.issubset(restype_heavy)


def _prepare_ligand_via_smiles(
    ligand_info: NonStandardResidueInfo,
    *,
    ph: float,
    sample_proton_chi: bool,
    strict: bool = True,
) -> LigandPreparation:
    """Prepare one ligand through the unified CIF -> SMILES -> params path.

    Derives candidate SMILES from the ligand's atom array (existing-bonds then
    geometry; never a CCD lookup), runs each through the SMILES -> mol2 ->
    params pipeline, and returns the first preparation whose heavy-atom names
    cover the original ligand's heavy atoms (so CIF coordinates can be placed).

    Args:
        ligand_info: The detected (CIF/atom-array) ligand.
        ph: Target pH for protonation (applied in the SMILES -> mol2 step).
        sample_proton_chi: Whether to emit proton-chi samples.
        strict: If True (default), raise when no SMILES candidate fully covers
            the CIF heavy-atom names. If False, fall back to the last
            successful (best-effort) preparation, which may leave some CIF
            coordinates unplaceable.

    Returns:
        The chosen :class:`LigandPreparation`.

    Raises:
        ValueError: If no SMILES candidate could be derived or prepared.
        LigandPreparationError: If ``strict`` and no candidate fully matches
            the CIF heavy-atom names.
    """
    candidates = ligand_smiles_candidates_from_atom_array(
        ligand_info.atom_array, res_name=ligand_info.res_name
    )
    if not candidates:
        raise ValueError(
            f"{ligand_info.res_name}: could not derive a SMILES from the ligand "
            "atom array (no usable bonds or geometry)."
        )

    cif_heavy_names = _cif_heavy_atom_names(ligand_info)
    last_prep: LigandPreparation | None = None
    last_error: Exception | None = None

    for smiles in candidates:
        try:
            smiles_info = nonstandard_residue_info_from_smiles_via_mol2(
                smiles, res_name=ligand_info.res_name, ph=ph
            )
            prep = prepare_single_ligand(
                smiles_info,
                sample_proton_chi=sample_proton_chi,
                name_source=ligand_info,
            )
        except Exception as err:  # noqa: BLE001  try the next candidate
            last_error = err
            logger.warning(
                "SMILES candidate %r failed for %s: %s",
                smiles,
                ligand_info.res_name,
                err,
            )
            continue

        last_prep = prep
        if _residue_covers_cif_heavy_atoms(prep, cif_heavy_names):
            return prep
        logger.info(
            "SMILES candidate for %s did not cover all CIF heavy-atom names; "
            "trying next candidate",
            ligand_info.res_name,
        )

    if last_prep is not None:
        if strict:
            raise LigandPreparationError(
                f"{ligand_info.res_name}: no SMILES candidate fully matched the "
                "CIF heavy-atom names, so some ligand coordinates cannot be "
                "placed. Pass strict_ligands=False to accept a best-effort "
                "preparation, or supply prebuilt params via ligand_params_files."
            )
        logger.warning(
            "No SMILES candidate fully matched CIF atom names for %s; "
            "using best-effort preparation",
            ligand_info.res_name,
        )
        return last_prep

    raise ValueError(
        f"{ligand_info.res_name}: failed to prepare ligand via SMILES"
    ) from last_error


def prepare_ligands(
    atom_array: struc.AtomArray,
    param_db: Optional[ParameterDatabase] = None,
    ph: float = 7.4,
    strict_atom_types: bool = False,
    cache: LigandPreparationCache | None = None,
    params_files: list[str] | None = None,
    params_output: str | None = None,
    sample_proton_chi: bool = True,
    strict_ligands: bool = True,
) -> tuple[ParameterDatabase, CanonicalOrdering]:
    """Detect, prepare, and register all non-standard residues.

    Scans the input AtomArray for residues not in the ParameterDatabase,
    runs each through the unified SMILES→OpenBabel mol2→typing→residue-build
    pipeline, and returns a **new** ParameterDatabase with the ligand data
    injected.

    Args:
        atom_array: A biotite AtomArray from a CIF or PDB file.
        param_db: The base ParameterDatabase (not modified). If None, the
            default database is used.
        ph: Target pH for ligand protonation (Dimorphite-DL on derived SMILES).
        strict_atom_types: If True, fail when unknown atom-type element
            mappings are encountered during registration.
        cache: Optional cache object controlling ligand reuse behavior.
            If None, uses the process-global default cache.
        params_files: Optional list of tmol YAML params file paths to
            inject before detection. Residues defined in these files
            skip the RDKit/OB preparation pipeline.
        params_output: Optional path to write all prepared ligand data
            to a tmol YAML params file for later reuse.
        sample_proton_chi: Whether to emit PROTON_CHI samples in the
            built residue type (also part of the in-process cache key).
        strict_ligands: If True (default), raise :class:`LigandPreparationError`
            when a detected non-standard residue is skipped (metal-containing or
            covalently linked) or fails preparation, instead of silently
            dropping it. If False, such residues are logged as warnings and
            skipped, leaving them to be filtered out during pose construction.

    Returns:
        A (ParameterDatabase, CanonicalOrdering) tuple. The returned
        ParameterDatabase is a new instance with all detected ligands
        injected; the input ``param_db`` is not modified.

    Raises:
        LigandPreparationError: If ``strict_ligands`` and any detected ligand
            cannot be prepared and registered.
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
            _skip_or_raise(
                strict_ligands,
                f"{lig.res_name}: ligands containing metal atoms "
                f"({metals_present}) are not supported",
            )
            continue

        if lig.covalently_linked:
            _skip_or_raise(
                strict_ligands,
                f"{lig.res_name}: ligand is covalently linked to another residue "
                "(e.g. glycan attached to protein) — not supported",
            )
            continue

        cache_key = (
            lig.res_name,
            round(ph, 3),
            sample_proton_chi,
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
        try:
            prep = _prepare_ligand_via_smiles(
                lig,
                ph=ph,
                sample_proton_chi=sample_proton_chi,
                strict=strict_ligands,
            )
        except LigandPreparationError:
            raise
        except Exception as err:  # noqa: BLE001  SMILES/typing/build failure
            if strict_ligands:
                raise LigandPreparationError(
                    f"{lig.res_name}: failed to prepare ligand ({err}). Pass "
                    "strict_ligands=False to skip it with a warning, or supply "
                    "prebuilt params via ligand_params_files."
                ) from err
            logger.warning(
                "Skipping %s: ligand preparation failed (%s)", lig.res_name, err
            )
            continue
        cache_ligand(
            lig.res_name,
            prep.residue_type,
            charges=prep.partial_charges,
            cache_key=cache_key,
            cache=cache,
        )
        preparations.append(prep)

    if strict_ligands and not preparations:
        raise LigandPreparationError(
            "All "
            f"{len(ligands)} detected non-standard residue(s) "
            f"({', '.join(sorted({lig.res_name for lig in ligands}))}) were "
            "skipped; none could be prepared. Pass strict_ligands=False to "
            "continue with these residues dropped."
        )

    if preparations:
        param_db = inject_ligand_preparations(
            param_db, preparations, strict_atom_types=strict_atom_types
        )
        canonical_ordering = rebuild_canonical_ordering(param_db)

        if params_output:
            from tmol.ligand.params_io import write_params_file

            write_params_file(preparations, params_output, format="tmol")
            logger.info("Wrote params to %s", params_output)

    return param_db, canonical_ordering


def _ligand_info_from_cif(
    cif_path: str, res_name: str | None
) -> NonStandardResidueInfo:
    """Read a ligand CIF file into a :class:`NonStandardResidueInfo`.

    Loads the atom array (with the ``_chem_comp_bond`` table when present),
    atom names, and elements. Bond orders/chemistry are intentionally *not*
    trusted here — they are re-derived as a SMILES by the unified path; this
    only needs connectivity (for graph matching) and CIF atom names/coords.
    """
    import biotite.structure.io.pdbx as pdbx

    from tmol.ligand.detect import get_chem_comp_type

    cif = pdbx.CIFFile.read(str(cif_path))
    arr = pdbx.get_structure(cif, model=1, include_bonds=True, extra_fields=["charge"])
    if isinstance(arr, struc.AtomArrayStack):
        arr = arr[0]

    atom_site = cif.block["atom_site"]
    atom_names = [str(v) for v in atom_site["label_atom_id"].as_array()]
    resolved = (res_name or str(arr.res_name[0])).strip()
    arr.res_name = np.array([resolved] * len(arr), dtype=arr.res_name.dtype)

    return NonStandardResidueInfo(
        res_name=resolved,
        ccd_type=get_chem_comp_type(resolved) or "UNKNOWN",
        atom_names=tuple(atom_names),
        elements=tuple(str(e) for e in arr.element),
        coords=arr.coord.copy(),
        atom_array=arr,
    )


def prepare_ligand_from_cif(
    cif_path: str,
    *,
    param_db: Optional[ParameterDatabase] = None,
    ph: float = 7.4,
    strict_atom_types: bool = False,
    strict_ligands: bool = True,
    res_name: str | None = None,
    sample_proton_chi: bool = True,
) -> tuple[ParameterDatabase, CanonicalOrdering]:
    """Prepare a single ligand from a CIF file and inject it into a database.

    Routes through the unified path: the CIF ligand is converted to a SMILES
    (existing-bonds / geometry; never a CCD lookup), run through the SMILES ->
    params pipeline, and the resulting residue's atom names are graph-matched
    back to the CIF names so pose-build can place CIF coordinates.

    Args:
        cif_path: Path to the ligand CIF file.
        param_db: Base database (not modified); defaults to the tmol default.
        ph: Target pH for protonation.
        strict_atom_types: Fail on unknown atom-type element mappings.
        strict_ligands: If True (default), raise when no SMILES candidate fully
            matches the CIF heavy-atom names. Pass False to accept a best-effort
            preparation.
        res_name: Optional residue name override.
        sample_proton_chi: Whether to emit proton-chi samples.

    Returns:
        A ``(ParameterDatabase, CanonicalOrdering)`` with the ligand injected.
    """
    if param_db is None:
        param_db = ParameterDatabase.get_default()

    lig = _ligand_info_from_cif(cif_path, res_name)
    prep = _prepare_ligand_via_smiles(
        lig, ph=ph, sample_proton_chi=sample_proton_chi, strict=strict_ligands
    )
    param_db = inject_ligand_preparations(
        param_db, [prep], strict_atom_types=strict_atom_types
    )
    return param_db, rebuild_canonical_ordering(param_db)


def prepare_ligand_from_smiles(
    smiles: str,
    *,
    param_db: Optional[ParameterDatabase] = None,
    ph: float = 7.4,
    strict_atom_types: bool = False,
    res_name: str | None = None,
    protonate: bool = True,
    sample_proton_chi: bool = True,
    conformer_search: bool = True,
) -> tuple[ParameterDatabase, CanonicalOrdering]:
    """Prepare a single ligand from a SMILES string and inject it into a database.

    Follows the canonical ligand-prep protocol: Dimorphite-DL pKa-protonates
    the SMILES at ``ph``, OpenBabel generates a 3D mol2 with MMFF94 partial
    charges, and that mol2 is read verbatim (atom names, coordinates, charges,
    and bond orders preserved). The MMFF94 charges flow through untouched —
    there is no biotite atom-array round-trip or MMFF recompute. This path
    requires the optional ``openbabel`` package.

    Args:
        protonate: When ``True`` (default) Dimorphite protonates ``smiles``
            first; set ``False`` to pin an already-protonated SMILES verbatim.
        conformer_search: When ``True`` (default) run a rotor conformer search
            during 3D mol2 generation (matching the reference pipeline); set
            ``False`` for faster single-conformer generation.
    """
    if param_db is None:
        param_db = ParameterDatabase.get_default()

    lig = nonstandard_residue_info_from_smiles_via_mol2(
        smiles,
        res_name=res_name,
        ph=ph,
        protonate=protonate,
        conformer_search=conformer_search,
    )
    prep = prepare_single_ligand(lig, sample_proton_chi=sample_proton_chi)
    param_db = inject_ligand_preparations(
        param_db, [prep], strict_atom_types=strict_atom_types
    )
    return param_db, rebuild_canonical_ordering(param_db)


def prepare_ligand_from_mol2(
    mol2_path: str,
    *,
    param_db: Optional[ParameterDatabase] = None,
    strict_atom_types: bool = False,
    res_name: str | None = None,
    sample_proton_chi: bool = True,
) -> tuple[ParameterDatabase, CanonicalOrdering]:
    """Prepare a single ligand from a Tripos mol2 file and inject it.

    Reads atom names, coordinates, bond orders, and MMFF94 partial charges
    verbatim from the mol2 (no SMILES or OpenBabel 3D generation step).

    Args:
        mol2_path: Path to the ligand mol2 file.
        param_db: Base database (not modified); defaults to the tmol default.
        strict_atom_types: Fail on unknown atom-type element mappings.
        res_name: Optional residue name override.
        sample_proton_chi: Whether to emit proton-chi samples.

    Returns:
        A ``(ParameterDatabase, CanonicalOrdering)`` with the ligand injected.
    """
    from tmol.ligand.detect import nonstandard_residue_info_from_mol2

    if param_db is None:
        param_db = ParameterDatabase.get_default()

    lig = nonstandard_residue_info_from_mol2(mol2_path, res_name=res_name)
    prep = prepare_single_ligand(lig, sample_proton_chi=sample_proton_chi)
    param_db = inject_ligand_preparations(
        param_db, [prep], strict_atom_types=strict_atom_types
    )
    return param_db, rebuild_canonical_ordering(param_db)
