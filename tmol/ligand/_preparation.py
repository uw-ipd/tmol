"""Ligand preparation implementation for tmol.

This module contains the concrete preparation pipeline implementation.
`tmol.ligand.__init__` re-exports the public API from here.
"""

import itertools
import logging
from typing import Optional

import biotite.structure as struc
import numpy as np
from rdkit import Chem

from tmol.database import ParameterDatabase
from tmol.io import CanonicalOrdering
from tmol.ligand._atom_typing import AtomTypeAssignment, assign_tmol_atom_types
from tmol.ligand._detect import (
    NonStandardResidueInfo,
    _METAL_SYMBOLS,
    detect_nonstandard_residues,
    nonstandard_residue_info_from_smiles_via_mol2,
)
from tmol.ligand._registry import (
    LigandPreparation,
    _build_cartbonded_params,
    inject_ligand_preparations,
    rebuild_canonical_ordering,
)
from tmol.ligand._mol3d import authoritative_charges_by_index
from tmol.ligand._residue_builder import build_residue_type
from tmol.ligand._structure_to_smiles import ligand_smiles_from_atom_array
from tmol.ligand._rdkit_mol import ligand_atom_array_to_rdkit_mol

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


def _partial_charges_for_residue(param_db, residue_name: str) -> dict[str, float]:
    return {
        parameter.atom: parameter.charge
        for parameter in param_db.scoring.elec.atom_charge_parameters
        if parameter.res == residue_name
    }


def _assert_fragment_names_available(param_db, fragment_preparations) -> None:
    existing = {residue.name: residue for residue in param_db.chemical.residues}
    for prep in fragment_preparations:
        previous = existing.get(prep.residue_type.name)
        if previous is None:
            continue
        previous_charges = _partial_charges_for_residue(
            param_db, prep.residue_type.name
        )
        previous_cartbonded = param_db.scoring.cartbonded.residue_params.get(
            prep.residue_type.name
        )
        if (
            previous != prep.residue_type
            or previous_charges != prep.partial_charges
            or previous_cartbonded != prep.cartbonded_params
        ):
            raise LigandPreparationError(
                f"{prep.residue_type.name}: generated fragment name already "
                "exists with different chemistry"
            )


def _rename_atoms_to_cif(
    pipeline_mol: Chem.Mol,
    atom_types: list[AtomTypeAssignment],
    ligand_info: NonStandardResidueInfo,
    source_atom_order: Optional[tuple[int, ...]],
) -> list[AtomTypeAssignment]:
    """Rename pipeline heavy atoms to CIF names via the source-atom-order map.

    source_atom_order[k] is the CIF atom-array index of the k-th heavy atom in
    pipeline (mol2) order, carried through the SMILES->mol2 round trip by atom-map
    numbers. Hydrogens keep pipeline names. Unchanged if the map is absent or its
    size disagrees with the pipeline heavy-atom count.
    """
    if not source_atom_order:
        return atom_types
    heavy_pipeline_idx = [
        a.GetIdx() for a in pipeline_mol.GetAtoms() if a.GetAtomicNum() != 1
    ]
    if len(heavy_pipeline_idx) != len(source_atom_order):
        logger.warning(
            "Atom-order map size %d != pipeline heavy count %d for %s; "
            "keeping pipeline names",
            len(source_atom_order),
            len(heavy_pipeline_idx),
            ligand_info.res_name,
        )
        return atom_types

    names = ligand_info.atom_names
    pipeline_to_cif = {}
    for pos, pipeline_idx in enumerate(heavy_pipeline_idx):
        cif_idx = source_atom_order[pos]
        if 0 <= cif_idx < len(names):
            pipeline_to_cif[pipeline_idx] = names[cif_idx]

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
    :func:`tmol.ligand._detect.nonstandard_residue_info_from_smiles_via_mol2`.

    Charges are mapped onto atoms by stable RDKit index (source atom order),
    so they are independent of the atom renaming below and never recomputed.

    Returns a :class:`LigandPreparation` -- the same struct
    :func:`tmol.ligand._params_file.load_params_file` produces for each residue
    defined in a ``.tmol`` file, so the AtomArray-driven path and the params-file
    path converge on a single abstraction that
    :func:`inject_ligand_preparations` consumes.

    Args:
        ligand_info: A SMILES-derived ligand (``skip_protonation=True`` with
            authoritative ``partial_charges``). Raw CIF/atom-array ligands must
            be routed through :func:`prepare_ligands` / :func:`prepare_ligand_from_cif`.
        sample_proton_chi: Whether to emit proton-chi samples.
        name_source: Optional ligand whose atom names the prepared residue should
            adopt (mapped to the prepared heavy atoms via the atom-order map). On
            the unified CIF path this is the original CIF ligand. Defaults to
            ``ligand_info``.

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

    from tmol.ligand._atom_typing import sanitize_tolerant
    from tmol.ligand._generated_geometry import correct_generated_geometry

    protonated = ligand_atom_array_to_rdkit_mol(ligand_info, keep_hydrogens=True)
    sanitize_tolerant(protonated)

    # The generated conformer has correct chemistry but wrong local geometry.
    # Repair it before computing icoors
    for applied in correct_generated_geometry(protonated):
        logger.info("%s: %s", ligand_info.res_name, applied)

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
        protonated,
        atom_types,
        name_source if name_source is not None else ligand_info,
        ligand_info.source_atom_order,
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
) -> LigandPreparation:
    """Prepare one ligand through the unified CIF -> SMILES -> params path.

    Derives a SMILES from the ligand's atom array (from its explicit bond table;
    never geometry perception, never a CCD lookup) and runs it through the
    SMILES -> mol2 -> params pipeline. The prepared residue's heavy-atom names
    should cover the original ligand's heavy atoms; if they don't, the
    preparation is still returned best-effort.

    Args:
        ligand_info: The detected (CIF/atom-array) ligand.
        ph: Target pH for protonation (applied in the SMILES -> mol2 step).
        sample_proton_chi: Whether to emit proton-chi samples.

    Returns:
        The :class:`LigandPreparation`.

    Raises:
        ValueError: If a SMILES could not be derived or prepared.
    """
    smiles = ligand_smiles_from_atom_array(
        ligand_info.atom_array, res_name=ligand_info.res_name, with_atom_map=True
    )

    try:
        smiles_info = nonstandard_residue_info_from_smiles_via_mol2(
            smiles, res_name=ligand_info.res_name, ph=ph
        )
        prep = prepare_single_ligand(
            smiles_info,
            sample_proton_chi=sample_proton_chi,
            name_source=ligand_info,
        )
    except Exception as err:
        raise ValueError(
            f"{ligand_info.res_name}: failed to prepare ligand via SMILES {smiles!r}"
        ) from err

    if not _residue_covers_cif_heavy_atoms(prep, _cif_heavy_atom_names(ligand_info)):
        logger.warning(
            "Prepared SMILES for %s did not cover all CIF heavy-atom names; "
            "using best-effort preparation",
            ligand_info.res_name,
        )
    return prep


def _supported_elements(param_db: ParameterDatabase) -> set[str]:
    """Element symbols tmol has atom types for."""
    return {
        at.element.strip().capitalize()
        for at in param_db.chemical.atom_types
        if at.element and at.element.strip()
    }


def _ligand_unsupported_reason(
    lig: NonStandardResidueInfo, supported_elements: set[str]
) -> str | None:
    """Why this ligand cannot be prepared, or None if it can."""
    metals_present = sorted(
        {
            e.strip().capitalize()
            for e in lig.elements
            if e.strip().capitalize() in _METAL_SYMBOLS
        }
    )
    if metals_present:
        return (
            f"{lig.res_name}: ligands containing metal atoms "
            f"({metals_present}) are not supported"
        )

    unsupported = sorted(
        {
            e.strip().capitalize()
            for e in lig.elements
            if e and e.strip() and e.strip().capitalize() not in supported_elements
        }
    )
    if unsupported:
        return (
            f"{lig.res_name}: ligands containing unsupported element(s) "
            f"({', '.join(unsupported)}) are not supported; tmol has no atom "
            "types for them"
        )

    return None


def prepare_ligands(  # noqa: C901
    atom_array: struc.AtomArray,
    param_db: Optional[ParameterDatabase] = None,
    ph: float = 7.4,
    strict_atom_types: bool = False,
    params_files: list[str] | None = None,
    params_output: str | None = None,
    sample_proton_chi: bool = True,
    strict_ligands: bool = True,
    return_fragment_definitions: bool = False,
) -> tuple:
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
        params_files: Optional list of tmol YAML params file paths to
            inject before detection. Residues defined in these files
            skip the RDKit/OB preparation pipeline.
        params_output: Optional path to write all prepared ligand data
            to a tmol YAML params file for later reuse.
        sample_proton_chi: Whether to emit PROTON_CHI samples in the
            built residue type.
        strict_ligands: If True (default), raise :class:`LigandPreparationError`
            when a detected non-standard residue is skipped (metal-containing or
            covalently linked) or fails preparation, instead of silently
            dropping it. If False, such residues are logged as warnings and
            skipped, leaving them to be filtered out during pose construction.
        return_fragment_definitions: Internal/context-building option. If True,
            include definitions derived from ``tmol_fragment_id`` annotations
            as the third return value.

    Returns:
        A (ParameterDatabase, CanonicalOrdering) tuple. When
        ``return_fragment_definitions`` is true, a third element containing
        the structure-independent ligand fragment definitions is returned. The returned
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

    from tmol.ligand._fragmentation import (
        FRAGMENT_ID_ANNOTATION,
        build_ligand_fragment_definition,
        fragment_ids_from_atom_array,
    )

    fragment_layouts_by_name = {}
    first_residue_by_name: dict[str, struc.AtomArray] = {}
    starts = struc.get_residue_starts(atom_array)
    ends = np.append(starts[1:], atom_array.array_length())
    for start, end in zip(starts, ends):
        residue = atom_array[start:end]
        fragment_ids = fragment_ids_from_atom_array(residue)
        layout = (
            None
            if fragment_ids is None
            else tuple(sorted(zip(map(str, residue.atom_name), map(int, fragment_ids))))
        )
        ligand_name = str(residue.res_name[0])
        if (
            ligand_name in fragment_layouts_by_name
            and fragment_layouts_by_name[ligand_name] != layout
        ):
            raise LigandPreparationError(
                f"{ligand_name}: all residues with the same name must use the "
                f"same {FRAGMENT_ID_ANNOTATION} annotation"
            )
        fragment_layouts_by_name[ligand_name] = layout
        first_residue_by_name.setdefault(ligand_name, residue)

    params_preparations: list[LigandPreparation] = []
    if params_files:
        from tmol.ligand._params_file import load_params_file

        for params_file in params_files:
            params_preparations.extend(load_params_file(params_file))
        param_db = inject_ligand_preparations(
            param_db,
            params_preparations,
            strict_atom_types=strict_atom_types,
        )

    fragment_source_preparations = {
        prep.residue_type.name: prep for prep in params_preparations
    }
    existing_residues = {
        residue.name: residue for residue in param_db.chemical.residues
    }
    for ligand_name, layout in fragment_layouts_by_name.items():
        if layout is None or ligand_name in fragment_source_preparations:
            continue
        restype = existing_residues.get(ligand_name)
        if restype is None:
            continue
        cartbonded_params = param_db.scoring.cartbonded.residue_params.get(ligand_name)
        if cartbonded_params is None:
            cartbonded_params = _build_cartbonded_params(restype)
        fragment_source_preparations[ligand_name] = LigandPreparation(
            residue_type=restype,
            partial_charges=_partial_charges_for_residue(param_db, ligand_name),
            cartbonded_params=cartbonded_params,
            atom_type_elements=None,
        )

    canonical_ordering = rebuild_canonical_ordering(param_db)
    fragment_definitions_by_name = {}

    def add_fragment_definition(ligand_name, definition):
        if ligand_name in fragment_definitions_by_name or definition is None:
            return False
        fragment_definitions_by_name[ligand_name] = definition
        return True

    if fragment_source_preparations:
        for prep in fragment_source_preparations.values():
            source = first_residue_by_name.get(prep.residue_type.name)
            if source is None:
                continue
            try:
                definition = build_ligand_fragment_definition(prep, source)
            except Exception as err:
                raise LigandPreparationError(
                    f"{prep.residue_type.name}: invalid "
                    f"{FRAGMENT_ID_ANNOTATION} annotation ({err})"
                ) from err
            add_fragment_definition(prep.residue_type.name, definition)
        if fragment_definitions_by_name:
            fragment_preparations = [
                fragment_prep
                for definition in fragment_definitions_by_name.values()
                for fragment_prep in definition.fragment_preparations
            ]
            _assert_fragment_names_available(param_db, fragment_preparations)
            param_db = inject_ligand_preparations(
                param_db,
                fragment_preparations,
                strict_atom_types=strict_atom_types,
            )
            canonical_ordering = rebuild_canonical_ordering(param_db)

    ligands = detect_nonstandard_residues(atom_array, canonical_ordering)

    if not ligands:
        logger.info("No non-standard residues detected")
        if return_fragment_definitions:
            return (
                param_db,
                canonical_ordering,
                tuple(fragment_definitions_by_name.values()),
            )
        return param_db, canonical_ordering

    logger.info("Found %d non-standard residue type(s) to prepare", len(ligands))

    supported_elements = _supported_elements(param_db)

    preparations: list[LigandPreparation] = []
    prepared_ligands: list[tuple[NonStandardResidueInfo, LigandPreparation]] = []
    for lig in ligands:
        reason = _ligand_unsupported_reason(lig, supported_elements)
        if reason:
            _skip_or_raise(strict_ligands, reason)
            continue

        logger.info("Preparing %s (CCD type: %s)", lig.res_name, lig.ccd_type)
        try:
            prep = _prepare_ligand_via_smiles(
                lig, ph=ph, sample_proton_chi=sample_proton_chi
            )
            from tmol.ligand._polymer import specialize_component_preparation

            prep, profile = specialize_component_preparation(prep, lig, param_db)
            logger.info("Prepared %s through the %s path", lig.res_name, profile.kind)
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
        preparations.append(prep)
        prepared_ligands.append((lig, prep))

    if prepared_ligands:
        for lig, prep in prepared_ligands:
            try:
                definition = build_ligand_fragment_definition(prep, lig.atom_array)
            except Exception as err:
                raise LigandPreparationError(
                    f"{lig.res_name}: invalid {FRAGMENT_ID_ANNOTATION} annotation "
                    f"({err})"
                ) from err
            if add_fragment_definition(lig.res_name, definition):
                preparations.extend(definition.fragment_preparations)

    if strict_ligands and not preparations:
        raise LigandPreparationError(
            "All "
            f"{len(ligands)} detected non-standard residue(s) "
            f"({', '.join(sorted({lig.res_name for lig in ligands}))}) were "
            "skipped; none could be prepared. Pass strict_ligands=False to "
            "continue with these residues dropped."
        )

    if preparations:
        fragment_preparations = [
            fragment_prep
            for definition in fragment_definitions_by_name.values()
            for fragment_prep in definition.fragment_preparations
        ]
        _assert_fragment_names_available(param_db, fragment_preparations)
        param_db = inject_ligand_preparations(
            param_db, preparations, strict_atom_types=strict_atom_types
        )
        canonical_ordering = rebuild_canonical_ordering(param_db)

        if params_output:
            from tmol.ligand._params_io import write_params_file

            # Fragment residue types are an in-memory representation in this
            # first API version. Persist only the fully prepared source ligand.
            write_params_file(
                [prep for _, prep in prepared_ligands],
                params_output,
                format="tmol",
            )
            logger.info("Wrote params to %s", params_output)

    if return_fragment_definitions:
        return (
            param_db,
            canonical_ordering,
            tuple(fragment_definitions_by_name.values()),
        )
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

    from tmol.ligand._detect import get_chem_comp_type

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


def _inject_single(
    prep: LigandPreparation,
    param_db: Optional[ParameterDatabase],
    strict_atom_types: bool,
) -> tuple[ParameterDatabase, CanonicalOrdering]:
    """Inject one prepared ligand and return the new ``(db, canonical_ordering)``.

    Shared tail of the single-ligand ``prepare_ligand_from_*`` entry points:
    resolve the default database, inject the preparation, and rebuild the
    canonical ordering for the extended database.
    """
    if param_db is None:
        param_db = ParameterDatabase.get_default()
    param_db = inject_ligand_preparations(
        param_db, [prep], strict_atom_types=strict_atom_types
    )
    return param_db, rebuild_canonical_ordering(param_db)


def prepare_ligand_from_cif(
    cif_path: str,
    *,
    param_db: Optional[ParameterDatabase] = None,
    ph: float = 7.4,
    strict_atom_types: bool = False,
    res_name: str | None = None,
    sample_proton_chi: bool = True,
) -> tuple[ParameterDatabase, CanonicalOrdering]:
    """Prepare a single ligand from a CIF file and inject it into a database.

    Runs the same full pipeline as :func:`prepare_ligand_from_smiles`; the only
    CIF-specific step is the front end. A SMILES is derived from the CIF ligand's
    explicit bond table (never geometry perception, never a CCD lookup) and run
    through the SMILES -> mol2 -> params path (protonation, 3D conformer, MMFF94
    charges). The prepared residue's heavy-atom names are then mapped back to the
    CIF atom names via the atom-order map carried through the round-trip.

    Args:
        cif_path: Path to the ligand CIF file.
        param_db: Base database (not modified); defaults to the tmol default.
        ph: Target pH for protonation.
        strict_atom_types: Fail on unknown atom-type element mappings.
        res_name: Optional residue name override.
        sample_proton_chi: Whether to emit proton-chi samples.

    Returns:
        A ``(ParameterDatabase, CanonicalOrdering)`` with the ligand injected.
    """
    lig = _ligand_info_from_cif(cif_path, res_name)
    prep = _prepare_ligand_via_smiles(lig, ph=ph, sample_proton_chi=sample_proton_chi)
    return _inject_single(prep, param_db, strict_atom_types)


def prepare_ligand_from_smiles(
    smiles: str,
    *,
    param_db: Optional[ParameterDatabase] = None,
    ph: float = 7.4,
    strict_atom_types: bool = False,
    res_name: str | None = None,
    protonate: bool = True,
    sample_proton_chi: bool = True,
    seed: int | None = None,
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
        seed: Fixed RNG seed for reproducible 3D coordinates; ``None`` is random.
    """
    lig = nonstandard_residue_info_from_smiles_via_mol2(
        smiles,
        res_name=res_name,
        ph=ph,
        protonate=protonate,
        seed=seed,
    )
    prep = prepare_single_ligand(lig, sample_proton_chi=sample_proton_chi)
    return _inject_single(prep, param_db, strict_atom_types)


def unused_ligand_name(taken) -> str:
    """First "LGn" name not already present in ``taken``."""
    for i in itertools.count(1):
        name = f"LG{i}"
        if name not in taken:
            return name


def prepare_ligands_from_smiles(
    smiles,
    *,
    param_db: Optional[ParameterDatabase] = None,
    ph: float = 7.4,
    strict_atom_types: bool = False,
    protonate: bool = True,
    sample_proton_chi: bool = True,
    seed: int | None = None,
) -> tuple[ParameterDatabase, dict]:
    """Prepare one residue type per SMILES, naming them LG1, LG2, ...

    Returns the extended database and a {smiles: residue name} mapping.
    """
    if param_db is None:
        param_db = ParameterDatabase.get_default()
    taken = {residue.name for residue in param_db.chemical.residues}
    names = {}
    for smi in smiles:
        if smi in names:
            continue
        name = unused_ligand_name(taken)
        taken.add(name)
        param_db, _ = prepare_ligand_from_smiles(
            smi,
            param_db=param_db,
            ph=ph,
            strict_atom_types=strict_atom_types,
            res_name=name,
            protonate=protonate,
            sample_proton_chi=sample_proton_chi,
            seed=seed,
        )
        names[smi] = name
    return param_db, names


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
    from tmol.ligand._detect import nonstandard_residue_info_from_mol2

    lig = nonstandard_residue_info_from_mol2(mol2_path, res_name=res_name)
    prep = prepare_single_ligand(lig, sample_proton_chi=sample_proton_chi)
    return _inject_single(prep, param_db, strict_atom_types)
