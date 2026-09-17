"""Ligand preparation implementation for tmol.

This module contains the concrete preparation pipeline implementation.
`tmol.ligand.__init__` re-exports the public API from here.
"""

import itertools
import logging
from dataclasses import replace
from typing import Optional

import attr

import biotite.structure as struc
import numpy as np
from rdkit import Chem

from tmol.database import ParameterDatabase
from tmol.database.chemical import AtomAlias
from tmol.io import CanonicalOrdering
from tmol.ligand._atom_typing import AtomTypeAssignment, assign_tmol_atom_types
from tmol.ligand._detect import (
    NonStandardResidueInfo,
    _METAL_SYMBOLS,
    detect_nonstandard_residues,
    is_polymer_linking_component_type,
    with_resolved_coordinates,
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
from tmol.ligand._polymer_profile import na_base_reference
from tmol.ligand._registry import collect_new_atom_types
from tmol.ligand._rotamer_reference import _cached_profiles

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


def _without_unprepared_partners(atom_array, chemistry, strict_ligands):
    """The input with covalent bonds to unprepared residues removed, and their names.

    A residue the preparation loop skipped is still covalently bonded here, and
    a conjugate model refuses to guess connectivity for one it has no definition
    for. Strict preparation owns that refusal in its own terms rather than
    letting it escape as a bare ValueError from inside the model builder;
    strict_ligands=False means "drop what you cannot prepare", so there the link
    is cut and the rest of the input still parameterizes.
    """
    from tmol.ligand._conjugate_model import (
        unprepared_conjugate_partners,
        without_cross_bonds_to,
    )

    if atom_array.bonds is None:
        return atom_array, frozenset()
    unprepared = unprepared_conjugate_partners(atom_array, chemistry)
    if not unprepared:
        return atom_array, frozenset()
    listed = ", ".join(sorted(unprepared))
    if strict_ligands:
        raise LigandPreparationError(
            f"{listed}: covalently bonded but not prepared, so the connection "
            "cannot be parameterized. Pass strict_ligands=False to cut the bond "
            "and skip it with a warning, or supply prebuilt params via "
            "ligand_params_files"
        )
    logger.warning(
        "Cutting covalent bonds to unprepared residue(s) %s; their connections "
        "are not parameterized",
        listed,
    )
    return without_cross_bonds_to(atom_array, unprepared), frozenset(unprepared)


def _prepare_conjugate_params(atom_array, param_db, ph, strict_ligands=True):
    from tmol.ligand._conjugate_context import contextual_conjugate_types
    from tmol.ligand._local_conjugate_params import (
        ConjugateContextConflict,
        generate_conjugate_parameters,
        install_conjugate_parameters,
    )

    if atom_array.bonds is None:
        return param_db, (), ()
    atom_array, _ = _without_unprepared_partners(
        atom_array, param_db.chemical, strict_ligands
    )
    models = None
    while True:
        try:
            result = generate_conjugate_parameters(
                atom_array,
                param_db,
                ph=ph,
                existing=param_db.scoring.cartbonded.connection_params,
                _models=models,
            )
            break
        except ConjugateContextConflict as error:
            extended, models = contextual_conjugate_types(
                atom_array, param_db, error.residue_name
            )
            if extended is param_db:
                raise
            param_db = extended
    if not result.connections:
        return param_db, (), ()
    param_db = install_conjugate_parameters(param_db, result)
    replacements = tuple(
        LigandPreparation(
            residue_type=row.residue_type,
            partial_charges=row.partial_charges,
            cartbonded_params=row.cartbonded_params,
            baseline_sha256=(
                None if row.residue_type.conjugation_context else row.baseline_sha256
            ),
        )
        for row in result.residues
    )
    return param_db, result.connections, replacements


def _write_preparations(preparations, replacements, records, param_db, path):
    from tmol.ligand._params_io import write_params_file
    from tmol.ligand._parameter_replacements import _baseline_charges, _local_identity

    preparations = [*preparations, *replacements]
    if records and not preparations:
        # A supplied database may need only a new connection record. Carry it
        # on an unchanged, guarded endpoint so loading still checks its baseline.
        rt = next(
            r for r in param_db.chemical.residues if r.name == records[0].block_type1
        )
        charges = _baseline_charges(
            {
                (p.res, p.atom): p.charge
                for p in param_db.scoring.elec.atom_charge_parameters
            },
            rt,
        )
        cart = param_db.scoring.cartbonded.residue_params
        bonded = cart.get(rt.name, cart.get(rt.base_name))
        preparations.append(
            LigandPreparation(
                residue_type=rt,
                partial_charges=charges,
                cartbonded_params=bonded,
                baseline_sha256=_local_identity(rt, charges, bonded),
            )
        )
    if preparations:
        first = preparations[0]
        preparations[0] = replace(
            first, connection_params=(*first.connection_params, *records)
        )
        write_params_file(preparations, path)
        logger.info("Wrote params to %s", path)


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
    name_source: Optional[NonStandardResidueInfo] = None,
    assign_ring_chis: bool = False,
    generate_heavy_chi_samples: bool = False,
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

    source = name_source if name_source is not None else ligand_info
    template = getattr(source.atom_array, "_custom_ccd_registry", {}).get(
        source.res_name
    )
    frame_excluded = frozenset()
    if (
        template is not None
        and "is_leaving_atom" in template.get_annotation_categories()
    ):
        frame_excluded = frozenset(template.atom_name[template.is_leaving_atom])
    restype = build_residue_type(
        protonated,
        ligand_info.res_name,
        atom_types,
        typing_state=typing_state,
        assign_ring_chis=assign_ring_chis,
        generate_heavy_chi_samples=generate_heavy_chi_samples,
        original_single_bonds=ligand_info.original_single_bonds,
        frame_excluded_atoms=frame_excluded,
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


def _with_sampling_reference(residue_type, profile, param_db, atom_type_elements):
    """Name the canonical residue whose torsion distributions this one borrows.

    A nucleotide borrows base torsion tables, which it is scored on. An amino
    acid borrows a rotamer library, which is read for sampling only and is
    chosen while its chi are defined, since the two are the same decision.
    """
    if profile.backbone_type in ("dna", "rna"):
        return attr.evolve(
            residue_type,
            na_base_reference=na_base_reference(
                residue_type, profile, param_db.chemical
            ),
        )
    # sampled chi are stored whole. What they enumerate depends on whether the
    #    residue is sampled alone or together with something bonded to it, which
    #    is not knowable here, so the packer bounds them when it builds rotamers.
    return residue_type


def prepare_polymer_residue(
    atom_array,
    canonical_ordering,
    param_db: Optional[ParameterDatabase] = None,
    *,
    ph: float = 7.4,
    profile=None,
    connection_atoms=None,
    use_ccd: bool = True,
    seed: int | None = None,
) -> LigandPreparation:
    """Prepare one polymer residue: cap it, run the ligand pipeline, restore the
    polymer connections.

    atom_array is the residue as it appears in a structure, with its polymer
    connections open; profile defaults to whichever backbone its atoms match.
    connection_atoms names the atoms that bond to neighbouring residues, which
    is what distinguishes a residue linked through its carbonyl from one linked
    through a sidechain carbon.
    """
    from tmol.ligand._polymer_builder import (
        add_capped_junction_parameters,
        backbone_renames,
        to_polymer_residue_type,
    )
    from tmol.ligand._terminus_patches import patch_charge_entries, terminus_patches
    from tmol.ligand._polymer_profile import (
        canonical_backbone_renames,
        complete_backbone_from_reference,
        cap_residue,
        profile_for_atom_array,
        _CANONICAL_RING_ATOM,
        ring_nitrogen_angle_atom,
        noncanonical_junction_substitutions,
        substituted_wildcard_rows,
    )

    if param_db is None:
        param_db = ParameterDatabase.get_default()
    # the backbone is read from the bonds, so say so before classifying: with
    #    no bond table every residue looks like a backbone we do not support.
    #    A one-atom residue has no bonds to read and is not the same thing.
    if atom_array.bonds is None or (
        atom_array.array_length() > 1 and atom_array.bonds.get_bond_count() == 0
    ):
        raise ValueError(
            "residue carries no bond table; read the structure with "
            "include_bonds=True so its chemistry can be derived"
        )
    if profile is None:
        profile = profile_for_atom_array(
            atom_array, connection_atoms, param_db.chemical
        )
    if profile is None:
        if connection_atoms and len(connection_atoms) == 1:
            known = next(iter(connection_atoms))
            raise LigandPreparationError(
                f"{atom_array.res_name[0]}: bonded to a neighbour only at "
                f"{known}, and its chemistry does not say where the "
                "backbone's other end is. Prepare it "
                "once with prepare_polymer_residue(..., connection_atoms="
                f'("{known}", <other end>)), save it with write_params_file, '
                "and pass that file as params_files"
            )
        raise LigandPreparationError(
            f"{atom_array.res_name[0]}: declared a polymer residue but its atoms "
            "match no supported backbone"
        )

    # CCD geometry uses source names before backbone renaming or cap donation.
    resolved = with_resolved_coordinates(
        atom_array, str(atom_array.res_name[0]), use_ccd
    )
    atom_array = atom_array if resolved is None else resolved

    # Scoring and reference completion use canonical backbone names. Preserve
    # source names as aliases so construction still accepts the original input.
    backbone_renames_to_canonical = canonical_backbone_renames(
        atom_array, profile, param_db.chemical, connection_atoms
    )
    atom_aliases = ()
    if backbone_renames_to_canonical:
        atom_array = atom_array.copy()
        for i, name in enumerate(atom_array.atom_name):
            atom_array.atom_name[i] = backbone_renames_to_canonical.get(str(name), name)
        atom_aliases = tuple(
            AtomAlias(name=canonical, alt_name=actual)
            for actual, canonical in sorted(backbone_renames_to_canonical.items())
        )

    # a residue seen only at a chain end is missing what that end does not
    #    carry; put it back before capping, once the names are settled
    atom_array = complete_backbone_from_reference(atom_array, profile, param_db)

    # an atom the structure did not resolve arrives at NaN; the molecule has to
    #    be readable before it can be typed
    resolved = with_resolved_coordinates(
        atom_array, str(atom_array.res_name[0]), use_ccd
    )
    if resolved is None:
        unresolved = sorted(
            str(n)
            for n, missing in zip(
                atom_array.atom_name, np.isnan(atom_array.coord).any(axis=-1)
            )
            if missing
        )
        raise LigandPreparationError(
            f"{atom_array.res_name[0]}: {', '.join(unresolved)} are declared "
            "but unresolved, and no component definition can place them. "
            "Preparing the residue without them would define a different "
            "molecule, and placing them arbitrarily would assert a "
            "stereochemistry nothing observed. Supply a structure that "
            "resolves them, or prebuilt params via params_files"
        )
    atom_array = resolved

    capped, cap_names = cap_residue(atom_array, profile)
    # the capped molecule is built here, so its completeness is not in doubt
    detected = detect_nonstandard_residues(capped, canonical_ordering)
    if not detected:
        raise LigandPreparationError(
            f"{atom_array.res_name[0]}: capped residue was not detected as "
            "non-standard; is its name already in the database?"
        )
    prep = _prepare_ligand_via_smiles(
        detected[0],
        ph=ph,
        seed=seed,
        # a sidechain ring rotates the way proline's does; a nucleotide's rings
        #    are its sugar and its base, whose torsions the na_torsion term owns
        assign_ring_chis=profile.polymer_type == "amino_acid",
        # heavy chi are sampled to fill in what a borrowed rotamer library does
        #    not cover; a nucleotide has its own sampler and needs none
        generate_heavy_chi_samples=profile.polymer_type == "amino_acid",
    )

    # only a profile that transplants backbone icoors needs a donor residue
    reference = (
        {r.name: r for r in param_db.chemical.residues}[profile.reference_restype]
        if profile.reference_restype is not None
        else None
    )
    default_elements = {at.name: at.element for at in param_db.chemical.atom_types}
    elements = {
        a.name: (prep.atom_type_elements or {}).get(
            a.atom_type, default_elements.get(a.atom_type, "")
        )
        for a in prep.residue_type.atoms
    }
    coords = _ideal_coords_by_name(prep.residue_type)
    library_names = {m.residue_name for m in param_db.scoring.dun.dun_lookup}
    atom_type_index = {a.name: a for a in param_db.chemical.atom_types}
    atom_type_index.update(
        {
            a.name: a
            for a in collect_new_atom_types(
                param_db.chemical,
                prep.residue_type,
                atom_type_elements=prep.atom_type_elements,
            )
        }
    )
    residue_type = to_polymer_residue_type(
        prep.residue_type,
        coords,
        profile,
        reference,
        elements,
        cap_names,
        rotamer_references=_cached_profiles(param_db.chemical, library_names),
        atom_type_index=atom_type_index,
    )

    if atom_aliases:
        residue_type = attr.evolve(
            residue_type, atom_aliases=residue_type.atom_aliases + atom_aliases
        )
    residue_type = _with_sampling_reference(
        residue_type, profile, param_db, prep.atom_type_elements
    )

    kept = {a.name for a in residue_type.atoms}
    renamed, _dropped = backbone_renames(
        prep.residue_type, profile, elements, cap_names
    )
    charges = {
        renamed.get(name, name): charge
        for name, charge in prep.partial_charges.items()
        if renamed.get(name, name) in kept
    }
    # measure cartbonded off the polymer residue's own icoors, so the backbone
    #    terms follow the transplanted geometry rather than the conformer's
    kept_coords = {
        name: xyz
        for name, xyz in _ideal_coords_by_name(residue_type).items()
        if name in kept
    }
    cartbonded_params = _build_cartbonded_params(residue_type, coords=kept_coords)
    # Keep the existing ring-specific rows for continuing backbones.
    substitutions = []
    if profile.name != "cap":
        ring_atom = ring_nitrogen_angle_atom(atom_array, connection_atoms)
        if ring_atom is not None and ring_atom in kept:
            substitutions.append(((_CANONICAL_RING_ATOM,), ring_atom))
    # Both continuing backbones and terminal caps need these chemical roles.
    # Scoring resolves either side of a junction without renaming source atoms.
    junctions = noncanonical_junction_substitutions(
        residue_type, atom_type_index, atom_array
    )
    references = {
        actual: canonical
        for mapping in junctions
        for canonical, actual in mapping.items()
        if actual != canonical
    }
    if references:
        residue_type = attr.evolve(
            residue_type,
            atoms=tuple(
                attr.evolve(atom, cartbonded_reference=references.get(atom.name))
                for atom in residue_type.atoms
            ),
        )

    cartbonded_params = add_capped_junction_parameters(
        cartbonded_params,
        prep.cartbonded_params,
        profile,
        kept,
        renamed,
        references,
        elements,
    )

    for substitution in substitutions:
        rows = substituted_wildcard_rows(
            param_db.scoring.cartbonded, *substitution, kept
        )
        cartbonded_params = attr.evolve(
            cartbonded_params,
            length_parameters=cartbonded_params.length_parameters
            + tuple(attr.evolve(p, atm1=a[0], atm2=a[1]) for a, p in rows["length"]),
            angle_parameters=cartbonded_params.angle_parameters
            + tuple(
                attr.evolve(p, atm1=a[0], atm2=a[1], atm3=a[2])
                for a, p in rows["angle"]
            ),
            torsion_parameters=cartbonded_params.torsion_parameters
            + tuple(
                attr.evolve(p, atm1=a[0], atm2=a[1], atm3=a[2], atm4=a[3])
                for a, p in rows["torsion"]
            ),
            improper_parameters=cartbonded_params.improper_parameters
            + tuple(
                attr.evolve(p, atm1=a[0], atm2=a[1], atm3=a[2], atm4=a[3])
                for a, p in rows["improper"]
            ),
        )

    # the database's termini patches are written for an alpha backbone; any
    #    other one brings its own, named around the atoms it already has
    generated, variant_charges = (), None
    if profile.backbone_type in ("nonstandard_aa", "nonstandard_na"):
        generated = terminus_patches(
            param_db.chemical, residue_type, profile, atom_array, ph, charges
        )
        if generated:
            patched = param_db.chemical.with_added_residues(
                [residue_type],
                atom_types=(
                    *param_db.chemical.atom_types,
                    *collect_new_atom_types(
                        param_db.chemical,
                        residue_type,
                        atom_type_elements=prep.atom_type_elements,
                    ),
                ),
                variants=[patch for patch, _t, _c in generated],
            )
            variant_charges = patch_charge_entries(
                param_db, patched, residue_type, generated
            )
            # a patch whose terminal charges could not be measured would leave
            #    the residue unscoreable wherever it applied, so it does not ship
            generated = _patches_with_charges(
                residue_type, generated, variant_charges, patched
            )

    return LigandPreparation(
        residue_type=residue_type,
        partial_charges=charges,
        cartbonded_params=cartbonded_params,
        atom_type_elements=prep.atom_type_elements,
        adds_patches=tuple(patch for patch, _t, _c in generated),
        variant_partial_charges=variant_charges or None,
    )


def _patches_with_charges(residue_type, generated, variant_charges, patched):
    """The generated patches that have charges to go with them.

    A patch without them would leave the residue unscoreable wherever it
    applied, so it does not ship; which of the two ways it failed is worth
    saying, since one means the generated pattern is wrong.
    """
    built = {r.name for r in patched.residues}
    kept = []
    for entry in generated:
        variant = f"{residue_type.name}:{entry[0].display_name}"
        if variant in variant_charges:
            kept.append(entry)
        elif variant not in built:
            logger.warning(
                "%s %s patch: applying it produced no variant. Its pattern did "
                "not match the residue it was generated for, so the residue "
                "cannot sit at that end.",
                residue_type.name,
                entry[0].display_name,
            )
        else:
            logger.warning(
                "%s %s patch: charges could not be mapped onto it. The terminal "
                "molecule did not match the patched residue atom for atom, so "
                "the residue cannot sit at that end.",
                residue_type.name,
                entry[0].display_name,
            )
    return kept


def _ideal_coords_by_name(residue_type) -> dict:
    """Conformer positions by atom name, rebuilt from the residue's icoors."""
    import cattr
    from tmol.chemical._restypes import RefinedResidueType

    refined = cattr.structure(cattr.unstructure(residue_type), RefinedResidueType)
    xyz = refined.compute_ideal_coords()
    return {ic.name: np.asarray(xyz[i]) for i, ic in enumerate(refined.icoors)}


def _cif_heavy_atom_names(ligand_info: NonStandardResidueInfo) -> set[str]:
    """Heavy-atom names of the (CIF) ligand, used to verify name matching."""
    return {
        name
        for name, element in zip(ligand_info.atom_names, ligand_info.elements)
        if str(element).strip().upper() not in ("H", "D")
    }


def _missing_cif_heavy_atom_names(
    prep: LigandPreparation, cif_heavy_names: set[str]
) -> set[str]:
    """Return authored heavy-atom names absent from the prepared residue."""
    if not cif_heavy_names:
        return set()
    elements = prep.atom_type_elements or {}
    restype_heavy: set[str] = set()
    for atom in prep.residue_type.atoms:
        element = elements.get(atom.atom_type)
        if element is not None and element.upper() == "H":
            continue
        restype_heavy.add(atom.name)
    return cif_heavy_names - restype_heavy


def _residue_covers_cif_heavy_atoms(
    prep: LigandPreparation, cif_heavy_names: set[str]
) -> bool:
    """Return True if the prepared residue carries every CIF heavy-atom name.

    When the SMILES-derived residue's heavy-atom names are a superset of the
    CIF ligand's heavy-atom names, pose-build can place every CIF heavy-atom
    coordinate by ``(res_name, atom_name)`` match.
    """
    return not _missing_cif_heavy_atom_names(prep, cif_heavy_names)


def _prepare_ligand_via_smiles(
    ligand_info: NonStandardResidueInfo,
    *,
    ph: float,
    protonate: bool = True,
    seed: int | None = None,
    assign_ring_chis: bool = False,
    generate_heavy_chi_samples: bool = False,
) -> LigandPreparation:
    """Prepare one ligand through the unified CIF -> SMILES -> params path.

    Derives a SMILES from the ligand's atom array (from its explicit bond table;
    never geometry perception, never a CCD lookup) and runs it through the
    SMILES -> mol2 -> params pipeline. The prepared residue's heavy-atom names
    must cover the original ligand's heavy atoms.

    Args:
        ligand_info: The detected (CIF/atom-array) ligand.
        ph: Target pH for protonation (applied in the SMILES -> mol2 step).
        seed: Fixed RNG seed for reproducible 3D coordinates; ``None`` is
            random, which makes the measured bonded parameters differ between
            runs of the same input.

    Returns:
        The :class:`LigandPreparation`.

    Raises:
        ValueError: If a SMILES could not be derived or prepared.
    """
    # Source-index atom maps must not change the seeded conformer when readers
    # or callers order the same named atoms differently.
    array = ligand_info.atom_array[np.argsort(ligand_info.atom_array.atom_name)]
    ligand_info = attr.evolve(
        ligand_info,
        atom_array=array,
        atom_names=tuple(array.atom_name),
        elements=tuple(array.element),
        coords=array.coord,
    )
    smiles = ligand_smiles_from_atom_array(
        array, res_name=ligand_info.res_name, with_atom_map=True
    )

    try:
        smiles_info = nonstandard_residue_info_from_smiles_via_mol2(
            smiles, res_name=ligand_info.res_name, ph=ph, protonate=protonate, seed=seed
        )
        prep = prepare_single_ligand(
            smiles_info,
            name_source=ligand_info,
            assign_ring_chis=assign_ring_chis,
            generate_heavy_chi_samples=generate_heavy_chi_samples,
        )
    except Exception as err:
        raise ValueError(
            f"{ligand_info.res_name}: failed to prepare ligand via SMILES {smiles!r}: {err}"
        ) from err

    missing_heavy_names = sorted(
        _missing_cif_heavy_atom_names(prep, _cif_heavy_atom_names(ligand_info))
    )
    if missing_heavy_names:
        raise ValueError(
            f"{ligand_info.res_name}: generated chemistry does not cover authored "
            "source heavy-atom names: "
            f"{', '.join(missing_heavy_names)}"
        )
    return prep


def _supported_elements(param_db: ParameterDatabase) -> set[str]:
    """Element symbols tmol has atom types for."""
    return {
        at.element.strip().capitalize()
        for at in param_db.chemical.atom_types
        if at.element and at.element.strip()
    }


def _polymer_connection_atoms(res_name, lig, canonical_ordering, chemdb):
    """The atoms of a residue where a polymer chain attaches, or an empty set.

    A canonical residue reports its own down and up connections; anything else
    is asked for its backbone, which a sugar or a free ligand does not have.
    """
    from tmol.ligand._polymer_profile import (
        _chain_end_candidates,
        profile_for_atom_array,
    )

    classes = canonical_ordering.restype_io_equiv_classes
    if res_name in classes:
        conn_inds = canonical_ordering.polymer_conn_inds
        equiv_ind = classes.index(res_name)
        names = canonical_ordering.restypes_ordered_atom_names[res_name]
        return frozenset(
            names[atom_ind]
            for atom_ind in (
                conn_inds.down_atom_for_co_restype[equiv_ind],
                conn_inds.up_atom_for_co_restype[equiv_ind],
            )
            if atom_ind >= 0
        )
    if lig is None:
        return frozenset()
    profile = profile_for_atom_array(lig.atom_array, lig.connection_atom_names, chemdb)
    if profile is None and len(lig.connection_atom_names or ()) > 2:
        elements = dict(zip(lig.atom_array.atom_name, lig.atom_array.element))
        pairs = {
            frozenset((nitrogen, carbonyl))
            for nitrogen in lig.connection_atom_names
            if elements[nitrogen] == "N"
            for carbonyl in (_chain_end_candidates(lig.atom_array, nitrogen) or ())
            if carbonyl in lig.connection_atom_names
        }
        # Sidechain crosslinks do not erase an unambiguous peptide backbone.
        # Resolve every partner from its chemistry, independently of loop order.
        if len(pairs) == 1:
            profile = profile_for_atom_array(lig.atom_array, pairs.pop(), chemdb)
    if profile is None:
        return frozenset()
    return frozenset(atom for _, atom in profile.connections)


def conjugation_atoms(lig, polymer_ports):
    """Sites outside an ordinary connection between two polymer ports.

    Both endpoints matter: an acyl cap's inferred polymer port does not turn
    a glycan amine or a polymer sidechain into a backbone connection.
    """
    partners = lig.connection_partners
    if not partners:
        return frozenset()
    conjugations = set()
    for atom, far_side in partners.items():
        if atom not in polymer_ports.get(lig.res_name, ()) or not any(
            partner_atom in polymer_ports[partner_name]
            for partner_name, partner_atom in far_side
        ):
            conjugations.add(atom)
    return frozenset(conjugations)


def _retain_polymer_connected_ports(ports, ligands):
    """A terminal-cap profile needs a polymer partner to define a backbone.

    Two linked small molecules can each fit a one-port cap profile. Without a
    declared or chemically complete backbone, their bond is a conjugation.
    """
    neighbors = {name: set() for name in ports}
    members = {
        name
        for name, atoms in ports.items()
        if atoms
        and (
            name not in ligands
            or ligands[name].in_polymer_entity
            or is_polymer_linking_component_type(ligands[name].component_type)
            or len(atoms) > 1
        )
    }
    for name, lig in ligands.items():
        for atom, partners in (lig.connection_partners or {}).items():
            for partner, other_atom in partners:
                if atom in ports[name] and other_atom in ports[partner]:
                    neighbors[name].add(partner)
                    neighbors[partner].add(name)
    pending = list(members)
    while pending:
        for partner in neighbors[pending.pop()] - members:
            members.add(partner)
            pending.append(partner)
    return {
        name: atoms if name in members else frozenset() for name, atoms in ports.items()
    }


def _bond_lengths_by_site(atom_array):
    """Finite, positive measurements of declared cross-residue bonds.

    An unresolved or coincident endpoint supplies no measurement, so it must
    neither override another instance's observation nor become a patch icoor.
    """
    if atom_array.bonds is None:
        return {}
    bonds = atom_array.bonds.as_array()
    if len(bonds) == 0:
        return {}
    starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    lengths = {}
    # Bound bond/coordinate temporaries independently of the structure size.
    # Search residue boundaries instead of allocating an index for every atom.
    for start in range(0, len(bonds), 4096):
        pairs = bonds[start : start + 4096, :2]
        residues = np.searchsorted(starts, pairs, side="right")
        pairs = pairs[residues[:, 0] != residues[:, 1]]
        endpoints = atom_array.coord[pairs]
        resolved = np.isfinite(endpoints).all(axis=(1, 2))
        pairs, endpoints = pairs[resolved], endpoints[resolved]
        distances = np.linalg.norm(endpoints[:, 0] - endpoints[:, 1], axis=1)
        for (first, second), distance in zip(pairs, distances):
            if not np.isfinite(distance) or distance <= 0:
                continue
            for index in (first, second):
                lengths[
                    (
                        str(atom_array.res_name[index]).strip(),
                        str(atom_array.atom_name[index]).strip(),
                    )
                ] = float(distance)
    return lengths


def _with_conjugation(
    prep, atoms, chemdb, lengths, hydrogens, bond_types, heavy_leaving, open_geometries
):
    """``prep`` carrying a patch and charge entry for each attachment site."""
    from dataclasses import replace

    from tmol.ligand._conjugation_patches import (
        conjugation_charge_entries,
        conjugation_patches,
    )

    if not atoms:
        return prep
    residue_type = prep.residue_type
    distances = {atom: lengths.get((residue_type.name, atom)) for atom in atoms}
    patches = conjugation_patches(
        residue_type,
        atoms,
        chemdb,
        distances,
        hydrogens,
        bond_types,
        heavy_leaving,
        open_geometries,
    )
    if not patches:
        return prep
    charges = conjugation_charge_entries(
        residue_type,
        atoms,
        chemdb,
        prep.partial_charges or {},
        hydrogens,
        heavy_leaving,
    )
    return replace(
        prep,
        adds_patches=(*prep.adds_patches, *patches),
        variant_partial_charges={**(prep.variant_partial_charges or {}), **charges},
    )


def canonical_conjugation_sites(atom_array, chemical_database):
    """Undeclared attachment sites on existing types, including canonical pairs."""
    from tmol.ligand._conjugate_model import attachment_connection_name

    if atom_array.bonds is None:
        return {}
    definitions = {
        r.name: r for r in chemical_database.residues if r.name == r.base_name
    }
    prepared_sites = {
        (r.io_equiv_class, c.atom)
        for r in chemical_database.residues
        for c in r.connections
        if c.name.startswith("conj_")
    }
    disulfides = {
        (r.io_equiv_class, c.atom)
        for r in definitions.values()
        for c in r.connections
        if c.name == "dslf"
    }
    for residue in list(definitions.values()):
        definitions.setdefault(residue.io_equiv_class, residue)
    starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    bonds = atom_array.bonds.as_array()[:, :2]
    residues = np.searchsorted(starts, bonds, side="right")
    cross = bonds[residues[:, 0] != residues[:, 1]]
    sites = {}
    for pair in cross:
        if all(
            (str(atom_array.res_name[i]), str(atom_array.atom_name[i])) in disulfides
            for i in pair
        ):
            continue
        for index, partner in (pair, pair[::-1]):
            index, partner = int(index), int(partner)
            name = str(atom_array.res_name[index])
            atom = str(atom_array.atom_name[index])
            residue = definitions.get(name)
            if residue is None:
                continue
            connection = attachment_connection_name(atom_array, index, partner, residue)
            if connection in ("up", "down"):
                continue
            if (name, atom) in prepared_sites:
                continue
            sites.setdefault(name, set()).add(atom)
    return sites


def _conjugation_chemistry(atom_array, chemical_database, ph):
    """Attachment states and missing frames from the chemistry used for scoring.

    Both sides need the bonded state: an acylated generated amine loses two
    hydrogens just as a canonical lysine does. Preserve source instance identity
    while protonating; reusable patches must agree across every instance.
    """
    from tmol.ligand._conjugate_model import iter_capped_conjugate_models
    from tmol.ligand._connection_params import _model_identity, _protonated_model
    from rdkit.Chem import AllChem

    if atom_array.bonds is None:
        return {}, {}, {}
    counts, bond_types, open_geometries, seen = {}, {}, {}, set()
    free_types = {
        rt.io_equiv_class: rt
        for rt in chemical_database.residues
        if rt.name == rt.base_name
    }
    for model in iter_capped_conjugate_models(atom_array, chemical_database):
        identity = _model_identity(model)
        if identity in seen:
            continue
        seen.add(identity)
        mol, mapping = _protonated_model(model, ph)
        props = None
        local_by_mol = {value: key for key, value in mapping.items()}
        local_by_source = {
            int(source): local
            for local, source in enumerate(model.source_atom_indices)
            if source >= 0
        }
        for first, second, _ in model.connections:
            indices = [mapping[local_by_source[s]] for s in (first, second)]
            order = str(mol.GetBondBetweenAtoms(*indices).GetBondType())
            for source, partner in ((first, second), (second, first)):
                key = (
                    str(atom_array.res_name[source]).strip(),
                    str(atom_array.atom_name[source]).strip(),
                )
                atom = mol.GetAtomWithIdx(mapping[local_by_source[source]])
                count = sum(n.GetAtomicNum() == 1 for n in atom.GetNeighbors())
                previous = counts.setdefault(key, count)
                previous_order = bond_types.setdefault(key, order)
                if (previous, previous_order) != (count, order):
                    raise ValueError(
                        f"Incompatible conjugate chemistry for {key}: "
                        f"{previous} H/{previous_order} and {count} H/{order}; "
                        "distinct chemistry needs distinct residue identities"
                    )
                if atom.GetAtomicNum() == 8 and atom.GetDegree() == 2 and count == 0:
                    restype = free_types.get(key[0])
                    if (
                        restype is None
                        or sum(key[1] in bond[:2] for bond in restype.bonds) != 1
                    ):
                        continue
                    # Existing departing-H frames remain reusable even when
                    # different partners have different bonded-state targets.
                    if props is None:
                        props = AllChem.MMFFGetMoleculeProperties(
                            mol, mmffVariant="MMFF94"
                        )
                    if props is None:
                        raise ValueError("MMFF94 does not cover this capped conjugate")
                    remote = mapping[local_by_source[partner]]
                    local = next(
                        n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != remote
                    )
                    target = props.GetMMFFAngleBendParams(
                        mol, local, atom.GetIdx(), remote
                    )
                    bond = props.GetMMFFBondStretchParams(mol, atom.GetIdx(), remote)
                    if target is not None and bond is not None and 0 < target[2] < 180:
                        value = (
                            str(model.atom_array.atom_name[local_by_mol[local]]),
                            np.deg2rad(target[2]),
                            bond[2],
                        )
                        if open_geometries.setdefault(key, value) != value:
                            raise ValueError(
                                f"Incompatible attachment geometry for {key}"
                            )
    return counts, bond_types, open_geometries


def _canonical_conjugation_parameters(
    param_db,
    sites,
    lengths,
    hydrogen_counts,
    bond_types,
    heavy_leaving,
    open_geometries,
):
    """Patches and charges for the database residues a component attaches to."""
    from tmol.ligand._conjugation_patches import (
        _hydrogens_on,
        charges_for_database_residue,
        combined_terminal_conjugation_patch,
        conjugation_patch,
        conjugation_charge_entries,
    )
    from tmol.ligand._parameter_replacements import _baseline_charges

    variants, charges = [], {}
    charge_index = {
        (entry.res, entry.atom): entry.charge
        for entry in param_db.scoring.elec.atom_charge_parameters
    }
    for res_name, partners in sorted(sites.items()):
        residue_type = next(
            (
                r
                for r in param_db.chemical.residues
                if r.io_equiv_class == res_name and r.name == r.base_name
            ),
            None,
        )
        if residue_type is None:
            continue
        base = charges_for_database_residue(param_db, residue_type.name)
        connections = {conn.atom: conn.name for conn in residue_type.connections}
        for atom in sorted(partners):
            count = hydrogen_counts.get((res_name, atom))
            leaving = heavy_leaving.get((res_name, atom), ())
            distance = lengths.get((res_name, atom))
            order = bond_types.get((res_name, atom), "SINGLE")
            geometry = open_geometries.get((res_name, atom))
            terminal_type = next(
                (
                    rt
                    for rt in param_db.chemical.residues
                    if rt.base_name == residue_type.base_name
                    and rt.name.split(":")[1:] == ["nterm"]
                ),
                None,
            )
            base_hydrogens = len(_hydrogens_on(residue_type, atom, param_db.chemical))
            is_terminal_amine = (
                terminal_type is not None
                and connections.get(atom) == "down"
                and count is not None
                and count > base_hydrogens
            )
            patch_source = terminal_type if is_terminal_amine else residue_type
            patch = conjugation_patch(
                patch_source,
                atom,
                param_db.chemical,
                distance,
                n_hydrogens=count,
                bond_type=order,
                heavy_leaving=leaving,
                open_geometry=geometry,
            )
            if patch is None:
                continue
            if is_terminal_amine:
                patch = combined_terminal_conjugation_patch(
                    residue_type, terminal_type, patch, param_db.chemical
                )
                if patch is None:
                    continue
                terminal_charges = _baseline_charges(charge_index, terminal_type)
                entry = conjugation_charge_entries(
                    terminal_type,
                    {atom},
                    param_db.chemical,
                    terminal_charges,
                    {atom: count},
                    {atom: leaving},
                )
                charges[f"{residue_type.name}:{patch.display_name}"] = next(
                    iter(entry.values())
                )
            else:
                charges.update(
                    conjugation_charge_entries(
                        residue_type,
                        {atom},
                        param_db.chemical,
                        base,
                        {atom: count},
                        {atom: leaving},
                    )
                )
            variants.append(patch)
    return tuple(variants), charges


def _routes_to_polymer_path(
    lig: NonStandardResidueInfo, chemdb=None, conjugations=None
) -> bool:
    """Whether this residue is prepared as a chain member rather than a molecule.

    A chain member is bonded to a neighbour, so nothing unlinked is one however
    the file describes it. Given that, the file settles what kind of thing it
    is: a polymer entity's residues are numbered along its sequence, and a
    declared linking component type says the same. Failing both, the residue is
    routed by its chemistry; the cap profiles match their whole atom set, so an
    unrelated linked fragment does not qualify.
    """
    from tmol.ligand._polymer_profile import profile_for_atom_array

    if not lig.covalently_linked:
        return False
    # every attachment lands on a sidechain: a conjugate, not a chain member
    if conjugations is not None and conjugations >= (
        lig.connection_atom_names or set()
    ):
        return False
    if lig.in_polymer_entity:
        return True
    if is_polymer_linking_component_type(lig.component_type):
        return True
    backbone_connections = (
        (lig.connection_atom_names - (conjugations or frozenset()))
        if lig.connection_atom_names is not None
        else None
    )
    return (
        profile_for_atom_array(lig.atom_array, backbone_connections, chemdb) is not None
    )


def _ligand_unsupported_reason(
    lig: NonStandardResidueInfo,
    supported_elements: set[str],
    is_polymer: bool = False,
) -> str | None:
    """Why this residue cannot be prepared, or None if it can."""
    if lig.chemistry_problem:
        return lig.chemistry_problem

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
    strict_ligands: bool = True,
    return_fragment_definitions: bool = False,
    return_cut_partners: bool = False,
    chem_comp_types: dict[str, str] | None = None,
    use_ccd: bool = True,
    seed: int | None = None,
) -> tuple:
    """Detect, prepare, and register all non-standard residues.

    Scans the input AtomArray for residues not in the ParameterDatabase,
    runs each through the unified SMILES→OpenBabel mol2→typing→residue-build
    pipeline, and returns a **new** ParameterDatabase with the ligand data
    injected. Ligand/glycan attachment chemistry uses the connected capped
    model for local typing, bonded terms and atom construction, preserving
    prepared residue charges. Lengths and angles use the same Cartesian
    constants as ordinary ligand parameters. Supplied explicit connection
    records take precedence and preserve both endpoint residue types.

    Args:
        atom_array: A biotite AtomArray from a CIF or PDB file.
        param_db: The base ParameterDatabase (not modified). If None, the
            default database is used.
        ph: Target pH for ligand protonation (Dimorphite-DL on derived SMILES).
        strict_atom_types: If True, fail when unknown atom-type element
            mappings are encountered during registration.
        params_files: Optional list of tmol YAML params file paths to
            inject before detection. Residues defined in these files
            skip residue generation. Missing attachment records are generated;
            existing records are reused without regeneration.
        params_output: Optional path to write all prepared ligand data
            to a tmol YAML params file for later reuse.
        strict_ligands: If True (default), raise :class:`LigandPreparationError`
            when a detected non-standard residue is skipped (metal-containing or
            covalently linked) or fails preparation, instead of silently
            dropping it. If False, such residues are logged as warnings and
            skipped, leaving them to be filtered out during pose construction.
        return_cut_partners: Internal option. If True, append the residue
            names whose covalent bonds were cut. A ligand skipped under
            ``strict_ligands=False`` loses its bonds here, and a caller
            that goes on to build a pose must know that, or it will reject
            the dangling partner it was told to drop.
        return_fragment_definitions: Internal/context-building option. If True,
            include definitions derived from ``tmol_fragment_id`` annotations
            as the third return value.
        chem_comp_types: ``{comp_id: type}`` read from the input file's
            ``_chem_comp`` table (see
            :func:`tmol.ligand.chem_comp_types_from_cif`), consulted where the
            file does not number the residue along a polymer sequence. A
            polymer-linking type routes the residue to
            :func:`prepare_polymer_residue` instead of the free-molecule
            ligand path.
        use_ccd: Whether a residue the input declares no chemistry for may be
            completed from the component dictionary by its code. Pass False for
            a source that supplies whole molecules under codes of its own, such
            as a mol2 or a ligand file naming its residue LG1; the dictionary
            defines those codes as unrelated molecules.
        seed: Fixed RNG seed for the 3D conformer each residue is built from.
            ``None`` is random, which makes the prepared residue types differ
            between runs.

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
    has_fragments = FRAGMENT_ID_ANNOTATION in atom_array.get_annotation_categories()
    for start, end in zip(starts, ends):
        ligand_name = str(atom_array.res_name[start])
        if not has_fragments and ligand_name in first_residue_by_name:
            continue
        residue = atom_array[start:end]
        fragment_ids = fragment_ids_from_atom_array(residue)
        layout = (
            None
            if fragment_ids is None
            else tuple(sorted(zip(map(str, residue.atom_name), map(int, fragment_ids))))
        )
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
        from tmol.ligand._params_file import _load_params_files

        params_preparations = _load_params_files(params_files)
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

    ligands = detect_nonstandard_residues(
        atom_array,
        canonical_ordering,
        chem_comp_types=chem_comp_types,
    )

    canonical_sites = canonical_conjugation_sites(atom_array, param_db.chemical)
    if not ligands and not canonical_sites:
        logger.info("No non-standard residues detected")
        prepared_db, records, replacements = _prepare_conjugate_params(
            atom_array, param_db, ph, strict_ligands=strict_ligands
        )
        if prepared_db is not param_db:
            canonical_ordering = rebuild_canonical_ordering(prepared_db)
        param_db = prepared_db
        if params_output:
            _write_preparations(
                params_preparations, replacements, records, param_db, params_output
            )
        if return_fragment_definitions:
            return (
                param_db,
                canonical_ordering,
                tuple(fragment_definitions_by_name.values()),
                *((frozenset(),) if return_cut_partners else ()),
            )
        if return_cut_partners:
            return param_db, canonical_ordering, frozenset()
        return param_db, canonical_ordering

    logger.info("Found %d non-standard residue type(s) to prepare", len(ligands))
    cut_partners: frozenset[str] = frozenset()

    supported_elements = _supported_elements(param_db)

    preparations: list[LigandPreparation] = []
    prepared_ligands: list[tuple[NonStandardResidueInfo, LigandPreparation]] = []
    prepared_polymers: list[LigandPreparation] = []
    ligands_by_name = {lig.res_name: lig for lig in ligands}
    partner_names = set(ligands_by_name) | {
        name
        for lig in ligands
        for partners in (lig.connection_partners or {}).values()
        for name, _atom in partners
    }
    polymer_ports = {
        name: _polymer_connection_atoms(
            name, ligands_by_name.get(name), canonical_ordering, param_db.chemical
        )
        for name in partner_names
    }
    polymer_ports = _retain_polymer_connected_ports(polymer_ports, ligands_by_name)
    bond_lengths = _bond_lengths_by_site(atom_array)
    for lig in ligands:
        conjugations = conjugation_atoms(lig, polymer_ports)
        is_polymer = _routes_to_polymer_path(lig, param_db.chemical, conjugations)
        reason = _ligand_unsupported_reason(lig, supported_elements, is_polymer)
        if reason:
            _skip_or_raise(strict_ligands, reason)
            continue

        logger.info(
            "Preparing %s (declared component type: %s)",
            lig.res_name,
            lig.component_type,
        )
        try:
            bonds = lig.atom_array.bonds
            if len(lig.atom_array) > 1 and (
                bonds is None
                or bonds.get_bond_count() == 0
                or np.any(bonds.as_array()[:, 2] == struc.BondType.ANY)
            ):
                raise ValueError(
                    f"{lig.res_name}: parameter generation requires chemical bond orders. "
                    "Supply a typed CIF/MOL2, enable use_ccd for a CCD component, "
                    "or load its prepared .tmol parameters with params_files."
                )
            if is_polymer:
                prep = prepare_polymer_residue(
                    lig.atom_array,
                    canonical_ordering,
                    param_db,
                    ph=ph,
                    # An attached sidechain (e.g. BCX's disulfide SG) does
                    # not change the polymer backbone's two endpoints.
                    connection_atoms=(
                        lig.connection_atom_names - conjugations
                        if lig.connection_atom_names is not None
                        else None
                    ),
                    use_ccd=use_ccd,
                    seed=seed,
                )
            else:
                prep = _prepare_ligand_via_smiles(
                    lig,
                    ph=ph,
                    seed=seed,
                    generate_heavy_chi_samples=bool(conjugations),
                )
        except LigandPreparationError as err:
            # strict_ligands=False means "drop what you cannot prepare". That has
            # to hold for ligands too, not just polymer residues: a conjugated
            # residue whose partner needs unsupported chemistry raises here, and
            # re-raising regardless of the flag turned a skip into a crash.
            if strict_ligands:
                raise
            kind = "polymer residue" if is_polymer else "ligand"
            logger.warning(
                "Skipping %s: %s preparation failed (%s)", lig.res_name, kind, err
            )
            continue
        except Exception as err:  # noqa: BLE001  SMILES/typing/build failure
            kind = "polymer residue" if is_polymer else "ligand"
            if strict_ligands:
                raise LigandPreparationError(
                    f"{lig.res_name}: failed to prepare {kind} ({err}). Pass "
                    "strict_ligands=False to skip it with a warning, or supply "
                    "prebuilt params via ligand_params_files."
                ) from err
            logger.warning(
                "Skipping %s: %s preparation failed (%s)", lig.res_name, kind, err
            )
            continue
        preparations.append(prep)
        # a polymer residue is not a free molecule, so it is never fragmented
        if is_polymer:
            prepared_polymers.append(prep)
        else:
            prepared_ligands.append((lig, prep))

    from tmol.ligand._conjugation_patches import declared_heavy_leaving_groups

    heavy_leaving = declared_heavy_leaving_groups(atom_array)
    hydrogen_counts, bond_types, open_geometries = {}, {}, {}
    if preparations or canonical_sites:
        # Base definitions suffice for identifying polymer caps. Avoid a second
        # database injection/patch expansion merely to read the bonded state.
        chemistry = attr.evolve(
            param_db.chemical,
            residues=(
                *param_db.chemical.residues,
                *(p.residue_type for p in preparations),
            ),
        )
        # Everything below reads conjugate models off this array, and a residue
        # the loop above skipped is still covalently bonded in it. Cut those
        # bonds once, here, rather than at each model-building call site.
        atom_array, cut = _without_unprepared_partners(
            atom_array, chemistry, strict_ligands
        )
        cut_partners |= cut
        hydrogen_counts, bond_types, open_geometries = _conjugation_chemistry(
            atom_array, chemistry, ph
        )
        prepared_by_name = {}
        for prep in preparations:
            name = prep.residue_type.name
            atoms = conjugation_atoms(ligands_by_name[name], polymer_ports)
            prepared_by_name[name] = _with_conjugation(
                prep,
                atoms,
                param_db.chemical,
                bond_lengths,
                {atom: hydrogen_counts.get((name, atom)) for atom in atoms},
                {atom: bond_types.get((name, atom), "SINGLE") for atom in atoms},
                {atom: heavy_leaving.get((name, atom), ()) for atom in atoms},
                {atom: open_geometries.get((name, atom)) for atom in atoms},
            )
        preparations = list(prepared_by_name.values())
        prepared_ligands = [
            (lig, prepared_by_name[prep.residue_type.name])
            for lig, prep in prepared_ligands
        ]
        prepared_polymers = [
            prepared_by_name[prep.residue_type.name] for prep in prepared_polymers
        ]

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

    if strict_ligands and ligands and not preparations:
        raise LigandPreparationError(
            "All "
            f"{len(ligands)} detected non-standard residue(s) "
            f"({', '.join(sorted({lig.res_name for lig in ligands}))}) were "
            "skipped; none could be prepared. Pass strict_ligands=False to "
            "continue with these residues dropped."
        )

    if preparations or canonical_sites:
        fragment_preparations = [
            fragment_prep
            for definition in fragment_definitions_by_name.values()
            for fragment_prep in definition.fragment_preparations
        ]
        _assert_fragment_names_available(param_db, fragment_preparations)
        partner_patches, partner_charges = _canonical_conjugation_parameters(
            param_db,
            canonical_sites,
            bond_lengths,
            hydrogen_counts,
            bond_types,
            heavy_leaving,
            open_geometries,
        )
        # Keep shared partner metadata with the source preparation so export
        # and direct injection follow the same path.
        if not preparations:
            residue = next(
                r
                for r in param_db.chemical.residues
                if r.io_equiv_class in canonical_sites and r.name == r.base_name
            )
            preparations.append(
                LigandPreparation(
                    residue_type=residue,
                    partial_charges=_partial_charges_for_residue(
                        param_db, residue.name
                    ),
                    cartbonded_params=param_db.scoring.cartbonded.residue_params.get(
                        residue.name
                    ),
                )
            )
        first = preparations[0]
        preparations[0] = replace(
            first,
            adds_patches=(*first.adds_patches, *partner_patches),
            variant_partial_charges={
                **(first.variant_partial_charges or {}),
                **partner_charges,
            },
        )
        param_db = inject_ligand_preparations(
            param_db, preparations, strict_atom_types=strict_atom_types
        )
        param_db, records, replacements = _prepare_conjugate_params(
            atom_array, param_db, ph, strict_ligands=strict_ligands
        )
        canonical_ordering = rebuild_canonical_ordering(param_db)

        if params_output:
            # Fragment residue types are an in-memory representation in this
            # first API version. Persist only the fully prepared source ligand.
            sources = {p.residue_type.name for _, p in prepared_ligands}
            sources.update(p.residue_type.name for p in prepared_polymers)
            sources.add(preparations[0].residue_type.name)  # Shared canonical patches.
            _write_preparations(
                [
                    *params_preparations,
                    *(p for p in preparations if p.residue_type.name in sources),
                ],
                replacements,
                records,
                param_db,
                params_output,
            )

    if return_fragment_definitions:
        return (
            param_db,
            canonical_ordering,
            tuple(fragment_definitions_by_name.values()),
            *((frozenset(cut_partners),) if return_cut_partners else ()),
        )
    if return_cut_partners:
        return param_db, canonical_ordering, frozenset(cut_partners)
    return param_db, canonical_ordering


def _ligand_info_from_cif(
    cif_path: str, res_name: str | None, *, use_ccd: bool = False
) -> NonStandardResidueInfo:
    """Read one ligand through the shared format, identity and completion path."""
    from tmol.io import atom_array_from_cif

    arr = atom_array_from_cif(cif_path, use_ccd=use_ccd)
    if struc.get_residue_count(arr) != 1:
        raise ValueError("Single-ligand preparation requires exactly one residue")
    resolved = (res_name or str(arr.res_name[0])).strip()
    component_type = str(getattr(arr, "chem_comp_type", ["UNKNOWN"])[0]) or "UNKNOWN"
    arr.res_name = np.full(len(arr), resolved)

    return NonStandardResidueInfo(
        res_name=resolved,
        component_type=component_type,
        atom_names=tuple(str(n) for n in arr.atom_name),
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
    seed: int | None = None,
    strict_atom_types: bool = False,
    res_name: str | None = None,
    use_ccd: bool = False,
) -> tuple[ParameterDatabase, CanonicalOrdering]:
    """Prepare a single ligand from a CIF file and inject it into a database.

    Runs the same full pipeline as :func:`prepare_ligand_from_smiles`; the only
    CIF-specific step is the front end. A SMILES is derived from the CIF ligand's
    bond table (optionally supplemented from CCD, never geometry perception) and run
    through the SMILES -> mol2 -> params path (protonation, 3D conformer, MMFF94
    charges). The prepared residue's heavy-atom names are then mapped back to the
    CIF atom names via the atom-order map carried through the round-trip.

    Args:
        cif_path: One ligand residue in a text, compressed or binary CIF file;
            the first model is selected. Use :func:`prepare_ligands` for arrays
            containing multiple residues.
        param_db: Base database (not modified); defaults to the tmol default.
        ph: Target pH for protonation.
        seed: Reproducible conformer seed, shared with the MOL2/SMILES entry paths.
        strict_atom_types: Fail on unknown atom-type element mappings.
        res_name: Optional residue name override.
        use_ccd: Allow dictionary supplementation for a recognized component code.
            False (default) requires authored bonds and permits custom residue codes.

    Returns:
        A ``(ParameterDatabase, CanonicalOrdering)`` with the ligand injected.
    """
    lig = _ligand_info_from_cif(cif_path, res_name, use_ccd=use_ccd)
    prep = _prepare_ligand_via_smiles(lig, ph=ph, seed=seed)
    return _inject_single(prep, param_db, strict_atom_types)


def prepare_ligand_from_smiles(
    smiles: str,
    *,
    param_db: Optional[ParameterDatabase] = None,
    ph: float = 7.4,
    strict_atom_types: bool = False,
    res_name: str | None = None,
    protonate: bool = True,
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
    prep = prepare_single_ligand(lig)
    return _inject_single(prep, param_db, strict_atom_types)


def unused_ligand_name(taken) -> str:
    """First "L_n" name not already present in ``taken``.

    The underscore keeps the name out of the component dictionary, which uses
    only letters and digits: a generated code that collided with a real entry
    would invite completing the molecule from an unrelated one.
    """
    for i in itertools.count(1):
        name = f"L_{i}"
        if name not in taken:
            return name


def prepare_ligands_from_smiles(
    smiles,
    *,
    param_db: Optional[ParameterDatabase] = None,
    ph: float = 7.4,
    strict_atom_types: bool = False,
    protonate: bool = True,
    seed: int | None = None,
) -> tuple[ParameterDatabase, dict]:
    """Prepare one residue type per SMILES, naming them L_1, L_2, ...

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
            seed=seed,
        )
        names[smi] = name
    return param_db, names


def _prepare_mol2(mol2_path, res_name=None, *, ph=7.4, mode="auto", seed=None):
    from tmol.ligand._detect import nonstandard_residue_info_from_mol2

    if mode not in ("keep", "auto", "regenerate"):
        raise ValueError("MOL2 mode must be keep, auto, or regenerate")
    lig = nonstandard_residue_info_from_mol2(mol2_path, res_name=res_name)
    try:
        return (
            prepare_single_ligand(lig)
            if mode == "keep" or (mode == "auto" and lig.skip_protonation)
            else _prepare_ligand_via_smiles(lig, ph=ph, seed=seed)
        )
    except ValueError as error:
        raise ValueError(f"{mol2_path}: {error}") from error


def prepare_ligand_from_mol2(
    mol2_path: str,
    *,
    param_db: Optional[ParameterDatabase] = None,
    ph: float = 7.4,
    mode: str = "auto",
    seed: int | None = None,
    strict_atom_types: bool = False,
    res_name: str | None = None,
) -> tuple[ParameterDatabase, CanonicalOrdering]:
    """Prepare a single ligand from a Tripos mol2 file and inject it.

    By default, preserve complete molecules with supported partial charges;
    otherwise run shared protonation and charge generation. Use ``regenerate``
    to apply the requested pH even to a neutralized, fully hydrogenated input.

    Args:
        mol2_path: Path to the ligand mol2 file.
        param_db: Base database (not modified); defaults to the tmol default.
        ph: Target pH for protonation.
        mode: ``"keep"`` requires prepared input and retains its protonation and
            charges; ``"auto"`` (default) regenerates only unprepared input;
            ``"regenerate"`` always runs pH-dependent preparation. Complete
            hydrogens and supported charges do not establish the intended pH.
        seed: Reproducible conformer seed when charges must be generated.
        strict_atom_types: Fail on unknown atom-type element mappings.
        res_name: Optional residue name override.

    Returns:
        A ``(ParameterDatabase, CanonicalOrdering)`` with the ligand injected.
    """
    prep = _prepare_mol2(mol2_path, res_name, ph=ph, mode=mode, seed=seed)
    return _inject_single(prep, param_db, strict_atom_types)
