"""Reading a structure from an mmCIF, with the chemistry it declares.

A structure records only the atoms its density resolved, but it declares the
whole of every component it names, in ``chem_comp_atom`` and ``chem_comp_bond``.
The two are different things and downstream code needs both: what the residue
IS decides its residue type, and what was SEEN decides which of its atoms have
coordinates.

The reader here resolves them into one AtomArray. Every atom the file declares
is present; the ones it did not resolve carry NaN coordinates, which is already
how the rest of tmol marks an atom as absent (see
``tmol.io._pose_stack_construction``), so pose construction rebuilds them
through the usual sidechain path rather than through anything new.
"""

import logging

import biotite.structure as struc
import numpy as np

logger = logging.getLogger(__name__)

_MISSING_CIF_VALUE = ("", ".", "?")
_FORMAL_CHARGE_SPECIFIED = "tmol_formal_charge_specified"
_SOURCE_FORMAL_CHARGE = "tmol_source_formal_charge"


def component_chemistry_from_cif(cif_path) -> dict:
    """The complete chemistry a CIF declares for each component it names.

    A structure records only the atoms it resolved, but it declares the whole
    component in ``chem_comp_atom`` and ``chem_comp_bond`` -- names, elements
    and bond orders, independent of what the density showed. That declaration
    is what residue-type generation needs, since a residue built from resolved
    atoms alone is a different molecule: an unresolved sidechain tip becomes a
    real methyl once the pipeline protonates it.

    Returns ``{comp_id: AtomArray}`` with no coordinates -- the pipeline
    generates its own conformer -- and only for components the file bonds:
    connectivity cannot be guessed, so a multi-atom component with no authored
    bonds is left out for the caller to resolve another way.
    """
    from atomworks.io.utils.io_utils import read_any

    cif = read_any(cif_path)
    entries: dict[str, struc.AtomArray] = {}
    for block in cif.values():
        entries.update(component_chemistry_from_block(block))
    return entries


def component_chemistry_from_block(block, *, use_ccd=False) -> dict:
    """Read authored chemistry, optionally supplementing missing annotations."""
    from atomworks.io.utils.ccd import build_ccd_entries_from_cif_block

    entries = build_ccd_entries_from_cif_block(block, supplement_from_ccd=use_ccd)
    return _completion_templates(entries)


def _completion_templates(entries):
    """Normalize owned templates for tmol's author-identifier completion policy."""
    for array in entries.values():
        # Preserve the legacy template contract; structure-level metadata is
        # attached separately before parameter preparation.
        for name in set(array.get_annotation_categories()) - {
            "chain_id",
            "res_id",
            "ins_code",
            "res_name",
            "hetero",
            "atom_name",
            "element",
            "charge",
            "is_leaving_atom",
        }:
            array.del_annotation(name)
        array.coord[:] = np.nan
        array.chain_id[:] = "A"
        array.res_id[:] = 1
        array.hetero[:] = True
        array.element = np.array(
            [
                str(e) or _element_from_name(str(n))
                for n, e in zip(array.atom_name, array.element)
            ],
            dtype="U4",
        )
        bonds = array.bonds.as_array()
        aromatic = np.isin(
            bonds[:, 2],
            [
                struc.BondType.AROMATIC_SINGLE,
                struc.BondType.AROMATIC_DOUBLE,
                struc.BondType.AROMATIC_TRIPLE,
            ],
        )
        bonds[aromatic, 2] = struc.BondType.AROMATIC
        array.bonds = struc.BondList(len(array), bonds)
    return entries


def _element_from_name(name: str) -> str:
    """Element from a PDB atom name, for a file that declares no type_symbol."""
    from tmol.chemical import get_element_from_atom_name

    return get_element_from_atom_name(name)


def _component_dictionary_template(res_name: str):
    """Read complete chemical annotations from the shared component dictionary."""
    from atomworks.io.utils.ccd import atom_array_from_ccd_code

    try:
        component = atom_array_from_ccd_code(
            res_name, ccd_mirror_path=None, coords=None
        )
    except (ValueError, AttributeError):
        return None
    return component if component.bonds is not None else None


def atom_array_from_cif(
    cif_path,
    *,
    model: int = 1,
    assembly_id: str | None = None,
    use_ccd: bool = True,
):
    """Read a PDB, CIF, compressed CIF or binary CIF into an AtomArray.

    Preserve supplied names, coordinates, hydrogens and covalent connections.
    Declared unresolved heavy atoms have NaN coordinates. ``use_ccd`` controls
    whether missing chemistry may be supplemented from the component dictionary;
    False never consults it, including for PDB input and custom residue codes.

    Missing chemical metadata is allowed here. Pose construction uses existing
    parameters for known residues and requires additional chemistry only when
    preparing an unknown residue. Pose construction controls hydrogen optimization
    and whether to trust regenerated ligand hydrogen names.
    """
    from tmol.io._atomworks_reader import read_structure

    if assembly_id is not None and (
        not isinstance(assembly_id, str) or not assembly_id
    ):
        raise ValueError("assembly_id must be a nonempty string or None")
    array, block = read_structure(
        cif_path, model=model, assembly_id=assembly_id, use_ccd=use_ccd
    )
    array = _with_polymer_entity_flag(array, block)
    array = _with_component_type_annotation(array, block)
    templates = _completion_templates(
        {name: entry.copy() for name, entry in array._custom_ccd_registry.items()}
    )
    from atomworks.io.utils.ccd import (
        add_annotations_from_ccd,
        custom_ccd_residues,
        get_available_ccd_codes,
    )
    from atomworks.io.utils.leaving_atoms import resolve_leaving_atoms

    authored_charges = _component_formal_charges(block)
    available_ccds = get_available_ccd_codes("") if use_ccd else ()
    authoritative_charges = set(authored_charges)
    for template in templates.values():
        charge_specified = np.array(
            [
                (
                    str(template.res_name[index]).strip().upper(),
                    str(template.atom_name[index]).strip(),
                )
                in authored_charges
                or str(template.res_name[index]) in available_ccds
                for index in range(len(template))
            ],
            dtype=bool,
        )
        template.set_annotation(
            _FORMAL_CHARGE_SPECIFIED,
            charge_specified,
        )
        authoritative_charges.update(
            (
                str(template.res_name[index]).strip().upper(),
                str(template.atom_name[index]).strip(),
            )
            for index in np.flatnonzero(charge_specified)
        )

    # AtomWorks skips sanitation when its own completion is disabled. Repair
    # the final author-view graph with the same shared attachment chemistry.
    with custom_ccd_residues(array._custom_ccd_registry):
        specified = getattr(array, "pdbx_formal_charge", np.full(len(array), "?"))
        charge_specified = ~np.isin(specified, _MISSING_CIF_VALUE)
        for index in np.flatnonzero(charge_specified):
            array.charge[index] = int(specified[index])
        if hasattr(array, "pdbx_formal_charge"):
            # ``charge`` cannot distinguish an authored zero from its default.
            # Keep that provenance for fail-closed ligand regeneration checks.
            array.set_annotation(
                _SOURCE_FORMAL_CHARGE, np.asarray(specified, dtype=str).copy()
            )
        missing = ~charge_specified & np.isin(
            array.res_name, list(array._custom_ccd_registry)
        )
        if missing.any():
            if use_ccd:
                supplement = add_annotations_from_ccd(
                    array[missing],
                    annotations=["charge"],
                    overwrite=True,
                    ccd_mirror_path=None,
                )
                array.charge[missing] = supplement.charge
            for index in np.flatnonzero(missing):
                key = (
                    str(array.res_name[index]).strip().upper(),
                    str(array.atom_name[index]).strip(),
                )
                if key in authored_charges:
                    array.charge[index] = authored_charges[key]
                    charge_specified[index] = True
                elif key in authoritative_charges:
                    charge_specified[index] = True
        array.set_annotation(_FORMAL_CHARGE_SPECIFIED, charge_specified)
        if hasattr(array, "pdbx_formal_charge"):
            array.del_annotation("pdbx_formal_charge")
        array = with_unresolved_atoms(array, templates, use_ccd=use_ccd)
        return resolve_leaving_atoms(array)[0] if use_ccd else array


def _component_formal_charges(block) -> dict[tuple[str, str], int]:
    """Return formal charges explicitly authored in ``chem_comp_atom``."""
    if (
        block is None
        or "chem_comp_atom" not in block
        or "charge" not in block["chem_comp_atom"]
    ):
        return {}
    atoms = block["chem_comp_atom"]
    return {
        (str(comp).strip().upper(), str(atom).strip()): int(charge)
        for comp, atom, charge in zip(
            atoms["comp_id"].as_array(str),
            atoms["atom_id"].as_array(str),
            atoms["charge"].as_array(str),
        )
        if str(charge).strip() not in _MISSING_CIF_VALUE
    }


def _with_component_type_annotation(array, block):
    """Carry declared component types from the already parsed CIF into preparation."""
    if block is None or "chem_comp" not in block:
        return array
    category = block["chem_comp"]
    if "id" not in category or "type" not in category:
        return array
    declared = {
        str(name).strip().upper(): str(kind).strip().upper()
        for name, kind in zip(
            category["id"].as_array(str), category["type"].as_array(str)
        )
        if str(kind).strip() not in ("", ".", "?")
    }
    previous = getattr(array, "chem_comp_type", np.full(array.array_length(), ""))
    array.set_annotation(
        "chem_comp_type",
        np.array(
            [
                declared.get(str(name).upper(), str(old))
                for name, old in zip(array.res_name, previous)
            ]
        ),
    )
    return array


def _with_polymer_entity_flag(atom_array, block):
    """Mark each atom with whether its entity is a polymer, if the file says.

    ``_entity.type`` is what distinguishes a chain from a ligand, a glycan or a
    solvent. ``label_seq_id`` is only a proxy for it and a generated file may
    number a ligand along a sequence it does not belong to, so the entity is
    read directly.
    """
    categories = atom_array.get_annotation_categories()
    if block is None:
        if "is_polymer" in categories:
            atom_array.set_annotation(
                "tmol_polymer_entity", atom_array.is_polymer.copy()
            )
        return atom_array
    if "label_entity_id" not in categories or "entity" not in block:
        return atom_array
    entity = block["entity"]
    if "id" not in entity or "type" not in entity:
        return atom_array
    polymer = {
        str(i).strip()
        for i, t in zip(
            entity["id"].as_array(str), np.char.lower(entity["type"].as_array(str))
        )
        if str(t).strip() == "polymer"
    }
    ids = np.char.strip(atom_array.get_annotation("label_entity_id").astype(str))
    atom_array.set_annotation("tmol_polymer_entity", np.isin(ids, sorted(polymer)))
    return atom_array


def with_unresolved_atoms(atom_array, declared: dict, *, use_ccd: bool = True):
    """``atom_array`` plus the declared atoms it does not resolve, at NaN.

    Declared leaving groups and unobserved carbonyl/phosphoryl substitution branches are
    absent at their connection sites. Other unresolved atoms retain identity.
    """
    from atomworks.io.utils.leaving_atoms import (
        get_absent_substitution_leaving_groups,
        get_leaving_atom_groups,
    )
    from tmol.ligand._polymer_profile import completed_connection_atoms

    boundaries = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    starts = boundaries[:-1]
    connections = _connection_atoms_by_residue(atom_array, boundaries)
    templates = dict(declared)
    chemistry = {}
    warned: set = set()
    additions: dict[int, list] = {}
    for ri, (start, stop) in enumerate(zip(starts, boundaries[1:])):
        res_name = str(atom_array.res_name[start]).strip()
        if res_name not in templates and use_ccd:
            templates[res_name] = _component_dictionary_template(res_name)
        template = templates.get(res_name)
        if template is None:
            continue
        residue = atom_array[start:stop]
        present = {str(n) for n in residue.atom_name}
        if res_name not in chemistry:
            chemistry[res_name] = (
                frozenset(
                    str(n)
                    for n, e in zip(template.atom_name, template.element)
                    if e not in ("H", "D")
                ),
                get_leaving_atom_groups(template),
            )
        heavy, leaving = chemistry[res_name]
        observed_heavy = {
            str(n)
            for n, e in zip(residue.atom_name, residue.element)
            if e not in ("H", "D")
        }
        if not observed_heavy or not observed_heavy <= heavy:
            if res_name in warned:
                continue
            warned.add(res_name)
            logger.warning(
                "%s: the chemistry available for it does not account for the "
                "atoms the file resolved, so it is taken as resolved",
                res_name,
            )
            continue
        # Endpoint valence is meaningful only on the complete component
        # topology, before unresolved heavy atoms are inserted below.
        ends = completed_connection_atoms(template, frozenset(connections.get(ri, ())))
        missing = set(heavy - present)
        for end in ends or ():
            for group in leaving.get(end, ()):
                if present.isdisjoint(group):
                    missing.difference_update(group)
        for group in get_absent_substitution_leaving_groups(
            template, residue, set(ends or ())
        ).values():
            missing.difference_update(group)
        missing = sorted(missing)
        if missing:
            additions[int(start)] = (missing, template)
    if not additions:
        return atom_array
    return _inserted(atom_array, starts, additions)


def _connection_atoms_by_residue(atom_array, boundaries=None) -> dict:
    """Covalent attachment atoms keyed by contiguous residue instance."""
    if atom_array.bonds is None:
        return {}
    if boundaries is None:
        boundaries = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    n_residues = len(boundaries) - 1
    residue_index = np.repeat(
        np.arange(n_residues, dtype=np.min_scalar_type(n_residues)),
        np.diff(boundaries),
    )
    bonds = atom_array.bonds.as_array()[:, :2]
    cross = bonds[residue_index[bonds[:, 0]] != residue_index[bonds[:, 1]]]
    linked = {}
    for atom in cross.flat:
        linked.setdefault(int(residue_index[atom]), set()).add(
            str(atom_array.atom_name[atom]).strip()
        )
    return linked


def _inserted(atom_array, starts, additions: dict):
    """``atom_array`` with each residue's missing atoms appended after it.

    Existing bonds are carried across by index, and the added atoms are bonded
    as the template declares, so the result is bonded throughout rather than
    only where the density reached.
    """
    from atomworks.io.utils.atom_array_plus import concatenate_atom_array_plus

    boundaries = list(starts) + [atom_array.array_length()]

    # final layout: each residue's own atoms, then the ones it did not resolve
    pieces = []
    remap = np.empty(atom_array.array_length(), dtype=np.uint32)
    added_at = {}
    total = 0
    for begin, end in zip(boundaries, boundaries[1:]):
        pieces.append(atom_array[begin:end])
        remap[begin:end] = np.arange(total, total + end - begin, dtype=np.uint32)
        total += end - begin
        entry = additions.get(int(begin))
        if entry is None:
            continue
        missing, template = entry
        pieces.append(_placeholder_atoms(atom_array, begin, missing, template))
        for name in missing:
            added_at[(int(begin), name)] = total
            total += 1

    combined = concatenate_atom_array_plus(pieces)

    bond_tables = []
    if atom_array.bonds is not None:
        original_bonds = atom_array.bonds.as_array()
        original_bonds[:, :2] = remap[original_bonds[:, :2]]
        bond_tables.append(original_bonds)

    ends = dict(zip(boundaries[:-1], boundaries[1:]))
    added_bonds = []
    for begin, (missing, template) in additions.items():
        missing = set(missing)
        position = {
            str(atom_array.atom_name[old]): remap[old]
            for old in range(begin, ends[begin])
        }
        position.update({name: added_at[(begin, name)] for name in missing})
        for i, j, order in template.bonds.as_array():
            a, b = str(template.atom_name[i]), str(template.atom_name[j])
            if (a in missing or b in missing) and a in position and b in position:
                added_bonds.append((position[a], position[b], int(order)))

    bond_tables.append(np.asarray(added_bonds, dtype=np.uint32).reshape(-1, 3))
    combined.bonds = struc.BondList(
        combined.array_length(), np.concatenate(bond_tables)
    )
    return combined


def _placeholder_atoms(atom_array, begin, missing, template):
    """The unresolved atoms of one residue, at NaN, annotated like their residue."""
    positions = {str(n): i for i, n in enumerate(template.atom_name)}
    indices = [positions[n] for n in missing]
    extra = struc.AtomArray(len(missing))
    extra.coord = np.full((len(missing), 3), np.nan, dtype=np.float32)
    for field in atom_array.get_annotation_categories():
        source = atom_array.get_annotation(field)
        if field == "atom_name":
            value = np.array(missing, dtype=str)
        elif field in ("element", "charge", _FORMAL_CHARGE_SPECIFIED):
            values = getattr(
                template, field, np.zeros(len(template), dtype=source.dtype)
            )
            value = values[indices].astype(source.dtype)
        elif field == _SOURCE_FORMAL_CHARGE:
            value = np.full(len(missing), "?", dtype=source.dtype)
        else:
            value = np.array([source[begin]] * len(missing), dtype=source.dtype)
        extra.set_annotation(field, value)
    return extra


def pose_stack_from_cif(
    cif_path,
    device,
    *,
    model: int = 1,
    assembly_id: str | None = None,
    use_ccd: bool = True,
    residue_start: int | None = None,
    residue_end: int | None = None,
    **kwargs,
):
    """Construct a PoseStack from a PDB, CIF or binary/compressed CIF structure.

    Reads the structure with :func:`atom_array_from_cif`, so the residues carry
    every atom their chemistry declares and the unresolved ones arrive at NaN
    for pose construction to rebuild. Further keyword arguments are passed to
    :func:`tmol.io.pose_stack_from_biotite`.
    """
    from tmol.io._pose_stack_from_biotite import pose_stack_from_biotite

    array = atom_array_from_cif(
        cif_path,
        model=model,
        assembly_id=assembly_id,
        use_ccd=use_ccd,
    )
    if residue_start is not None or residue_end is not None:
        boundaries = struc.get_residue_starts(array, add_exclusive_stop=True)
        start, stop, _ = slice(residue_start, residue_end).indices(len(boundaries) - 1)
        array = array[boundaries[start] : boundaries[stop]]
    return pose_stack_from_biotite(
        array,
        device,
        use_ccd=use_ccd,
        **kwargs,
    )


atom_array_from_file = atom_array_from_cif
pose_stack_from_file = pose_stack_from_cif
