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
import biotite.structure.io.pdbx as pdbx
import numpy as np

logger = logging.getLogger(__name__)


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
    cif = pdbx.CIFFile.read(str(cif_path))
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
    include_bonds: bool = True,
    extra_fields=None,
    reader: str = "tmol",
    hydrogen_policy: str = "rebuild",
):
    """A structure's atoms, including the ones its density did not resolve.

    Reads the coordinates with their bonds, then adds back every atom the file
    declares but ``atom_site`` omits, at NaN. What a residue is and what was
    seen of it are then both carried by one array, so nothing downstream has to
    be told the difference.

    Args:
        cif_path: Path to an mmCIF file.
        model: Which model to read.
        assembly_id: Biological assembly ID, or None for the asymmetric unit.
            AtomWorks applies the file's assembly operations. Assembly chain IDs
            identify each label-chain/transformation instance on both readers;
            the author reader also retains ``auth_asym_id``. Requires bonds.
        use_ccd: Whether the component dictionary may say which atoms a residue
            has, for a component the file declares no chemistry for. False for
            a source whose residue codes mean nothing outside it, since the
            dictionary defines tens of thousands of codes and would answer for
            an unrelated molecule.
        include_bonds: Whether to read the bond table. Without it the chemistry
            cannot be read, so unresolved atoms are not added either.
        extra_fields: Further ``atom_site`` columns to keep, alongside the
            ``label_entity_id`` this always reads.
        reader: Both bonded routes use AtomWorks parsing and bond sanitation.
            ``"tmol"`` restores author identifiers and applies tmol's legacy
            completion policy. ``"atomworks"`` uses label identifiers and
            AtomWorks NaN completion.
            AtomWorks also supports compressed and binary CIF. Its label route requires
            ``use_ccd=True`` and ``include_bonds=True``; extra fields are not
            supported across the AtomWorks completion API.
        hydrogen_policy: ``"rebuild"`` discards supplied hydrogens except observed
            histidine ring protons used for tautomer selection. ``"preserve"``
            retains them and requires compatible hydrogen names/connectivity.
            Applies to bonded reading; without bonds the raw coordinates are kept.

    Returns:
        An AtomWorks AtomArrayPlus when reading bonds, retaining the component
        templates needed for unresolved stereochemistry. Slicing, copying and
        AtomWorks concatenation preserve these templates. Coordinate-only reading
        returns a plain Biotite AtomArray.
    """
    if assembly_id is not None:
        if not isinstance(assembly_id, str) or not assembly_id:
            raise ValueError("assembly_id must be a nonempty string or None")
        if not include_bonds:
            raise ValueError("Assembly reading requires include_bonds=True")
    if reader == "atomworks":
        if not use_ccd or not include_bonds or extra_fields:
            raise ValueError(
                "The AtomWorks reader requires use_ccd=True, include_bonds=True "
                "and no extra_fields. Use reader='tmol' for those input policies."
            )
        from tmol.io._atomworks_reader import read_cif

        return read_cif(
            cif_path,
            model=model,
            assembly_id=assembly_id,
            hydrogen_policy=hydrogen_policy,
        )[0]
    if reader != "tmol":
        raise ValueError(f"Unknown CIF reader {reader!r}; choose 'tmol' or 'atomworks'")
    if include_bonds:
        from tmol.io._atomworks_reader import read_cif

        array, block = read_cif(
            cif_path,
            model=model,
            assembly_id=assembly_id,
            author_fields=True,
            extra_fields=extra_fields,
            hydrogen_policy=hydrogen_policy,
        )
    else:
        cif = pdbx.CIFFile.read(str(cif_path))
        block = cif[next(iter(cif.keys()))]
        array = pdbx.get_structure(
            cif,
            model=model,
            include_bonds=False,
            extra_fields=["label_entity_id", *(extra_fields or [])],
        )
    array = _with_polymer_entity_flag(array, block)
    array = _with_component_type_annotation(array, block)
    if not include_bonds:
        return array
    if not use_ccd:
        from atomworks.io.utils.ccd import build_ccd_entries_from_cif_block

        array._custom_ccd_registry = build_ccd_entries_from_cif_block(
            block, supplement_from_ccd=False
        )
    templates = _completion_templates(
        {name: entry.copy() for name, entry in array._custom_ccd_registry.items()}
    )
    from atomworks.io.utils.leaving_atoms import resolve_overvalent_atoms

    # AtomWorks skips sanitation when its own completion is disabled. Repair
    # the final author-view graph with the same shared attachment chemistry.
    return resolve_overvalent_atoms(
        with_unresolved_atoms(array, templates, use_ccd=use_ccd)
    )


def _with_component_type_annotation(array, block):
    """Carry declared component types from the already parsed CIF into preparation."""
    if "chem_comp" not in block:
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

    Only declared leaving groups at this instance's connections are absent by
    chemistry. Unresolved ring and sidechain atoms retain their chemical identity.
    """
    from atomworks.io.utils.leaving_atoms import get_leaving_atom_groups
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
            # nothing describes this residue, so there is no telling whether
            #    what was resolved is all of it. A code no dictionary defines
            #    is rare, so this stays quiet for ordinary structures
            if res_name not in warned:
                warned.add(res_name)
                logger.warning(
                    "%s: nothing describes this residue -- the file declares "
                    "no chemistry for it and no component dictionary defines "
                    "it -- so it is taken to be complete as resolved. An "
                    "unresolved atom would be protonated rather than left "
                    "open, which would make its residue type a different "
                    "molecule.",
                    res_name,
                )
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
        ends = completed_connection_atoms(residue, frozenset(connections.get(ri, ())))
        missing = set(heavy - present)
        for end in ends or ():
            for group in leaving.get(end, ()):
                if present.isdisjoint(group):
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
        elif field in ("element", "charge"):
            values = getattr(
                template, field, np.zeros(len(template), dtype=source.dtype)
            )
            value = values[indices].astype(source.dtype)
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
    reader: str = "tmol",
    hydrogen_policy: str = "rebuild",
    **kwargs,
):
    """Construct a PoseStack from an mmCIF file.

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
        reader=reader,
        hydrogen_policy=hydrogen_policy,
    )
    return pose_stack_from_biotite(
        array,
        device,
        use_ccd=use_ccd,
        **kwargs,
    )
