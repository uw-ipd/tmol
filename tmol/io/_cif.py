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
import biotite.structure.info as info
import biotite.structure.io.pdbx as pdbx
import numpy as np

logger = logging.getLogger(__name__)

_BOND_ORDERS = {
    "SING": struc.BondType.SINGLE,
    "DOUB": struc.BondType.DOUBLE,
    "TRIP": struc.BondType.TRIPLE,
    "QUAD": struc.BondType.QUADRUPLE,
}


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


def component_chemistry_from_block(block) -> dict:
    """The declared chemistry of one CIF block; see the path form above."""
    if "chem_comp_atom" not in block:
        return {}
    atoms = block["chem_comp_atom"]
    if "comp_id" not in atoms or "atom_id" not in atoms:
        return {}

    comp = np.char.strip(atoms["comp_id"].as_array(str))
    name = np.char.strip(atoms["atom_id"].as_array(str))
    element = (
        np.char.strip(atoms["type_symbol"].as_array(str))
        if "type_symbol" in atoms
        else np.full(len(name), "", dtype="U4")
    )
    bonds = _authored_bonds(block)

    entries: dict[str, struc.AtomArray] = {}
    for comp_id in sorted(set(str(c) for c in comp)):
        if not comp_id:
            continue
        mask = comp == comp_id
        names = [str(n) for n in name[mask]]
        if len(names) > 1 and comp_id not in bonds:
            logger.warning(
                "%s: the file declares %d atoms but bonds none of them, so "
                "its chemistry cannot be read from the file; falling back "
                "to the component dictionary",
                comp_id,
                len(names),
            )
            continue
        entries[comp_id] = _component_array(
            comp_id, names, [str(e) for e in element[mask]], bonds.get(comp_id, ())
        )
    return entries


def _authored_bonds(block) -> dict:
    """``{comp_id: [(name_a, name_b, BondType), ...]}`` from ``chem_comp_bond``."""
    if "chem_comp_bond" not in block:
        return {}
    category = block["chem_comp_bond"]
    required = ("comp_id", "atom_id_1", "atom_id_2")
    if any(field not in category for field in required):
        return {}
    comp = np.char.strip(category["comp_id"].as_array(str))
    first = np.char.strip(category["atom_id_1"].as_array(str))
    second = np.char.strip(category["atom_id_2"].as_array(str))
    order = (
        np.char.upper(np.char.strip(category["value_order"].as_array(str)))
        if "value_order" in category
        else np.full(len(comp), "SING", dtype="U8")
    )
    aromatic = (
        np.char.strip(category["pdbx_aromatic_flag"].as_array(str)) == "Y"
        if "pdbx_aromatic_flag" in category
        else np.zeros(len(comp), dtype=bool)
    )
    bonds: dict[str, list] = {}
    for c, a, b, o, aro in zip(comp, first, second, order, aromatic):
        kind = (
            struc.BondType.AROMATIC
            if aro
            else _BOND_ORDERS.get(str(o)[:4], struc.BondType.SINGLE)
        )
        bonds.setdefault(str(c), []).append((str(a), str(b), int(kind)))
    return bonds


def _component_array(comp_id, names, elements, bonds) -> struc.AtomArray:
    """One component as an AtomArray with its bonds and no coordinates."""
    array = struc.AtomArray(len(names))
    array.coord = np.full((len(names), 3), np.nan, dtype=np.float32)
    array.atom_name = np.array(names, dtype="U16")
    array.element = np.array(
        [e if e else _element_from_name(n) for n, e in zip(names, elements)], dtype="U4"
    )
    array.res_name = np.array([comp_id] * len(names), dtype="U8")
    array.chain_id = np.array(["A"] * len(names), dtype="U4")
    array.res_id = np.array([1] * len(names), dtype=np.int32)
    array.hetero = np.array([True] * len(names), dtype=bool)
    index = {n: i for i, n in enumerate(names)}
    table = struc.BondList(len(names))
    for a, b, kind in bonds:
        if a in index and b in index:
            table.add_bond(index[a], index[b], kind)
    array.bonds = table
    return array


def _element_from_name(name: str) -> str:
    """Element from a PDB atom name, for a file that declares no type_symbol."""
    from tmol.chemical import get_element_from_atom_name

    return get_element_from_atom_name(name)


def expected_atoms_from_chemistry(template, observed, ends) -> frozenset:
    """The heavy atoms a copy of this residue in this structure should carry.

    ``template`` declares the free molecule, so it also names the atoms
    polymerizing displaces -- the hydroxyl of an acid, a proton of an amine.
    Those are absent by chemistry rather than by density, and are told apart by
    hanging off one of the backbone's ends and appearing in no copy at all.
    Both ends count, not only the ends the structure shows bonded: a residue at
    a chain terminus sheds the same atoms as one in the middle, and the variant
    for that terminus is what puts them back.
    """
    heavy = {
        str(n)
        for n, e in zip(template.atom_name, template.element)
        if str(e).strip().upper() != "H"
    }
    connections = set(ends or ())
    if not connections:
        return frozenset(heavy)
    adjacency: dict = {}
    for i, j, _order in template.bonds.as_array():
        a, b = str(template.atom_name[i]), str(template.atom_name[j])
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    shed = {
        name for name in heavy - observed if adjacency.get(name, set()) & connections
    }
    return frozenset(heavy - shed)


def _component_dictionary_template(res_name: str):
    """The component dictionary's account of this residue, or None."""
    try:
        component = info.residue(res_name)
    except Exception:
        return None
    return component if component.bonds is not None else None


def _accounts_for(observed, template) -> bool:
    """Whether a template names every heavy atom the structure resolved."""
    declared = {
        str(n)
        for n, e in zip(template.atom_name, template.element)
        if str(e).strip().upper() != "H"
    }
    heavy = {n for n in observed if not n.startswith("H")}
    return bool(heavy) and heavy <= declared


def atom_array_from_cif(
    cif_path,
    *,
    model: int = 1,
    use_ccd: bool = True,
    include_bonds: bool = True,
    extra_fields=None,
):
    """A structure's atoms, including the ones its density did not resolve.

    Reads the coordinates with their bonds, then adds back every atom the file
    declares but ``atom_site`` omits, at NaN. What a residue is and what was
    seen of it are then both carried by one array, so nothing downstream has to
    be told the difference.

    Args:
        cif_path: Path to an mmCIF file.
        model: Which model to read.
        use_ccd: Whether the component dictionary may say which atoms a residue
            has, for a component the file declares no chemistry for. False for
            a source whose residue codes mean nothing outside it, since the
            dictionary defines tens of thousands of codes and would answer for
            an unrelated molecule.
        include_bonds: Whether to read the bond table. Without it the chemistry
            cannot be read, so unresolved atoms are not added either.
        extra_fields: Further ``atom_site`` columns to keep, alongside the
            ``label_entity_id`` this always reads.

    Returns:
        A biotite AtomArray.
    """
    cif = pdbx.CIFFile.read(str(cif_path))
    block = cif[next(iter(cif.keys()))]
    fields = ["label_entity_id", *(extra_fields or [])]
    array = pdbx.get_structure(
        cif, model=model, include_bonds=include_bonds, extra_fields=fields
    )
    if isinstance(array, struc.AtomArrayStack):
        array = array[0]
    array = _with_polymer_entity_flag(array, block)
    if not include_bonds:
        return array
    return with_unresolved_atoms(
        array, component_chemistry_from_block(block), use_ccd=use_ccd
    )


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

    An atom the chain sheds is not unresolved -- a residue in a chain never had
    the hydroxyl of its acid -- so those are left out; they are told apart by
    hanging off one of the backbone's ends.
    """
    from tmol.ligand._polymer_profile import completed_connection_atoms

    connections = _connection_atoms_by_name(atom_array)
    starts = struc.get_residue_starts(atom_array)
    warned: set = set()
    additions: dict[int, list] = {}
    for start in starts:
        res_name = str(atom_array.res_name[start]).strip()
        template = declared.get(res_name)
        if template is None and use_ccd:
            template = _component_dictionary_template(res_name)
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
        residue = atom_array[_residue_mask(atom_array, start)]
        present = {str(n) for n in residue.atom_name}
        if not _accounts_for(present, template):
            if res_name in warned:
                continue
            warned.add(res_name)
            logger.warning(
                "%s: the chemistry available for it does not account for the "
                "atoms the file resolved, so it is taken as resolved",
                res_name,
            )
            continue
        ends = completed_connection_atoms(
            residue, frozenset(connections.get(res_name, ()))
        )
        expected = expected_atoms_from_chemistry(template, present, ends)
        missing = sorted(expected - present)
        if missing:
            additions[int(start)] = (missing, template)
    if not additions:
        return atom_array
    return _inserted(atom_array, starts, additions)


def _residue_mask(atom_array, start):
    """Boolean mask of the residue instance the atom at ``start`` belongs to."""
    mask = atom_array.res_id == atom_array.res_id[start]
    mask &= atom_array.res_name == atom_array.res_name[start]
    if hasattr(atom_array, "chain_id"):
        mask &= atom_array.chain_id == atom_array.chain_id[start]
    return mask


def _connection_atoms_by_name(atom_array) -> dict:
    """Atoms each residue name bonds a neighbouring residue through.

    Collected across every copy: a copy at a chain end is bonded on one side
    only, and every copy of a name shares one chemistry.
    """
    if atom_array.bonds is None or atom_array.bonds.get_bond_count() == 0:
        return {}
    names = atom_array.res_name
    ids = atom_array.res_id
    chains = atom_array.chain_id if hasattr(atom_array, "chain_id") else None
    linked: dict = {}
    for i, j, _order in atom_array.bonds.as_array():
        i, j = int(i), int(j)
        same = names[i] == names[j] and ids[i] == ids[j]
        if same and (chains is None or chains[i] == chains[j]):
            continue
        for at in (i, j):
            linked.setdefault(str(names[at]).strip(), set()).add(
                str(atom_array.atom_name[at]).strip()
            )
    return linked


def _inserted(atom_array, starts, additions: dict):
    """``atom_array`` with each residue's missing atoms appended after it.

    Existing bonds are carried across by index, and the added atoms are bonded
    as the template declares, so the result is bonded throughout rather than
    only where the density reached.
    """
    boundaries = list(starts) + [atom_array.array_length()]

    # final layout: each residue's own atoms, then the ones it did not resolve
    pieces = []
    remap = {}
    added_at = {}
    total = 0
    for begin, end in zip(boundaries, boundaries[1:]):
        pieces.append(atom_array[begin:end])
        for old in range(begin, end):
            remap[old] = total
            total += 1
        entry = additions.get(int(begin))
        if entry is None:
            continue
        missing, template = entry
        pieces.append(_placeholder_atoms(atom_array, begin, missing, template))
        for name in missing:
            added_at[(int(begin), name)] = total
            total += 1

    combined = pieces[0]
    for piece in pieces[1:]:
        combined = combined + piece

    bonds = struc.BondList(combined.array_length())
    if atom_array.bonds is not None:
        for i, j, order in atom_array.bonds.as_array():
            bonds.add_bond(remap[int(i)], remap[int(j)], int(order))

    for begin, (missing, template) in additions.items():
        position = {
            str(atom_array.atom_name[old]): remap[old]
            for old in range(begin, _residue_end(boundaries, begin))
        }
        position.update({name: added_at[(begin, name)] for name in missing})
        for i, j, order in template.bonds.as_array():
            a, b = str(template.atom_name[i]), str(template.atom_name[j])
            if (a in missing or b in missing) and a in position and b in position:
                bonds.add_bond(position[a], position[b], int(order))

    combined.bonds = bonds
    return combined


def _residue_end(boundaries, begin) -> int:
    """Where the residue starting at ``begin`` ends in the original array."""
    return boundaries[boundaries.index(begin) + 1]


def _placeholder_atoms(atom_array, begin, missing, template):
    """The unresolved atoms of one residue, at NaN, annotated like their residue."""
    element = {str(n): str(e) for n, e in zip(template.atom_name, template.element)}
    extra = struc.AtomArray(len(missing))
    extra.coord = np.full((len(missing), 3), np.nan, dtype=np.float32)
    for field in atom_array.get_annotation_categories():
        source = atom_array.get_annotation(field)
        if field == "atom_name":
            value = np.array(missing, dtype=source.dtype)
        elif field == "element":
            value = np.array([element.get(n, "") for n in missing], dtype=source.dtype)
        else:
            value = np.array([source[begin]] * len(missing), dtype=source.dtype)
        extra.set_annotation(field, value)
    return extra


def pose_stack_from_cif(cif_path, device, *, use_ccd: bool = True, **kwargs):
    """Construct a PoseStack from an mmCIF file.

    Reads the structure with :func:`atom_array_from_cif`, so the residues carry
    every atom their chemistry declares and the unresolved ones arrive at NaN
    for pose construction to rebuild. Further keyword arguments are passed to
    :func:`tmol.io.pose_stack_from_biotite`.
    """
    from tmol.ligand import chem_comp_types_from_cif
    from tmol.io._pose_stack_from_biotite import pose_stack_from_biotite

    kwargs.setdefault("chem_comp_types", chem_comp_types_from_cif(cif_path))
    return pose_stack_from_biotite(
        atom_array_from_cif(cif_path, use_ccd=use_ccd),
        device,
        use_ccd=use_ccd,
        **kwargs,
    )
