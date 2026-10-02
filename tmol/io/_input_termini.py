"""Preserve terminal chemistry stated by atoms, independently of gap geometry."""

from collections import Counter, defaultdict

import biotite.structure as struc
import numpy as np
from atomworks.constants import METAL_ELEMENTS

from tmol.utility.weak_identity_cache import WeakIdentityLRU

EXPLICIT_TERMINI = "tmol_explicit_termini"  # lower=1, upper=2
_TERMINAL_CHEMISTRY = WeakIdentityLRU()
_CONNECTION_HYDROGEN_CAPACITIES = WeakIdentityLRU()
_NONCOVALENT_NEIGHBORS = {*METAL_ELEMENTS, "H", "D"}


def _connection_hydrogen_capacities(database):
    """Maximum H count on each polymer connection atom across all database forms."""
    element = {a.name: a.element.upper() for a in database.atom_types}
    connections = {
        (r.io_equiv_class, c.atom)
        for r in database.residues
        for c in r.connections
        if c.name in ("up", "down")
    }
    capacity = defaultdict(int)
    for residue in database.residues:
        hydrogens = {
            a.name for a in residue.atoms if element[a.atom_type] in ("H", "D")
        }
        counts = Counter(
            parent
            for a, b, *_ in residue.bonds
            for parent, hydrogen in ((a, b), (b, a))
            if hydrogen in hydrogens
        )
        for atom in residue.atoms:
            key = residue.io_equiv_class, atom.name
            if key in connections:
                capacity[key] = max(capacity[key], counts[atom.name])
    return capacity


def validate_connection_hydrogens(structure, database):
    """Reject stated terminal H that no available residue form can retain.

    An aldehyde hydrogen on the backbone carbon must not silently become OXT.
    Hydrogen names are irrelevant; supplied covalent bonds establish the parent.
    """
    template = (
        structure[0] if isinstance(structure, struc.AtomArrayStack) else structure
    )
    if template.bonds is None:
        return
    is_h = np.isin(np.char.upper(template.element.astype(str)), ("H", "D"))
    if not is_h.any():
        return
    capacities = _CONNECTION_HYDROGEN_CAPACITIES.get_or_create(
        database, (), lambda: _connection_hydrogen_capacities(database)
    )
    aliases = {a.name3: a.read_as for a in database.name3_aliases}
    bonds = template.bonds.as_array()
    bonds = bonds[bonds[:, 2] != struc.BondType.COORDINATION, :2]
    parents = np.r_[bonds[is_h[bonds[:, 1]], 0], bonds[is_h[bonds[:, 0]], 1]]
    for parent, count in Counter(parents.tolist()).items():
        name = str(template.res_name[parent])
        atom = str(template.atom_name[parent])
        maximum = capacities.get((aliases.get(name, name), atom))
        if maximum is not None and count > maximum:
            raise ValueError(
                "Unsupported terminal hydrogen count at "
                f"{template.chain_id[parent]}:{template.res_id[parent]}"
                f"{template.ins_code[parent]}:{name}/{atom}: supplied {count}, "
                f"database forms support at most {maximum}; provide parameters "
                "for the declared terminal chemistry"
            )


def _terminal_chemistry(database, co):
    if co is None:
        from tmol.io._canonical_ordering import CanonicalOrdering

        co = CanonicalOrdering.from_chemdb(database)
    element = {a.name: a.element for a in database.atom_types}
    neighbors, aliases, ports = defaultdict(set), defaultdict(set), {}
    elements = defaultdict(set)
    equivalence = {}
    for residue in database.residues:
        component = residue.io_equiv_class
        equivalence[residue.base_name] = component
        names = {a.name: element[a.atom_type] for a in residue.atoms}
        for name, kind in names.items():
            elements[component, name].add(kind)
        for alias in residue.atom_aliases:
            aliases[component, alias.name].add(alias.alt_name)
        for a, b, *_ in residue.bonds:
            neighbors[component, a].add(b)
            neighbors[component, b].add(a)
        for connection in residue.connections:
            if connection.name in ("down", "up"):
                ports[component, 1 if connection.name == "down" else 2] = (
                    connection.atom
                )
    end_atoms = defaultdict(int)
    for (base, patch), atoms in co.termini_only_atoms.items():
        side = int(patch in co.down_termini_patches) + 2 * int(
            patch in co.up_termini_patches
        )
        for atom in atoms:
            end_atoms[equivalence[base], atom] |= side
    for (component, name), alternatives in aliases.items():
        for alias in alternatives:
            neighbors[component, alias].update(neighbors[component, name])
            elements[component, alias].update(elements[component, name])
            end_atoms[component, alias] |= end_atoms[component, name]
    residue_aliases = {alias.name3: alias.read_as for alias in database.name3_aliases}
    return neighbors, end_atoms, ports, residue_aliases, elements


def _inferred_residue_bonds(template):
    """Use source component definitions before generated database hydrogen names."""
    custom = {
        name: {
            (str(entry.atom_name[i]), str(entry.atom_name[j])): int(kind)
            for i, j, kind in entry.bonds.as_array()
        }
        for name, entry in getattr(template, "_custom_ccd_registry", {}).items()
        if entry.bonds is not None
    }
    return struc.connect_via_residue_names(
        template, inter_residue=False, custom_bond_dict=custom
    )


def with_stated_termini(structure, database, co=None):
    """Fill missing hydrogen bonds and preserve stated terminal chemistry.

    A terminal atom's element and bonded parent must match the database. Source
    component definitions and supplied bonds override generated hydrogen names.
    Terminal-only atoms state a closed polymer connection even at a short gap.
    An existing covalent bond through that connection is an input contradiction.
    Neither atom names nor coordinates change.
    """
    owners = (database,) if co is None else (database, co)
    parents, terminal_atoms, ports, residue_aliases, elements = (
        _TERMINAL_CHEMISTRY.get_or_create_many(
            owners, (), lambda: _terminal_chemistry(database, co)
        )
    )
    template = (
        structure[0] if isinstance(structure, struc.AtomArrayStack) else structure
    )
    bounds = struc.get_residue_starts(template, add_exclusive_stop=True)
    bonds = template.bonds
    if bonds is None:
        bonds = _inferred_residue_bonds(template)
    bound = (bonds.get_all_bonds()[0] >= 0).any(axis=1)
    flags = np.zeros(len(template), dtype=np.uint8)
    resolved = np.isfinite(template.coord).all(axis=-1)
    added_bonds = []
    inferred = (
        _inferred_residue_bonds(template)
        if np.any(~bound & np.isin(template.element, ("H", "D")))
        else bonds
    )
    for begin, end in zip(bounds[:-1], bounds[1:]):
        name = template.res_name[begin]
        component = residue_aliases.get(name, name)
        atoms = {str(template.atom_name[i]): i for i in range(begin, end)}
        stated = 0
        valid_parents = {
            i
            for n, i in atoms.items()
            if template.element[i].upper() in elements.get((component, n), ())
        }
        for name, index in atoms.items():
            candidates = parents.get((component, name), ())
            kind = template.element[index].upper()
            kind = "H" if kind == "D" else kind
            terminal = terminal_atoms.get((component, name), 0)
            if kind not in elements.get((component, name), ()) or (
                not terminal and not (kind == "H" and not bound[index])
            ):
                continue
            expected = {atoms[n] for n in candidates if n in atoms} & valid_parents
            neighbors, bond_types = bonds.get_bonds(index)
            covalent = (bond_types != struc.BondType.COORDINATION) & ~np.isin(
                template.element[neighbors], list(METAL_ELEMENTS)
            )
            neighbors = neighbors[covalent]
            if kind == "H" and not bound[index]:
                neighbors = inferred.get_bonds(index)[0]
                if not len(neighbors) and len(candidates) == 1:
                    neighbors = list(expected)
                if len(neighbors) == 1 and neighbors[0] in valid_parents:
                    added_bonds.append((index, neighbors[0], struc.BondType.SINGLE))
            if kind != "H":
                neighbors = [
                    n for n in neighbors if template.element[n] not in ("H", "D")
                ]
            if resolved[index] and set(neighbors) <= expected:
                stated |= terminal
        flags[begin:end] = stated
        for side in (1, 2):
            parent = atoms.get(ports.get((component, side)))
            if not stated & side or parent is None:
                continue
            neighbors, kinds = bonds.get_bonds(parent)
            for neighbor, kind in zip(neighbors, kinds):
                if (
                    not begin <= neighbor < end
                    and kind != struc.BondType.COORDINATION
                    and template.element[neighbor].upper() not in _NONCOVALENT_NEIGHBORS
                ):
                    raise ValueError(
                        "Stated terminus conflicts with a covalent connection: "
                        f"{template.chain_id[parent]}:{template.res_id[parent]}:"
                        f"{component}/{template.atom_name[parent]} -- "
                        f"{template.chain_id[neighbor]}:{template.res_id[neighbor]}:"
                        f"{template.res_name[neighbor]}/{template.atom_name[neighbor]}"
                    )
    if (
        not added_bonds
        and not flags.any()
        and EXPLICIT_TERMINI not in template.get_annotation_categories()
    ):
        return structure
    result = structure.copy()
    if added_bonds:
        result.bonds = bonds.merge(
            struc.BondList(len(template), np.asarray(added_bonds))
        )
    elif template.bonds is None:
        result.bonds = bonds
    result.set_annotation(EXPLICIT_TERMINI, flags)
    return result
