"""Preserve terminal chemistry stated by atoms, independently of gap geometry."""

from collections import defaultdict

import biotite.structure as struc
import numpy as np
from atomworks.constants import METAL_ELEMENTS

from tmol.utility.weak_identity_cache import WeakIdentityLRU

EXPLICIT_TERMINI = "tmol_explicit_termini"  # lower=1, upper=2
_TERMINAL_CHEMISTRY = WeakIdentityLRU()
_NONCOVALENT_NEIGHBORS = {*METAL_ELEMENTS, "H", "D"}


def _terminal_chemistry(database, co):
    if co is None:
        from tmol.io._canonical_ordering import CanonicalOrdering

        co = CanonicalOrdering.from_chemdb(database)
    element = {a.name: a.element for a in database.atom_types}
    hydrogen_parents, aliases, ports = defaultdict(set), defaultdict(set), {}
    equivalence = {}
    for residue in database.residues:
        component = residue.io_equiv_class
        equivalence[residue.base_name] = component
        names = {a.name: element[a.atom_type] for a in residue.atoms}
        for alias in residue.atom_aliases:
            aliases[component, alias.name].add(alias.alt_name)
        for a, b, *_ in residue.bonds:
            for hydrogen, parent in ((a, b), (b, a)):
                if names.get(hydrogen) == "H" and names.get(parent) not in (None, "H"):
                    hydrogen_parents[component, hydrogen].add(parent)
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
            hydrogen_parents[component, alias].update(hydrogen_parents[component, name])
            end_atoms[component, alias] |= end_atoms[component, name]
    residue_aliases = {alias.name3: alias.read_as for alias in database.name3_aliases}
    return hydrogen_parents, end_atoms, ports, residue_aliases


def with_stated_termini(structure, database, co=None):
    """Bond otherwise unbonded named hydrogens to their unique database parent.

    Terminal-only atoms state a closed polymer connection even at a short gap.
    An existing covalent bond through that connection is an input contradiction.
    Neither atom names nor coordinates change.
    """
    owners = (database,) if co is None else (database, co)
    parents, terminal_atoms, ports, residue_aliases = (
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
        bonds = struc.connect_via_residue_names(template, inter_residue=False)
    bound = (bonds.get_all_bonds()[0] >= 0).any(axis=1)
    flags = np.zeros(len(template), dtype=np.uint8)
    resolved = np.isfinite(template.coord).all(axis=-1)
    added_bonds = []
    for begin, end in zip(bounds[:-1], bounds[1:]):
        name = template.res_name[begin]
        component = residue_aliases.get(name, name)
        atoms = {str(template.atom_name[i]): i for i in range(begin, end)}
        stated = 0
        for name, index in atoms.items():
            if resolved[index]:
                stated |= terminal_atoms.get((component, name), 0)
            candidates = parents.get((component, name), ())
            if (
                template.element[index] in ("H", "D")
                and not bound[index]
                and len(candidates) == 1
            ):
                parent = atoms.get(next(iter(candidates)))
                if parent is not None:
                    added_bonds.append((index, parent, struc.BondType.SINGLE))
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
