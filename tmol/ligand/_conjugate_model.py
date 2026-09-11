"""Complete, capped chemical models for non-polymer connections in an input.

Models retain residue-instance and source-atom identity. Caps replace ordinary
polymer neighbours, while every attachment within a connected group is kept.
They are topology-only inputs for parameter generation, not scored coordinates.
"""

from dataclasses import dataclass

import biotite.structure as struc
import networkx as nx
import numpy as np

from tmol.ligand._polymer_profile import cap_residue, profile_for_atom_array


@dataclass(frozen=True)
class CappedConjugateModel:
    atom_array: struc.AtomArray
    # Original input indices; synthetic cap atoms have index -1.
    source_atom_indices: np.ndarray
    source_residue_indices: np.ndarray
    # Original input atom indices and declared Biotite bond order.
    connections: tuple[tuple[int, int, int], ...]


def _polymer_connection(residue, atom):
    if not residue.properties.polymer.is_polymer:
        return None
    return next(
        (
            c.name
            for c in residue.connections
            if c.name in ("up", "down") and c.atom == atom
        ),
        None,
    )


def capped_conjugate_models(atom_array, chemical_database):
    """Build owned models using a database in which input residues are prepared.

    Ordinary up/down polymer bonds are cut and replaced by the existing cap
    profiles. Covalent groups may contain repeated residue names, multiple
    polymer anchors, or no polymer anchor. No coordinates or residue-name
    matching are used to infer attachment bonds. Unknown residue definitions
    at a cross-residue bond raise instead of guessing polymer connectivity.
    """
    if atom_array.bonds is None:
        raise ValueError("Conjugate models require explicit input bonds")
    starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
    indices = np.repeat(np.arange(len(starts) - 1), np.diff(starts))
    definitions = {
        r.name: r for r in chemical_database.residues if r.name == r.base_name
    }
    for alias in chemical_database.name3_aliases:
        if alias.read_as in definitions:
            definitions.setdefault(alias.name3, definitions[alias.read_as])
    residue_definitions = {}

    def definition(ri):
        if ri not in residue_definitions:
            name = str(atom_array.res_name[starts[ri]])
            if name not in definitions:
                raise ValueError(
                    f"Unprepared residue {ri} ({name}) in a conjugate model"
                )
            residue_definitions[ri] = definitions[name]
        return residue_definitions[ri]

    links_by_residue = {}
    graph = nx.Graph()
    bonds = atom_array.bonds.as_array()
    cross = bonds[indices[bonds[:, 0]] != indices[bonds[:, 1]]]
    for first, second, order in cross:
        first, second = int(first), int(second)
        ri, rj = int(indices[first]), int(indices[second])
        ci = _polymer_connection(definition(ri), str(atom_array.atom_name[first]))
        cj = _polymer_connection(definition(rj), str(atom_array.atom_name[second]))
        if {ci, cj} == {"up", "down"}:
            continue
        if atom_array.element[first] == "H" or atom_array.element[second] == "H":
            raise ValueError(
                "Cross-residue hydrogen bonds are not covalent heavy-atom attachments"
            )
        links_by_residue.setdefault(ri, []).append((first, second, int(order)))
        graph.add_edge(ri, rj)

    models = []
    for members in nx.connected_components(graph):
        arrays, source_atoms, source_residues = [], [], []
        for ri in sorted(members):
            start, stop = starts[ri : ri + 2]
            source = atom_array[start:stop]
            names = list(map(str, source.atom_name))
            if len(set(names)) != len(names):
                raise ValueError(f"Duplicate atom names in residue instance {ri}")
            source_by_name = {name: int(start + i) for i, name in enumerate(names)}
            residue_type = definition(ri)
            if residue_type.properties.polymer.is_polymer:
                connections = frozenset(
                    c.atom for c in residue_type.connections if c.name in ("up", "down")
                )
                profile = profile_for_atom_array(source, connections, chemical_database)
                if profile is None:
                    raise ValueError(
                        f"No cap profile for polymer residue {ri} ({residue_type.name})"
                    )
                prepared, caps = cap_residue(source, profile, include_coordinates=False)
                cap_names = set(caps.values())
            else:
                prepared = source[source.element != "H"]
                prepared.coord[:] = np.nan
                cap_names = set()
            arrays.append(prepared)
            source_atoms.extend(
                -1 if name in cap_names else source_by_name[str(name)]
                for name in prepared.atom_name
            )
            source_residues.extend([ri] * len(prepared))
        combined = struc.concatenate(arrays)
        remap = {old: new for new, old in enumerate(source_atoms) if old >= 0}
        group_links = tuple(
            link for ri in sorted(members) for link in links_by_residue.get(ri, ())
        )
        if any(a not in remap or b not in remap for a, b, _ in group_links):
            raise ValueError("Capping removed an atom involved in an attachment")
        connections = np.asarray(
            [(remap[a], remap[b], order) for a, b, order in group_links], dtype=np.int64
        )
        combined.bonds = struc.BondList(
            len(combined), np.concatenate((combined.bonds.as_array(), connections))
        )
        models.append(
            CappedConjugateModel(
                combined,
                np.asarray(source_atoms, dtype=np.int64),
                np.asarray(source_residues, dtype=np.int64),
                group_links,
            )
        )
    return tuple(models)
