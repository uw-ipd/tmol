"""Import explicit, non-polymeric covalent bonds from Biotite structures."""

from collections import defaultdict
import math

import attr
import biotite.structure as struc
import numpy as np
import torch

from tmol.database.chemical import Connection, Icoor
from tmol.pose import InterResidueConnection, connect_pose_blocks


def _template(structure):
    return structure[0] if isinstance(structure, struc.AtomArrayStack) else structure


def _residue_keys(array):
    starts = struc.get_residue_starts(array)
    ends = np.append(starts[1:], array.array_length())
    ins_codes = (
        array.ins_code if "ins_code" in array.get_annotation_categories() else None
    )
    keys = []
    for start, end in zip(starts, ends):
        keys.append(
            (
                str(array.chain_id[start]),
                int(array.res_id[start]),
                str(ins_codes[start]) if ins_codes is not None else "",
                str(array.res_name[start]),
            )
        )
    atom_to_residue = (
        np.searchsorted(starts, np.arange(array.array_length()), side="right") - 1
    )
    return starts, ends, keys, atom_to_residue


def _is_standard_polymer_bond(keys, res1, atom1, res2, atom2):
    """Return whether a cross-residue bond is an adjacent polymer link."""

    if abs(res1 - res2) != 1:
        return False
    if keys[res1][0] != keys[res2][0]:
        return False
    first_atom, second_atom = (atom1, atom2) if res1 < res2 else (atom2, atom1)
    return (first_atom, second_atom) in (("C", "N"), ("O3'", "P"))


def _explicit_cross_residue_bonds(structure):
    """Return nonstandard cross-residue bonds from a Biotite bond table."""

    array = _template(structure)
    if array.bonds is None:
        return ()
    _, _, keys, atom_to_residue = _residue_keys(array)
    result = []
    occupied = set()
    for atom1, atom2, _bond_type in array.bonds.as_array():
        res1, res2 = int(atom_to_residue[atom1]), int(atom_to_residue[atom2])
        if res1 == res2:
            continue
        name1, name2 = str(array.atom_name[atom1]), str(array.atom_name[atom2])
        if _is_standard_polymer_bond(keys, res1, name1, res2, name2) or {
            name1,
            name2,
        } == {"SG"}:
            continue
        endpoints = ((keys[res1], name1), (keys[res2], name2))
        for endpoint in endpoints:
            if endpoint in occupied:
                raise ValueError(
                    f"covalent attachment atom {endpoint[0]}:{endpoint[1]} "
                    "has more than one inter-residue bond"
                )
            occupied.add(endpoint)
        result.append(endpoints)
    return tuple(result)


def _angle(a, b, c):
    ba, bc = a - b, c - b
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-12:
        return 0.0
    return float(np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)))


def _dihedral(a, b, c, d):
    b1, b2, b3 = b - a, c - b, d - c
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    norms = (
        float(np.linalg.norm(n1)),
        float(np.linalg.norm(n2)),
        float(np.linalg.norm(b2)),
    )
    if min(norms) < 1e-12:
        return 0.0
    n1, n2 = n1 / norms[0], n2 / norms[1]
    return float(np.arctan2(np.dot(np.cross(n1, b2 / norms[2]), n2), np.dot(n1, n2)))


def _connection_icoor(raw, connection_name, atom, local_coords, remote_coord):
    adjacency = defaultdict(list)
    for name1, name2, *_ in raw.bonds:
        adjacency[name1].append(name2)
        adjacency[name2].append(name1)
    gp_candidates = [name for name in adjacency[atom] if name in local_coords]
    if not gp_candidates:
        raise ValueError(f"connection atom {raw.name}:{atom} has no local frame")
    gp = sorted(gp_candidates)[0]
    ggp_candidates = [
        name for name in adjacency[gp] if name != atom and name in local_coords
    ]
    if not ggp_candidates:
        ggp_candidates = [name for name in local_coords if name not in (atom, gp)]
    if not ggp_candidates:
        raise ValueError(f"connection atom {raw.name}:{atom} lacks a third frame atom")
    ggp = sorted(ggp_candidates)[0]
    return Icoor(
        name=connection_name,
        phi=-_dihedral(
            remote_coord, local_coords[atom], local_coords[gp], local_coords[ggp]
        ),
        theta=math.pi - _angle(remote_coord, local_coords[atom], local_coords[gp]),
        d=float(np.linalg.norm(remote_coord - local_coords[atom])),
        parent=atom,
        grand_parent=gp,
        great_grand_parent=ggp,
    )


def _virtualize_leaving_hydrogens(raw, attachment_atoms, atom_type_elements):
    """Turn one bonded hydrogen per new attachment into an inert virtual atom."""

    adjacency = defaultdict(list)
    for atom1, atom2, *_ in raw.bonds:
        adjacency[atom1].append(atom2)
        adjacency[atom2].append(atom1)
    atom_by_name = {atom.name: atom for atom in raw.atoms}
    leaving = []
    for attachment in attachment_atoms:
        candidates = sorted(
            neighbor
            for neighbor in adjacency[attachment]
            if atom_type_elements[atom_by_name[neighbor].atom_type] == "H"
            and neighbor not in leaving
        )
        if candidates:
            leaving.append(candidates[0])
    if not leaving:
        return raw.atoms, raw.properties
    atoms = tuple(
        attr.evolve(atom, atom_type="Vrt") if atom.name in leaving else atom
        for atom in raw.atoms
    )
    properties = attr.evolve(
        raw.properties,
        virtual=tuple(dict.fromkeys((*raw.properties.virtual, *leaving))),
    )
    return atoms, properties


def _is_carbohydrate(residue_key):
    from tmol.ligand._detect import get_chem_comp_type

    return "SACCHARIDE" in (get_chem_comp_type(residue_key[3]) or "").upper()


def _carbohydrate_connection_roles(structure, bonds):
    """Name sugar connections from the input covalent graph, not residue names.

    ``down`` points toward the non-carbohydrate anchor (or deterministic root),
    ``up`` points toward the first child, and further children are branches.
    """

    if not bonds:
        return {}
    array = _template(structure)
    _, _, keys, _ = _residue_keys(array)
    key_to_index = {key: index for index, key in enumerate(keys)}
    sugars = {key for bond in bonds for key, _ in bond if _is_carbohydrate(key)}
    if not sugars:
        return {}

    sugar_edges = []
    anchors = defaultdict(list)
    adjacency = defaultdict(list)
    for endpoint1, endpoint2 in bonds:
        key1, key2 = endpoint1[0], endpoint2[0]
        sugar1, sugar2 = key1 in sugars, key2 in sugars
        if sugar1 and sugar2:
            edge_index = len(sugar_edges)
            sugar_edges.append((endpoint1, endpoint2))
            adjacency[key1].append((key2, edge_index))
            adjacency[key2].append((key1, edge_index))
        elif sugar1:
            anchors[key1].append(endpoint1)
        elif sugar2:
            anchors[key2].append(endpoint2)

    roles = {}
    unvisited = set(sugars)
    while unvisited:
        seed = min(unvisited, key=key_to_index.__getitem__)
        component = set()
        pending = [seed]
        while pending:
            current = pending.pop()
            if current in component:
                continue
            component.add(current)
            pending.extend(neighbor for neighbor, _ in adjacency[current])

        anchored = sorted(
            (key for key in component if anchors[key]), key=key_to_index.__getitem__
        )
        root = anchored[0] if anchored else min(component, key=key_to_index.__getitem__)
        parent = {root: None}
        queue = [root]
        tree_edges = set()
        while queue:
            current = queue.pop(0)
            for neighbor, edge_index in sorted(
                adjacency[current], key=lambda item: key_to_index[item[0]]
            ):
                if neighbor in parent:
                    continue
                parent[neighbor] = current
                tree_edges.add(edge_index)
                queue.append(neighbor)

        for key in sorted(component, key=key_to_index.__getitem__):
            for anchor_index, endpoint in enumerate(
                sorted(anchors[key], key=lambda item: item[1])
            ):
                roles[endpoint] = (
                    "down"
                    if key == root and anchor_index == 0
                    else (f"branch_{endpoint[1]}")
                )

        children = defaultdict(list)
        for edge_index in tree_edges:
            endpoint1, endpoint2 = sugar_edges[edge_index]
            if parent[endpoint2[0]] == endpoint1[0]:
                parent_endpoint, child_endpoint = endpoint1, endpoint2
            else:
                parent_endpoint, child_endpoint = endpoint2, endpoint1
            children[parent_endpoint[0]].append((child_endpoint[0], parent_endpoint))
            roles[child_endpoint] = "down"
        for key, entries in children.items():
            for child_index, (_, endpoint) in enumerate(
                sorted(entries, key=lambda item: key_to_index[item[0]])
            ):
                roles[endpoint] = "up" if child_index == 0 else f"branch_{endpoint[1]}"

        for edge_index, (endpoint1, endpoint2) in enumerate(sugar_edges):
            if edge_index in tree_edges:
                continue
            for endpoint in (endpoint1, endpoint2):
                roles[endpoint] = f"branch_{endpoint[1]}"
        unvisited -= component
    return roles


def _shortest_atom_path(raw, start, end):
    adjacency = defaultdict(list)
    for atom1, atom2, *_ in raw.bonds:
        adjacency[atom1].append(atom2)
        adjacency[atom2].append(atom1)
    parent = {start: None}
    queue = [start]
    while queue and end not in parent:
        current = queue.pop(0)
        for neighbor in sorted(adjacency[current]):
            if neighbor not in parent:
                parent[neighbor] = current
                queue.append(neighbor)
    if end not in parent:
        raise ValueError(f"{raw.name}: no bonded path from {start} to {end}")
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = parent[current]
    return tuple(reversed(path))


def _carbohydrate_properties(raw, connections, base_properties):
    named = {name: atom for atom, name in connections}
    if not ({"down", "up"} & set(named)):
        return base_properties
    if "down" in named and "up" in named:
        mainchain = _shortest_atom_path(raw, named["down"], named["up"])
    else:
        mainchain = (named.get("down") or named["up"],)
    polymer = attr.evolve(
        base_properties.polymer,
        is_polymer=True,
        polymer_type="carbohydrate",
        backbone_type="carbohydrate",
        mainchain_atoms=mainchain,
        sidechain_chirality="NA",
        termini_variants=(),
    )
    return attr.evolve(
        base_properties,
        polymer=polymer,
        chemical_modifications=tuple(
            dict.fromkeys(
                (*base_properties.chemical_modifications, "generated_carbohydrate")
            )
        ),
    )


def augment_database_for_covalent_bonds(structure, param_db):
    """Add same-atom-layout residue variants needed by explicit input bonds."""

    bonds = _explicit_cross_residue_bonds(structure)
    if not bonds:
        return param_db, {}

    array = _template(structure)
    starts, ends, keys, _ = _residue_keys(array)
    key_to_residue = {key: i for i, key in enumerate(keys)}
    connection_roles = _carbohydrate_connection_roles(structure, bonds)
    attachment_connections = defaultdict(dict)
    partner_coord = {}
    for (key1, atom1), (key2, atom2) in bonds:
        attachment_connections[key1][atom1] = connection_roles.get(
            (key1, atom1), f"covalent_{atom1}"
        )
        attachment_connections[key2][atom2] = connection_roles.get(
            (key2, atom2), f"covalent_{atom2}"
        )
        res1, res2 = key_to_residue[key1], key_to_residue[key2]
        inds1 = range(starts[res1], ends[res1])
        inds2 = range(starts[res2], ends[res2])
        idx1 = next(i for i in inds1 if array.atom_name[i] == atom1)
        idx2 = next(i for i in inds2 if array.atom_name[i] == atom2)
        partner_coord[(key1, atom1)] = np.asarray(array.coord[idx2], dtype=np.float64)
        partner_coord[(key2, atom2)] = np.asarray(array.coord[idx1], dtype=np.float64)

    patterns = set()
    geometry = {}
    for key, connection_by_atom in attachment_connections.items():
        res_ind = key_to_residue[key]
        local = {
            str(array.atom_name[i]): np.asarray(array.coord[i], dtype=np.float64)
            for i in range(starts[res_ind], ends[res_ind])
        }
        connection_spec = tuple(sorted(connection_by_atom.items()))
        pattern = (key[3], connection_spec)
        patterns.add(pattern)
        geometry.setdefault(
            pattern,
            (local, {atom: partner_coord[(key, atom)] for atom in connection_by_atom}),
        )

    clones = []
    variant_names = {}
    supported_patterns = set()
    atom_type_elements = {
        atom_type.name: atom_type.element for atom_type in param_db.chemical.atom_types
    }
    for raw in param_db.chemical.residues:
        for res_name, connection_spec in sorted(patterns):
            atoms = tuple(atom for atom, _ in connection_spec)
            if raw.name3 != res_name or not set(atoms) <= {a.name for a in raw.atoms}:
                continue
            local, remotes = geometry[(res_name, connection_spec)]
            generic_names = all(
                name == f"covalent_{atom}" for atom, name in connection_spec
            )
            suffix = ",".join(
                atom if generic_names else name for atom, name in connection_spec
            )
            clone_name = f"{raw.name}:covalent_{suffix}"
            added_connections = tuple(
                Connection(name=name, atom=atom, type="SINGLE")
                for atom, name in connection_spec
            )
            added_icoors = tuple(
                _connection_icoor(raw, name, atom, local, remotes[atom])
                for atom, name in connection_spec
            )
            clone_atoms, clone_properties = _virtualize_leaving_hydrogens(
                raw, atoms, atom_type_elements
            )
            is_carbohydrate = _is_carbohydrate(("", 0, "", res_name))
            clone = attr.evolve(
                raw,
                name=clone_name,
                # Keep the input-facing NAG/MAN/etc. equivalence class
                # non-polymeric while the pose is first assigned.  The
                # connection-capable clone is swapped in afterwards, so its
                # graph-derived polymer semantics must not make every member
                # of the source equivalence class look like a sequence
                # polymer.
                io_equiv_class=clone_name if is_carbohydrate else raw.io_equiv_class,
                atoms=clone_atoms,
                connections=(*raw.connections, *added_connections),
                icoors=(*raw.icoors, *added_icoors),
                properties=(
                    _carbohydrate_properties(raw, connection_spec, clone_properties)
                    if is_carbohydrate
                    else clone_properties
                ),
            )
            clones.append(clone)
            variant_names[(raw.name, connection_spec)] = clone_name
            if generic_names:
                variant_names[(raw.name, atoms)] = clone_name
            supported_patterns.add((res_name, connection_spec))

    missing = sorted(set(patterns) - supported_patterns)
    if missing:
        raise ValueError(
            f"no residue type supports covalent attachment pattern(s): {missing}"
        )
    chemical = attr.evolve(
        param_db.chemical, residues=(*param_db.chemical.residues, *clones)
    )
    return attr.evolve(param_db, chemical=chemical), variant_names


def apply_covalent_bonds_from_biotite(pose_stack, structure, variant_names):
    """Select connection-capable variants and install explicit input bonds."""

    bonds = _explicit_cross_residue_bonds(structure)
    if not bonds:
        return pose_stack
    connection_roles = _carbohydrate_connection_roles(structure, bonds)
    attachment_connections = defaultdict(dict)
    for endpoint1, endpoint2 in bonds:
        for key, atom in (endpoint1, endpoint2):
            attachment_connections[key][atom] = connection_roles.get(
                (key, atom), f"covalent_{atom}"
            )

    pbt = pose_stack.packed_block_types
    name_to_ind = {bt.name: i for i, bt in enumerate(pbt.active_block_types)}
    type64 = pose_stack.block_type_ind64.clone()
    key_to_block = {}
    for pose_index in range(len(pose_stack)):
        for block in range(pose_stack.max_n_blocks):
            if type64[pose_index, block] < 0:
                continue
            key = (
                str(pose_stack.pdb_info.chain_labels[pose_index, block]),
                int(pose_stack.pdb_info.residue_labels[pose_index, block]),
                str(pose_stack.pdb_info.residue_insertion_codes[pose_index, block]),
            )
            key_to_block[(pose_index, key)] = block

    for pose_index in range(len(pose_stack)):
        for input_key, connection_by_atom in attachment_connections.items():
            block = key_to_block[(pose_index, input_key[:3])]
            original = pbt.active_block_types[int(type64[pose_index, block])]
            connection_spec = tuple(sorted(connection_by_atom.items()))
            clone_name = variant_names[(original.name, connection_spec)]
            clone = pbt.active_block_types[name_to_ind[clone_name]]
            if tuple(a.name for a in clone.atoms) != tuple(
                a.name for a in original.atoms
            ):
                raise ValueError("covalent variants must preserve atom layout")
            type64[pose_index, block] = name_to_ind[clone_name]

    pose_stack = attr.evolve(
        pose_stack,
        coords=pose_stack.coords.clone(),
        block_type_ind64=type64,
        block_type_ind=type64.to(torch.int32),
    )
    connections = []
    for pose_index in range(len(pose_stack)):
        for (key1, atom1), (key2, atom2) in bonds:
            connections.append(
                InterResidueConnection(
                    pose_index,
                    key_to_block[(pose_index, key1[:3])],
                    attachment_connections[key1][atom1],
                    key_to_block[(pose_index, key2[:3])],
                    attachment_connections[key2][atom2],
                )
            )
    return connect_pose_blocks(pose_stack, connections)
