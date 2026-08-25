"""Import explicit metal coordination bonds from Biotite structures."""

from collections import Counter, defaultdict
import math

import attr
import numpy as np
import torch

from tmol.database import ParameterDatabase, inject_residue_params
from tmol.database.chemical import (
    Atom,
    AtomType,
    ChemicalProperties,
    Connection,
    Element,
    Icoor,
    PolymerProperties,
    ProtonationProperties,
    RawResidueType,
)
from tmol.database.scoring._ljlk import LJLKAtomTypeParameters
from tmol.io._covalent_bonds import (
    _connection_icoor,
    _residue_keys,
    _template,
    _virtualize_leaving_hydrogens,
)
from tmol.pose import ConstraintSet, InterResidueConnection, connect_pose_blocks
from tmol.score.constraint import ConstraintEnergyTerm

MAX_COORDINATION_CONNECTIONS = 8

# Rosetta fa_standard ion physics.  This is an element-level force-field table,
# not a residue-name catalog: deposited component and atom names are preserved
# when residue types are generated below.
_METAL_PARAMETERS = {
    "Mg": (12, "Mg2p", 2.0, (1.185, 0.015, -5.0, 3.5, 7.0)),
    "Ca": (20, "Ca2p", 2.0, (1.367, 0.120, 0.0, 2.0, 10.7)),
    "Zn": (30, "Zn2p", 2.0, (1.090, 0.250, -5.0, 3.5, 5.4)),
}


def _element(value):
    value = str(value).strip()
    return value[:1].upper() + value[1:].lower()


def _metal_residues(structure):
    """Return supported single-ion components keyed by deposited name."""

    array = _template(structure)
    starts, ends, keys, _ = _residue_keys(array)
    result = {}
    for start, end, key in zip(starts, ends, keys):
        supported = [
            index
            for index in range(start, end)
            if _element(array.element[index]) in _METAL_PARAMETERS
        ]
        if not supported:
            continue
        if len(supported) != 1 or end - start != 1:
            raise ValueError(
                f"{key}: supported metal components must contain one ion atom"
            )
        index = supported[0]
        value = (_element(array.element[index]), str(array.atom_name[index]))
        previous = result.setdefault(key[3], value)
        if previous != value:
            raise ValueError(
                f"metal component name {key[3]!r} denotes inconsistent ions"
            )
    return result


def _metal_residue_type(name, atom_name, atom_type):
    virtuals = tuple(f"V{index}" for index in range(1, 9))
    atoms = (Atom(atom_name, atom_type), *(Atom(name, "Vrt") for name in virtuals))
    bonds = tuple((atom_name, name, "SINGLE", False) for name in virtuals)
    icoors = (
        Icoor(atom_name, 0.0, 0.0, 0.0, atom_name, "V1", "V2"),
        Icoor("V1", 0.0, math.pi, 1.0, atom_name, "V1", "V2"),
        Icoor("V2", 0.0, math.pi / 2, 1.0, atom_name, "V1", "V2"),
        Icoor("V3", math.pi / 2, math.pi / 2, 1.0, atom_name, "V2", "V1"),
        Icoor("V4", math.pi, math.pi / 2, 1.0, atom_name, "V2", "V1"),
        Icoor("V5", math.pi, math.pi / 2, 1.0, atom_name, "V1", "V2"),
        Icoor("V6", -math.pi / 2, math.pi / 2, 1.0, atom_name, "V2", "V1"),
        Icoor("V7", math.pi / 4, math.pi / 2, 1.0, atom_name, "V1", "V2"),
        Icoor("V8", -math.pi / 4, math.pi / 2, 1.0, atom_name, "V1", "V2"),
    )
    properties = ChemicalProperties(
        is_canonical=False,
        polymer=PolymerProperties(False, None, None, (), "NA", ()),
        chemical_modifications=("generated_metal",),
        connectivity=(),
        protonation=ProtonationProperties((), "positively_charged", 7.0),
        virtual=virtuals,
    )
    return RawResidueType(
        name=name,
        base_name=name,
        name3=name,
        io_equiv_class=name,
        atoms=atoms,
        atom_aliases=(),
        bonds=bonds,
        connections=(),
        torsions=(),
        icoors=icoors,
        properties=properties,
        chi_samples=(),
        default_jump_connection_atom=atom_name,
    )


def augment_database_for_metals(structure, param_db: ParameterDatabase):
    """Generate deposited Mg/Ca/Zn components and Rosetta ion parameters."""

    components = _metal_residues(structure)
    existing = {residue.name for residue in param_db.chemical.residues}
    additions = []
    charges = {}
    added_elements = set()
    for name, (element, atom_name) in components.items():
        if name in existing:
            continue
        _atomic_number, atom_type, charge, _ljlk = _METAL_PARAMETERS[element]
        additions.append(_metal_residue_type(name, atom_name, atom_type))
        charges[name] = {atom_name: charge, **{f"V{i}": 0.0 for i in range(1, 9)}}
        added_elements.add(element)
    if not additions:
        return param_db

    result = inject_residue_params(
        param_db,
        additions,
        atom_types=[
            AtomType(_METAL_PARAMETERS[element][1], element)
            for element in sorted(added_elements)
        ],
        partial_charges=charges,
    )
    existing_elements = {element.name for element in result.chemical.element_types}
    elements = tuple(
        Element(element, _METAL_PARAMETERS[element][0])
        for element in sorted(added_elements)
        if element not in existing_elements
    )
    existing_ljlk = {
        parameter.name for parameter in result.scoring.ljlk.atom_type_parameters
    }
    ljlk = tuple(
        LJLKAtomTypeParameters(atom_type, *values)
        for element in sorted(set(value[0] for value in components.values()))
        for _atomic_number, atom_type, _charge, values in [_METAL_PARAMETERS[element]]
        if atom_type not in existing_ljlk
    )
    chemical = attr.evolve(
        result.chemical,
        element_types=(*result.chemical.element_types, *elements),
    )
    scoring = attr.evolve(
        result.scoring,
        ljlk=attr.evolve(
            result.scoring.ljlk,
            atom_type_parameters=(*result.scoring.ljlk.atom_type_parameters, *ljlk),
        ),
    )
    return attr.evolve(result, chemical=chemical, scoring=scoring)


def _metal_cross_residue_bonds(structure):
    """Return sorted metal--donor contacts from an explicit bond table."""

    array = _template(structure)
    if array.bonds is None:
        return ()
    _, _, keys, atom_to_residue = _residue_keys(array)
    contacts = []
    for atom1, atom2, _bond_type in array.bonds.as_array():
        res1, res2 = int(atom_to_residue[atom1]), int(atom_to_residue[atom2])
        if res1 == res2:
            continue
        is_metal1 = _element(array.element[atom1]) in _METAL_PARAMETERS
        is_metal2 = _element(array.element[atom2]) in _METAL_PARAMETERS
        if is_metal1 == is_metal2:
            continue
        metal_i, donor_i = (atom1, atom2) if is_metal1 else (atom2, atom1)
        metal_res = int(atom_to_residue[metal_i])
        donor_res = int(atom_to_residue[donor_i])
        contacts.append(
            (
                (keys[metal_res], str(array.atom_name[metal_i])),
                (keys[donor_res], str(array.atom_name[donor_i])),
            )
        )
    return tuple(sorted(contacts))


def _connection_names(contacts):
    """Assign deterministic unique names, including bridging donors."""

    totals = Counter(endpoint for contact in contacts for endpoint in contact)
    seen = Counter()
    named = []
    for contact in contacts:
        names = []
        for key, atom in contact:
            endpoint = (key, atom)
            seen[endpoint] += 1
            suffix = f"_{seen[endpoint]}" if totals[endpoint] > 1 else ""
            names.append(f"metal_{atom}{suffix}")
        named.append((contact[0], names[0], contact[1], names[1]))
    return tuple(named)


def _metal_connection_icoor(raw, connection_name, atom, distance):
    """Create an ion connection frame using its built-in virtual geometry."""

    virtuals = tuple(raw.properties.virtual)
    if len(virtuals) < 2:
        raise ValueError(f"metal residue {raw.name} lacks two virtual frame atoms")
    return Icoor(
        name=connection_name,
        phi=0.0,
        theta=math.pi / 2,
        d=distance,
        parent=atom,
        grand_parent=virtuals[0],
        great_grand_parent=virtuals[1],
    )


def _fallback_connection_icoor(raw, connection_name, atom, distance):
    """Return a topology-only frame when deposited local frame atoms are absent."""

    adjacency = defaultdict(list)
    for atom1, atom2, *_ in raw.bonds:
        adjacency[atom1].append(atom2)
        adjacency[atom2].append(atom1)
    if not adjacency[atom]:
        raise ValueError(f"connection atom {raw.name}:{atom} has no local frame")
    grand_parent = sorted(adjacency[atom])[0]
    great_grandparents = sorted(
        name for name in adjacency[grand_parent] if name != atom
    )
    if not great_grandparents:
        great_grandparents = sorted(
            atom_info.name
            for atom_info in raw.atoms
            if atom_info.name not in (atom, grand_parent)
        )
    if not great_grandparents:
        raise ValueError(f"connection atom {raw.name}:{atom} lacks a third frame atom")
    return Icoor(
        name=connection_name,
        phi=0.0,
        theta=math.pi / 2,
        d=distance,
        parent=atom,
        grand_parent=grand_parent,
        great_grand_parent=great_grandparents[0],
    )


def augment_database_for_metal_bonds(structure, param_db):
    """Add same-layout residue variants for explicit metal coordination."""

    contacts = _metal_cross_residue_bonds(structure)
    if not contacts:
        return param_db, {}

    array = _template(structure)
    starts, ends, keys, _ = _residue_keys(array)
    key_to_residue = {key: i for i, key in enumerate(keys)}
    named_contacts = _connection_names(contacts)
    endpoint_connections = defaultdict(list)
    remote_coords = {}
    for endpoint1, name1, endpoint2, name2 in named_contacts:
        endpoint_connections[endpoint1[0]].append((endpoint1[1], name1))
        endpoint_connections[endpoint2[0]].append((endpoint2[1], name2))
        res1, res2 = key_to_residue[endpoint1[0]], key_to_residue[endpoint2[0]]
        idx1 = next(
            i
            for i in range(starts[res1], ends[res1])
            if array.atom_name[i] == endpoint1[1]
        )
        idx2 = next(
            i
            for i in range(starts[res2], ends[res2])
            if array.atom_name[i] == endpoint2[1]
        )
        remote_coords[(endpoint1[0], name1)] = np.asarray(
            array.coord[idx2], dtype=np.float64
        )
        remote_coords[(endpoint2[0], name2)] = np.asarray(
            array.coord[idx1], dtype=np.float64
        )

    patterns = {}
    for key, connections in endpoint_connections.items():
        signature = tuple(sorted(connections))
        res_ind = key_to_residue[key]
        local = {
            str(array.atom_name[i]): np.asarray(array.coord[i], dtype=np.float64)
            for i in range(starts[res_ind], ends[res_ind])
        }
        patterns.setdefault((key[3], signature), (key, local))

    atom_type_elements = {
        atom_type.name: atom_type.element for atom_type in param_db.chemical.atom_types
    }
    metal_residue_names = set(_metal_residues(structure))
    clones = []
    variant_names = {}
    supported = set()
    for raw in param_db.chemical.residues:
        raw_atom_names = {atom.name for atom in raw.atoms}
        for pattern, (input_key, local) in sorted(patterns.items()):
            res_name, signature = pattern
            attachment_atoms = tuple(atom for atom, _ in signature)
            if raw.name3 != res_name or not set(attachment_atoms) <= raw_atom_names:
                continue
            if len(raw.connections) + len(signature) > MAX_COORDINATION_CONNECTIONS:
                raise ValueError(
                    f"residue {raw.name} would require "
                    f"{len(raw.connections) + len(signature)} connections; "
                    f"the scoring kernels support {MAX_COORDINATION_CONNECTIONS}"
                )
            clone_name = f"{raw.name}:" + ",".join(name for _, name in signature)
            added_connections = tuple(
                Connection(name=name, atom=atom, type="SINGLE")
                for atom, name in signature
            )
            if raw.name in metal_residue_names:
                added_icoors = tuple(
                    _metal_connection_icoor(
                        raw,
                        name,
                        atom,
                        float(
                            np.linalg.norm(
                                remote_coords[(input_key, name)] - local[atom]
                            )
                        ),
                    )
                    for atom, name in signature
                )
            else:
                added_icoors = []
                for atom, name in signature:
                    remote = remote_coords[(input_key, name)]
                    distance = float(np.linalg.norm(remote - local[atom]))
                    try:
                        icoor = _connection_icoor(raw, name, atom, local, remote)
                    except ValueError:
                        icoor = _fallback_connection_icoor(raw, name, atom, distance)
                    added_icoors.append(icoor)
                added_icoors = tuple(added_icoors)
            clone_atoms, clone_properties = _virtualize_leaving_hydrogens(
                raw, attachment_atoms, atom_type_elements
            )
            if raw.name3 == "HOH":
                # Deposited crystallographic waters normally contain oxygen only.
                # Keep the coordinating oxygen and make absent hydrogens inert.
                water_hydrogens = tuple(
                    atom.name
                    for atom in clone_atoms
                    if atom_type_elements[atom.atom_type] == "H"
                )
                clone_atoms = tuple(
                    (
                        attr.evolve(atom, atom_type="Vrt")
                        if atom.name in water_hydrogens
                        else atom
                    )
                    for atom in clone_atoms
                )
                clone_properties = attr.evolve(
                    clone_properties,
                    virtual=tuple(
                        dict.fromkeys((*clone_properties.virtual, *water_hydrogens))
                    ),
                )
            clones.append(
                attr.evolve(
                    raw,
                    name=clone_name,
                    atoms=clone_atoms,
                    connections=(*raw.connections, *added_connections),
                    icoors=(*raw.icoors, *added_icoors),
                    properties=clone_properties,
                )
            )
            variant_names[(raw.name, signature)] = clone_name
            supported.add(pattern)

    missing = sorted(set(patterns) - supported)
    if missing:
        raise ValueError(
            f"no residue type supports metal attachment pattern(s): {missing}"
        )
    chemical = attr.evolve(
        param_db.chemical, residues=(*param_db.chemical.residues, *clones)
    )
    return attr.evolve(param_db, chemical=chemical), variant_names


def _add_metal_constraints(
    pose_stack,
    named_contacts,
    endpoint_connections,
    key_to_block,
    distance_multiplier,
    angle_multiplier,
):
    """Add Rosetta SetupMetalsMover-equivalent deposited-geometry constraints."""

    if distance_multiplier < 0 or angle_multiplier < 0:
        raise ValueError("metal constraint multipliers must be nonnegative")
    if distance_multiplier == 0 and angle_multiplier == 0:
        return pose_stack

    device = pose_stack.device
    distance_atoms = []
    distance_params = []
    angle_atoms = []
    angle_params = []
    proxies_by_metal = defaultdict(list)

    def coord(pose_index, block, atom_index):
        offset = int(pose_stack.block_coord_offset64[pose_index, block])
        return pose_stack.coords[pose_index, offset + atom_index]

    for pose_index in range(len(pose_stack)):
        for metal_endpoint, metal_name, donor_endpoint, _donor_name in named_contacts:
            metal_block = key_to_block[(pose_index, metal_endpoint[0][:3])]
            donor_block = key_to_block[(pose_index, donor_endpoint[0][:3])]
            metal_type = pose_stack.block_type(pose_index, metal_block)
            donor_type = pose_stack.block_type(pose_index, donor_block)
            metal_atom = metal_type.atom_to_idx[metal_endpoint[1]]
            donor_atom = donor_type.atom_to_idx[donor_endpoint[1]]
            signature = tuple(sorted(endpoint_connections[metal_endpoint[0]]))
            virtual_number = signature.index((metal_endpoint[1], metal_name)) + 1
            virtual_atom = metal_type.atom_to_idx[f"V{virtual_number}"]
            metal_ref = (pose_index, metal_block, metal_atom)
            virtual_ref = (pose_index, metal_block, virtual_atom)
            donor_ref = (pose_index, donor_block, donor_atom)
            proxies_by_metal[(pose_index, metal_endpoint[0])].append(
                (virtual_ref, coord(pose_index, metal_block, virtual_atom))
            )

            if distance_multiplier > 0:
                sd = 0.1 / math.sqrt(distance_multiplier)
                distance_atoms.extend(
                    ((virtual_ref, donor_ref), (metal_ref, virtual_ref))
                )
                metal_virtual_distance = torch.linalg.norm(
                    coord(pose_index, metal_block, metal_atom)
                    - coord(pose_index, metal_block, virtual_atom)
                ).item()
                distance_params.extend(((0.0, sd), (metal_virtual_distance, sd)))

            if angle_multiplier > 0:
                icoor = donor_type.icoors[donor_type.icoors_index[donor_endpoint[1]]]
                if (
                    icoor.parent in donor_type.atom_to_idx
                    and icoor.parent != donor_endpoint[1]
                ):
                    parent_atom = donor_type.atom_to_idx[icoor.parent]
                    parent_ref = (pose_index, donor_block, parent_atom)
                    vector1 = coord(pose_index, metal_block, metal_atom) - coord(
                        pose_index, donor_block, donor_atom
                    )
                    vector2 = coord(pose_index, donor_block, parent_atom) - coord(
                        pose_index, donor_block, donor_atom
                    )
                    cosine = torch.dot(vector1, vector2) / (
                        torch.linalg.norm(vector1) * torch.linalg.norm(vector2)
                    )
                    target = torch.acos(cosine.clamp(-1.0, 1.0)).item()
                    angle_atoms.append((metal_ref, donor_ref, parent_ref))
                    angle_params.append((target, 0.05 / math.sqrt(angle_multiplier)))

    if distance_multiplier > 0:
        sd = 0.1 / math.sqrt(distance_multiplier)
        for proxies in proxies_by_metal.values():
            for first in range(len(proxies)):
                for second in range(first + 1, len(proxies)):
                    first_ref, first_coord = proxies[first]
                    second_ref, second_coord = proxies[second]
                    distance_atoms.append((first_ref, second_ref))
                    distance_params.append(
                        (torch.linalg.norm(first_coord - second_coord).item(), sd)
                    )

    constraints = pose_stack.constraint_set or ConstraintSet.create_empty(
        device=device, n_poses=len(pose_stack)
    )
    if distance_atoms:
        constraints = constraints.add_constraints(
            ConstraintEnergyTerm.harmonic,
            torch.tensor(distance_atoms, dtype=torch.int32, device=device),
            torch.tensor(distance_params, dtype=torch.float32, device=device),
        )
    if angle_atoms:
        constraints = constraints.add_constraints(
            ConstraintEnergyTerm.harmonic_angle,
            torch.tensor(angle_atoms, dtype=torch.int32, device=device),
            torch.tensor(angle_params, dtype=torch.float32, device=device),
        )
    return attr.evolve(pose_stack, constraint_set=constraints)


def apply_metal_bonds_from_biotite(
    pose_stack,
    structure,
    variant_names,
    distance_constraint_multiplier=1.0,
    angle_constraint_multiplier=1.0,
):
    """Select coordination variants and install explicit metal bonds."""

    contacts = _metal_cross_residue_bonds(structure)
    if not contacts:
        return pose_stack
    named_contacts = _connection_names(contacts)
    endpoint_connections = defaultdict(list)
    for endpoint1, name1, endpoint2, name2 in named_contacts:
        endpoint_connections[endpoint1[0]].append((endpoint1[1], name1))
        endpoint_connections[endpoint2[0]].append((endpoint2[1], name2))

    pbt = pose_stack.packed_block_types
    name_to_ind = {bt.name: i for i, bt in enumerate(pbt.active_block_types)}
    type64 = pose_stack.block_type_ind64.clone()
    coords = pose_stack.coords.clone()
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
        for input_key, connections in endpoint_connections.items():
            block = key_to_block[(pose_index, input_key[:3])]
            original = pbt.active_block_types[int(type64[pose_index, block])]
            signature = tuple(sorted(connections))
            clone_name = variant_names[(original.name, signature)]
            clone = pbt.active_block_types[name_to_ind[clone_name]]
            if tuple(a.name for a in clone.atoms) != tuple(
                a.name for a in original.atoms
            ):
                raise ValueError("metal variants must preserve atom layout")
            type64[pose_index, block] = name_to_ind[clone_name]
            atom_start = int(pose_stack.block_coord_offset[pose_index, block])
            block_coords = coords[
                pose_index, atom_start : atom_start + len(clone.atoms)
            ]
            missing = torch.isnan(block_coords).any(dim=-1)
            if torch.any(missing):
                virtual_inds = [
                    clone.atom_to_idx[name] for name in clone.properties.virtual
                ]
                anchor_candidates = [
                    i
                    for i in range(len(clone.atoms))
                    if i not in virtual_inds and not bool(missing[i])
                ]
                if anchor_candidates:
                    anchor = anchor_candidates[0]
                    ideal = torch.as_tensor(
                        clone.ideal_coords,
                        dtype=coords.dtype,
                        device=coords.device,
                    )
                    placed = ideal + (block_coords[anchor] - ideal[anchor])
                    for atom_ind in virtual_inds:
                        if bool(missing[atom_ind]):
                            block_coords[atom_ind] = placed[atom_ind]

    # Rosetta places each metal virtual proxy exactly on its deposited donor.
    # Copy from each pose independently so AtomArrayStack models retain their
    # own coordination geometry.
    for pose_index in range(len(pose_stack)):
        for metal_endpoint, metal_name, donor_endpoint, _donor_name in named_contacts:
            metal_block = key_to_block[(pose_index, metal_endpoint[0][:3])]
            donor_block = key_to_block[(pose_index, donor_endpoint[0][:3])]
            metal_type = pbt.active_block_types[int(type64[pose_index, metal_block])]
            donor_type = pbt.active_block_types[int(type64[pose_index, donor_block])]
            signature = tuple(sorted(endpoint_connections[metal_endpoint[0]]))
            virtual_index = signature.index((metal_endpoint[1], metal_name)) + 1
            metal_offset = int(pose_stack.block_coord_offset[pose_index, metal_block])
            donor_offset = int(pose_stack.block_coord_offset[pose_index, donor_block])
            coords[
                pose_index,
                metal_offset + metal_type.atom_to_idx[f"V{virtual_index}"],
            ] = coords[
                pose_index,
                donor_offset + donor_type.atom_to_idx[donor_endpoint[1]],
            ]

    pose_stack = attr.evolve(
        pose_stack,
        coords=coords,
        block_type_ind64=type64,
        block_type_ind=type64.to(torch.int32),
    )
    connections = []
    for pose_index in range(len(pose_stack)):
        for endpoint1, name1, endpoint2, name2 in named_contacts:
            connections.append(
                InterResidueConnection(
                    pose_index,
                    key_to_block[(pose_index, endpoint1[0][:3])],
                    name1,
                    key_to_block[(pose_index, endpoint2[0][:3])],
                    name2,
                )
            )
    pose_stack = connect_pose_blocks(pose_stack, connections)
    return _add_metal_constraints(
        pose_stack,
        named_contacts,
        endpoint_connections,
        key_to_block,
        distance_constraint_multiplier,
        angle_constraint_multiplier,
    )
