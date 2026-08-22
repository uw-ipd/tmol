"""User-defined fragmentation of fully prepared ligands.

Fragments are specified by the integer ``tmol_fragment_id`` annotation on the
input Biotite AtomArray.  Chemistry is perceived once for the complete ligand;
the functions in this module only partition that prepared chemistry.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math
from typing import Mapping, Sequence

import biotite.structure as struc
import numpy as np

from tmol.chemical import build_coords_from_icoors
from tmol.database.chemical import Connection, Icoor, RawResidueType
from tmol.ligand._registry import LigandPreparation

FRAGMENT_ID_ANNOTATION = "tmol_fragment_id"
MAX_FRAGMENT_CONNECTIONS = 4
MIN_FRAGMENT_HEAVY_ATOMS = 3


@dataclass(frozen=True)
class FragmentConnection:
    """One directed side of a cut bond."""

    fragment_id: int
    partner_fragment_id: int
    connection_name: str
    partner_connection_name: str
    atom_name: str
    partner_atom_name: str
    bond_type: str


@dataclass(frozen=True)
class LigandFragmentDefinition:
    """Structure-independent definition of one fragmented ligand type."""

    ligand_name: str
    atom_to_fragment: Mapping[str, int]
    fragment_ids: tuple[int, ...]
    fragment_preparations: tuple[LigandPreparation, ...]
    connections: tuple[FragmentConnection, ...]

    def fragment_name(self, fragment_id: int) -> str:
        return f"{self.ligand_name}.{fragment_id}"


@dataclass(frozen=True)
class LigandFragmentBlockMapping:
    """Map a user fragment ID onto its block in a built pose."""

    pose_index: int
    ligand_name: str
    residue_label: int
    pose_residue_label: int
    chain_label: str
    insertion_code: str
    fragment_id: int
    block_index: int
    atom_names: tuple[str, ...]

    @property
    def fragment_name(self) -> str:
        return f"{self.ligand_name}.{self.fragment_id}"


@dataclass(frozen=True)
class FragmentedLigandPoseMapping:
    """Runtime mapping and connection list for a fragmented pose."""

    blocks: tuple[LigandFragmentBlockMapping, ...]
    connection_pairs: tuple[tuple[int, str, int, str], ...]

    def split(self, pose_index: int) -> "FragmentedLigandPoseMapping":
        """Return the mapping for one pose, reindexed as pose zero."""

        return replace(
            self,
            blocks=tuple(
                replace(block, pose_index=0)
                for block in self.blocks
                if block.pose_index == pose_index
            ),
        )


def recombine_fragmented_ligands(
    structure: struc.AtomArray | struc.AtomArrayStack,
    pose_stack,
) -> struc.AtomArray | struc.AtomArrayStack:
    """Restore original residue identities on exported ligand fragments.

    Uses ``pose_stack.split_block_mapping`` to identify which residues in
    *structure* are split-block fragments and what their original PDB identity
    should be.  Atoms are matched by the residue label stored in the pose's
    PDBInfo (which is what ``biotite_from_pose_stack`` writes to ``res_id``).
    """
    sbm = pose_stack.split_block_mapping
    if sbm is None or not sbm.entries:
        return structure

    pbt = pose_stack.packed_block_types
    result = structure.copy()

    for entry in sbm.entries:
        if entry.pose_ind != 0:
            continue
        pose_res_label = int(
            pose_stack.pdb_info.residue_labels[entry.pose_ind, entry.block_ind]
        )
        frag_name = pbt.active_block_types[
            int(pose_stack.block_type_ind[entry.pose_ind, entry.block_ind])
        ].name
        orig_name = pbt.active_block_types[entry.orig_block_type_ind].name

        fragment_atoms = (result.res_id == pose_res_label) & (
            result.res_name == frag_name
        )
        result.res_name[fragment_atoms] = orig_name
        result.res_id[fragment_atoms] = entry.orig_residue_label
        result.chain_id[fragment_atoms] = entry.orig_chain_label
        result.ins_code[fragment_atoms] = entry.orig_ins_code
    return result


def fragment_ids_from_atom_array(atom_array: struc.AtomArray) -> np.ndarray | None:
    """Return validated fragment IDs, or ``None`` when no split is requested."""

    if FRAGMENT_ID_ANNOTATION not in atom_array.get_annotation_categories():
        return None
    raw_ids = np.asarray(getattr(atom_array, FRAGMENT_ID_ANNOTATION))
    if raw_ids.ndim != 1 or raw_ids.shape[0] != atom_array.array_length():
        raise ValueError(f"{FRAGMENT_ID_ANNOTATION} must contain one integer per atom")
    if not np.issubdtype(raw_ids.dtype, np.integer):
        raise ValueError(f"{FRAGMENT_ID_ANNOTATION} values must be integers")
    fragment_ids = raw_ids.astype(np.int64)
    if np.any(fragment_ids < 0):
        raise ValueError(f"{FRAGMENT_ID_ANNOTATION} values must be non-negative")
    if np.unique(fragment_ids).size <= 1:
        return None
    return fragment_ids


def _bond_type_name(bond: tuple) -> str:
    if len(bond) <= 2:
        return "SINGLE"
    value = bond[2]
    if isinstance(value, str):
        return value.upper()
    if hasattr(value, "name"):
        return str(value.name)
    return {1: "SINGLE", 2: "DOUBLE", 3: "TRIPLE", 4: "AROMATIC"}.get(
        int(value), "SINGLE"
    )


def _full_ideal_coords(restype: RawResidueType) -> dict[str, np.ndarray]:
    icoor_index = {icoor.name: i for i, icoor in enumerate(restype.icoors)}
    ancestors = np.empty((len(restype.icoors), 3), dtype=np.int32)
    geom = np.empty((len(restype.icoors), 3), dtype=np.float64)
    for i, icoor in enumerate(restype.icoors):
        ancestors[i] = (
            icoor_index[icoor.parent],
            icoor_index[icoor.grand_parent],
            icoor_index[icoor.great_grand_parent],
        )
        geom[i] = (icoor.phi, icoor.theta, icoor.d)
    coords = build_coords_from_icoors(ancestors, geom)
    return {
        icoor.name: coords[i].astype(np.float64)
        for i, icoor in enumerate(restype.icoors)
    }


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-12:
        return 0.0
    return float(np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)))


def _dihedral(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    b1 = b - a
    b2 = c - b
    b3 = d - c
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1_norm = float(np.linalg.norm(n1))
    n2_norm = float(np.linalg.norm(n2))
    b2_norm = float(np.linalg.norm(b2))
    if min(n1_norm, n2_norm, b2_norm) < 1e-12:
        return 0.0
    n1 /= n1_norm
    n2 /= n2_norm
    m1 = np.cross(n1, b2 / b2_norm)
    return float(np.arctan2(np.dot(m1, n2), np.dot(n1, n2)))


def _fragment_atom_tree(  # noqa: C901
    atom_names: Sequence[str],
    bonds: Sequence[tuple],
    coords: Mapping[str, np.ndarray],
) -> tuple[list[str], dict[str, str], dict[str, tuple[str, str]]]:
    adjacency = {name: [] for name in atom_names}
    for a, b, *_ in bonds:
        adjacency[a].append(b)
        adjacency[b].append(a)

    root = max(
        atom_names, key=lambda name: (len(adjacency[name]), -atom_names.index(name))
    )
    order: list[str] = []
    parent = {root: root}
    queue = deque([root])
    while queue:
        current = queue.popleft()
        order.append(current)
        for neighbor in sorted(adjacency[current]):
            if neighbor in parent:
                continue
            parent[neighbor] = current
            queue.append(neighbor)
    if len(order) != len(atom_names):
        raise ValueError("fragment atoms must form one connected component")

    position = {name: i for i, name in enumerate(order)}
    grandparents: dict[str, tuple[str, str]] = {}
    for name in order:
        par = parent[name]
        gp = parent.get(par, par)
        if name != root and gp == par:
            alternatives = [
                n
                for n in adjacency[par]
                if n != name and position.get(n, len(order)) < position[name]
            ]
            if alternatives:
                gp = alternatives[0]
        ggp = parent.get(gp, gp)
        if name != root and ggp in (name, par, gp):
            alternatives = [
                n
                for n in (*adjacency[par], *adjacency.get(gp, []))
                if n not in (name, par, gp)
                and position.get(n, len(order)) < position[name]
            ]
            if alternatives:
                ggp = alternatives[0]
        grandparents[name] = (gp, ggp)
    return order, parent, grandparents


def _atom_icoors(
    order: Sequence[str],
    parent: Mapping[str, str],
    grandparents: Mapping[str, tuple[str, str]],
    coords: Mapping[str, np.ndarray],
) -> list[Icoor]:
    result: list[Icoor] = []
    for i, name in enumerate(order):
        par = parent[name]
        gp, ggp = grandparents[name]
        if i == 0:
            phi = theta = d = 0.0
        elif i == 1:
            phi, theta = 0.0, math.pi
            d = float(np.linalg.norm(coords[name] - coords[par]))
        elif i == 2:
            phi = 0.0
            theta = math.pi - _angle(coords[name], coords[par], coords[gp])
            d = float(np.linalg.norm(coords[name] - coords[par]))
        else:
            phi = -_dihedral(coords[name], coords[par], coords[gp], coords[ggp])
            theta = math.pi - _angle(coords[name], coords[par], coords[gp])
            d = float(np.linalg.norm(coords[name] - coords[par]))
        result.append(
            Icoor(
                name=name,
                phi=phi,
                theta=theta,
                d=d,
                parent=par,
                grand_parent=gp,
                great_grand_parent=ggp,
            )
        )
    return result


def _connection_icoor(
    connection: FragmentConnection,
    local_atoms: Sequence[str],
    local_bonds: Sequence[tuple],
    coords: Mapping[str, np.ndarray],
) -> Icoor:
    adjacency = {name: [] for name in local_atoms}
    for a, b, *_ in local_bonds:
        adjacency[a].append(b)
        adjacency[b].append(a)
    parent = connection.atom_name
    gp_candidates = adjacency[parent]
    if not gp_candidates:
        raise ValueError(f"connection atom {parent} has no local frame atom")
    gp = sorted(gp_candidates)[0]
    ggp_candidates = [name for name in adjacency[gp] if name != parent]
    if not ggp_candidates:
        ggp_candidates = [name for name in local_atoms if name not in (parent, gp)]
    if not ggp_candidates:
        raise ValueError(f"connection at {parent} lacks a third local frame atom")
    ggp = sorted(ggp_candidates)[0]
    remote = coords[connection.partner_atom_name]
    return Icoor(
        name=connection.connection_name,
        phi=-_dihedral(remote, coords[parent], coords[gp], coords[ggp]),
        theta=math.pi - _angle(remote, coords[parent], coords[gp]),
        d=float(np.linalg.norm(remote - coords[parent])),
        parent=parent,
        grand_parent=gp,
        great_grand_parent=ggp,
    )


def _unresolved_atom_name(unresolved) -> str | None:
    return unresolved.atom


def _validate_bonded_cut_layout(
    ligand_name: str,
    adjacency: Mapping[str, Sequence[str]],
    cut_bonds: Sequence[tuple],
) -> None:
    """Reject layouts containing bonded terms that span three or more blocks."""

    cut_edges = {frozenset(bond[:2]) for bond in cut_bonds}
    cut_degree = {
        atom: sum(frozenset((atom, neighbor)) in cut_edges for neighbor in neighbors)
        for atom, neighbors in adjacency.items()
    }
    if any(degree > 1 for degree in cut_degree.values()):
        raise ValueError(
            f"{ligand_name}: no atom may participate in more than one fragment "
            "cut; impropers spanning three blocks are not supported"
        )

    def visit(path: tuple[str, ...]) -> None:
        if len(path) == 4:
            n_cuts = sum(
                frozenset((path[i], path[i + 1])) in cut_edges for i in range(3)
            )
            if n_cuts > 1:
                raise ValueError(
                    f"{ligand_name}: fragment cuts are too close along bonded path "
                    f"{'-'.join(path)}; torsions spanning three blocks are not supported"
                )
            return
        for neighbor in adjacency[path[-1]]:
            if neighbor not in path:
                visit(path + (neighbor,))

    for atom in adjacency:
        visit((atom,))


def _validate_scoring_cut_layout(restype: RawResidueType, cut_bonds: Sequence[tuple]):
    """Reject cuts that break nonbonded scoring geometry assumptions."""

    from tmol.ligand._chemistry_tables import get_hbond_properties

    hbond_properties = get_hbond_properties()
    atom_type_by_name = {atom.name: atom.atom_type for atom in restype.atoms}
    bad_cuts: list[str] = []
    for bond in cut_bonds:
        a, b = bond[:2]
        for atom_name, neighbor_name in ((a, b), (b, a)):
            atom_type = atom_type_by_name[atom_name]
            if hbond_properties.get(atom_type, {}).get("is_acceptor", False):
                bad_cuts.append(f"{atom_name}-{neighbor_name} ({atom_type})")

    if bad_cuts:
        raise ValueError(
            f"{restype.name}: fragment cuts through hbond/lk_ball acceptor "
            "geometry are not supported; acceptor atoms must remain in the "
            "same fragment as their bonded frame atoms. Unsupported cut(s): "
            + ", ".join(bad_cuts)
        )


def build_ligand_fragment_definition(  # noqa: C901
    preparation: LigandPreparation,
    source_atom_array: struc.AtomArray,
) -> LigandFragmentDefinition | None:
    """Partition a fully prepared ligand according to its source annotation."""

    restype = preparation.residue_type
    source_fragment_ids = fragment_ids_from_atom_array(source_atom_array)
    if source_fragment_ids is None:
        return None
    if np.any(source_fragment_ids == 0):
        zero_atom_names = list(
            map(str, source_atom_array.atom_name[source_fragment_ids == 0][:8])
        )
        raise ValueError(
            f"{restype.name}: every atom in a fragmented residue must have a "
            f"positive {FRAGMENT_ID_ANNOTATION}; found 0 for "
            f"{zero_atom_names}"
        )

    source_name_to_id = {
        str(name): int(fragment_id)
        for name, fragment_id in zip(source_atom_array.atom_name, source_fragment_ids)
    }
    source_elements = {
        str(name): str(element).upper()
        for name, element in zip(source_atom_array.atom_name, source_atom_array.element)
    }
    atom_names = [atom.name for atom in restype.atoms]
    adjacency = {name: [] for name in atom_names}
    for a, b, *_ in restype.bonds:
        adjacency[a].append(b)
        adjacency[b].append(a)

    atom_to_fragment: dict[str, int] = {
        name: source_name_to_id[name]
        for name in atom_names
        if name in source_name_to_id
    }
    unresolved = set(atom_names) - set(atom_to_fragment)
    while unresolved:
        progressed = False
        for name in tuple(unresolved):
            neighbor_ids = {
                atom_to_fragment[n] for n in adjacency[name] if n in atom_to_fragment
            }
            if len(neighbor_ids) == 1:
                atom_to_fragment[name] = neighbor_ids.pop()
                unresolved.remove(name)
                progressed = True
        if not progressed:
            raise ValueError(
                f"{restype.name}: could not assign prepared atoms to fragments: "
                f"{sorted(unresolved)}"
            )
    # Keep source-only hydrogen names in the public/input mapping too. Prepared
    # ligands may regenerate and rename these hydrogens.
    atom_to_fragment.update(
        {
            name: fragment_id
            for name, fragment_id in source_name_to_id.items()
            if name not in atom_to_fragment
        }
    )

    # Only prepared atoms define fragment blocks. Source-only names may retain
    # orphan IDs in the public mapping but cannot create empty blocks.
    fragment_ids = sorted({atom_to_fragment[atom.name] for atom in restype.atoms})
    orphan_ids = set(atom_to_fragment.values()) - set(fragment_ids)
    if orphan_ids:
        raise ValueError(
            f"{restype.name}: fragment id(s) {sorted(orphan_ids)} have no "
            "prepared atoms"
        )
    cut_bonds = [
        bond
        for bond in restype.bonds
        if atom_to_fragment[bond[0]] != atom_to_fragment[bond[1]]
    ]
    _validate_bonded_cut_layout(restype.name, adjacency, cut_bonds)
    _validate_scoring_cut_layout(restype, cut_bonds)
    connections_by_fragment: dict[int, list[FragmentConnection]] = {
        fragment_id: [] for fragment_id in fragment_ids
    }
    directed_connections: list[FragmentConnection] = []
    for cut_index, bond in enumerate(cut_bonds, start=1):
        a, b = bond[:2]
        fa, fb = atom_to_fragment[a], atom_to_fragment[b]
        name_a = f"conn_{cut_index}_{fa}_to_{fb}"
        name_b = f"conn_{cut_index}_{fb}_to_{fa}"
        conn_a = FragmentConnection(fa, fb, name_a, name_b, a, b, _bond_type_name(bond))
        conn_b = FragmentConnection(fb, fa, name_b, name_a, b, a, _bond_type_name(bond))
        connections_by_fragment[fa].append(conn_a)
        connections_by_fragment[fb].append(conn_b)
        directed_connections.extend((conn_a, conn_b))

    coords = _full_ideal_coords(restype)
    fragment_preparations: list[LigandPreparation] = []
    for fragment_id in fragment_ids:
        names = [
            atom.name
            for atom in restype.atoms
            if atom_to_fragment[atom.name] == fragment_id
        ]
        name_set = set(names)
        local_bonds = tuple(
            bond
            for bond in restype.bonds
            if bond[0] in name_set and bond[1] in name_set
        )

        seen = set()
        queue = deque([names[0]])
        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            queue.extend(
                b if a == current else a
                for a, b, *_ in local_bonds
                if a == current or b == current
            )
        if seen != name_set:
            raise ValueError(
                f"{restype.name}.{fragment_id}: fragment must be one connected component"
            )

        element_by_name = {
            atom.name: (
                source_elements.get(atom.name)
                or (preparation.atom_type_elements or {}).get(atom.atom_type, "")
                or ("H" if atom.atom_type.upper().startswith("H") else "")
            ).upper()
            for atom in restype.atoms
            if atom.name in name_set
        }
        n_heavy = sum(element != "H" for element in element_by_name.values())
        if n_heavy < MIN_FRAGMENT_HEAVY_ATOMS:
            raise ValueError(
                f"{restype.name}.{fragment_id}: fragments require at least "
                f"{MIN_FRAGMENT_HEAVY_ATOMS} heavy atoms; found {n_heavy}"
            )
        if len(connections_by_fragment[fragment_id]) > MAX_FRAGMENT_CONNECTIONS:
            raise ValueError(
                f"{restype.name}.{fragment_id}: at most {MAX_FRAGMENT_CONNECTIONS} "
                "inter-block connections are supported"
            )

        order, parent, grandparents = _fragment_atom_tree(names, local_bonds, coords)
        icoors = _atom_icoors(order, parent, grandparents, coords)
        icoors.extend(
            _connection_icoor(conn, names, local_bonds, coords)
            for conn in connections_by_fragment[fragment_id]
        )

        torsions = tuple(
            torsion
            for torsion in restype.torsions
            if all(
                _unresolved_atom_name(atom) in name_set
                for atom in (torsion.a, torsion.b, torsion.c, torsion.d)
            )
        )
        torsion_names = {torsion.name for torsion in torsions}
        chi_samples = tuple(
            sample
            for sample in restype.chi_samples
            if sample.chi_dihedral in torsion_names
        )
        fragment_name = f"{restype.name}.{fragment_id}"
        fragment_restype = RawResidueType(
            name=fragment_name,
            base_name=restype.name,
            name3=fragment_name,
            io_equiv_class=fragment_name,
            atoms=tuple(atom for atom in restype.atoms if atom.name in name_set),
            atom_aliases=tuple(
                alias for alias in restype.atom_aliases if alias.name in name_set
            ),
            bonds=local_bonds,
            connections=tuple(
                Connection(
                    name=conn.connection_name, atom=conn.atom_name, type=conn.bond_type
                )
                for conn in connections_by_fragment[fragment_id]
            ),
            torsions=torsions,
            icoors=tuple(icoors),
            properties=restype.properties,
            chi_samples=chi_samples,
            default_jump_connection_atom=order[0],
            # Fragment poses should honor provided H coordinates; absent leaf H
            # atoms can still be rebuilt from the fragment icoors.
            hydrogens_regenerated=False,
            is_ligand_fragment=True,
        )
        fragment_preparations.append(
            LigandPreparation(
                residue_type=fragment_restype,
                partial_charges={
                    name: charge
                    for name, charge in preparation.partial_charges.items()
                    if name in name_set
                },
                cartbonded_params=preparation.cartbonded_params,
                atom_type_elements=preparation.atom_type_elements,
            )
        )

    return LigandFragmentDefinition(
        ligand_name=restype.name,
        atom_to_fragment=atom_to_fragment,
        fragment_ids=tuple(fragment_ids),
        fragment_preparations=tuple(fragment_preparations),
        connections=tuple(directed_connections),
    )


def expand_fragmented_ligands(  # noqa: C901
    structure: struc.AtomArray | struc.AtomArrayStack,
    definitions: Sequence[LigandFragmentDefinition],
) -> tuple[struc.AtomArray | struc.AtomArrayStack, FragmentedLigandPoseMapping]:
    """Replace each annotated ligand residue with contiguous fragment residues."""

    if not definitions:
        return (
            structure,
            FragmentedLigandPoseMapping(blocks=(), connection_pairs=()),
        )
    definition_by_name = {
        definition.ligand_name: definition for definition in definitions
    }
    representative = (
        structure[0] if isinstance(structure, struc.AtomArrayStack) else structure
    )
    residue_starts = struc.get_residue_starts(representative)
    residue_ends = np.append(residue_starts[1:], representative.array_length())

    atom_order: list[int] = []
    output_names: list[str] = []
    output_residue_labels: list[int] = []
    block_records: list[LigandFragmentBlockMapping] = []
    connection_pairs: list[tuple[int, str, int, str]] = []
    output_block_index = 0
    next_synthetic_residue_label = int(np.max(representative.res_id)) + 1

    for start, end in zip(residue_starts, residue_ends):
        ligand_name = str(representative.res_name[start])
        definition = definition_by_name.get(ligand_name)
        if definition is None:
            inds = list(range(int(start), int(end)))
            atom_order.extend(inds)
            output_names.extend(str(representative.res_name[i]) for i in inds)
            output_residue_labels.extend(int(representative.res_id[i]) for i in inds)
            output_block_index += 1
            continue

        local_names = [str(name) for name in representative.atom_name[start:end]]
        local_ids: list[int] = []
        annotation = (
            np.asarray(getattr(representative, FRAGMENT_ID_ANNOTATION))[start:end]
            if FRAGMENT_ID_ANNOTATION in representative.get_annotation_categories()
            else None
        )
        for local_index, atom_name in enumerate(local_names):
            if atom_name not in definition.atom_to_fragment:
                raise ValueError(
                    f"{ligand_name}: atom {atom_name!r} is absent from the prepared "
                    "fragment mapping"
                )
            fragment_id = definition.atom_to_fragment[atom_name]
            if annotation is not None and int(annotation[local_index]) != fragment_id:
                raise ValueError(
                    f"{ligand_name}: {FRAGMENT_ID_ANNOTATION} changed for atom "
                    f"{atom_name!r} after the build context was created"
                )
            local_ids.append(fragment_id)

        block_for_fragment: dict[int, int] = {}
        for fragment_id in definition.fragment_ids:
            selected = [
                int(start) + i
                for i, value in enumerate(local_ids)
                if value == fragment_id
            ]
            if not selected:
                raise ValueError(
                    f"{ligand_name}.{fragment_id}: no atoms found in this structure"
                )
            fragment_name = definition.fragment_name(fragment_id)
            atom_order.extend(selected)
            output_names.extend([fragment_name] * len(selected))
            synthetic_residue_label = next_synthetic_residue_label
            next_synthetic_residue_label += 1
            output_residue_labels.extend([synthetic_residue_label] * len(selected))
            block_for_fragment[fragment_id] = output_block_index
            block_records.append(
                LigandFragmentBlockMapping(
                    pose_index=0,
                    ligand_name=ligand_name,
                    residue_label=int(representative.res_id[start]),
                    pose_residue_label=synthetic_residue_label,
                    chain_label=str(representative.chain_id[start]),
                    insertion_code=str(representative.ins_code[start]),
                    fragment_id=fragment_id,
                    block_index=output_block_index,
                    atom_names=tuple(
                        str(representative.atom_name[i]) for i in selected
                    ),
                )
            )
            output_block_index += 1

        seen_pairs: set[frozenset[str]] = set()
        for connection in definition.connections:
            key = frozenset(
                (connection.connection_name, connection.partner_connection_name)
            )
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            connection_pairs.append(
                (
                    block_for_fragment[connection.fragment_id],
                    connection.connection_name,
                    block_for_fragment[connection.partner_fragment_id],
                    connection.partner_connection_name,
                )
            )

    if isinstance(structure, struc.AtomArrayStack):
        expanded = structure[:, atom_order].copy()
    else:
        expanded = structure[atom_order].copy()
    # Biotite's built-in res_name annotation is commonly U3. Replace it instead
    # of assigning into it so deterministic names such as ``XYZ.12`` are retained.
    expanded.set_annotation(
        "res_name",
        np.asarray(
            output_names, dtype=f"U{max(3, max(map(len, output_names), default=3))}"
        ),
    )
    expanded.set_annotation("res_id", np.asarray(output_residue_labels, dtype=np.int32))

    n_poses = len(structure) if isinstance(structure, struc.AtomArrayStack) else 1
    blocks = tuple(
        replace(record, pose_index=pose_index)
        for pose_index in range(n_poses)
        for record in block_records
    )
    return (
        expanded,
        FragmentedLigandPoseMapping(
            blocks=blocks, connection_pairs=tuple(connection_pairs)
        ),
    )


def apply_fragment_connections(pose_stack, mapping: FragmentedLigandPoseMapping):
    """Install fragment cut bonds and rebuild all inter-block bond separations."""

    resolved_blocks = []
    temporary_to_actual: dict[int, int] = {}
    for record in mapping.blocks:
        candidates = np.flatnonzero(
            pose_stack.pdb_info.residue_labels[record.pose_index]
            == record.pose_residue_label
        ).tolist()
        if len(candidates) != 1:
            raise ValueError(
                f"Could not uniquely map {record.fragment_name} "
                f"{record.chain_label}:{record.residue_label}{record.insertion_code}; "
                f"found blocks {candidates}"
            )
        actual_index = candidates[0]
        block_type_index = int(
            pose_stack.block_type_ind64[record.pose_index, actual_index].item()
        )
        block_type = pose_stack.packed_block_types.active_block_types[block_type_index]
        if block_type.name != record.fragment_name:
            raise ValueError(
                f"Mapped residue label {record.pose_residue_label} to "
                f"{block_type.name}, expected {record.fragment_name}"
            )
        previous_index = temporary_to_actual.setdefault(
            record.block_index, actual_index
        )
        if previous_index != actual_index:
            raise ValueError(
                "Fragmented AtomArrayStack models must have identical block "
                "topology; fragment block indices differ between models"
            )
        resolved_blocks.append(replace(record, block_index=actual_index))

    mapping = FragmentedLigandPoseMapping(
        blocks=tuple(resolved_blocks),
        connection_pairs=tuple(
            (
                temporary_to_actual[block_a],
                name_a,
                temporary_to_actual[block_b],
                name_b,
            )
            for block_a, name_a, block_b, name_b in mapping.connection_pairs
        ),
    )
    if not mapping.connection_pairs:
        pose_stack, sbm = build_split_block_mapping(pose_stack, mapping)
        import attr as _attr

        return _attr.evolve(pose_stack, split_block_mapping=sbm)

    from tmol.pose import InterResidueConnection, connect_pose_blocks

    result = connect_pose_blocks(
        pose_stack,
        (
            InterResidueConnection(pose_index, block_a, name_a, block_b, name_b)
            for pose_index in range(len(pose_stack))
            for block_a, name_a, block_b, name_b in mapping.connection_pairs
        ),
    )
    result, sbm = build_split_block_mapping(result, mapping)
    import attr as _attr

    return _attr.evolve(result, split_block_mapping=sbm)


def build_split_block_mapping(
    pose_stack,
    resolved_mapping: FragmentedLigandPoseMapping,
):
    """Build a SplitBlockMapping from a fragmented PoseStack.

    Ensures the original (unfragmented) block types are present in the
    PackedBlockTypes of the returned PoseStack, then records for each
    fragment block: its pose/block indices, its group within that pose,
    the index of the original block type, and the per-atom mapping
    split_atom → orig_atom.

    Fragment block-type names are expected to follow the convention
    ``"ORIGNAME.FRAGMENT_ID"`` (e.g. ``"LIG.0"``).

    Returns the (possibly PBT-extended) PoseStack and the SplitBlockMapping.
    """
    import attr as _attr

    from tmol.pose._packed_block_types import PackedBlockTypes
    from tmol.pose._split_block_mapping import SplitBlockEntry, SplitBlockMapping

    pbt = pose_stack.packed_block_types
    active_name_to_idx = {bt.name: i for i, bt in enumerate(pbt.active_block_types)}

    # ── collect original types that are not yet in the PBT ──────────────────
    missing_orig: dict[str, object] = {}  # name → RefinedResidueType
    for record in resolved_mapping.blocks:
        frag_name = pbt.active_block_types[
            int(
                pose_stack.block_type_ind64[
                    record.pose_index, record.block_index
                ].item()
            )
        ].name
        orig_name = frag_name.rsplit(".", 1)[0]
        if orig_name not in active_name_to_idx and orig_name not in missing_orig:
            orig_rt = next(
                (rt for rt in pbt.restype_set.residue_types if rt.name == orig_name),
                None,
            )
            if orig_rt is None:
                raise ValueError(
                    f"Original block type {orig_name!r} not found in ResidueTypeSet"
                )
            missing_orig[orig_name] = orig_rt

    if missing_orig:
        extended = list(pbt.active_block_types) + list(missing_orig.values())
        new_pbt = PackedBlockTypes.from_restype_list(
            pbt.chem_db, pbt.restype_set, extended, pose_stack.device
        )
        pose_stack = _attr.evolve(pose_stack, packed_block_types=new_pbt)
        pbt = new_pbt
        active_name_to_idx = {bt.name: i for i, bt in enumerate(pbt.active_block_types)}

    # ── build entries ────────────────────────────────────────────────────────
    group_ind_map: dict[tuple[int, str], int] = {}
    pose_group_counter: dict[int, int] = {}
    entries = []

    for record in resolved_mapping.blocks:
        pose_ind = record.pose_index
        block_ind = record.block_index
        bt_idx = int(pose_stack.block_type_ind64[pose_ind, block_ind].item())
        frag_bt = pbt.active_block_types[bt_idx]
        orig_name = frag_bt.name.rsplit(".", 1)[0]

        key = (pose_ind, orig_name)
        if key not in group_ind_map:
            pose_group_counter.setdefault(pose_ind, 0)
            group_ind_map[key] = pose_group_counter[pose_ind]
            pose_group_counter[pose_ind] += 1
        group_ind = group_ind_map[key]

        orig_bt_ind = active_name_to_idx[orig_name]
        orig_bt = pbt.active_block_types[orig_bt_ind]
        orig_atom_to_idx = orig_bt.atom_to_idx  # {name: local_index}

        split_to_orig = np.array(
            [orig_atom_to_idx[atom.name] for atom in frag_bt.atoms],
            dtype=np.int32,
        )

        entries.append(
            SplitBlockEntry(
                pose_ind=pose_ind,
                block_ind=block_ind,
                group_ind=group_ind,
                orig_block_type_ind=orig_bt_ind,
                split_to_orig_atom_inds=split_to_orig,
                orig_residue_label=record.residue_label,
                orig_chain_label=record.chain_label,
                orig_ins_code=record.insertion_code,
            )
        )

    return pose_stack, SplitBlockMapping(entries=tuple(entries))


def _unsplit_group_entries(sbm):
    """Return (groups, split_block_set, entry_lookup) from a SplitBlockMapping."""
    from collections import defaultdict

    groups = defaultdict(list)
    split_block_set = defaultdict(set)
    for e in sbm.entries:
        groups[(e.pose_ind, e.group_ind)].append(e)
        split_block_set[e.pose_ind].add(e.block_ind)
    for key in groups:
        groups[key].sort(key=lambda e: e.block_ind)
    entry_lookup = {(e.pose_ind, e.block_ind): e for e in sbm.entries}
    return groups, split_block_set, entry_lookup


def _unsplit_per_pose_blocks(pose_stack, split_block_set, entry_lookup):
    """Build per-pose new block sequence (bt_ind, kind, src) for unsplitting."""
    n_poses = len(pose_stack)
    old_max = int(pose_stack.block_type_ind64.shape[1])
    per_pose = []
    for p in range(n_poses):
        seen, blocks = set(), []
        for b in range(old_max):
            bt = int(pose_stack.block_type_ind64[p, b].item())
            if bt < 0:
                continue
            if b not in split_block_set[p]:
                blocks.append((bt, "orig", b))
            else:
                e = entry_lookup[(p, b)]
                gkey = (p, e.group_ind)
                if gkey not in seen:
                    seen.add(gkey)
                    blocks.append((e.orig_block_type_ind, "group", gkey))
        per_pose.append(blocks)
    return per_pose


def _unsplit_build_coords(
    per_pose_blocks, pbt, groups, pose_stack, new_bco, new_max_n_atoms, n_poses
):
    """Gather atom coordinates from split blocks into original block layout."""
    old_coords = pose_stack.coords.cpu().numpy()
    old_bco = pose_stack.block_coord_offset.cpu().numpy()
    new_bco_np = new_bco.cpu().numpy()
    out = np.zeros((n_poses, new_max_n_atoms, 3), dtype=np.float32)
    for p, blocks in enumerate(per_pose_blocks):
        for new_b, (bt_idx, kind, src) in enumerate(blocks):
            new_off = int(new_bco_np[p, new_b])
            n_atoms = int(pbt.n_atoms[bt_idx].item())
            if kind == "orig":
                old_off = int(old_bco[p, src])
                out[p, new_off : new_off + n_atoms] = old_coords[
                    p, old_off : old_off + n_atoms
                ]
            else:
                dest = np.zeros((n_atoms, 3), dtype=np.float32)
                for fe in groups[src]:
                    fo = int(old_bco[p, fe.block_ind])
                    for li, oi in enumerate(fe.split_to_orig_atom_inds):
                        dest[oi] = old_coords[p, fo + li]
                out[p, new_off : new_off + n_atoms] = dest
    return out, new_bco_np


def _unsplit_old_to_new(per_pose_blocks, pose_stack, split_block_set, entry_lookup):
    """Build per-pose mapping from old block indices to new (or -1 if absorbed)."""
    old_max = int(pose_stack.block_type_ind64.shape[1])
    old_to_new = []
    for p, blocks in enumerate(per_pose_blocks):  # noqa: B007
        seen, m, idx = set(), {}, 0
        for b in range(old_max):
            if int(pose_stack.block_type_ind64[p, b].item()) < 0:
                continue
            if b not in split_block_set[p]:
                m[b] = idx
                idx += 1
            else:
                e = entry_lookup[(p, b)]
                gkey = (p, e.group_ind)
                if gkey not in seen:
                    seen.add(gkey)
                    m[b] = idx
                    idx += 1
                else:
                    m[b] = -1
        old_to_new.append(m)
    return old_to_new


def _unsplit_connections(
    pose_stack,
    pbt,
    n_poses,
    new_max_n_blocks,
    split_block_set,
    entry_lookup,
    old_to_new,
    device,
):
    """Remap inter-residue connections, discarding intra-fragment ones."""
    import torch

    max_n_conn = int(pose_stack.inter_residue_connections64.shape[2])
    out = torch.full(
        (n_poses, new_max_n_blocks, max_n_conn, 2), -1, dtype=torch.int64, device=device
    )
    old_irc = pose_stack.inter_residue_connections64
    for p in range(n_poses):
        m = old_to_new[p]
        for old_b, new_b in m.items():
            if new_b < 0:
                continue
            if old_b in split_block_set[p]:
                e = entry_lookup[(p, old_b)]
                n_conn = len(pbt.active_block_types[e.orig_block_type_ind].connections)
            else:
                n_conn = len(
                    pbt.active_block_types[
                        int(pose_stack.block_type_ind64[p, old_b].item())
                    ].connections
                )
            for c in range(n_conn):
                partner = old_irc[p, old_b, c]
                pb = int(partner[0].item())
                if pb == -1 or (
                    old_b in split_block_set[p] and pb in split_block_set[p]
                ):
                    continue
                pc = int(partner[1].item())
                npb = m.get(pb, -1)
                if npb >= 0:
                    out[p, new_b, c] = torch.tensor(
                        [npb, pc], dtype=torch.int64, device=device
                    )
    return out


def _unsplit_chain_and_pdb(
    pose_stack,
    old_to_new,
    pbt,
    per_pose_blocks,
    new_bco_np,
    n_poses,
    new_max_n_blocks,
    new_max_n_atoms,
    device,
):
    """Build new chain_id tensor and PDBInfo for the unsplit pose."""
    import torch
    from tmol.pose._pdb_info import (
        PDBInfo,
        DEFAULT_ATOM_OCCUPANCY,
        DEFAULT_ATOM_B_FACTOR,
    )

    old_pdb = pose_stack.pdb_info
    old_bco = pose_stack.block_coord_offset.cpu().numpy()
    chain_np = np.full((n_poses, new_max_n_blocks), -1, dtype=np.int32)
    res_labels = np.zeros((n_poses, new_max_n_blocks), dtype=int)
    ins_codes = np.full((n_poses, new_max_n_blocks), "", dtype=object)
    chain_labels = np.full((n_poses, new_max_n_blocks), "", dtype=object)
    occ = np.full((n_poses, new_max_n_atoms), DEFAULT_ATOM_OCCUPANCY, dtype=np.float32)
    bf = np.full((n_poses, new_max_n_atoms), DEFAULT_ATOM_B_FACTOR, dtype=np.float32)
    old_chain = pose_stack.chain_id.cpu().numpy()
    for p, (blocks, m) in enumerate(zip(per_pose_blocks, old_to_new)):
        for old_b, new_b in m.items():
            if new_b < 0:
                continue
            chain_np[p, new_b] = old_chain[p, old_b]
            res_labels[p, new_b] = old_pdb.residue_labels[p, old_b]
            ins_codes[p, new_b] = old_pdb.residue_insertion_codes[p, old_b]
            chain_labels[p, new_b] = old_pdb.chain_labels[p, old_b]
        for new_b, (bt_idx, kind, src) in enumerate(blocks):
            if kind != "orig":
                continue
            new_off = int(new_bco_np[p, new_b])
            n_at = int(pbt.n_atoms[bt_idx].item())
            old_off = int(old_bco[p, src])
            occ[p, new_off : new_off + n_at] = old_pdb.atom_occupancy[
                p, old_off : old_off + n_at
            ]
            bf[p, new_off : new_off + n_at] = old_pdb.atom_b_factor[
                p, old_off : old_off + n_at
            ]
    new_chain_id = torch.from_numpy(chain_np).to(device)
    new_pdb = PDBInfo(
        residue_labels=res_labels,
        residue_insertion_codes=ins_codes,
        chain_labels=chain_labels,
        atom_occupancy=occ,
        atom_b_factor=bf,
    )
    return new_chain_id, new_pdb


def unsplit_pose_stack(pose_stack):
    """Reconstruct an unsplit PoseStack from one containing split (fragment) blocks.

    Each group of split blocks sharing a ``(pose_ind, group_ind)`` in the
    PoseStack's ``split_block_mapping`` is collapsed back into a single block
    of the corresponding original block type.  Atom coordinates are gathered
    from the fragment blocks using the per-entry ``split_to_orig_atom_inds``
    arrays; atoms present in the original type but absent from all fragments
    are left at zero.

    Inter-residue connections *between* fragment blocks are removed (they
    become intra-block interactions in the original).  External connections
    from a fragment block to a non-fragment block are transferred to the
    reconstructed original block.

    Returns a new PoseStack with ``split_block_mapping=None``.
    """
    import attr
    import torch
    from tmol.pose import PoseStackBuilder
    from tmol.utility.tensor import exclusive_cumsum2d

    sbm = pose_stack.split_block_mapping
    if sbm is None or not sbm.entries:
        return attr.evolve(pose_stack, split_block_mapping=None)

    pbt = pose_stack.packed_block_types
    device = pose_stack.device
    n_poses = len(pose_stack)

    groups, split_block_set, entry_lookup = _unsplit_group_entries(sbm)
    per_pose_blocks = _unsplit_per_pose_blocks(
        pose_stack, split_block_set, entry_lookup
    )

    new_max_n_blocks = max((len(bl) for bl in per_pose_blocks), default=0)
    new_bt64 = torch.full(
        (n_poses, new_max_n_blocks), -1, dtype=torch.int64, device=device
    )
    for p, blocks in enumerate(per_pose_blocks):
        for new_b, (bt_idx, *_) in enumerate(blocks):
            new_bt64[p, new_b] = bt_idx
    new_bt32 = new_bt64.to(torch.int32)

    real_new = new_bt64 >= 0
    new_n_atoms_blk = torch.zeros(
        (n_poses, new_max_n_blocks), dtype=torch.int32, device=device
    )
    new_n_atoms_blk[real_new] = pbt.n_atoms[new_bt64[real_new]]
    new_bco = exclusive_cumsum2d(new_n_atoms_blk)
    new_max_n_atoms = int(torch.max(torch.sum(new_n_atoms_blk, dim=1)).item())

    coords_np, new_bco_np = _unsplit_build_coords(
        per_pose_blocks, pbt, groups, pose_stack, new_bco, new_max_n_atoms, n_poses
    )
    new_coords = torch.from_numpy(coords_np).to(device)

    old_to_new = _unsplit_old_to_new(
        per_pose_blocks, pose_stack, split_block_set, entry_lookup
    )
    new_irc64 = _unsplit_connections(
        pose_stack,
        pbt,
        n_poses,
        new_max_n_blocks,
        split_block_set,
        entry_lookup,
        old_to_new,
        device,
    )

    pconn_matrix, pconn_offsets, block_n_conn, pose_n_pconn = (
        PoseStackBuilder._take_real_conn_conn_intrablock_pairs(pbt, new_bt64, real_new)
    )
    PoseStackBuilder._incorporate_inter_residue_connections_into_connectivity_graph(
        new_irc64, pconn_offsets, pconn_matrix
    )
    new_ibs64 = PoseStackBuilder._calculate_interblock_bondsep_from_connectivity_graph(
        pbt, block_n_conn, pose_n_pconn, pconn_matrix
    )

    new_chain_id, new_pdb_info = _unsplit_chain_and_pdb(
        pose_stack,
        old_to_new,
        pbt,
        per_pose_blocks,
        new_bco_np,
        n_poses,
        new_max_n_blocks,
        new_max_n_atoms,
        device,
    )

    return attr.evolve(
        pose_stack,
        coords=new_coords,
        block_coord_offset=new_bco,
        block_coord_offset64=new_bco.to(torch.int64),
        inter_residue_connections=new_irc64.to(torch.int32),
        inter_residue_connections64=new_irc64,
        inter_block_bondsep=new_ibs64.to(torch.int32),
        inter_block_bondsep64=new_ibs64,
        block_type_ind=new_bt32,
        block_type_ind64=new_bt64,
        chain_id=new_chain_id,
        chain_id64=new_chain_id.to(torch.int64),
        pdb_info=new_pdb_info,
        split_block_mapping=None,
    )
