"""Docking-only ligand rigid-body and torsional DOF setup.

This module deliberately lives under :mod:`tmol.dock`: resolving a logical
ligand chi from either end of its dihedral is a docking policy, not a change to
the residue type or to tmol's shared kinematic scan ordering.  In particular,
the resolver below stores its logical mapping only in the returned object.  A
``PoseStackKinematicsModule`` may lazily populate the ordinary kinematics
caches on block types and ``PackedBlockTypes``; the builder does not alter
existing cache contents, the default jump connection atom, or global
``MoveMap`` defaults.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy
import torch

from tmol.kinematics.compiled import inverse_kin
from tmol.kinematics import BondDOFTypes, NodeType
from tmol.kinematics import EdgeType, FoldForest
from tmol.kinematics import MinimizerMap, MoveMap
from tmol.kinematics import PoseStackKinematicsModule
from tmol.pose import PoseStack
from tmol.score.gridscore._movable import (
    validated_sidechain_blocks,
)


def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _coord_dihedrals(a, b, c, d):
    """Measure dihedrals without down-casting float64 coordinates."""

    ba = a - b
    bc = c - b
    cd = d - c
    unit_bc = bc / torch.linalg.vector_norm(bc, dim=1, keepdim=True)
    v = ba - torch.sum(ba * unit_bc, dim=1, keepdim=True) * unit_bc
    w = cd - torch.sum(cd * unit_bc, dim=1, keepdim=True) * unit_bc
    x = torch.sum(v * w, dim=1)
    y = torch.sum(torch.linalg.cross(unit_bc, v) * w, dim=1)
    return torch.atan2(y, x)


class _ChiConversion:
    """Raw-DOF <-> logical-chi conversion shared by torsion mappings."""

    def logical_from_raw(self, raw_dof: torch.Tensor) -> torch.Tensor:
        """Convert raw values to logical chi values, modulo 2*pi."""

        return _wrap_angle(self.sign * raw_dof + self.offset)

    def raw_from_logical(self, logical_chi: torch.Tensor) -> torch.Tensor:
        """Convert logical chi values to the equivalent raw values."""

        return _wrap_angle(self.sign * (logical_chi - self.offset))


@dataclass(frozen=True)
class LigandTorsionDOF(_ChiConversion):
    """A docking-local logical-chi to raw-kinematic-DOF mapping.

    ``raw_atom`` is an atom index local to the ligand block.  Angles obey
    ``logical_chi = wrap(sign * raw_dof + offset[pose])``.  The per-pose
    offset is needed because raw internal-coordinate zero and named-dihedral
    zero use different frame conventions.
    """

    name: str
    torsion_index: int
    raw_atom: int
    raw_atom_name: str
    dof_type: BondDOFTypes
    sign: int
    offset: torch.Tensor

    @property
    def dof_name(self) -> str:
        return self.dof_type.name


@dataclass(frozen=True)
class SidechainTorsionDOF(_ChiConversion):
    """A receptor sidechain chi to raw-kinematic-DOF mapping."""

    block: int
    name: str
    torsion_index: int
    sc_torsion_index: int
    raw_atom: int
    raw_atom_name: str
    dof_type: BondDOFTypes
    sign: int
    offset: torch.Tensor

    @property
    def dof_name(self) -> str:
        return self.dof_type.name


@dataclass(frozen=True)
class LigandDockingDOFs:
    """Fold forest, MoveMap, and logical docking torsion mappings."""

    fold_forest: FoldForest
    move_map: MoveMap
    ligand_block: int
    torsions: tuple[LigandTorsionDOF, ...]
    sidechain_blocks: tuple[int, ...] = ()
    sidechain_torsions: tuple[SidechainTorsionDOF, ...] = ()

    @property
    def logical_chi_to_raw_dof(self) -> dict[str, LigandTorsionDOF]:
        """Return the docking-owned mapping keyed by logical torsion name."""

        return {mapping.name: mapping for mapping in self.torsions}


def _block_kinforest_nodes(
    pose_stack: PoseStack,
    block: int,
    n_atoms: int,
    forest_ids: torch.Tensor,
) -> torch.Tensor:
    """Return the ``[pose, block_atom]`` kinematic-node lookup."""

    n_targets = pose_stack.n_poses * pose_stack.max_n_pose_atoms
    id_to_node = torch.full(
        (n_targets,), -1, dtype=torch.int64, device=pose_stack.device
    )
    nodes = torch.arange(
        1, forest_ids.shape[0], dtype=torch.int64, device=pose_stack.device
    )
    id_to_node.scatter_(0, forest_ids[1:].to(torch.int64), nodes)

    poses = torch.arange(
        pose_stack.n_poses, dtype=torch.int64, device=pose_stack.device
    )
    local_atoms = torch.arange(n_atoms, dtype=torch.int64, device=pose_stack.device)
    target_ids = (
        poses[:, None] * pose_stack.max_n_pose_atoms
        + pose_stack.block_coord_offset64[:, block, None]
        + local_atoms[None, :]
    )
    block_nodes = id_to_node[target_ids]
    if bool(torch.any(block_nodes < 0)):
        raise ValueError(
            f"an atom in block {block} is absent from the kinematic forest"
        )
    return block_nodes


def _ligand_kinforest_nodes(
    pose_stack: PoseStack,
    ligand_block: int,
    ligand_n_atoms: int,
    forest_ids: torch.Tensor,
) -> torch.Tensor:
    """Return the ``[pose, ligand_atom]`` kinematic-node lookup in one gather."""

    return _block_kinforest_nodes(pose_stack, ligand_block, ligand_n_atoms, forest_ids)


def _raw_dof_affected_nodes(
    forest,
    controller_node: torch.Tensor,
    candidate_nodes: torch.Tensor,
    dof_type: BondDOFTypes,
) -> torch.Tensor:
    """Return the candidate nodes moved by one raw torsion DOF."""

    ancestors = candidate_nodes.clone()
    affected = torch.zeros_like(ancestors, dtype=torch.bool)
    if dof_type == BondDOFTypes.phi_p:
        affected |= ancestors == controller_node
    while bool(torch.any(ancestors != 0)):
        ancestors = forest.parent[ancestors].to(torch.int64)
        affected |= ancestors == controller_node
    return affected


def _sidechain_end_local_atoms(
    block_type,
    torsion_atoms: tuple[int, int, int, int],
    device: torch.device,
) -> torch.Tensor:
    """Return the chemical-graph component beyond a chi's central bond."""

    _, atom_b, atom_c, atom_d = torsion_atoms
    adjacency = [set() for _ in range(block_type.n_atoms)]
    for left, right in block_type.bond_indices:
        left = int(left)
        right = int(right)
        if {left, right} == {atom_b, atom_c}:
            continue
        adjacency[left].add(right)
        adjacency[right].add(left)

    downstream = {atom_d}
    pending = [atom_d]
    while pending:
        atom = pending.pop()
        for neighbor in adjacency[atom] - downstream:
            downstream.add(neighbor)
            pending.append(neighbor)
    if atom_b in downstream:
        return torch.empty(0, dtype=torch.int64, device=device)
    return torch.tensor(sorted(downstream), dtype=torch.int64, device=device)


def _validate_sidechain_controller(
    forest,
    block: int,
    block_nodes: torch.Tensor,
    sidechain_local_atoms: torch.Tensor,
    raw_atom: int,
    dof_type: BondDOFTypes,
    required_local_atom: int,
    max_n_pose_atoms: int,
) -> torch.Tensor | None:
    """Return a topology-local downstream mask, or ``None`` if unsafe."""

    # Kinforest nodes are grouped by pose and target-coordinate IDs are ordered
    # within each pose.  Validate the shared topology on pose zero instead of
    # expanding every pose-local controller over the full multi-pose forest.
    # The previous [n_poses, n_forest_nodes] masks scaled quadratically because
    # n_forest_nodes itself grows with n_poses.
    first_pose_stop = int(
        torch.searchsorted(
            forest.id,
            forest.id.new_tensor(max_n_pose_atoms),
        )
    )
    pose_nodes = torch.arange(
        1, first_pose_stop, dtype=torch.int64, device=forest.parent.device
    )
    affected = _raw_dof_affected_nodes(
        forest,
        block_nodes[0, raw_atom],
        pose_nodes,
        dof_type,
    )
    affected_nodes = pose_nodes[affected]
    allowed_nodes = block_nodes[0, sidechain_local_atoms]
    required_node = block_nodes[0, required_local_atom]
    if (
        affected_nodes.numel() == 0
        or not bool(torch.any(affected_nodes == required_node))
        or not bool(torch.all(torch.isin(affected_nodes, allowed_nodes)))
    ):
        return None
    return torch.isin(block_nodes[0], affected_nodes)


def _sidechain_coordinate_metadata(
    pose_stack: PoseStack,
    docking_dofs: LigandDockingDOFs,
    forest,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather sidechain atoms and their per-chi downstream relationships."""

    affected_by_block: dict[int, set[int]] = {
        block: set() for block in docking_dofs.sidechain_blocks
    }
    block_nodes_by_block = {}
    mapping_local_masks = []
    for block in docking_dofs.sidechain_blocks:
        block_type_index = int(pose_stack.block_type_ind64[0, block])
        block_type = pose_stack.packed_block_types.active_block_types[block_type_index]
        block_nodes_by_block[block] = _block_kinforest_nodes(
            pose_stack, block, block_type.n_atoms, forest.id
        )

    for mapping in docking_dofs.sidechain_torsions:
        block_nodes = block_nodes_by_block[mapping.block]
        block_type = pose_stack.block_type(0, mapping.block)
        torsion_atoms = _local_torsion_atoms(block_type, mapping.torsion_index)
        sidechain_local_atoms = _sidechain_end_local_atoms(
            block_type, torsion_atoms, pose_stack.device
        )
        local_mask = _validate_sidechain_controller(
            forest,
            mapping.block,
            block_nodes,
            sidechain_local_atoms,
            mapping.raw_atom,
            mapping.dof_type,
            torsion_atoms[3],
            pose_stack.max_n_pose_atoms,
        )
        if local_mask is None:
            raise ValueError(
                f"sidechain chi {mapping.name!r} in block {mapping.block} "
                "does not control only its own downstream sidechain atoms"
            )
        local_atoms = torch.nonzero(local_mask, as_tuple=False).flatten()
        affected_by_block[mapping.block].update(local_atoms.cpu().tolist())
        mapping_local_masks.append((mapping.block, local_mask))

    node_parts = []
    atom_index_parts = []
    ordered_atoms = []
    for block in docking_dofs.sidechain_blocks:
        local_atoms_host = sorted(affected_by_block[block])
        if not local_atoms_host:
            continue
        local_atoms = torch.tensor(
            local_atoms_host, dtype=torch.int64, device=pose_stack.device
        )
        node_parts.append(block_nodes_by_block[block][:, local_atoms])
        atom_index_parts.append(
            pose_stack.block_coord_offset64[:, block, None] + local_atoms[None, :]
        )
        ordered_atoms.extend((block, atom) for atom in local_atoms_host)

    if not node_parts:
        empty = torch.empty(
            (pose_stack.n_poses, 0),
            dtype=torch.int64,
            device=pose_stack.device,
        )
        empty_influence = torch.empty(
            (len(docking_dofs.sidechain_torsions), 0),
            dtype=torch.bool,
            device=pose_stack.device,
        )
        return empty, empty.clone(), empty_influence

    influence = torch.zeros(
        (len(mapping_local_masks), len(ordered_atoms)),
        dtype=torch.bool,
        device=pose_stack.device,
    )
    for mapping_index, (mapping_block, local_mask) in enumerate(mapping_local_masks):
        influence[mapping_index] = torch.tensor(
            [
                block == mapping_block and bool(local_mask[local_atom])
                for block, local_atom in ordered_atoms
            ],
            dtype=torch.bool,
            device=pose_stack.device,
        )
    return (
        torch.cat(node_parts, dim=1),
        torch.cat(atom_index_parts, dim=1),
        influence,
    )


def _sidechain_coordinate_indices(
    pose_stack: PoseStack,
    docking_dofs: LigandDockingDOFs,
    forest,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather atoms downstream of the selected sidechain chi controllers."""

    nodes, atom_indices, _ = _sidechain_coordinate_metadata(
        pose_stack, docking_dofs, forest
    )
    return nodes, atom_indices


def _local_torsion_atoms(block_type, torsion_index: int) -> tuple[int, int, int, int]:
    uaids = block_type.ordered_torsions[torsion_index]
    if any(int(uaid[1]) != -1 for uaid in uaids):
        raise ValueError(
            f"torsion {block_type.torsions[torsion_index].name!r} reaches "
            "another block; docking torsions must be block-local"
        )
    atoms = tuple(int(uaid[0]) for uaid in uaids)
    if any(atom < 0 for atom in atoms):
        raise ValueError(
            f"torsion {block_type.torsions[torsion_index].name!r} "
            "contains an unresolved atom"
        )
    return atoms


def _inverse_dofs(pose_stack: PoseStack, kin_module: PoseStackKinematicsModule):
    kmd = kin_module.kmd
    dtype = pose_stack.coords.dtype
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("ligand docking kinematics require float32 or float64 coords")
    kincoords = torch.zeros(
        (kmd.forest.id.shape[0], 3), dtype=dtype, device=pose_stack.device
    )
    kincoords[1:] = pose_stack.coords.reshape(-1, 3)[kmd.forest.id[1:].to(torch.int64)]
    return inverse_kin(
        kincoords,
        kmd.forest.parent,
        kmd.forest.frame_x,
        kmd.forest.frame_y,
        kmd.forest.frame_z,
        kmd.forest.doftype,
    )


def _batched_coord_dihedrals(
    coords: torch.Tensor,
    atom_nodes: torch.Tensor,
) -> torch.Tensor:
    """Measure ``[pose, torsion, 4]`` atom-node dihedrals in one batch."""

    selected = coords[atom_nodes].double()
    shape = atom_nodes.shape[:2]
    flattened = selected.reshape(-1, 4, 3)
    values = _coord_dihedrals(*(flattened[:, i] for i in range(4)))
    return values.reshape(shape)


def _sidechain_logical_torsions_from_coords(
    pose_stack: PoseStack,
    docking_dofs: LigandDockingDOFs,
) -> torch.Tensor:
    """Measure selected sidechain chis with explicit per-pose atom indices."""

    mappings = docking_dofs.sidechain_torsions
    if not mappings:
        return torch.empty(
            (pose_stack.n_poses, 0),
            dtype=pose_stack.coords.dtype,
            device=pose_stack.device,
        )

    blocks = torch.tensor(
        [mapping.block for mapping in mappings],
        dtype=torch.int64,
        device=pose_stack.device,
    )
    torsion_atoms = torch.tensor(
        [
            _local_torsion_atoms(
                pose_stack.block_type(0, mapping.block), mapping.torsion_index
            )
            for mapping in mappings
        ],
        dtype=torch.int64,
        device=pose_stack.device,
    )
    atom_indices = (
        pose_stack.block_coord_offset64[:, blocks, None] + torsion_atoms[None, :, :]
    )
    pose_indices = torch.arange(
        pose_stack.n_poses, dtype=torch.int64, device=pose_stack.device
    )
    flat_atom_indices = (
        pose_indices[:, None, None] * pose_stack.max_n_pose_atoms + atom_indices
    )
    return _batched_coord_dihedrals(
        pose_stack.coords.reshape(-1, 3), flat_atom_indices
    ).to(dtype=pose_stack.coords.dtype)


def _empirical_torsion_signs(
    kin_module: PoseStackKinematicsModule,
    raw_dofs: torch.Tensor,
    ligand_nodes: torch.Tensor,
    torsion_atoms: torch.Tensor,
    raw_nodes: torch.Tensor,
    dof_types: torch.Tensor,
) -> torch.Tensor:
    """Measure and validate declared-dihedral responses to each raw DOF."""

    if torsion_atoms.shape[0] == 0:
        return torch.empty(0, dtype=torch.int64, device=raw_dofs.device)

    # Measure the declared-dihedral responses in float64 regardless of the
    # pose/docking dtype.  This is a one-shot, build-time geometric check, and
    # float32 forward kinematics near the ligand root jump is not accurate
    # enough for a 1e-3 finite-difference dihedral-response measurement: in
    # float32 it produces spurious non-unit self-slopes and cross-coupling that
    # are pure numerical artifacts, whereas the true geometry is clean unit-
    # slope with near-zero coupling in float64 (worst observed cross-response
    # ~3e-10 rad/rad across the validated ligands).  Only this
    # validation is fp32-sensitive; the docking itself runs in the pose dtype.
    measure_dofs = raw_dofs.double()
    delta = 1.0e-3
    base_coords = kin_module(measure_dofs)
    atom_nodes = ligand_nodes[:, torsion_atoms]
    base = _batched_coord_dihedrals(base_coords, atom_nodes)

    response_rows = []
    for torsion_index in range(torsion_atoms.shape[0]):
        perturbed_dofs = measure_dofs.clone()
        perturbed_dofs[raw_nodes[:, torsion_index], dof_types[torsion_index]] += delta
        perturbed_coords = kin_module(perturbed_dofs)
        changed = _batched_coord_dihedrals(perturbed_coords, atom_nodes)
        response_rows.append(_wrap_angle(changed - base) / delta)
    responses = torch.stack(response_rows, dim=1)
    self_slopes = torch.diagonal(responses, dim1=1, dim2=2)

    if not bool(torch.all(torch.isfinite(responses))):
        raise ValueError(
            "ligand torsion raw-DOF perturbation must have finite declared-"
            f"dihedral responses; measured {responses.detach().cpu().tolist()}"
        )

    # Require UNIT magnitude, not merely non-degeneracy. The gene<->raw affine
    # mapping (LigandTorsionDOF: logical = wrap(sign*raw + offset), with sign in
    # {-1,+1} and a single-pose offset) carries NO scale term, so the public
    # "gene == named logical chi" contract holds only when d(dihedral)/d(raw) is
    # exactly +/-1. Measured in float64 the true self-slopes are unit to ~1e-10
    # for every real ligand, so this costs nothing and still unblocks ache/egfr;
    # a genuinely non-unit isolated response would silently miscalibrate the
    # logical chi and must be rejected.
    unit_tolerance = 5.0e-3
    if not bool(torch.all(torch.abs(torch.abs(self_slopes) - 1.0) <= unit_tolerance)):
        raise ValueError(
            "ligand torsion raw-DOF perturbation must change its declared "
            f"dihedral with unit magnitude (tol {unit_tolerance}); measured "
            f"{self_slopes.detach().cpu().tolist()}"
        )

    isolation_tolerance = 1.0e-2
    off_diagonal = ~torch.eye(
        responses.shape[1], dtype=torch.bool, device=responses.device
    )
    if bool(torch.any(torch.abs(responses[:, off_diagonal]) > isolation_tolerance)):
        raise ValueError(
            "ligand torsion raw-DOF perturbation changes another declared "
            f"dihedral by more than {isolation_tolerance} rad/rad; measured "
            f"response matrices {responses.detach().cpu().tolist()}"
        )

    positive = self_slopes > 0
    if bool(torch.any(positive != positive[0:1])):
        raise ValueError(
            "ligand torsion raw-DOF slope has inconsistent signs across poses: "
            f"{self_slopes.detach().cpu().tolist()}"
        )
    return torch.where(positive[0], 1, -1)


def _validate_ligand_root_jumps(
    pose_stack: PoseStack, ligand_block: int, fold_forest: FoldForest
) -> None:
    edges = fold_forest.edges
    active = numpy.arange(edges.shape[1])[None, :] < fold_forest.n_edges[:, None]
    ligand_root = (
        active
        & (edges[:, :, 0] == int(EdgeType.root_jump))
        & (edges[:, :, 2] == ligand_block)
    )
    counts = numpy.sum(ligand_root, axis=1)
    if numpy.any(counts != 1):
        raise ValueError(
            f"ligand block {ligand_block} must have exactly one root jump per "
            f"pose; found {counts.tolist()}"
        )


def _resolve_sidechain_torsions(
    pose_stack: PoseStack,
    sidechain_blocks: tuple[int, ...],
    kin_module: PoseStackKinematicsModule,
    raw_dofs: torch.Tensor,
) -> tuple[SidechainTorsionDOF, ...]:
    """Resolve protein chis to controllers on the isolated sidechain end."""

    kmd = kin_module.kmd
    pose_indices = torch.arange(
        pose_stack.n_poses, dtype=torch.int64, device=pose_stack.device
    )
    mappings: list[SidechainTorsionDOF] = []

    for block in sidechain_blocks:
        block_type_index = int(pose_stack.block_type_ind64[0, block])
        block_type = pose_stack.packed_block_types.active_block_types[block_type_index]
        torsion_indices = [int(index) for index in block_type.sc_torsions]
        if not torsion_indices:
            continue

        block_nodes = _block_kinforest_nodes(
            pose_stack, block, block_type.n_atoms, kmd.forest.id
        )
        torsion_atoms_host = [
            _local_torsion_atoms(block_type, torsion_index)
            for torsion_index in torsion_indices
        ]
        torsion_atoms = torch.tensor(
            torsion_atoms_host, dtype=torch.int64, device=pose_stack.device
        ).reshape(-1, 4)

        first_nodes = block_nodes[0]
        atom_a, atom_b, atom_c, atom_d = torsion_atoms.unbind(dim=1)
        node_b = first_nodes[atom_b]
        node_c = first_nodes[atom_c]
        c_is_child = kmd.forest.parent[node_c] == node_b
        b_is_child = kmd.forest.parent[node_b] == node_c
        if bool(torch.any(c_is_child == b_is_child)):
            raise ValueError(
                f"central atoms for a sidechain torsion in block {block} are "
                "not a parent/child pair in the docking fold forest"
            )

        child_central = torch.where(c_is_child, atom_c, atom_b)
        central_parent = torch.where(c_is_child, atom_b, atom_c)
        reverse_outer = torch.where(c_is_child, atom_a, atom_d)
        child_nodes = block_nodes[:, child_central]
        parent_nodes = block_nodes[:, central_parent]
        if bool(torch.any(kmd.forest.parent[child_nodes] != parent_nodes)):
            raise ValueError(
                f"a sidechain torsion in block {block} has inconsistent tree "
                "orientation across poses"
            )
        if bool(torch.any(kmd.forest.doftype[child_nodes] != int(NodeType.bond))):
            raise ValueError(
                f"a normal sidechain torsion controller in block {block} is "
                "not a bond node"
            )

        normal_fixed = kmd.keep_atom_fixed[child_nodes, int(BondDOFTypes.phi_c)]
        if bool(torch.any(normal_fixed != normal_fixed[0:1])):
            raise ValueError(
                f"a sidechain torsion in block {block} has inconsistent fixed-"
                "DOF status across poses"
            )
        raw_atoms = torch.empty_like(child_central)
        dof_types = torch.empty_like(child_central)
        for chi_index in range(torsion_atoms.shape[0]):
            sidechain_local_atoms = _sidechain_end_local_atoms(
                block_type,
                torsion_atoms_host[chi_index],
                pose_stack.device,
            )
            candidates = []
            if not bool(normal_fixed[0, chi_index]):
                candidates.append(
                    (
                        int(child_central[chi_index]),
                        BondDOFTypes.phi_c,
                    )
                )

            reversed_atom = int(reverse_outer[chi_index])
            reversed_nodes = block_nodes[:, reversed_atom]
            reversed_fixed = kmd.keep_atom_fixed[
                reversed_nodes, int(BondDOFTypes.phi_p)
            ]
            if bool(torch.any(reversed_fixed != reversed_fixed[0:1])):
                raise ValueError(
                    f"a reversed sidechain torsion controller in block {block} "
                    "has inconsistent fixed-DOF status across poses"
                )
            reversed_is_valid = (
                bool(
                    torch.all(
                        kmd.forest.parent[reversed_nodes] == parent_nodes[:, chi_index]
                    )
                )
                and bool(
                    torch.all(kmd.forest.doftype[reversed_nodes] == int(NodeType.bond))
                )
                and not bool(reversed_fixed[0])
            )
            if reversed_is_valid:
                candidates.append((reversed_atom, BondDOFTypes.phi_p))

            selected = None
            for raw_atom, dof_type in candidates:
                affected = _validate_sidechain_controller(
                    kmd.forest,
                    block,
                    block_nodes,
                    sidechain_local_atoms,
                    raw_atom,
                    dof_type,
                    int(atom_d[chi_index]),
                    pose_stack.max_n_pose_atoms,
                )
                if affected is not None:
                    selected = (raw_atom, dof_type)
                    break
            if selected is None:
                torsion_name = block_type.torsions[torsion_indices[chi_index]].name
                raise ValueError(
                    f"sidechain chi {torsion_name!r} in block {block} has no "
                    "raw controller whose downstream set is confined to that "
                    "residue's sidechain; exclude this residue from flexible "
                    "sidechain selection"
                )
            raw_atoms[chi_index] = selected[0]
            dof_types[chi_index] = int(selected[1])

        raw_nodes = block_nodes[:, raw_atoms]

        raw_keys = raw_atoms * raw_dofs.shape[1] + dof_types
        if torch.unique(raw_keys).numel() != raw_keys.numel():
            raise ValueError(
                f"two sidechain torsions in block {block} share the same raw DOF"
            )

        signs = _empirical_torsion_signs(
            kin_module,
            raw_dofs,
            block_nodes,
            torsion_atoms,
            raw_nodes,
            dof_types,
        )
        coord_atom_indices = (
            pose_stack.block_coord_offset64[:, block, None, None]
            + torsion_atoms[None, :, :]
        )
        coord_pose_indices = pose_indices[:, None, None].expand_as(coord_atom_indices)
        logical = _batched_coord_dihedrals(
            pose_stack.coords.reshape(-1, 3),
            coord_pose_indices * pose_stack.max_n_pose_atoms + coord_atom_indices,
        )
        raw = raw_dofs[raw_nodes, dof_types[None, :]].double()
        offsets = _wrap_angle(logical - signs[None, :] * raw).detach().clone()

        mapping_metadata = (
            torch.stack((raw_atoms, dof_types, signs), dim=1).cpu().tolist()
        )
        for sc_torsion_index, (
            torsion_index,
            (raw_atom, dof_type, sign),
        ) in enumerate(zip(torsion_indices, mapping_metadata)):
            dof_type = BondDOFTypes(dof_type)
            mappings.append(
                SidechainTorsionDOF(
                    block=block,
                    name=block_type.torsions[torsion_index].name,
                    torsion_index=torsion_index,
                    sc_torsion_index=sc_torsion_index,
                    raw_atom=raw_atom,
                    raw_atom_name=block_type.atoms[raw_atom].name,
                    dof_type=dof_type,
                    sign=sign,
                    offset=offsets[:, sc_torsion_index],
                )
            )

    return tuple(mappings)


def build_ligand_docking_dofs(
    pose_stack: PoseStack,
    ligand_block: int,
    sidechain_blocks=(),
) -> LigandDockingDOFs:
    """Build ligand and selected receptor-sidechain docking freedoms.

    The ordinary ``reasonable_fold_forest`` is retained: the disconnected
    ligand gets its existing root jump and its existing jump atom/frame.  Six
    DOFs are enabled on that root jump.  Each named ligand torsion first tries
    the normal child-central ``phi_c`` controller.  If the jump stub fixes that
    controller, the resolver reverses the dihedral and enables ``phi_p`` on the
    opposite outer atom, matching Rosetta's bidirectional torsion lookup.

    The affine sign between the selected raw DOF and the declared four-atom
    dihedral is measured by small, one-at-a-time perturbations through the
    actual forward kinematics.  Each response must be non-degenerate, isolated
    from the other declared torsions, and sign-stable across poses.

    Selected polymer blocks contribute their sidechain chi torsions only. A
    protein chi controller is accepted only when its full kinematic downstream
    set lies in the chemical-graph component beyond that chi's central bond; a
    residue with a non-isolatable chi is rejected so it can be excluded from
    flexible selection. Mainchain torsions and receptor root jumps remain fixed.
    Every pose in the stack must place the same ligand block type at
    ``ligand_block``.  The returned MoveMap contains independent entries for
    every pose, so the PoseStack batch axis is preserved.  Constructing the
    kinematics module may populate its normal lazy caches, but this builder
    does not modify their existing contents or store docking annotations on
    the block type or ``PackedBlockTypes``.
    """

    if not isinstance(ligand_block, int):
        raise TypeError("ligand_block must be an integer block index")
    if ligand_block < 0 or ligand_block >= pose_stack.max_n_blocks:
        raise ValueError(f"ligand block {ligand_block} is out of range")

    block_type_indices = pose_stack.block_type_ind64[:, ligand_block]
    if bool(torch.any(block_type_indices < 0)):
        raise ValueError(
            f"ligand block {ligand_block} is absent from at least one pose"
        )
    if bool(torch.any(block_type_indices != block_type_indices[0])):
        raise ValueError("all poses must use the same ligand block type")

    block_type_index = int(block_type_indices[0])
    block_type = pose_stack.packed_block_types.active_block_types[block_type_index]
    if block_type.properties.polymer.is_polymer:
        raise ValueError(f"block {ligand_block} ({block_type.name}) is polymeric")
    sidechain_blocks = validated_sidechain_blocks(
        pose_stack, ligand_block, sidechain_blocks
    )

    fold_forest = FoldForest.reasonable_fold_forest(pose_stack)
    _validate_ligand_root_jumps(pose_stack, ligand_block, fold_forest)
    kin_module = PoseStackKinematicsModule(pose_stack, fold_forest)
    kmd = kin_module.kmd
    raw_dofs = _inverse_dofs(pose_stack, kin_module)

    ligand_nodes = _ligand_kinforest_nodes(
        pose_stack, ligand_block, block_type.n_atoms, kmd.forest.id
    )
    torsion_atoms_host = [
        _local_torsion_atoms(block_type, torsion_index)
        for torsion_index in range(block_type.n_torsions)
    ]
    torsion_atoms = torch.tensor(
        torsion_atoms_host, dtype=torch.int64, device=pose_stack.device
    ).reshape(-1, 4)

    first_nodes = ligand_nodes[0]
    atom_a, atom_b, atom_c, atom_d = torsion_atoms.unbind(dim=1)
    node_b = first_nodes[atom_b]
    node_c = first_nodes[atom_c]
    c_is_child = kmd.forest.parent[node_c] == node_b
    b_is_child = kmd.forest.parent[node_b] == node_c
    if bool(torch.any(c_is_child == b_is_child)):
        raise ValueError(
            "central atoms for a ligand torsion are not a parent/child pair "
            "in the docking fold forest"
        )

    child_central = torch.where(c_is_child, atom_c, atom_b)
    central_parent = torch.where(c_is_child, atom_b, atom_c)
    reverse_outer = torch.where(c_is_child, atom_a, atom_d)
    child_nodes = ligand_nodes[:, child_central]
    parent_nodes = ligand_nodes[:, central_parent]
    if bool(torch.any(kmd.forest.parent[child_nodes] != parent_nodes)):
        raise ValueError(
            "a ligand torsion has inconsistent tree orientation across poses"
        )
    if bool(torch.any(kmd.forest.doftype[child_nodes] != int(NodeType.bond))):
        raise ValueError("a normal ligand torsion controller is not a bond node")

    normal_fixed = kmd.keep_atom_fixed[child_nodes, int(BondDOFTypes.phi_c)]
    if bool(torch.any(normal_fixed != normal_fixed[0:1])):
        raise ValueError(
            "a ligand torsion has inconsistent fixed-DOF status across poses"
        )
    use_reversed = normal_fixed[0]
    raw_atoms = torch.where(use_reversed, reverse_outer, child_central)
    dof_types = torch.where(
        use_reversed,
        int(BondDOFTypes.phi_p),
        int(BondDOFTypes.phi_c),
    )
    raw_nodes = ligand_nodes[:, raw_atoms]

    reversed_nodes = raw_nodes[:, use_reversed]
    reversed_parents = parent_nodes[:, use_reversed]
    if bool(torch.any(kmd.forest.parent[reversed_nodes] != reversed_parents)):
        raise ValueError(
            "a reversed ligand torsion endpoint is not a child of its central atom"
        )
    if bool(torch.any(kmd.forest.doftype[reversed_nodes] != int(NodeType.bond))):
        raise ValueError("a reversed ligand torsion controller is not a bond node")
    if bool(torch.any(kmd.keep_atom_fixed[reversed_nodes, int(BondDOFTypes.phi_p)])):
        raise ValueError("both ends of a ligand torsion are fixed by the fold forest")

    raw_keys = raw_atoms * raw_dofs.shape[1] + dof_types
    if torch.unique(raw_keys).numel() != raw_keys.numel():
        raise ValueError("two ligand torsions share the same raw DOF")

    signs = _empirical_torsion_signs(
        kin_module,
        raw_dofs,
        ligand_nodes,
        torsion_atoms,
        raw_nodes,
        dof_types,
    )

    pose_indices = torch.arange(
        pose_stack.n_poses, dtype=torch.int64, device=pose_stack.device
    )
    coord_atom_indices = (
        pose_stack.block_coord_offset64[:, ligand_block, None, None]
        + torsion_atoms[None, :, :]
    )
    coord_pose_indices = pose_indices[:, None, None].expand_as(coord_atom_indices)
    logical = _batched_coord_dihedrals(
        pose_stack.coords.reshape(-1, 3),
        coord_pose_indices * pose_stack.max_n_pose_atoms + coord_atom_indices,
    )
    raw = raw_dofs[raw_nodes, dof_types[None, :]].double()
    offsets = _wrap_angle(logical - signs[None, :] * raw).detach().clone()

    # One aggregate device-to-host transfer supplies the small immutable mapping
    # metadata. It is O(1) synchronizations, independent of poses and atoms.
    mapping_metadata = torch.stack((raw_atoms, dof_types, signs), dim=1).cpu().tolist()
    mappings: list[LigandTorsionDOF] = []
    for torsion_index, (raw_atom, dof_type, sign) in enumerate(mapping_metadata):
        dof_type = BondDOFTypes(dof_type)
        mappings.append(
            LigandTorsionDOF(
                name=block_type.torsions[torsion_index].name,
                torsion_index=torsion_index,
                raw_atom=raw_atom,
                raw_atom_name=block_type.atoms[raw_atom].name,
                dof_type=dof_type,
                sign=sign,
                offset=offsets[:, torsion_index],
            )
        )

    sidechain_mappings = _resolve_sidechain_torsions(
        pose_stack, sidechain_blocks, kin_module, raw_dofs
    )

    move_map = MoveMap.from_pose_stack(pose_stack)
    poses = torch.arange(
        pose_stack.n_poses, dtype=torch.int64, device=pose_stack.device
    )
    blocks = torch.full_like(poses, ligand_block)
    move_map.set_move_all_jump_dofs_for_root_jump(poses, blocks, True)
    for mapping in mappings:
        if mapping.dof_type == BondDOFTypes.phi_c:
            torsions = torch.full_like(poses, mapping.torsion_index)
            move_map.set_move_named_torsion_for_blocks(poses, blocks, torsions, True)
        else:
            atoms = torch.full_like(poses, mapping.raw_atom)
            dof_types = torch.full_like(poses, int(mapping.dof_type))
            move_map.set_move_atom_dof_for_blocks(poses, blocks, atoms, dof_types, True)
    for mapping in sidechain_mappings:
        blocks = torch.full_like(poses, mapping.block)
        if mapping.dof_type == BondDOFTypes.phi_c:
            torsions = torch.full_like(poses, mapping.sc_torsion_index)
            move_map.set_move_sc_tor_for_blocks(poses, blocks, torsions, True)
        else:
            atoms = torch.full_like(poses, mapping.raw_atom)
            dof_types = torch.full_like(poses, int(mapping.dof_type))
            move_map.set_move_atom_dof_for_blocks(poses, blocks, atoms, dof_types, True)

    minimizer_map = MinimizerMap(pose_stack, kmd, move_map)
    for mapping in sidechain_mappings:
        mapping_block_type = pose_stack.packed_block_types.active_block_types[
            int(pose_stack.block_type_ind64[0, mapping.block])
        ]
        mapping_nodes = _block_kinforest_nodes(
            pose_stack,
            mapping.block,
            mapping_block_type.n_atoms,
            kmd.forest.id,
        )[:, mapping.raw_atom]
        if not bool(
            torch.all(minimizer_map.dof_mask[mapping_nodes, int(mapping.dof_type)])
        ):
            raise ValueError(
                f"sidechain chi {mapping.name!r} in block {mapping.block} did "
                "not free its resolved raw DOF"
            )
    expected_per_pose = 6 + len(mappings) + len(sidechain_mappings)
    node_pose = torch.div(
        kmd.forest.id[1:],
        pose_stack.max_n_pose_atoms,
        rounding_mode="floor",
    ).to(torch.int64)
    movable_per_pose = torch.zeros(
        pose_stack.n_poses, dtype=torch.int64, device=pose_stack.device
    )
    movable_per_pose.scatter_add_(
        0,
        node_pose,
        torch.sum(minimizer_map.dof_mask[1:], dim=1).to(torch.int64),
    )
    if bool(torch.any(movable_per_pose != expected_per_pose)):
        raise ValueError(
            "ligand docking MoveMap has movable-DOF counts "
            f"{movable_per_pose.detach().cpu().tolist()}; expected six jump "
            f"DOFs plus {len(mappings)} ligand and {len(sidechain_mappings)} "
            f"sidechain torsions ({expected_per_pose} total)"
        )

    return LigandDockingDOFs(
        fold_forest=fold_forest,
        move_map=move_map,
        ligand_block=ligand_block,
        torsions=tuple(mappings),
        sidechain_blocks=sidechain_blocks,
        sidechain_torsions=sidechain_mappings,
    )
