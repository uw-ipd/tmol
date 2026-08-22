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
from tmol.kinematics.datatypes import BondDOFTypes, NodeType
from tmol.kinematics.fold_forest import EdgeType, FoldForest
from tmol.kinematics.move_map import MinimizerMap, MoveMap
from tmol.kinematics.script_modules import PoseStackKinematicsModule
from tmol.pose.pose_stack import PoseStack


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


@dataclass(frozen=True)
class LigandTorsionDOF:
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

    @property
    def uses_reversed_end(self) -> bool:
        return self.dof_type == BondDOFTypes.phi_p

    def logical_from_raw(self, raw_dof: torch.Tensor) -> torch.Tensor:
        """Convert raw values to logical chi values, modulo 2*pi."""

        return _wrap_angle(self.sign * raw_dof + self.offset)

    def raw_from_logical(self, logical_chi: torch.Tensor) -> torch.Tensor:
        """Convert logical chi values to the equivalent raw values."""

        return _wrap_angle(self.sign * (logical_chi - self.offset))


@dataclass(frozen=True)
class LigandDockingDOFs:
    """Fold forest, MoveMap, and logical ligand torsion mapping for docking."""

    fold_forest: FoldForest
    move_map: MoveMap
    ligand_block: int
    torsions: tuple[LigandTorsionDOF, ...]

    @property
    def logical_chi_to_raw_dof(self) -> dict[str, LigandTorsionDOF]:
        """Return the docking-owned mapping keyed by logical torsion name."""

        return {mapping.name: mapping for mapping in self.torsions}


def _ligand_kinforest_nodes(
    pose_stack: PoseStack,
    ligand_block: int,
    ligand_n_atoms: int,
    forest_ids: torch.Tensor,
) -> torch.Tensor:
    """Return the ``[pose, ligand_atom]`` kinematic-node lookup in one gather."""

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
    local_atoms = torch.arange(
        ligand_n_atoms, dtype=torch.int64, device=pose_stack.device
    )
    target_ids = (
        poses[:, None] * pose_stack.max_n_pose_atoms
        + pose_stack.block_coord_offset64[:, ligand_block, None]
        + local_atoms[None, :]
    )
    ligand_nodes = id_to_node[target_ids]
    if bool(torch.any(ligand_nodes < 0)):
        raise ValueError("a ligand atom is absent from the kinematic forest")
    return ligand_nodes


def _local_torsion_atoms(block_type, torsion_index: int) -> tuple[int, int, int, int]:
    uaids = block_type.ordered_torsions[torsion_index]
    if any(int(uaid[1]) != -1 for uaid in uaids):
        raise ValueError(
            f"ligand torsion {block_type.torsions[torsion_index].name!r} reaches "
            "another block; docking ligand torsions must be block-local"
        )
    atoms = tuple(int(uaid[0]) for uaid in uaids)
    if any(atom < 0 for atom in atoms):
        raise ValueError(
            f"ligand torsion {block_type.torsions[torsion_index].name!r} "
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
    kincoords[1:] = pose_stack.coords.reshape(-1, 3)[
        kmd.forest.id[1:].to(torch.int64)
    ]
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


def _empirical_torsion_signs(
    kin_module: PoseStackKinematicsModule,
    raw_dofs: torch.Tensor,
    ligand_nodes: torch.Tensor,
    torsion_atoms: torch.Tensor,
    raw_nodes: torch.Tensor,
    dof_types: torch.Tensor,
) -> torch.Tensor:
    """Measure all d(declared-dihedral)/d(raw-DOF) signs together."""

    delta = 1.0e-3
    perturbed_dofs = raw_dofs.clone()
    perturbed_dofs[raw_nodes, dof_types[None, :]] += delta

    base_coords = kin_module(raw_dofs)
    perturbed_coords = kin_module(perturbed_dofs)
    atom_nodes = ligand_nodes[:, torsion_atoms]
    base = _batched_coord_dihedrals(base_coords, atom_nodes)
    changed = _batched_coord_dihedrals(perturbed_coords, atom_nodes)
    slopes = _wrap_angle(changed - base) / delta
    unit_slopes = torch.ones_like(slopes)
    if not torch.allclose(
        torch.abs(slopes), unit_slopes, rtol=5.0e-3, atol=5.0e-3
    ):
        raise ValueError(
            "ligand torsion raw-DOF perturbation must change its declared "
            f"dihedral with unit slope; measured {slopes.detach().cpu().tolist()}"
        )

    positive = slopes > 0
    if bool(torch.any(positive != positive[0:1])):
        raise ValueError(
            "ligand torsion raw-DOF slope has inconsistent signs across poses: "
            f"{slopes.detach().cpu().tolist()}"
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


def build_ligand_docking_dofs(
    pose_stack: PoseStack, ligand_block: int
) -> LigandDockingDOFs:
    """Build ligand-only docking freedoms for a protein--ligand PoseStack.

    The ordinary ``reasonable_fold_forest`` is retained: the disconnected
    ligand gets its existing root jump and its existing jump atom/frame.  Six
    DOFs are enabled on that root jump.  Each named ligand torsion first tries
    the normal child-central ``phi_c`` controller.  If the jump stub fixes that
    controller, the resolver reverses the dihedral and enables ``phi_p`` on the
    opposite outer atom, matching Rosetta's bidirectional torsion lookup.

    The affine sign between the selected raw DOF and the declared four-atom
    dihedral is measured by a small perturbation through the actual forward
    kinematics and checked to have unit magnitude.  This makes the mapping
    independent of which valid atom order declares the torsion.

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

    normal_fixed = kmd.keep_atom_fixed[
        child_nodes, int(BondDOFTypes.phi_c)
    ]
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
    if bool(
        torch.any(kmd.forest.parent[reversed_nodes] != reversed_parents)
    ):
        raise ValueError(
            "a reversed ligand torsion endpoint is not a child of its central atom"
        )
    if bool(
        torch.any(kmd.forest.doftype[reversed_nodes] != int(NodeType.bond))
    ):
        raise ValueError("a reversed ligand torsion controller is not a bond node")
    if bool(
        torch.any(
            kmd.keep_atom_fixed[
                reversed_nodes, int(BondDOFTypes.phi_p)
            ]
        )
    ):
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

    move_map = MoveMap.from_pose_stack(pose_stack)
    poses = torch.arange(
        pose_stack.n_poses, dtype=torch.int64, device=pose_stack.device
    )
    blocks = torch.full_like(poses, ligand_block)
    move_map.set_move_all_jump_dofs_for_root_jump(poses, blocks, True)
    for mapping in mappings:
        if mapping.dof_type == BondDOFTypes.phi_c:
            torsions = torch.full_like(poses, mapping.torsion_index)
            move_map.set_move_named_torsion_for_blocks(
                poses, blocks, torsions, True
            )
        else:
            atoms = torch.full_like(poses, mapping.raw_atom)
            dof_types = torch.full_like(poses, int(mapping.dof_type))
            move_map.set_move_atom_dof_for_blocks(
                poses, blocks, atoms, dof_types, True
            )

    minimizer_map = MinimizerMap(pose_stack, kmd, move_map)
    expected_per_pose = 6 + len(mappings)
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
            f"DOFs plus {len(mappings)} torsions ({expected_per_pose} total)"
        )

    return LigandDockingDOFs(
        fold_forest=fold_forest,
        move_map=move_map,
        ligand_block=ligand_block,
        torsions=tuple(mappings),
    )
