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


def _kinforest_node_for_target_atom(kmd, target_atom: int) -> int:
    matches = torch.nonzero(kmd.forest.id == target_atom, as_tuple=False).flatten()
    if matches.numel() != 1:
        raise ValueError(
            f"expected one kinematic node for pose atom {target_atom}, "
            f"found {matches.numel()}"
        )
    return int(matches[0])


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


def _logical_torsion_values(
    pose_stack: PoseStack, ligand_block: int, atoms: tuple[int, int, int, int]
) -> torch.Tensor:
    coords = pose_stack.coords.double()
    offsets = pose_stack.block_coord_offset64[:, ligand_block]
    pose_indices = torch.arange(
        pose_stack.n_poses, dtype=torch.int64, device=pose_stack.device
    )
    selected = [coords[pose_indices, offsets + atom] for atom in atoms]
    return _coord_dihedrals(*selected)


def _empirical_torsion_sign(
    kin_module: PoseStackKinematicsModule,
    raw_dofs: torch.Tensor,
    pose_nodes: list[dict[int, int]],
    atoms: tuple[int, int, int, int],
    raw_atom: int,
    dof_type: BondDOFTypes,
) -> int:
    """Measure d(declared dihedral)/d(raw DOF) through forward kinematics."""

    delta = 1.0e-3
    perturbed_dofs = raw_dofs.clone()
    for nodes in pose_nodes:
        perturbed_dofs[nodes[raw_atom], dof_type] += delta

    base_coords = kin_module(raw_dofs)
    perturbed_coords = kin_module(perturbed_dofs)

    def declared_dihedrals(coords):
        selected = [
            torch.stack([coords[nodes[atom]] for nodes in pose_nodes]).double()
            for atom in atoms
        ]
        return _coord_dihedrals(*selected)

    base = declared_dihedrals(base_coords)
    changed = declared_dihedrals(perturbed_coords)
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
    if bool(torch.any(positive != positive[0])):
        raise ValueError(
            "ligand torsion raw-DOF slope has inconsistent signs across poses: "
            f"{slopes.detach().cpu().tolist()}"
        )
    return 1 if bool(positive[0]) else -1


def _validate_ligand_root_jumps(
    pose_stack: PoseStack, ligand_block: int, fold_forest: FoldForest
) -> None:
    for pose in range(pose_stack.n_poses):
        edges = fold_forest.edges[pose, : fold_forest.n_edges[pose]]
        ligand_roots = [
            edge
            for edge in edges
            if int(edge[0]) == int(EdgeType.root_jump)
            and int(edge[2]) == ligand_block
        ]
        if len(ligand_roots) != 1:
            raise ValueError(
                f"ligand block {ligand_block} in pose {pose} must have exactly "
                f"one root jump; found {len(ligand_roots)}"
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

    pose_nodes: list[dict[int, int]] = []
    for pose in range(pose_stack.n_poses):
        block_offset = int(pose_stack.block_coord_offset64[pose, ligand_block])
        pose_base = pose * pose_stack.max_n_pose_atoms
        pose_nodes.append(
            {
                atom: _kinforest_node_for_target_atom(
                    kmd, pose_base + block_offset + atom
                )
                for atom in range(block_type.n_atoms)
            }
        )

    mappings: list[LigandTorsionDOF] = []
    claimed_raw_dofs: set[tuple[int, BondDOFTypes]] = set()
    for torsion_index, torsion in enumerate(block_type.torsions):
        a, b, c, d = _local_torsion_atoms(block_type, torsion_index)

        first_nodes = pose_nodes[0]
        if int(kmd.forest.parent[first_nodes[c]]) == first_nodes[b]:
            child_central = c
            reverse_outer = a
        elif int(kmd.forest.parent[first_nodes[b]]) == first_nodes[c]:
            child_central = b
            reverse_outer = d
        else:
            raise ValueError(
                f"central atoms for ligand torsion {torsion.name!r} are not a "
                "parent/child pair in the docking fold forest"
            )

        normal_fixed = []
        for pose, nodes in enumerate(pose_nodes):
            child_node = nodes[child_central]
            central_parent = b if child_central == c else c
            if int(kmd.forest.parent[child_node]) != nodes[central_parent]:
                raise ValueError(
                    f"ligand torsion {torsion.name!r} has inconsistent tree "
                    f"orientation in pose {pose}"
                )
            if int(kmd.forest.doftype[child_node]) != int(NodeType.bond):
                raise ValueError(
                    f"normal controller for ligand torsion {torsion.name!r} "
                    "is not a bond node"
                )
            normal_fixed.append(
                bool(kmd.keep_atom_fixed[child_node, BondDOFTypes.phi_c])
            )

        if any(fixed != normal_fixed[0] for fixed in normal_fixed[1:]):
            raise ValueError(
                f"ligand torsion {torsion.name!r} has inconsistent fixed-DOF "
                "status across poses"
            )

        if not normal_fixed[0]:
            raw_atom = child_central
            dof_type = BondDOFTypes.phi_c
        else:
            raw_atom = reverse_outer
            dof_type = BondDOFTypes.phi_p
            for pose, nodes in enumerate(pose_nodes):
                raw_node = nodes[raw_atom]
                parent_central = b if child_central == c else c
                if int(kmd.forest.parent[raw_node]) != nodes[parent_central]:
                    raise ValueError(
                        f"reversed endpoint for ligand torsion {torsion.name!r} "
                        f"is not a child of its central atom in pose {pose}"
                    )
                if int(kmd.forest.doftype[raw_node]) != int(NodeType.bond):
                    raise ValueError(
                        f"reversed controller for ligand torsion {torsion.name!r} "
                        "is not a bond node"
                    )
                if bool(kmd.keep_atom_fixed[raw_node, dof_type]):
                    raise ValueError(
                        f"both ends of ligand torsion {torsion.name!r} are fixed "
                        f"by the docking fold forest in pose {pose}"
                    )

        sign = _empirical_torsion_sign(
            kin_module,
            raw_dofs,
            pose_nodes,
            (a, b, c, d),
            raw_atom,
            dof_type,
        )

        raw_key = (raw_atom, dof_type)
        if raw_key in claimed_raw_dofs:
            raise ValueError(
                f"ligand torsion {torsion.name!r} shares raw DOF "
                f"{block_type.atoms[raw_atom].name}.{dof_type.name}"
            )
        claimed_raw_dofs.add(raw_key)

        logical = _logical_torsion_values(
            pose_stack, ligand_block, (a, b, c, d)
        )
        raw = torch.stack(
            [
                raw_dofs[pose_nodes[pose][raw_atom], dof_type].double()
                for pose in range(pose_stack.n_poses)
            ]
        )
        offset = _wrap_angle(logical - sign * raw).detach().clone()
        mappings.append(
            LigandTorsionDOF(
                name=torsion.name,
                torsion_index=torsion_index,
                raw_atom=raw_atom,
                raw_atom_name=block_type.atoms[raw_atom].name,
                dof_type=dof_type,
                sign=sign,
                offset=offset,
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
    for pose in range(pose_stack.n_poses):
        first_target_atom = pose * pose_stack.max_n_pose_atoms
        last_target_atom = first_target_atom + pose_stack.max_n_pose_atoms
        nodes_for_pose = (kmd.forest.id >= first_target_atom) & (
            kmd.forest.id < last_target_atom
        )
        actual = int(torch.sum(minimizer_map.dof_mask[nodes_for_pose]))
        if actual != expected_per_pose:
            raise ValueError(
                f"ligand docking MoveMap for pose {pose} has {actual} movable "
                f"DOFs; expected six jump DOFs plus {len(mappings)} torsions "
                f"({expected_per_pose} total)"
            )

    return LigandDockingDOFs(
        fold_forest=fold_forest,
        move_map=move_map,
        ligand_block=ligand_block,
        torsions=tuple(mappings),
    )
