import torch

from typing import Optional

from tmol.pose.pose_stack import PoseStack
from tmol.kinematics.datatypes import JumpDOFTypes, BondDOFTypes, KinematicModuleData
from tmol.kinematics.compiled.compiled_ops import minimizer_map_from_movemap


class MoveMap:
    def __init__(
        self,
        max_n_poses: int,
        max_n_blocks: int,
        max_n_named_torsions: int,
        max_n_atoms_per_block: int,
        device: Optional[torch.device] = None,
    ):
        if device is None:
            device = torch.device("cpu")

        self.move_all_jumps = False
        self.move_all_mc = False
        self.move_all_sc = False
        self.move_all_named_torsions = False

        def z_bool(shape):
            return torch.zeros(shape, dtype=torch.bool, device=device)

        def z_bool_pb(remaining_shape=tuple()):
            return z_bool((max_n_poses, max_n_blocks) + remaining_shape)

        # data members on a per-residue basis
        self.move_jumps = z_bool_pb()
        self.move_jumps_mask = z_bool_pb()
        self.move_mcs = z_bool_pb()
        self.move_mcs_mask = z_bool_pb()
        self.move_scs = z_bool_pb()
        self.move_scs_mask = z_bool_pb()
        self.move_named_torsions = z_bool_pb()
        self.move_named_torsions_mask = z_bool_pb()

        # data members on a per-DOF basis
        self.move_jump_dof = z_bool_pb((JumpDOFTypes.n_movable_dofs,))
        self.move_jump_dof_mask = z_bool_pb((JumpDOFTypes.n_movable_dofs,))
        self.move_mc_dof = z_bool_pb((max_n_named_torsions,))
        self.move_mc_dof_mask = z_bool_pb((max_n_named_torsions,))
        self.move_sc_dof = z_bool_pb((max_n_named_torsions,))
        self.move_sc_dof_mask = z_bool_pb((max_n_named_torsions,))
        self.move_named_torsion_dof = z_bool_pb((max_n_named_torsions,))
        self.move_named_torsion_dof_mask = z_bool_pb((max_n_named_torsions,))

        # data members on a per-atom basis
        self.move_atom_dof = z_bool_pb(
            (max_n_atoms_per_block, BondDOFTypes.n_movable_dofs)
        )
        self.move_atom_dof_mask = z_bool_pb(
            (max_n_atoms_per_block, BondDOFTypes.n_movable_dofs)
        )


class MinimizerMap:
    def __init__(
        self,
        pose_stack: PoseStack,
        kmd: KinematicModuleData,
        mm: MoveMap,
    ):
        pbt = pose_stack.packed_block_types
        # fmt: off
        self.dof_mask = minimizer_map_from_movemap(
            kmd.forest.id,
            pose_stack.max_n_pose_atoms,
            pose_stack.block_coord_offset,
            pose_stack.block_type_ind,
            pose_stack.inter_residue_connections,
            kmd.block_in_and_first_out,
            kmd.pose_stack_atom_for_jump,
            kmd.keep_atom_fixed,
            pbt.n_torsions,
            pbt.torsion_uaids,
            pbt.gen_seg_scan_path_segs.torsion_direction,
            pbt.is_torsion_mc,
            pbt.which_mcsc_torsions,
            pbt.atom_downstream_of_conn,
            mm.move_all_jumps,
            mm.move_all_mc,
            mm.move_all_sc,
            mm.move_all_named_torsions,
            mm.move_jumps,
            mm.move_jumps_mask,
            mm.move_mcs,
            mm.move_mcs_mask,
            mm.move_scs,
            mm.move_scs_mask,
            mm.move_named_torsions,
            mm.move_named_torsions_mask,
            mm.move_jump_dof,
            mm.move_jump_dof_mask,
            mm.move_mc_dof,
            mm.move_mc_dof_mask,
            mm.move_sc_dof,
            mm.move_sc_dof_mask,
            mm.move_named_torsion_dof,
            mm.move_named_torsion_dof_mask,
            mm.move_atom_dof,
            mm.move_atom_dof_mask,
        )
        # fmt: on
