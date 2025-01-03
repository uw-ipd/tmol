import torch

from typing import Optional, Union
from tmol.types.torch import Tensor

from tmol.pose.pose_stack import PoseStack
from tmol.kinematics.datatypes import (
    JumpDOFTypes,
    BondDOFTypes,
    KinematicModuleData,
    n_movable_bond_dof_types,
    n_movable_jump_dof_types,
)
from tmol.kinematics.compiled.compiled_ops import minimizer_map_from_movemap


class MoveMap:

    @classmethod
    def from_pose_stack_and_kmd(cls, ps: PoseStack, kmd: KinematicModuleData):
        """Main construction utility for MoveMap."""
        return cls(
            ps.n_poses,
            ps.max_n_blocks,
            ps.packed_block_types.max_n_torsions,
            ps.max_n_block_atoms,
            ps.device,
        )

    def set_move_all_jump_dofs_for_jump(
        self,
        pose_selection: Union[Tensor, int],
        jump_selection: Optional[Union[Tensor, int]] = None,
        value: bool = True,
    ):
        """Enable or disable all jump dofs for a particular set of jumps on particular poses.

        If jump_selection is None, then the two dimensional settings tensor will be indexed by
        the pose_selection tensor only; if both are not None, then the settings tensor will be indexed by
        both the pose_selection tensor and the jump_selection tensor.
        """
        self._set_val_and_mask_for_2dim(pose_selection, jump_selection, "jumps", value)

    def set_move_all_mc_tors_for_blocks(
        self,
        pose_selection: Union[Tensor, int],
        block_selection: Optional[Union[Tensor, int]] = None,
        value: bool = True,
    ):
        """Enable or disable all DOFs for a partiular set of blocks on particular poses.

        If block_selection is None, then the two dimensional settings tensor will be indexed by
        the pose_selection tensor only; if both are not None, then the tensor will be indexed by
        the pose_selection tensor and the block_selection tensor.

        Valid combinations of pose_selection and block_selection are, e.g.:
          - pose_selection: int, block_selection: int == a particular block on a particular pose
          - pose_selection: int, block_selection: None == all blocks on a particular pose
          - pose_selection: Tensor[bool][n_poses, max_n_blocks], block_selection: None == pose/block pairs encoded in "pose_selection" tensor
          - pose_selection: Tensor[int][N], block_selection: Tensor[int][N] == different blocks on different poses, selected by index
        """
        self._assert_valid_pose_block_selection(pose_selection, block_selection)
        self._set_val_and_mask_for_2dim(pose_selection, block_selection, "mcs", value)

    def set_move_all_sc_tors_for_blocks(
        self,
        pose_selection: Union[Tensor, int],
        block_selection: Optional[Union[Tensor, int]] = None,
        value: bool = True,
    ):
        self._assert_valid_pose_block_selection(pose_selection, block_selection)
        self._set_val_and_mask_for_2dim(pose_selection, block_selection, "scs", value)

    def set_move_all_named_torsions_for_blocks(
        self,
        pose_selection: Union[Tensor, int],
        block_selection: Optional[Union[Tensor, int]] = None,
        value: bool = True,
    ):
        self._assert_valid_pose_block_selection(pose_selection, block_selection)
        self._set_val_and_mask_for_2dim(
            pose_selection, block_selection, "named_torsions", value
        )

    def set_move_mc_tor_for_blocks(
        self,
        pose_selection: Union[Tensor, int],
        block_selection: Optional[Union[Tensor, int]] = None,
        tor_selection: Optional[Union[Tensor, int]] = None,
        value: bool = True,
    ):
        """Enable or disable partiular main-chain torsions for a particular set of blocks on particular poses.

        Valid combinations of block_selection and tor_selection are:
          - pose_selection: int, block_selection: int, tor_selection: int == a single DOF
          - pose_selection: Tensor[bool][n_poses, max_n_blocks, max_n_dofs], block_selection: None, tor_selection: None == pose/block/tor triples encoded in "pose_selection" tensor
          - pose_selection: Tensor[int][N], block_selection: Tensor[int][N], dof_selection: Tensor[int][N] == different torsions on different blocks on different poses, selected by index
        """
        self._assert_valid_pose_block_dof_selection(
            pose_selection, block_selection, tor_selection, self.max_n_named_torsions
        )
        self._set_val_and_mask_for_3dim(
            pose_selection, block_selection, tor_selection, "mc", value
        )

    def set_move_sc_tor_for_blocks(
        self,
        pose_selection: Union[Tensor, int],
        block_selection: Optional[Union[Tensor, int]] = None,
        dof_selection: Optional[Union[Tensor, int]] = None,
        value: bool = True,
    ):
        """Enable or disable partiular side-chain torsions for a particular set of blocks on particular poses."""
        self._assert_valid_pose_block_dof_selection(
            pose_selection, block_selection, dof_selection, self.max_n_named_torsions
        )
        self._set_val_and_mask_for_3dim(
            pose_selection, block_selection, dof_selection, "sc", value
        )

    def set_move_named_torsion_for_blocks(
        self,
        pose_selection: Union[Tensor, int],
        block_selection: Optional[Union[Tensor, int]] = None,
        tor_selection: Optional[Union[Tensor, int]] = None,
        value: bool = True,
    ):
        """Enable or disable partiular named-torsions for a particular set of blocks on particular poses."""
        self._assert_valid_pose_block_dof_selection(
            pose_selection, block_selection, tor_selection, self.max_n_named_torsions
        )
        self._set_val_and_mask_for_3dim(
            pose_selection, block_selection, tor_selection, "named_torsion", value
        )

    def set_move_jump_dof_for_jumps(
        self,
        pose_selection: Union[Tensor, int],
        jump_selection: Optional[Union[Tensor, int]] = None,
        dof_selection: Optional[Union[Tensor, int]] = None,
        value: bool = True,
    ):
        """Enable or disable all jump dofs for a particular set of jumps on particular poses.

        If jump_selection is None, then the two dimensional settings tensor will be indexed by
        the pose_selection tensor only; if both are not None, then the settings tensor will be indexed by
        both the pose_selection tensor and the jump_selection tensor.
        """
        self._assert_valid_pose_block_dof_selection(
            pose_selection, jump_selection, dof_selection, n_movable_jump_dof_types
        )
        self._set_val_and_mask_for_3dim(
            pose_selection, jump_selection, dof_selection, "jump_dof", value
        )

    def set_move_atom_dof_for_blocks(
        self,
        pose_selection: Union[Tensor, int],
        block_selection: Optional[Union[Tensor, int]] = None,
        atom_selection: Optional[Union[Tensor, int]] = None,
        dof_selection: Optional[Union[Tensor, int]] = None,
        value: bool = True,
    ):
        """Enable or disable partiular atom dofs for a particular set of blocks on particular poses.

        Either only "pose_selection" should be not None or all four "selection" variables should be
        not None; in the former case, the settings tensor, self.move_atom_dof, will be indexed
        solely by the pose_selection tensor; in the latter case, the settings tensor will be indexed
        by all four selection tensors.

        This function offers the finest grain control over which dofs should be minimized and
        settings made using this function will override any settings made using the other
        settings tensors.
        """
        if block_selection is None:
            assert atom_selection is None
            assert dof_selection is None
            self.move_atom_dof[pose_selection] = value
            self.move_atom_dof_mask[pose_selection] = True
        else:
            self.move_atom_dof[
                pose_selection, block_selection, atom_selection, dof_selection
            ] = value
            self.move_atom_dof_mask[
                pose_selection, block_selection, atom_selection, dof_selection
            ] = True

    @property
    def n_poses(self):
        return self.move_jumps.shape[0]

    @property
    def max_n_jumps(self):
        return self.move_jumps.shape[1]

    @property
    def max_n_blocks(self):
        return self.move_mcs.shape[1]

    @property
    def max_n_named_torsions(self):
        return self.move_named_torsion.shape[2]

    @property
    def max_n_atoms_per_block(self):
        return self.move_atom_dof.shape[2]

    def __init__(
        self,
        n_poses: int,
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
            return z_bool((n_poses, max_n_blocks) + remaining_shape)

        # data members on a per-residue basis; plural, representing
        # all of the torsions of a particular class for a residue
        self.move_jumps = z_bool_pb()
        self.move_jumps_mask = z_bool_pb()
        self.move_mcs = z_bool_pb()
        self.move_mcs_mask = z_bool_pb()
        self.move_scs = z_bool_pb()
        self.move_scs_mask = z_bool_pb()
        self.move_named_torsions = z_bool_pb()
        self.move_named_torsions_mask = z_bool_pb()

        # data members on a per-torsion, per-residue basis; singular,
        # representing a single torsion of a particular class for a residue
        self.move_mc = z_bool_pb((max_n_named_torsions,))
        self.move_mc_mask = z_bool_pb((max_n_named_torsions,))
        self.move_sc = z_bool_pb((max_n_named_torsions,))
        self.move_sc_mask = z_bool_pb((max_n_named_torsions,))
        self.move_named_torsion = z_bool_pb((max_n_named_torsions,))
        self.move_named_torsion_mask = z_bool_pb((max_n_named_torsions,))

        # data members on a per-DOF basis:
        # RBx,y,z and RBdel_alpha,_beta,_gamma for jumps, and
        # phi_p, theta, d, phi_c for atoms
        self.move_jump_dof = z_bool_pb((n_movable_jump_dof_types,))
        self.move_jump_dof_mask = z_bool_pb((n_movable_jump_dof_types,))
        self.move_atom_dof = z_bool_pb(
            (max_n_atoms_per_block, n_movable_bond_dof_types)
        )
        self.move_atom_dof_mask = z_bool_pb(
            (max_n_atoms_per_block, n_movable_bond_dof_types)
        )

    def _assert_valid_pose_block_selection(self, pose_sel, block_sel):
        if block_sel is None:
            if isinstance(pose_sel, int):
                return
            if (
                isinstance(pose_sel, torch.Tensor)
                and pose_sel.dim() == 2
                and pose_sel.dtype == torch.bool
            ):
                assert pose_sel.shape == (self.n_poses, self.max_n_blocks)
        else:
            if isinstance(pose_sel, torch.Tensor) and isinstance(
                block_sel, torch.Tensor
            ):
                if pose_sel.dtype == torch.int64 and block_sel.dtype == torch.int64:
                    assert pose_sel.shape == block_sel.shape

    def _assert_valid_pose_block_dof_selection(
        self, pose_sel, block_sel, dof_sel, n_dofs
    ):
        if block_sel is None:
            assert dof_sel is None
            if isinstance(pose_sel, int):
                return
            if (
                isinstance(pose_sel, torch.Tensor)
                and pose_sel.dim() == 3
                and pose_sel.dtype == torch.bool
            ):
                assert pose_sel.shape == (
                    self.n_poses,
                    self.max_n_blocks,
                    n_dofs,
                )
        else:
            assert dof_sel is not None
            if isinstance(pose_sel, torch.Tensor) and isinstance(
                block_sel, torch.Tensor
            ):
                if pose_sel.dtype == torch.int64 and block_sel.dtype == torch.int64:
                    assert pose_sel.shape == block_sel.shape
            if isinstance(pose_sel, torch.Tensor) and isinstance(dof_sel, torch.Tensor):
                if pose_sel.dtype == torch.int64 and dof_sel.dtype == torch.int64:
                    assert pose_sel.shape == dof_sel.shape
            if isinstance(block_sel, torch.Tensor) and isinstance(
                dof_sel, torch.Tensor
            ):
                if block_sel.dtype == torch.int64 and dof_sel.dtype == torch.int64:
                    assert block_sel.shape == dof_sel.shape

    def _set_val_and_mask_for_2dim(self, sel1, sel2, varname, value):
        var_to_set = getattr(self, "move_" + varname)
        mask_to_set = getattr(self, "move_" + varname + "_mask")
        if sel2 is None:
            var_to_set[sel1] = value
            mask_to_set[sel1] = True
        else:
            var_to_set[sel1, sel2] = value
            mask_to_set[sel1, sel2] = True

    def _set_val_and_mask_for_3dim(self, sel1, sel2, sel3, varname, value):
        var_to_set = getattr(self, "move_" + varname)
        mask_to_set = getattr(self, "move_" + varname + "_mask")
        if sel2 is None and sel3 is None:
            var_to_set[sel1] = value
            mask_to_set[sel1] = True
        else:
            var_to_set[sel1, sel2, sel3] = value
            mask_to_set[sel1, sel2, sel3] = True


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
            mm.move_mc,
            mm.move_mc_mask,
            mm.move_sc,
            mm.move_sc_mask,
            mm.move_named_torsion,
            mm.move_named_torsion_mask,
            mm.move_atom_dof,
            mm.move_atom_dof_mask,
        )
        # fmt: on
