import numpy
import torch
import attr

from typing import Tuple

from tmol.types.torch import Tensor
from tmol.types.array import NDArray
from tmol.types.functional import validate_args

from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack
from tmol.kinematics.datatypes import KinForest


@attr.s(auto_attribs=True)
class ConformerSampler:
    @classmethod
    def sampler_name(cls):
        raise NotImplementedError()

    @validate_args
    def annotate_residue_type(self, rt: RefinedResidueType):
        pass

    @validate_args
    def annotate_packed_block_types(self, packed_block_types: PackedBlockTypes):
        pass

    @validate_args
    def defines_rotamers_for_rt(self, rt: RefinedResidueType):
        raise NotImplementedError()

    def create_samples_for_poses(
        self, pose_stack: PoseStack, task: "PackerTask"
    ) -> Tuple[  # noqa F821
        Tensor[torch.int32][:],  # n_rots_for_bt
        Tensor[torch.int32][:],  # bt_for_rotamer
        dict,  # anything else the sampler wants to save for later
    ]:
        raise NotImplementedError()

    def fill_dofs_for_samples(
        self,
        pose_stack,
        orig_kinforest,
        orig_dofs_kto,
        task,
        rot_inds_for_sampler,
        rot_dofs_kto,
    ):
        raise NotImpllementedError


@attr.s(auto_attribs=True)
class IdealizedRotamerConformerSampler(ConformerSampler):
    """Build rotamers using the starting bb coordinates and ideal side chain bond geometries."""

    @classmethod
    def sampler_name(cls):
        raise NotImplementedError()

    @validate_args
    def annotate_residue_type(self, rt: RefinedResidueType):
        pass

    @validate_args
    def annotate_packed_block_types(self, packed_block_types: PackedBlockTypes):
        pass

    @validate_args
    def defines_rotamers_for_rt(self, rt: RefinedResidueType):
        raise NotImplementedError()

    def create_samples_for_poses(
        self, pose_stack: PoseStack, task: "PackerTask"
    ) -> Tuple[  # noqa F821
        Tensor[torch.int32][:],  # n_rots_for_bt
        Tensor[torch.int32][:],  # bt_for_rotamer
        dict,  # anything else the sampler wants to save for later
    ]:
        (
            n_rots_for_bt,
            bt_for_rotamer,
            chi_defining_atom_for_rotamer,
            chi_for_rotamers,
        ) = self.create_rotamer_samples_for_pose(pose_stack, task)
        return (
            n_rots_for_bt,
            bt_for_rotamer,
            dct(
                chi_defining_atom_for_rotamer=chi_defining_atom_for_rotamer,
                chi_for_rotamers=chi_for_rotamers,
            ),
        )

    def create_rotamer_samples_for_pose(
        pose_stack, task
    ) -> Tuple[  # noqa F821
        Tensor[torch.int32][:],  # n_rots_for_bt
        Tensor[torch.int32][:],  # bt_for_rotamer
        Tensor[torch.int32][:, :],  # chi_defining_atom_for_rotamer
        Tensor[torch.float32][:, :],  # chi_for_rotamers
    ]:
        raise NotImplementedError()

    def fill_dofs_for_samples(
        self,
        pose_stack: PoseStack,
        task: "PackerTask",
        orig_kinforest: KinForest,
        orig_dofs_kto: Tensor[torch.float32][:, 9],
        gbt_for_conformer: Tensor[torch.int64][:],
        block_type_ind_for_conformer: Tensor[torch.int64][:],
        n_dof_atoms_offset_for_conformer: Tensor[torch.int64][:],
        # which of all conformers are built by this sampler
        conformer_built_by_sampler: Tensor[torch.bool][:],
        # mapping orig conformer samples to merged conformer samples for this sampler
        conf_inds_for_sampler: Tensor[torch.int64][:],
        sampler_n_rots_for_bt: Tensor[torch.int32][:],
        sampler_gbt_for_rotamer: Tensor[torch.int32][:],
        sample_dict: dict,
        conf_dofs_kto: Tensor[torch.float32][:, 9],
    ):
        raise NotImplementedError()

        # chi_defining_atom_for_rotamer = sample_dict["chi_defining_atom_for_rotamer"]
        # chi_for_rotamers = sample_dict["chi_for_rotamers"]
        #
        # copy_dofs_from_orig_to_rotamers(
        #     pose_stack, task, samplers, rt_for_rot, block_type_ind_for_rot, sampler_for_rotamer,
        #     n_dof_atoms_offset_for_rot, orig_dofs_kto, rot_dofs_kto
        # )
