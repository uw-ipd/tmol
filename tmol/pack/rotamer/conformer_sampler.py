import torch
import attr

from typing import Tuple

from tmol.types import Tensor
from tmol.types import validate_args

from tmol.chemical import RefinedResidueType
from tmol.pose import PackedBlockTypes
from tmol.pose import PoseStack
from tmol.kinematics import KinForest


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

    def defines_rotamers_for_bts(
        self, pbt: PackedBlockTypes, bt_inds: Tensor[torch.int64]
    ) -> Tensor[torch.bool]:
        raise NotImplementedError()

    @validate_args
    def first_sc_atoms_for_rt(self, rt: RefinedResidueType) -> Tuple[str, ...]:
        raise NotImplementedError()

    def create_samples_for_poses(
        self,
        pose_stack: PoseStack,
        task: "PackerTask",  # noqa: 821
    ) -> Tuple[  # noqa F821
        Tensor[torch.int32][:],  # n_rots_for_bt
        Tensor[torch.int32][:],  # bt_for_rotamer
        dict,  # anything else the sampler wants to save for later
    ]:
        raise NotImplementedError()

    def fill_dofs_for_samples(
        self,
        pose_stack: PoseStack,
        task: "PackerTask",  # noqa: 821
        orig_kinforest: KinForest,
        orig_dofs_kto: Tensor[torch.float32][:, 9],
        gbt_for_conformer: Tensor[torch.int64][:],
        block_type_ind_for_conformer: Tensor[torch.int64][:],
        n_dof_atoms_offset_for_conformer: Tensor[torch.int64][:],
        # which of all conformers are built by this sampler
        conformer_built_by_sampler: Tensor[torch.bool][:],
        # mapping orig conformer samples to merged conformer samples for this sampler
        conf_inds_for_sampler: Tensor[torch.int64][:],
        sampler_n_rots_for_gbt: Tensor[torch.int32][:],
        sampler_gbt_for_rotamer: Tensor[torch.int32][:],
        sample_dict: dict,
        conf_dofs_kto: Tensor[torch.float32][:, 9],
    ):
        raise NotImplementedError
