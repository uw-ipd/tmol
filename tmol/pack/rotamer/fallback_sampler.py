import numpy
import torch
import attr

from typing import Tuple

from tmol.types.torch import Tensor
from tmol.types.functional import validate_args

from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack
from tmol.kinematics.datatypes import KinForest
from tmol.pack.rotamer.conformer_sampler import ConformerSampler
from tmol.pack.rotamer.include_current_sampler import (
    create_full_dof_inds_to_copy_from_orig_to_rotamers_for_include_current_sampler,
)


@attr.s(auto_attribs=True, frozen=True)
class FallbackSampler(ConformerSampler):
    """Include the input conformation as a rotamer only for positions that have
    no rotamers from any other sampler.

    This is the default sampler in PackerPalette. Unlike IncludeCurrentSampler,
    it does not unconditionally add a rotamer for every position; instead it
    activates only when every other sampler in the block-level task returns
    False from defines_rotamers_for_rt for the original block type, ensuring
    that positions covered by, e.g., DunbrackChiSampler do not accumulate an
    extra current-conformation rotamer.

    The disable_packing case (all block types disallowed) is also handled: a
    rotamer from the input conformation is always produced so the packer has
    something to represent for fixed residues.
    """

    @classmethod
    def sampler_name(cls):
        return "FallbackSampler"

    @validate_args
    def annotate_residue_type(self, rt: RefinedResidueType):
        pass

    @validate_args
    def annotate_packed_block_types(self, packed_block_types: PackedBlockTypes):
        pass

    @validate_args
    def defines_rotamers_for_rt(self, rt: RefinedResidueType):
        return True

    @validate_args
    def first_sc_atoms_for_rt(self, rt: RefinedResidueType) -> Tuple[str, ...]:
        return (rt.default_jump_connection_atom,)

    def create_samples_for_poses(
        self,
        pose_stack: PoseStack,
        task: "PackerTask",  # noqa: F821
    ) -> Tuple[  # noqa F821
        Tensor[torch.int32][:],  # n_rots_for_gbt
        Tensor[torch.int32][:],  # gbt_for_rotamer
        dict,
    ]:
        n_rots_for_gbt_list = [
            (
                1
                if bt is blt.original_block_type
                and (
                    not numpy.any(blt.block_type_allowed)
                    or not any(
                        s
                        for s in blt.conformer_samplers
                        if not isinstance(s, FallbackSampler)
                        and s.defines_rotamers_for_rt(bt)
                    )
                )
                else 0
            )
            for one_pose_blts in task.blts
            for blt in one_pose_blts
            for bt in blt.considered_block_types
        ]
        n_rots_for_gbt = torch.tensor(
            n_rots_for_gbt_list, dtype=torch.int32, device=pose_stack.device
        )
        gbt_for_rotamer = torch.nonzero(n_rots_for_gbt, as_tuple=True)[0]
        return (n_rots_for_gbt, gbt_for_rotamer, {})

    def fill_dofs_for_samples(
        self,
        pose_stack: PoseStack,
        task: "PackerTask",  # noqa: F821
        orig_kinforest: KinForest,
        orig_dofs_kto: Tensor[torch.float32][:, 9],
        gbt_for_conformer: Tensor[torch.int64][:],
        block_type_ind_for_conformer: Tensor[torch.int64][:],
        n_dof_atoms_offset_for_conformer: Tensor[torch.int64][:],
        conformer_built_by_sampler: Tensor[torch.bool][:],
        conf_inds_for_sampler: Tensor[torch.int64][:],
        sampler_n_rots_for_gbt: Tensor[torch.int32][:],
        sampler_gbt_for_rotamer: Tensor[torch.int32][:],
        sample_dict: dict,
        conf_dofs_kto: Tensor[torch.float32][:, 9],
    ):
        n_rots = sampler_gbt_for_rotamer.shape[0]
        if n_rots == 0:
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dst, src = (
            create_full_dof_inds_to_copy_from_orig_to_rotamers_for_include_current_sampler(
                pose_stack,
                task,
                gbt_for_conformer,
                block_type_ind_for_conformer,
                conf_inds_for_sampler,
                sampler_n_rots_for_gbt,
                sampler_gbt_for_rotamer,
                n_dof_atoms_offset_for_conformer,
            )
        )

        conf_dofs_kto[dst + 1, :] = orig_dofs_kto[src + 1, :]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
