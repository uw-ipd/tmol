import torch
import attr

from typing import Tuple

from tmol.types import (
    Tensor,
    validate_args,
)
from tmol.chemical import RefinedResidueType
from tmol.pose import (
    PackedBlockTypes,
    PoseStack,
)
from tmol.kinematics import KinForest
from tmol.pack.rotamer import ConformerSampler


@attr.s(auto_attribs=True, frozen=True)
class IncludeCurrentSampler(ConformerSampler):

    @classmethod
    def sampler_name(cls):
        return "IncludeCurrentSampler"

    @validate_args
    def annotate_residue_type(self, rt: RefinedResidueType):
        pass

    @validate_args
    def annotate_packed_block_types(self, packed_block_types: PackedBlockTypes):
        pass

    @validate_args
    def defines_rotamers_for_rt(self, rt: RefinedResidueType):
        return True

    def defines_rotamers_for_bts(
        self, pbt: PackedBlockTypes, bt_inds: Tensor[torch.int64]
    ) -> Tensor[torch.bool]:
        return torch.ones_like(bt_inds, dtype=torch.bool)

    @validate_args
    def first_sc_atoms_for_rt(self, rt: RefinedResidueType) -> Tuple[str, ...]:
        return (rt.default_jump_connection_atom,)

    def create_samples_for_poses(
        self,
        pose_stack: PoseStack,
        task: "SetPackerTask",  # noqa: 821
    ) -> Tuple[  # noqa F821
        Tensor[torch.int32][:],  # n_rots_for_gbt
        Tensor[torch.int32][:],  # gbt_for_rotamer
        dict,  # anything else the sampler wants to save for later
    ]:
        assert (
            id(self) in task.conformer_sampler_index
        ), "This sampler is not in the PackerTask's conformer samplers"
        self_ind_in_packer_task = task.conformer_sampler_index[id(self)]

        is_gbt_allowed_and_buildable = torch.logical_and(
            task.per_block_is_block_type_allowed[
                task.cons_bt_pose, task.cons_bt_block, task.cons_bt_which_block_type
            ],
            task.per_block_conformer_sampler_allowed[
                task.cons_bt_pose, task.cons_bt_block, self_ind_in_packer_task
            ],
        )
        n_rots_for_gbt = torch.logical_and(
            is_gbt_allowed_and_buildable,
            task.per_block_considered_block_types_is_orig[
                task.cons_bt_pose, task.cons_bt_block, task.cons_bt_which_block_type
            ],
        ).to(torch.int32)

        gbt_for_rotamer = torch.nonzero(n_rots_for_gbt, as_tuple=True)[0]
        return (n_rots_for_gbt, gbt_for_rotamer, {})

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
        n_rots = sampler_gbt_for_rotamer.shape[0]
        if n_rots == 0:
            return

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


# @validate_args
def create_full_dof_inds_to_copy_from_orig_to_rotamers_for_include_current_sampler(
    poses: PoseStack,
    task: "SetPackerTask",  # noqa F821
    gbt_for_rot: Tensor[torch.int64][:],  # max-n-rots
    block_type_ind_for_rot: Tensor[torch.int64][:],
    conf_inds_for_sampler: Tensor[torch.int64][:],
    sampler_n_rots_for_gbt: Tensor[torch.int32][:],
    sampler_gbt_for_rotamer: Tensor[torch.int32][:],
    n_dof_atoms_offset_for_rot: Tensor[torch.int64][:],
) -> Tuple[Tensor[torch.int64][:], Tensor[torch.int64][:]]:
    if conf_inds_for_sampler.numel() == 0:
        return conf_inds_for_sampler, conf_inds_for_sampler
    pbt = poses.packed_block_types
    types = poses.block_type_ind.reshape(-1).long()
    counts = pbt.n_atoms[types.clamp_min(0)].long()
    counts.masked_fill_(types < 0, 0)
    source_offsets = torch.cumsum(counts, 0) - counts
    considered = gbt_for_rot[conf_inds_for_sampler]
    residues = task.global_block_ind_for_considered_block_types[considered]
    target_offsets = n_dof_atoms_offset_for_rot[conf_inds_for_sampler]
    source_minus_target = source_offsets[residues] - target_offsets
    del considered, residues, types, counts, source_offsets
    sizes = pbt.n_atoms[block_type_ind_for_rot[conf_inds_for_sampler]].long()
    packed_offsets = torch.cumsum(sizes, 0) - sizes

    # The one-argument form returns each conformer's index once per atom.
    # Work scales with copied atoms, independent of unrelated large types.
    source = torch.repeat_interleave(sizes)
    destination = torch.arange(source.numel(), device=poses.device)
    destination.add_((target_offsets - packed_offsets)[source])
    # Reuse the fresh conformer-index buffer for source atom indices. Neither
    # input metadata nor DOFs are modified, and no padded atom matrix is built.
    source.copy_(source_minus_target[source])
    source.add_(destination)
    # These exclude the virtual-root row; the caller adds one to both indices.
    return destination, source
