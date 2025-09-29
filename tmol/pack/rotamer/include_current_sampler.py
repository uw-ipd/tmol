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
from tmol.pack.rotamer.conformer_sampler import ConformerSampler


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

    @validate_args
    def first_sc_atoms_for_rt(self, rt: RefinedResidueType) -> Tuple[str, ...]:
        return (rt.default_jump_connection_atom,)

    def create_samples_for_poses(
        self, pose_stack: PoseStack, task: "PackerTask"
    ) -> Tuple[  # noqa F821
        Tensor[torch.int32][:],  # n_rots_for_gbt
        Tensor[torch.int32][:],  # gbt_for_rotamer
        dict,  # anything else the sampler wants to save for later
    ]:
        n_poses = pose_stack.n_poses
        n_rots_for_gbt_list = [
            (
                1
                if bt is blt.original_block_type
                and (blt.include_current or not numpy.any(blt.block_type_allowed))
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

        # add one for the virtual root
        conf_dofs_kto[dst + 1, :] = orig_dofs_kto[src + 1, :]


# @validate_args
def create_full_dof_inds_to_copy_from_orig_to_rotamers_for_include_current_sampler(
    poses: PoseStack,
    task: "PackerTask",  # noqa F821
    gbt_for_rot: Tensor[torch.int64][:],  # max-n-rots
    block_type_ind_for_rot: Tensor[torch.int64][:],
    conf_inds_for_sampler: Tensor[torch.int64][:],
    sampler_n_rots_for_gbt: Tensor[torch.int32][:],
    sampler_gbt_for_rotamer: Tensor[torch.int32][:],
    n_dof_atoms_offset_for_rot: Tensor[torch.int64][:],
) -> Tuple[Tensor[torch.int64][:], Tensor[torch.int64][:]]:
    # we want to copy from the orig_dofs tensor into the
    # rot_dofs tensor for the "mainchain" atoms in the
    # original residues into the appropriate positions
    # for the rotamers thta we are building at those
    # residues. This requires a good deal of reindexing.

    # print("Include current: n_dof_atoms_offset_for_rot")
    # print(n_dof_atoms_offset_for_rot)

    pbt = poses.packed_block_types
    n_rots_for_sampler = sampler_gbt_for_rotamer.shape[0]

    orig_block_type_ind = (
        poses.block_type_ind[poses.block_type_ind != -1].view(-1).to(torch.int64)
    )

    # consider making this an argument and passing in
    # print("poses.block_type_ind.shape", poses.block_type_ind.shape)
    poses_res_to_real_poses_res = torch.full(
        (poses.block_type_ind.shape[0] * poses.block_type_ind.shape[1],),
        -1,
        dtype=torch.int64,
        device=poses.device,
    )
    # print("poses_res_to_real_poses_res")
    # print(poses_res_to_real_poses_res.shape)
    # print(poses_res_to_real_poses_res[-10:])
    poses_res_to_real_poses_res[poses.block_type_ind.view(-1) != -1] = torch.arange(
        orig_block_type_ind.shape[0], dtype=torch.int64, device=poses.device
    )

    # get the residue index for each rotamer
    max_n_blocks = poses.block_coord_offset.shape[1]
    res_ind_for_gbt = torch.tensor(
        [
            i * max_n_blocks + j
            for i, one_pose_blts in enumerate(task.blts)
            for j, blt in enumerate(one_pose_blts)
            for _ in blt.considered_block_types
        ],
        dtype=torch.int64,
        device=poses.device,
    )
    pose_ind_for_gbt = torch.floor_divide(res_ind_for_gbt, max_n_blocks).to(torch.int64)

    # print("res_ind_for_gbt")
    # print(res_ind_for_gbt)
    gbt_for_samplers_rots = gbt_for_rot[conf_inds_for_sampler]
    # torch.set_printoptions(threshold=10000)
    # print("gbt_for_samplers_rots")
    # print(gbt_for_samplers_rots)
    res_ind_for_samplers_rots = res_ind_for_gbt[gbt_for_samplers_rots]
    print("res_ind_for_samplers_rots")
    print(res_ind_for_samplers_rots)
    real_res_ind_for_samplers_rots = poses_res_to_real_poses_res[
        res_ind_for_samplers_rots
    ]
    # print("real_res_ind_for_samplers_rots")
    # print(real_res_ind_for_samplers_rots)
    block_type_ind_for_samplers_rots = block_type_ind_for_rot[conf_inds_for_sampler]

    # find the number of atoms for each rotamer / orig_res
    orig_res_n_atoms = pbt.n_atoms[block_type_ind_for_samplers_rots]

    # now lets note which atoms are real
    dummy_rotamer_atom_inds = (
        torch.arange(pbt.max_n_atoms, dtype=torch.int64, device=poses.device)
        .view(1, pbt.max_n_atoms)
        .expand(n_rots_for_sampler, -1)
    )
    print("dummy_rotamer_atom_inds")
    print(dummy_rotamer_atom_inds.shape)
    print("orig_res_n_atoms")
    print(
        orig_res_n_atoms.unsqueeze(1).expand(n_rots_for_sampler, pbt.max_n_atoms).shape
    )

    atom_is_real_for_rot = dummy_rotamer_atom_inds < orig_res_n_atoms.unsqueeze(
        1
    ).expand(n_rots_for_sampler, pbt.max_n_atoms)
    orig_atom_inds = (
        (
            poses.block_coord_offset64.view(-1)[res_ind_for_samplers_rots]
            + pose_ind_for_gbt[gbt_for_samplers_rots] * poses.max_n_pose_atoms
        )
        .unsqueeze(1)
        .expand(-1, pbt.max_n_atoms)
        + dummy_rotamer_atom_inds
    )[atom_is_real_for_rot]

    print("conf_inds_for_sampler")
    print(conf_inds_for_sampler)
    print("n_dof_atoms_offset_for_rot[conf_inds_for_sampler]")
    print(n_dof_atoms_offset_for_rot[conf_inds_for_sampler])

    rot_atom_inds = (
        n_dof_atoms_offset_for_rot[conf_inds_for_sampler]
        .unsqueeze(1)
        .expand(-1, pbt.max_n_atoms)
        + dummy_rotamer_atom_inds
    )[atom_is_real_for_rot]
    return rot_atom_inds, orig_atom_inds
