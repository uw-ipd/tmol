import numpy
import torch
import attr

from typing import Tuple

from tmol.types.torch import Tensor
from tmol.types.array import NDArray
from tmol.types.functional import validate_args

from tmol.utility.tensor.common_operations import exclusive_cumsum1d, stretch
from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack
from tmol.kinematics.datatypes import KinForest
from tmol.pack.rotamer.conformer_sampler import ConformerSampler


@attr.s(auto_attribs=True)
class ChiSampler(ConformerSampler):
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

    @validate_args
    def first_sc_atoms_for_rt(self, rt_name: str) -> Tuple[str, ...]:
        raise NotImplementedError()

    def create_samples_for_poses(
        self, pose_stack: PoseStack, task: "PackerTask"
    ) -> Tuple[  # noqa F821
        Tensor[torch.int32][:],  # n_rots_for_gbt
        Tensor[torch.int32][:],  # bt_for_rotamer
        dict,  # anything else the sampler wants to save for later
    ]:
        (
            n_rots_for_gbt,
            gbt_for_rotamer,
            chi_defining_atom_for_rotamer,
            chi_for_rotamers,
        ) = self.sample_chi_for_poses(pose_stack, task)
        # print("Sampling:", self.sampler_name(), chi_for_rotamers.shape)
        return (
            n_rots_for_gbt,
            gbt_for_rotamer,
            dict(
                chi_defining_atom_for_rotamer=chi_defining_atom_for_rotamer,
                chi_for_rotamers=chi_for_rotamers,
            ),
        )

    def sample_chi_for_poses(
        self, systems: PoseStack, task: "PackerTask"  # noqa F821
    ) -> Tuple[
        Tensor[torch.int32][:, :, :],  # n_rots_for_rt
        Tensor[torch.int32][:],  # rt_for_rotamer
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
        sampler_n_rots_for_gbt: Tensor[torch.int32][:],
        sampler_gbt_for_rotamer: Tensor[torch.int32][:],
        sample_dict: dict,
        conf_dofs_kto: Tensor[torch.float32][:, 9],
    ):
        chi_defining_atom_for_rotamer = sample_dict["chi_defining_atom_for_rotamer"]
        chi_for_rotamers = sample_dict["chi_for_rotamers"]

        copy_dofs_from_orig_to_rotamers_for_sampler(
            pose_stack,
            task,
            self.sampler_name(),
            gbt_for_conformer,
            block_type_ind_for_conformer,
            conf_inds_for_sampler,
            sampler_n_rots_for_gbt,
            sampler_gbt_for_rotamer,
            n_dof_atoms_offset_for_conformer,
            orig_dofs_kto,
            conf_dofs_kto,
        )

        chi_atoms = sample_dict["chi_defining_atom_for_rotamer"]
        chi = sample_dict["chi_for_rotamers"]
        if chi.shape[0] == 0:
            return

        assign_chi_dofs_from_samples(
            pose_stack.packed_block_types,
            block_type_ind_for_conformer,
            conf_inds_for_sampler,
            sampler_n_rots_for_gbt,
            sampler_gbt_for_rotamer,
            n_dof_atoms_offset_for_conformer,
            chi_atoms,
            chi,
            conf_dofs_kto,
        )


@validate_args
def copy_dofs_from_orig_to_rotamers_for_sampler(
    poses: PoseStack,
    task,
    sampler_name: str,
    gbt_for_rot: Tensor[torch.int64][:],
    block_type_ind_for_rot: Tensor[torch.int64][:],
    conf_inds_for_sampler: Tensor[torch.int64][:],
    sampler_n_rots_for_gbt: Tensor[torch.int32][:],
    sampler_gbt_for_rotamer: Tensor[torch.int32][:],
    n_dof_atoms_offset_for_rot: Tensor[torch.int64][:],
    orig_dofs_kto: Tensor[torch.float32][:, 9],
    rot_dofs_kto: Tensor[torch.float32][:, 9],
):
    dst, src = create_dof_inds_to_copy_from_orig_to_rotamers_for_sampler(
        poses,
        task,
        sampler_name,
        gbt_for_rot,
        block_type_ind_for_rot,
        conf_inds_for_sampler,
        sampler_n_rots_for_gbt,
        sampler_gbt_for_rotamer,
        n_dof_atoms_offset_for_rot,
    )

    rot_dofs_kto[dst, :] = orig_dofs_kto[src, :]


def create_dof_inds_to_copy_from_orig_to_rotamers_for_sampler(
    poses: PoseStack,
    task: "PackerTask",  # noqa F821
    sampler_name: str,
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

    pbt = poses.packed_block_types
    n_rots_for_sampler = sampler_gbt_for_rotamer.shape[0]

    # This could 100% be pre-computed
    pbts_sampler_ind = pbt.mc_fingerprints.sampler_mapping[sampler_name]

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
    # print("res_ind_for_gbt")
    # print(res_ind_for_gbt)
    gbt_for_samplers_rots = gbt_for_rot[conf_inds_for_sampler]
    torch.set_printoptions(threshold=10000)
    print("gbt_for_samplers_rots")
    print(gbt_for_samplers_rots)
    res_ind_for_samplers_rots = res_ind_for_gbt[gbt_for_samplers_rots]
    # print("res_ind_for_samplers_rots")
    # print(res_ind_for_samplers_rots)
    real_res_ind_for_samplers_rots = poses_res_to_real_poses_res[
        res_ind_for_samplers_rots
    ]
    # print("real_res_ind_for_samplers_rots")
    # print(real_res_ind_for_samplers_rots)
    block_type_ind_for_samplers_rots = block_type_ind_for_rot[conf_inds_for_sampler]

    # look up which mainchain fingerprint each
    # original residue should use

    mcfp = pbt.mc_fingerprints

    sampler_ind_for_orig = mcfp.max_sampler[orig_block_type_ind]
    orig_res_mcfp = mcfp.max_fingerprint[orig_block_type_ind]
    orig_res_mcfp_for_samplers_rots = orig_res_mcfp[real_res_ind_for_samplers_rots]

    # now lets find the kinforest-ordered indices of the
    # mainchain atoms for the rotamers that represents
    # the destination for the dofs we're copying
    max_n_mcfp_atoms = mcfp.atom_mapping.shape[3]

    samplers_rots_mcfp_at_inds_rto = mcfp.atom_mapping[
        pbts_sampler_ind,
        orig_res_mcfp_for_samplers_rots,
        block_type_ind_for_samplers_rots,
        :,
    ].view(-1)

    is_samplers_rots_mcfp_at_inds_rto_real = samplers_rots_mcfp_at_inds_rto != -1
    real_samplers_rots_mcfp_at_inds_rto = samplers_rots_mcfp_at_inds_rto[
        is_samplers_rots_mcfp_at_inds_rto_real
    ]

    # print("block_type_ind_for_samplers_rots", block_type_ind_for_samplers_rots.shape)
    real_samplers_rots_block_type_ind_for_mcfp_ats = stretch(
        block_type_ind_for_samplers_rots, max_n_mcfp_atoms
    )[is_samplers_rots_mcfp_at_inds_rto_real]

    samplers_rots_mcfp_at_inds_kto = torch.full_like(samplers_rots_mcfp_at_inds_rto, -1)
    samplers_rots_mcfp_at_inds_kto[is_samplers_rots_mcfp_at_inds_rto_real] = (
        torch.tensor(
            pbt.rotamer_kinforest.kinforest_idx[
                real_samplers_rots_block_type_ind_for_mcfp_ats.cpu().numpy(),
                real_samplers_rots_mcfp_at_inds_rto.cpu().numpy(),
            ],
            dtype=torch.int64,
            device=pbt.device,
        )
    )
    # print(
    #     "real_samplers_rots_block_type_ind_for_mcfp_ats",
    #     real_samplers_rots_block_type_ind_for_mcfp_ats.shape,
    # )

    is_samplers_rots_mcfp_at_inds_kto_real = samplers_rots_mcfp_at_inds_kto != -1
    # print(
    #     "is_samplers_rots_mcfp_at_inds_kto_real",
    #     is_samplers_rots_mcfp_at_inds_kto_real.shape,
    # )
    # print(
    #     "n_rots_for_sampler * max_n_mcfp_atoms", n_rots_for_sampler * max_n_mcfp_atoms
    # )
    n_dof_atoms_offset_for_samplers_rot = n_dof_atoms_offset_for_rot[
        conf_inds_for_sampler
    ]
    samplers_rots_mcfp_at_inds_kto[
        is_samplers_rots_mcfp_at_inds_kto_real
    ] += n_dof_atoms_offset_for_samplers_rot[
        torch.div(  # to do: replace with expand
            torch.arange(
                n_rots_for_sampler * max_n_mcfp_atoms,
                dtype=torch.int64,
                device=poses.device,
            ),
            max_n_mcfp_atoms,
            rounding_mode="trunc",
        )[is_samplers_rots_mcfp_at_inds_kto_real]
    ]

    # now get the indices in the orig_dofs array for the atoms to copy from.
    # The steps:
    # 1. get the mainchain atom indices for each of the original residues
    #    in residue-type order (rto)
    # 2. sample 1. for each rotamer
    # 3. find the real subset of these atoms
    # 4. note the residue index for each of these real atoms
    # 5. remap these to kinforest order (kto)
    # 6. increment the indices with the original-residue dof-index offsets

    # orig_mcfp_at_inds_for_orig_rto:
    # 1. these are the mainchain fingerprint atoms from the original
    #    residues on the pose
    # 2. they are stored in residue-type order (rto)
    # 3. they are indexed by original residue index

    orig_mcfp_at_inds_rto = mcfp.atom_mapping[
        sampler_ind_for_orig, orig_res_mcfp, orig_block_type_ind, :
    ].view(-1)

    real_orig_block_type_ind_for_orig_mcfp_ats = stretch(
        orig_block_type_ind, max_n_mcfp_atoms
    )[orig_mcfp_at_inds_rto != -1]

    orig_dof_atom_offset = exclusive_cumsum1d(pbt.n_atoms[orig_block_type_ind]).to(
        torch.int64
    )

    orig_mcfp_at_inds_kto = torch.full_like(orig_mcfp_at_inds_rto, -1)
    orig_mcfp_at_inds_kto[orig_mcfp_at_inds_rto != -1] = (
        torch.tensor(
            pbt.rotamer_kinforest.kinforest_idx[
                real_orig_block_type_ind_for_orig_mcfp_ats.cpu().numpy(),
                orig_mcfp_at_inds_rto[orig_mcfp_at_inds_rto != -1].cpu().numpy(),
            ],
            dtype=torch.int64,
            device=pbt.device,
        )
        + orig_dof_atom_offset[
            torch.floor_divide(  # to do: replace w/ expand
                torch.arange(
                    orig_block_type_ind.shape[0] * max_n_mcfp_atoms,
                    dtype=torch.int64,
                    device=pbt.device,
                ),
                max_n_mcfp_atoms,
            )
        ][orig_mcfp_at_inds_rto != -1]
    )

    orig_mcfp_at_inds_kto = orig_mcfp_at_inds_kto.view(
        orig_block_type_ind.shape[0], max_n_mcfp_atoms
    )

    orig_mcfp_at_inds_for_samplers_rots_kto = orig_mcfp_at_inds_kto[
        real_res_ind_for_samplers_rots, :
    ].view(-1)

    # pare down the subset to those where the mc atom is present for
    # both the original block type and the alternate block type;
    # take the subset and also increment the indices of all the atoms
    # by one to take into account the virtual root atom at the origin

    both_present = torch.logical_and(
        samplers_rots_mcfp_at_inds_kto != -1,
        orig_mcfp_at_inds_for_samplers_rots_kto != -1,
    )

    # add one for the virtual root
    samplers_rots_mcfp_at_inds_kto = samplers_rots_mcfp_at_inds_kto[both_present] + 1
    orig_mcfp_at_inds_for_samplers_rots_kto = (
        orig_mcfp_at_inds_for_samplers_rots_kto[both_present] + 1
    )

    print("samplers_rots_mcfp_at_inds_kto")
    print(samplers_rots_mcfp_at_inds_kto.shape)
    print(samplers_rots_mcfp_at_inds_kto[:30])
    print("orig_mcfp_at_inds_for_samplers_rots_kto")
    print(orig_mcfp_at_inds_for_samplers_rots_kto.shape)
    print(orig_mcfp_at_inds_for_samplers_rots_kto[:30])

    return samplers_rots_mcfp_at_inds_kto, orig_mcfp_at_inds_for_samplers_rots_kto


@validate_args
def assign_chi_dofs_from_samples(
    pbt: PackedBlockTypes,
    block_type_ind_for_rot: Tensor[torch.int64][:],
    conf_inds_for_sampler: Tensor[torch.int64][:],
    sampler_n_rots_for_bt: Tensor[torch.int32][:],
    sampler_gbt_for_rotamer: Tensor[torch.int32][:],
    n_dof_atoms_offset_for_rot: Tensor[torch.int64][:],
    chi_atoms: Tensor[torch.int32][:, :],
    chi: Tensor[torch.float32][:, :],
    rot_dofs_kto: Tensor[torch.float32][:, 9],
):
    assert chi_atoms.shape == chi.shape

    n_rots_for_sampler = sampler_gbt_for_rotamer.shape[0]
    block_type_ind_for_samplers_rots = block_type_ind_for_rot[conf_inds_for_sampler]

    max_n_chi_atoms = chi_atoms.shape[1]
    real_atoms = chi_atoms.view(-1) != -1

    sampler_rot_ind_for_real_atom = torch.floor_divide(  # to do: replace w/ expand
        torch.arange(
            max_n_chi_atoms * n_rots_for_sampler, dtype=torch.int64, device=pbt.device
        ),
        max_n_chi_atoms,
    )[real_atoms]
    global_rot_ind_for_real_atom = conf_inds_for_sampler[sampler_rot_ind_for_real_atom]

    block_type_ind_for_rot_atom = (
        block_type_ind_for_rot[global_rot_ind_for_real_atom].cpu().numpy()
    )

    rot_chi_atoms_kto = torch.tensor(
        pbt.rotamer_kinforest.kinforest_idx[
            block_type_ind_for_rot_atom, chi_atoms.view(-1)[real_atoms].cpu().numpy()
        ],
        dtype=torch.int64,
        device=pbt.device,
    )
    # increment with the atom offsets for the source rotamer and by
    # one to include the virtual root

    rot_chi_atoms_kto += (
        n_dof_atoms_offset_for_rot[global_rot_ind_for_real_atom].to(torch.int64) + 1
    )

    # overwrite the "downstream torsion" for the atoms that control
    # each chi
    rot_dofs_kto[rot_chi_atoms_kto, 3] = chi.view(-1)[real_atoms]

    # print("rot_chi_atoms_kto", rot_chi_atoms_kto[:10])
    # print("chi", chi.view(-1)[real_atoms][:10])
