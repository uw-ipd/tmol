import torch
import attr

from typing import Tuple, Union

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
        self,
        pose_stack: PoseStack,
        task: "PackerTask",  # noqa: 821
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
        Tensor[torch.float32][:, :],  # chi_for_rotamers, in radians
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
        copy_dofs_from_orig_to_rotamers_for_sampler(
            pose_stack,
            task,
            id(self),
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
    sampler_name: Union[str, int],
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
    sampler_name: Union[str, int],
    gbt_for_rot: Tensor[torch.int64][:],  # max-n-rots
    block_type_ind_for_rot: Tensor[torch.int64][:],
    conf_inds_for_sampler: Tensor[torch.int64][:],
    sampler_n_rots_for_gbt: Tensor[torch.int32][:],
    sampler_gbt_for_rotamer: Tensor[torch.int32][:],
    n_dof_atoms_offset_for_rot: Tensor[torch.int64][:],
) -> Tuple[Tensor[torch.int64][:], Tensor[torch.int64][:]]:
    # A sampler with no states need not have an ownership fingerprint at all.
    if conf_inds_for_sampler.numel() == 0:
        return conf_inds_for_sampler, conf_inds_for_sampler
    pbt = poses.packed_block_types
    fingerprints = pbt.mc_fingerprints
    if fingerprints.atom_mapping.shape[1] == 0:
        empty = conf_inds_for_sampler[:0]
        return empty, empty
    from tmol.pack.rotamer._build_rotamers import _kinforest_device_indices

    kfo = _kinforest_device_indices(pbt, pbt.device)
    flat_types = poses.block_type_ind.reshape(-1).long()
    counts = pbt.n_atoms[flat_types.clamp_min(0)].long()
    counts = torch.where(flat_types >= 0, counts, 0)
    source_offsets = torch.cumsum(counts, 0) - counts
    considered = gbt_for_rot[conf_inds_for_sampler]
    source_residues = task.global_block_ind_for_considered_block_types[considered]
    source_types = flat_types[source_residues]
    target_types = block_type_ind_for_rot[conf_inds_for_sampler]
    source_fingerprints = fingerprints.source_fingerprint[source_types]
    source_atoms = fingerprints.source_atom_mapping[source_types]
    source_local = kfo[source_types[:, None], source_atoms.clamp_min(0)]
    source_local.masked_fill_(source_atoms < 0, -1)
    del source_atoms
    target_atoms = fingerprints.atom_mapping[
        fingerprints.sampler_mapping[sampler_name],
        source_fingerprints.clamp_min(0),
        target_types,
    ]
    target_local = kfo[target_types[:, None], target_atoms.clamp_min(0)]
    target_local.masked_fill_(target_atoms < 0, -1)
    del target_atoms
    present = (
        (source_fingerprints[:, None] >= 0) & (source_local >= 0) & (target_local >= 0)
    )
    # Both DOF arrays reserve their first row for the virtual root.
    source_local.add_(source_offsets[source_residues, None] + 1)
    target_local.add_(n_dof_atoms_offset_for_rot[conf_inds_for_sampler, None] + 1)
    return target_local[present], source_local[present]


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

    # A ring-closing chi cannot be measured from the built coordinates, so it
    # carries a precomputed offset; every other chi is corrected by measurement
    # and its entry here is zero.
    from tmol.pack.rotamer import _build_ring_chi_phi_c_corrections

    corrections = torch.tensor(
        _build_ring_chi_phi_c_corrections(pbt)[
            block_type_ind_for_rot_atom, chi_atoms.view(-1)[real_atoms].cpu().numpy()
        ],
        dtype=rot_dofs_kto.dtype,
        device=pbt.device,
    )

    # overwrite the "downstream torsion" for the atoms that control each chi
    rot_dofs_kto[rot_chi_atoms_kto, 3] = chi.view(-1)[real_atoms] - corrections
