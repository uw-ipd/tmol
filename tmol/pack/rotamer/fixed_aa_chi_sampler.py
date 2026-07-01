import torch
import attr

from typing import Tuple

from tmol.types.torch import Tensor
from tmol.types.functional import validate_args

from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack
from tmol.pack.packer_task import SetPackerTask
from tmol.pack.rotamer.chi_sampler import ChiSampler


@attr.s(auto_attribs=True, frozen=True)
class FixedAAChiSampler(ChiSampler):
    @classmethod
    def sampler_name(cls):
        return "FixedAAChiSampler"

    @validate_args
    def defines_rotamers_for_rt(self, rt: RefinedResidueType):
        # ugly hack for now:
        if not rt.properties.polymer.is_polymer:
            return False
        if rt.properties.polymer.polymer_type != "amino_acid":
            return False
        if rt.properties.polymer.backbone_type != "alpha":
            return False

        if rt.base_name[:3] == "GLY" or rt.base_name[:3] == "ALA":
            return True

        return False

    def defines_rotamers_for_bts(
        self, pbt: PackedBlockTypes, bt_inds: Tensor[torch.int64]
    ) -> Tensor[torch.bool]:
        return pbt.fixed_aa_chi_sampler_builds_bt[bt_inds]

    @validate_args
    def first_sc_atoms_for_rt(self, rt: RefinedResidueType) -> Tuple[str, ...]:
        if rt.base_name == "GLY":
            return ("HA3",)
        elif rt.base_name == "ALA":
            return ("CB",)

    def annotate_residue_type(self, block_type):
        if hasattr(block_type, "fixed_aa_chi_sampler_builds_bt"):
            return
        builds_bt = False
        if block_type.base_name == "GLY" or block_type.base_name[:3] == "ALA":
            builds_bt = True
        setattr(block_type, "fixed_aa_chi_sampler_builds_bt", builds_bt)

    def annotate_packed_block_types(self, packed_block_types):
        if hasattr(packed_block_types, "fixed_aa_chi_sampler_builds_bt"):
            return
        builds_bt = torch.tensor(
            [
                bt.fixed_aa_chi_sampler_builds_bt
                for bt in packed_block_types.active_block_types
            ],
            dtype=torch.bool,
            device=packed_block_types.device,
        )
        setattr(packed_block_types, "fixed_aa_chi_sampler_builds_bt", builds_bt)

    @validate_args
    def sample_chi_for_poses(
        self, poses: PoseStack, task: "SetPackerTask"
    ) -> Tuple[
        Tensor[torch.int32][:],  # n_rots_for_rt
        Tensor[torch.int32][:],  # rt_for_rotamer
        Tensor[torch.int32][:, :],  # chi_defining_atom_for_rotamer
        Tensor[torch.float32][:, :],  # chi_for_rotamers
    ]:
        pbt = poses.packed_block_types
        self_ind_in_task = task.conformer_sampler_index[id(self)]
        faas_allowed = task.per_block_conformer_sampler_allowed[:, :, self_ind_in_task]
        faas_allowed_for_cons_bt = faas_allowed[task.cons_bt_pose, task.cons_bt_block]
        faa_builds_bt = pbt.fixed_aa_chi_sampler_builds_bt
        faa_builds_bt_for_cons_bt = faa_builds_bt[task.cons_bt_block_type]
        cons_bt_is_allowed = task.per_block_is_block_type_allowed[
            task.cons_bt_pose, task.cons_bt_block, task.cons_bt_which_block_type
        ]

        is_bt_faas_allowed_and_built_by = torch.logical_and(
            faas_allowed_for_cons_bt, faa_builds_bt_for_cons_bt
        )
        n_rots_for_gbt = torch.logical_and(
            is_bt_faas_allowed_and_built_by, cons_bt_is_allowed
        ).to(torch.int32)

        n_fixed_rots = torch.sum(n_rots_for_gbt).item()
        gbt_for_rotamer = torch.nonzero(n_rots_for_gbt > 0, as_tuple=True)[0].to(
            torch.int32
        )
        chi_for_rotamers = torch.zeros(
            (n_fixed_rots, 1), dtype=torch.float32, device=poses.device
        )
        chi_defining_atom_for_rotamer = torch.full_like(
            chi_for_rotamers, -1, dtype=torch.int32
        )

        return (
            n_rots_for_gbt,
            gbt_for_rotamer,
            chi_defining_atom_for_rotamer,
            chi_for_rotamers,
        )
