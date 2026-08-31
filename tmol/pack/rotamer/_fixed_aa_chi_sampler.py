from __future__ import annotations

import torch
import attr

from typing import (
    Tuple,
    TYPE_CHECKING,
)

from tmol.types import (
    Tensor,
    validate_args,
)
from tmol.chemical import RefinedResidueType, l_base_name
from tmol.pose import (
    PackedBlockTypes,
    PoseStack,
)

if TYPE_CHECKING:
    from tmol.pack._packer_task import SetPackerTask  # noqa: F401
from tmol.pack.rotamer import ChiSampler


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
        if rt.properties.polymer.backbone_type != "alpha_aa":
            return False

        if l_base_name(rt) in ("GLY", "ALA"):
            return True

        return False

    def defines_rotamers_for_bts(
        self, pbt: PackedBlockTypes, bt_inds: Tensor[torch.int64]
    ) -> Tensor[torch.bool]:
        return pbt.fixed_aa_chi_sampler_builds_bt[bt_inds]

    @validate_args
    def first_sc_atoms_for_rt(self, rt: RefinedResidueType) -> Tuple[str, ...]:
        base = l_base_name(rt)
        if base == "GLY":
            return ("HA3",)
        elif base == "ALA":
            return ("CB",)

    def annotate_residue_type(self, block_type):
        if hasattr(block_type, "fixed_aa_chi_sampler_builds_bt"):
            return
        builds_bt = False
        if l_base_name(block_type) in ("GLY", "ALA"):
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
        self, poses: PoseStack, task
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
