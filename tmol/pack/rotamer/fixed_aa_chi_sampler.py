import torch
import attr
import numpy

from typing import Tuple

from tmol.types.torch import Tensor
from tmol.types.functional import validate_args

from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.pose_stack import PoseStack
from tmol.pack.packer_task import PackerTask
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

        if rt.base_name == "GLY" or rt.base_name[:3] == "ALA":
            return True

        return False

    @validate_args
    def first_sc_atoms_for_rt(self, rt: RefinedResidueType) -> Tuple[str, ...]:
        if rt.base_name == "GLY":
            return ("HA3",)
        elif rt.base_name == "ALA":
            return ("CB",)

    @validate_args
    def sample_chi_for_poses(
        self, poses: PoseStack, task: "PackerTask"
    ) -> Tuple[
        Tensor[torch.int32][:],  # n_rots_for_rt
        Tensor[torch.int32][:],  # rt_for_rotamer
        Tensor[torch.int32][:, :],  # chi_defining_atom_for_rotamer
        Tensor[torch.float32][:, :],  # chi_for_rotamers
    ]:
        all_restypes = numpy.array(
            [
                rt
                for one_pose_rlts in task.rlts
                for rlt in one_pose_rlts
                for rt in rlt.allowed_restypes
                if self in rlt.chi_samplers
            ],
            dtype=object,
        )

        rt_base_names = numpy.array([rt.base_name for rt in all_restypes], dtype=object)
        n_rots_for_rt = torch.zeros(
            len(all_restypes), dtype=torch.int32, device=poses.device
        )
        is_ala_rt = torch.tensor(
            (rt_base_names == "ALA").astype(numpy.uint8),
            dtype=torch.uint8,
            device=poses.device,
        )
        is_gly_rt = torch.tensor(
            (rt_base_names == "GLY").astype(numpy.uint8),
            dtype=torch.uint8,
            device=poses.device,
        )
        n_rots_for_rt[is_ala_rt] += 1
        n_rots_for_rt[is_gly_rt] += 1
        # either_ala_or_gly = torch.logical_or(is_ala_rt, is_gly_rt)
        either_ala_or_gly = (is_ala_rt + is_gly_rt) > 0

        n_fixed_rots = torch.sum(n_rots_for_rt).item()
        # rt_for_rotamer = torch.zeros(
        #     n_fixed_rots,
        #     dtype=torch.int32,
        #     device=poses.device
        # )
        rt_for_rotamer = torch.arange(
            len(rt_base_names), dtype=torch.int32, device=poses.device
        )[either_ala_or_gly]
        chi_for_rotamers = torch.zeros(
            (n_fixed_rots, 1), dtype=torch.float32, device=poses.device
        )
        chi_defining_atom_for_rotamer = torch.full_like(
            chi_for_rotamers, -1, dtype=torch.int32
        )

        return (
            n_rots_for_rt,
            rt_for_rotamer,
            chi_defining_atom_for_rotamer,
            chi_for_rotamers,
        )
