import torch
import attr

from typing import Tuple

from tmol.types.torch import Tensor
from tmol.types.functional import validate_args

from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack

# from tmol.system.score_support import indexed_atoms_for_dihedral


@attr.s(auto_attribs=True)
class ChiSampler:
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

    def sample_chi_for_poses(
        self, systems: PoseStack, task: "PackerTask"  # noqa F821
    ) -> Tuple[
        Tensor[torch.int32][:, :, :],  # n_rots_for_rt
        Tensor[torch.int32][:],  # rt_for_rotamer
        Tensor[torch.int32][:, :],  # chi_defining_atom_for_rotamer
        Tensor[torch.float32][:, :],  # chi_for_rotamers
    ]:
        raise NotImplementedError()
