import torch
import attr
import numpy

from typing import Tuple

from tmol.types.torch import Tensor
from tmol.types.functional import validate_args

from tmol.score.dunbrack.params import (
    SamplingDunbrackDatabaseView,
    DunbrackParamResolver,
)

# from tmol.pack.packer_task import PackerTask
# from tmol.system.packed import PackedResidueSystem
from tmol.system.restypes import RefinedResidueType
from tmol.system.pose import Poses, PackedBlockTypes
from tmol.system.score_support import indexed_atoms_for_dihedral


@attr.s(auto_attribs=True)
class ChiSampler:
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
    def first_sc_atom_for_rt(self, rt_name: str) -> str:
        raise NotImplementedError()

    def sample_chi_for_poses(
        self, systems: Poses, task: "PackerTask"
    ) -> Tuple[
        Tensor(torch.int32)[:, :, :],  # n_rots_for_rt
        Tensor(torch.int32)[:, :, :],  # n_rots_for_rt_offsets
        Tensor(torch.int32)[:, 3],  # rt_for_rotamer
        Tensor(torch.int32)[:],  # chi_defining_atom_for_rotamer
        Tensor(torch.float32)[:, :],  # chi_for_rotamers
    ]:
        raise NotImplementedError()
