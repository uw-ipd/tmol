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

from tmol.system.restypes import RefinedResidueType
from tmol.system.pose import Poses
from tmol.system.score_support import indexed_atoms_for_dihedral


@attr.s(auto_attribs=True)
class FixedAAChiSampler:
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
            return ("2HA",)
        elif rt.base_name == "ALA":
            return ("CB",)

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
