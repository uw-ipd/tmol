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


def chi_moving_roots(rt: RefinedResidueType, chi_name: str) -> Tuple[str, ...]:
    """The atoms a chi turns: everything bonded to its third atom but its second.

    Sidechain roots mark where a sampler stops copying degrees of freedom from
    the input structure and starts rebuilding them from ideal internal
    coordinates. The third atom of a torsion carries the degree of freedom but
    does not itself move, so it must not be a root.
    """
    uaids = rt.torsion_to_uaids.get(chi_name)
    if uaids is None:
        return ()
    held, turned = uaids[1][0], uaids[2][0]
    if held < 0 or turned < 0:
        return ()
    moved = {int(j) for i, j in rt.bond_indices if int(i) == turned and int(j) != held}
    return tuple(rt.atoms[at].name for at in sorted(moved))


def sc_roots_for_chis(rt: RefinedResidueType, chi_names) -> Tuple[str, ...]:
    """Sidechain roots for a sampler that turns the named chis."""
    roots = {}
    for chi_name in chi_names:
        for at in chi_moving_roots(rt, chi_name):
            roots[at] = None
    return tuple(roots)


@attr.s(auto_attribs=True)
class ConformerSampler:
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

    def defines_rotamers_for_bts(
        self, pbt: PackedBlockTypes, bt_inds: Tensor[torch.int64]
    ) -> Tensor[torch.bool]:
        raise NotImplementedError()

    @validate_args
    def first_sc_atoms_for_rt(self, rt: RefinedResidueType) -> Tuple[str, ...]:
        raise NotImplementedError()

    def create_samples_for_poses(
        self,
        pose_stack: PoseStack,
        task: "PackerTask",  # noqa: 821
    ) -> Tuple[  # noqa F821
        Tensor[torch.int32][:],  # n_rots_for_bt
        Tensor[torch.int32][:],  # bt_for_rotamer
        dict,  # anything else the sampler wants to save for later
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
        raise NotImplementedError
