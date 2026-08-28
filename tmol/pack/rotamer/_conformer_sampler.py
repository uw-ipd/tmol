import torch
import attr

from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from tmol.pack import PackerTask


ConformerSample = tuple[
    Tensor[torch.int32][:],
    Tensor[torch.int32][:],
    dict[str, Any],
]


@attr.s(auto_attribs=True)
class ConformerSampler:
    """Interface for creating and applying packing conformer samples."""

    @classmethod
    def sampler_name(cls) -> str:
        """Return the stable name used for sampler-specific annotations."""
        raise NotImplementedError()

    @validate_args
    def annotate_residue_type(self, rt: RefinedResidueType) -> None:
        """Attach optional sampler metadata to one residue type."""
        pass

    @validate_args
    def annotate_packed_block_types(self, packed_block_types: PackedBlockTypes) -> None:
        """Attach optional sampler metadata to packed block types."""
        pass

    @validate_args
    def defines_rotamers_for_rt(self, rt: RefinedResidueType) -> bool:
        """Return whether this sampler supports a residue type."""
        raise NotImplementedError()

    def defines_rotamers_for_bts(
        self, pbt: PackedBlockTypes, bt_inds: Tensor[torch.int64]
    ) -> Tensor[torch.bool]:
        raise NotImplementedError()

    @validate_args
    def first_sc_atoms_for_rt(self, rt: RefinedResidueType) -> tuple[str, ...]:
        """Return side-chain roots used to transfer main-chain geometry."""
        raise NotImplementedError()

    def create_samples_for_poses(
        self,
        pose_stack: PoseStack,
        task: "PackerTask",
    ) -> ConformerSample:
        """Create per-block sample counts, block indices, and metadata."""
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
        sample_dict: dict[str, Any],
        conf_dofs_kto: Tensor[torch.float32][:, 9],
    ) -> None:
        """Write this sampler's conformer degrees of freedom in place."""
        raise NotImplementedError
