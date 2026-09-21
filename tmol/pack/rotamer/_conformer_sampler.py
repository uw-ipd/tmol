import torch
import attr

from typing import TYPE_CHECKING, Any, ClassVar

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


def chi_moving_roots(rt: RefinedResidueType, chi_name: str) -> tuple[str, ...]:
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


def sc_roots_for_chis(rt: RefinedResidueType, chi_names) -> tuple[str, ...]:
    """Sidechain roots for a sampler that turns the named chis."""
    roots = {}
    for chi_name in chi_names:
        for at in chi_moving_roots(rt, chi_name):
            roots[at] = None
    return tuple(roots)


@attr.s(auto_attribs=True)
class ConformerSampler:
    """Interface for creating and applying packing conformer samples."""

    #: Sample only after every other sampler has run, receiving the rotamer
    #: counts they actually produced. A sampler that fills gaps left by others
    #: must measure what was built rather than trust what was declared: a
    #: sampler may report that it covers a block type and still return no
    #: rotamers for a particular block, which would otherwise leave that
    #: position with nothing to pack.
    samples_after_other_samplers: ClassVar[bool] = False

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
        """Return counts, considered-block index per rotamer, and sampler data.

        A sampler that sets :py:attr:`samples_after_other_samplers` is called
        with an extra ``built_rotamer_counts`` keyword holding the per
        considered-block total produced by every other sampler.

        A producer of joint conformers declares ``correlated_gbts`` in its data:
        a tuple of considered-block-index tuples, one per joint group. Rotamer k
        must correspond across all members. Merging rejects additional states
        from other samplers on those blocks. Without a declaration, samples are
        independent even when their residues are covalently connected.

        Producers of unchanged input conformers may set
        ``copy_input_coordinates=True``. Their rows must use the original block
        type; Cartesian coordinates are copied exactly after DOF construction
        to avoid rounding from an unnecessary inverse/forward kinematics cycle.
        """
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
