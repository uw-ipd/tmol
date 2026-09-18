import torch
import attr

from typing import ClassVar, Tuple

from tmol.types import (
    Tensor,
    validate_args,
)
from tmol.chemical import RefinedResidueType
from tmol.pose import (
    PackedBlockTypes,
    PoseStack,
)
from tmol.pack.rotamer import (
    ConformerSampler,
    IncludeCurrentSampler,
)


@attr.s(auto_attribs=True, frozen=True)
class FallbackSampler(ConformerSampler):
    """Include the input conformation as a rotamer only for positions that have
    no rotamers from any other sampler.

    This is the default sampler in PackerPalette. Unlike IncludeCurrentSampler,
    it does not unconditionally add a rotamer for every position; instead it
    activates only where the other samplers actually produced no rotamers,
    ensuring that positions covered by, e.g., DunbrackChiSampler do not
    accumulate an extra current-conformation rotamer.

    The trigger is measured rather than declared. A sampler may report through
    defines_rotamers_for_bts that it handles a block type and still build no
    rotamers for a given block; trusting the declaration would leave that
    position with nothing to pack. Sampling last and counting what was built
    also means a new sampler needs no special registration here.

    The disable_packing case (all block types disallowed) is also handled: a
    rotamer from the input conformation is always produced so the packer has
    something to represent for fixed residues.
    """

    samples_after_other_samplers: ClassVar[bool] = True

    @classmethod
    def sampler_name(cls):
        return "FallbackSampler"

    @validate_args
    def annotate_residue_type(self, rt: RefinedResidueType):
        pass

    @validate_args
    def annotate_packed_block_types(self, packed_block_types: PackedBlockTypes):
        pass

    @validate_args
    def defines_rotamers_for_rt(self, rt: RefinedResidueType):
        return True

    @validate_args
    def first_sc_atoms_for_rt(self, rt: RefinedResidueType) -> Tuple[str, ...]:
        return (rt.default_jump_connection_atom,)

    def create_samples_for_poses(
        self,
        pose_stack: PoseStack,
        task: "SetPackerTask",  # noqa: F821
        *,
        built_rotamer_counts: Tensor[torch.int32][:] = None,
    ) -> Tuple[  # noqa F821
        Tensor[torch.int32][:],  # n_rots_for_gbt
        Tensor[torch.int32][:],  # gbt_for_rotamer
        dict,
    ]:
        """Create rotamers for blocks the other samplers left uncovered.

        A rotamer of the input conformation is built where the block either
        (1) has no allowed block types, so the residue is fixed, or (2) received
        no rotamers from any other sampler.

        Args:
          built_rotamer_counts: Rotamers actually produced per considered block
            by every other sampler. Supplied by the rotamer builder, which runs
            this sampler last. ``None`` means no other sampler ran.
        """
        n_allowed_per_block = task.per_block_is_block_type_allowed.to(torch.int32).sum(
            dim=2
        )
        gbt_block_allows_none = (n_allowed_per_block == 0)[
            task.cons_bt_pose, task.cons_bt_block
        ]

        assert (
            id(self) in task.conformer_sampler_index
        ), "This sampler is not in the PackerTask's conformer samplers"
        self_ind_in_packer_task = task.conformer_sampler_index[id(self)]

        if built_rotamer_counts is None:
            # No other sampler ran, so nothing was built anywhere. This is
            # unusual but reachable when the fallback is the only sampler.
            other_sampler_builds_for_gbt = torch.zeros(
                (task.cons_bt_pose.shape[0]),
                dtype=torch.bool,
                device=pose_stack.device,
            )
        else:
            # Aggregate per block, not per considered block type. A design
            # position whose other candidates were built is packable, even if
            # one candidate -- often the original identity being designed away
            # from -- received nothing. Triggering per candidate would offer
            # the input conformation back and let it win the position.
            built = (built_rotamer_counts > 0).to(torch.int32)
            covered = torch.zeros(
                task.per_block_n_considered_block_types.shape[:2],
                dtype=torch.int32,
                device=pose_stack.device,
            )
            covered.index_put_(
                (task.cons_bt_pose, task.cons_bt_block), built, accumulate=True
            )
            other_sampler_builds_for_gbt = (
                covered[task.cons_bt_pose, task.cons_bt_block] > 0
            )

        is_gbt_orig_block_type = task.per_block_considered_block_types_is_orig
        all_samplers_disabled = ~task.per_block_conformer_sampler_allowed[
            task.cons_bt_pose, task.cons_bt_block
        ].any(dim=-1)

        n_rots_for_gbt = torch.logical_and(
            is_gbt_orig_block_type[
                task.cons_bt_pose, task.cons_bt_block, task.cons_bt_which_block_type
            ],
            torch.logical_or(
                gbt_block_allows_none | all_samplers_disabled,
                ~other_sampler_builds_for_gbt
                & task.per_block_conformer_sampler_allowed[
                    task.cons_bt_pose, task.cons_bt_block, self_ind_in_packer_task
                ],
            ),
        ).to(torch.int32)

        gbt_for_rotamer = n_rots_for_gbt.nonzero(as_tuple=True)[0].to(torch.int32)
        return (n_rots_for_gbt, gbt_for_rotamer, {"copy_input_coordinates": True})

    # Selection differs, but both samplers copy the same input conformation.
    fill_dofs_for_samples = IncludeCurrentSampler.fill_dofs_for_samples
