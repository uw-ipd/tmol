# import numpy
import torch
import attr

from tmol.types.torch import Tensor
from tmol.chemical.restypes import RefinedResidueType  # , ResidueTypeSet
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack
from tmol.pack.rotamer.conformer_sampler import ConformerSampler

# from tmol.pack.rotamer.chi_sampler import ChiSampler

# Architecture is borrowed from Rosetta3:
# PackerTask: a class holding data describing how the
#   packer should behave. Each position in the
#   PackerTask corresponds to a residue in the input
#   PackedResidueSystem. For each residue, it holds
#   a list of the allowed residue types that can be
#   built at those positions. The PackerTask can be
#   modified only by removing residue types from those
#   list, not by adding new ones.
#
# PackerPallete: a class that decides how to construct
#   a PackerTask, deciding which residue types to allow
#   based on the residue type of the input structure.
#   Different PackerPalletes will construct different
#   starting points which can be refined towards the
#   set of design choices that make sense for your
#   application


def set_compare(x, y):
    """Treat the collections x and y as if they are sets. Return true if they
    contain the same elements and false otherwise
    """
    if len(x) != len(y):
        return False
    for val in x:
        if val not in y:
            return False
    return True


@attr.define(slots=True, frozen=True)
class PackerPalleteAnnotation:
    max_n_allowed: int
    n_allowed_block_types_for_block_type: Tensor[torch.int64][:]
    allowed_block_types_for_block_type: Tensor[torch.bool][:, :]
    allowed_block_type_is_orig: Tensor[torch.bool][:, :]
    restrict_to_repacking_masks: Tensor[torch.bool][:, :]


class PackerPalette:
    def __init__(self):
        pass

    def block_types_from_original_old(self, orig: RefinedResidueType):
        # ok, this is where we figure out what the allowed restypes
        # are for a residue; this might be complex logic.
        # Derived versions of this class can override this method to
        # implement different logic, e.g., to allow HIS_POS or D-AAs.

        # TO BE DEPRECATED!

        keepers = []
        for bt in self.rts.residue_types:
            if (
                bt.properties.polymer.is_polymer == orig.properties.polymer.is_polymer
                and bt.properties.polymer.polymer_type
                == orig.properties.polymer.polymer_type
                and bt.properties.polymer.backbone_type
                == orig.properties.polymer.backbone_type
                and bt.connections
                == orig.connections  # fd  use this instead of terminal variant check
                and set_compare(
                    bt.properties.chemical_modifications,
                    orig.properties.chemical_modifications,
                )
                and set_compare(
                    bt.properties.connectivity, orig.properties.connectivity
                )
                and bt.properties.protonation.protonation_state
                == orig.properties.protonation.protonation_state
            ):
                if (
                    bt.properties.polymer.sidechain_chirality
                    == orig.properties.polymer.sidechain_chirality
                ):
                    keepers.append(bt)
                elif orig.properties.polymer.polymer_type == "amino_acid" and (
                    (
                        orig.properties.polymer.sidechain_chirality == "l"
                        and bt.properties.polymer.sidechain_chirality == "achiral"
                    )
                    or (
                        orig.properties.polymer.sidechain_chirality == "achiral"
                        and bt.properties.polymer.sidechain_chirality == "l"
                    )
                ):
                    # allow glycine <--> l-caa mutations
                    keepers.append(bt)
                elif (
                    orig.properties.polymer.polymer_type == "amino_acid"
                    and orig.properties.polymer.sidechain_chirality == "d"
                    and bt.properties.polymer.sidechain_chirality == "achiral"
                ):
                    # allow d-caa --> glycine mutations;
                    # dangerous because this packer pallete will allow
                    # your d-caa to become glycine, and then later
                    # to an l-caa, but not the other way around
                    keepers.append(bt)

        return keepers

    def block_types_from_original(
        self, pbt: PackedBlockTypes, orig: Tensor[torch.int64][:, :]
    ) -> tuple[
        Tensor[torch.int64][:, :],
        Tensor[torch.int64][:, :, :],
        Tensor[torch.bool][:, :, :],
    ]:
        # initialize the list of allowed block types for a PoseStack based on the
        # original block types; the logic of which block types are allowed given
        # the original block type is pre-computed and cached in the PackedBlockTypes
        # object, so all we will do is take the subset of real block types (ind >= 0)
        # and read from the cached logic
        _annotate_packed_block_types_for_default_packer_palette(pbt)
        assert orig.device == pbt.device

        dppann = pbt.default_packer_palette_annotations
        allowed_block_types_for_block_type = torch.full(
            (orig.shape[0], orig.shape[1], dppann.max_n_allowed),
            -1,
            dtype=torch.int64,
            device=pbt.device,
        )
        allowed_block_type_is_orig = torch.zeros_like(
            allowed_block_types_for_block_type, dtype=torch.bool
        )
        n_allowed_block_types_for_block_type = torch.zeros_like(orig, dtype=torch.int64)
        is_real_bt = orig >= 0

        n_allowed_bt_from_orig = dppann.n_allowed_block_types_for_block_type
        allowed_bt_from_orig = dppann.allowed_block_types_for_block_type
        allowed_bt_is_orig_from_orig = dppann.allowed_block_type_is_orig
        n_allowed_block_types_for_block_type[is_real_bt] = n_allowed_bt_from_orig[
            orig[is_real_bt]
        ]
        allowed_block_types_for_block_type[is_real_bt] = allowed_bt_from_orig[
            orig[is_real_bt]
        ]
        allowed_block_type_is_orig[is_real_bt] = allowed_bt_is_orig_from_orig[
            orig[is_real_bt]
        ]

        return (
            n_allowed_block_types_for_block_type,
            allowed_block_types_for_block_type,
            allowed_block_type_is_orig,
        )

    def create_restrict_to_repacking_mask(
        self, pbt: PackedBlockTypes, orig: Tensor[torch.int64][:, :]
    ):
        # initialize the restrict-to-repacking mask for a PoseStack based on the
        # original block types; the logic of which block types are allowed for each
        # starting block type can be pre-computed and cached in the PackedBlockTypes
        # object and then looked up here. For the default PackerPalette, the logic is
        # simply to allow only those residue types with the same name3 as the original
        # residue type
        _annotate_packed_block_types_for_default_packer_palette(pbt)
        dppann = pbt.default_packer_palette_annotations
        rtr_mask_for_orig = torch.zeros(
            (orig.shape[0], orig.shape[1], dppann.max_n_allowed),
            dtype=torch.bool,
            device=pbt.device,
        )
        is_real_bt = orig >= 0
        rtr_mask = dppann.restrict_to_repacking_masks
        rtr_mask_for_orig[is_real_bt] = rtr_mask[orig[is_real_bt]]
        return rtr_mask_for_orig

    def default_conformer_samplers(self):
        """All positions must build one rotamer, even if they are not being optimized.

        Each block must have coordinates represented in the tensor with the other
        rotamers, and the easiest way to do that is to create a rotamer with the
        DOFs of the input conformation. The FallbackSampler copies these DOFs
        from the inverse-folded coordinates of the starting Pose's blocks, but
        only for positions where no other sampler provides rotamers (e.g. residue
        types not covered by DunbrackChiSampler). Positions with at least one
        other sampler are left to that sampler exclusively.
        Future versions of PackerPalette have the option to override this method.
        """
        from tmol.pack.rotamer.fallback_sampler import FallbackSampler

        return [FallbackSampler()]


def _annotate_packed_block_types_for_default_packer_palette(pbt: PackedBlockTypes):
    # Annotate the PackedBlockTypes object with the block-type to block-type comparisons
    if hasattr(pbt, "default_packer_palette_annotations"):
        return
    allowed_block_types_for_block_type = [list() for _ in range(pbt.n_types)]
    allowed_block_is_orig = [list() for _ in range(pbt.n_types)]
    restrict_to_repacking_masks = [list() for _ in range(pbt.n_types)]
    for i, orig_bt in enumerate(pbt.active_block_types):
        for j, alt_bt in enumerate(pbt.active_block_types):
            j_allowed_for_restrict_to_repack = alt_bt.name3 == orig_bt.name3
            if (
                alt_bt.properties.polymer.is_polymer
                == orig_bt.properties.polymer.is_polymer
                and alt_bt.properties.polymer.polymer_type
                == orig_bt.properties.polymer.polymer_type
                and alt_bt.properties.polymer.backbone_type
                == orig_bt.properties.polymer.backbone_type
                and alt_bt.connections
                == orig_bt.connections  # fd  use this instead of terminal variant check
                and set_compare(
                    alt_bt.properties.chemical_modifications,
                    orig_bt.properties.chemical_modifications,
                )
                and set_compare(
                    alt_bt.properties.connectivity, orig_bt.properties.connectivity
                )
                and alt_bt.properties.protonation.protonation_state
                == orig_bt.properties.protonation.protonation_state
            ):
                if (
                    alt_bt.properties.polymer.sidechain_chirality
                    == orig_bt.properties.polymer.sidechain_chirality
                ):
                    allowed_block_types_for_block_type[i].append(j)
                    allowed_block_is_orig[i].append(j == i)
                    restrict_to_repacking_masks[i].append(
                        j_allowed_for_restrict_to_repack
                    )
                elif orig_bt.properties.polymer.polymer_type == "amino_acid" and (
                    (
                        orig_bt.properties.polymer.sidechain_chirality == "l"
                        and alt_bt.properties.polymer.sidechain_chirality == "achiral"
                    )
                    or (
                        orig_bt.properties.polymer.sidechain_chirality == "achiral"
                        and alt_bt.properties.polymer.sidechain_chirality == "l"
                    )
                ):
                    # allow glycine <--> l-caa mutations
                    allowed_block_types_for_block_type[i].append(j)
                    allowed_block_is_orig[i].append(j == i)
                    restrict_to_repacking_masks[i].append(
                        j_allowed_for_restrict_to_repack
                    )
                elif (
                    orig_bt.properties.polymer.polymer_type == "amino_acid"
                    and orig_bt.properties.polymer.sidechain_chirality == "d"
                    and alt_bt.properties.polymer.sidechain_chirality == "achiral"
                ):
                    # allow d-caa --> glycine mutations;
                    # dangerous because this packer pallete will allow
                    # your d-caa to become glycine, and then later
                    # to an l-caa, but not the other way around
                    allowed_block_types_for_block_type[i].append(j)
                    allowed_block_is_orig[i].append(j == i)
                    restrict_to_repacking_masks[i].append(
                        j_allowed_for_restrict_to_repack
                    )
            elif i == j:
                allowed_block_types_for_block_type[i].append(j)
                allowed_block_is_orig[i].append(j == i)
                restrict_to_repacking_masks[i].append(j_allowed_for_restrict_to_repack)

    max_n_allowed = max(len(lst) for lst in allowed_block_types_for_block_type)
    n_allowed_block_types_for_block_type = torch.tensor(
        [len(lst) for lst in allowed_block_types_for_block_type],
        dtype=torch.int64,
        device=pbt.device,
    )
    allowed_block_types_for_block_type = torch.tensor(
        [
            lst + [-1] * (max_n_allowed - len(lst))
            for lst in allowed_block_types_for_block_type
        ],
        dtype=torch.int64,
        device=pbt.device,
    )
    allowed_block_type_is_orig = torch.tensor(
        [lst + [False] * (max_n_allowed - len(lst)) for lst in allowed_block_is_orig],
        dtype=torch.bool,
        device=pbt.device,
    )
    restrict_to_repacking_masks = torch.tensor(
        [
            lst + [False] * (max_n_allowed - len(lst))
            for lst in restrict_to_repacking_masks
        ],
        dtype=torch.bool,
        device=pbt.device,
    )

    annotation = PackerPalleteAnnotation(
        max_n_allowed=max_n_allowed,
        n_allowed_block_types_for_block_type=n_allowed_block_types_for_block_type,
        allowed_block_types_for_block_type=allowed_block_types_for_block_type,
        allowed_block_type_is_orig=allowed_block_type_is_orig,
        restrict_to_repacking_masks=restrict_to_repacking_masks,
    )
    setattr(pbt, "default_packer_palette_annotations", annotation)


class PackerTask:

    def __init__(self, systems: PoseStack, palette: PackerPalette):
        self.pbt = systems.packed_block_types
        self.device = systems.device
        self.is_real_block = systems.block_type_ind64 != -1
        self.real_block_pose, self.real_block_block = torch.nonzero(
            self.is_real_block, as_tuple=True
        )
        self.per_block_orig_block_type = torch.zeros(
            (systems.n_poses, systems.max_n_blocks),
            dtype=torch.int32,
            device=systems.device,
        )
        (
            self.per_block_n_considered_block_types,
            self.per_block_considered_block_types,
            self.per_block_considered_block_types_is_orig,
        ) = palette.block_types_from_original(
            systems.packed_block_types, systems.block_type_ind64
        )
        self.restrict_to_repacking_masks = palette.create_restrict_to_repacking_mask(
            systems.packed_block_types, systems.block_type_ind64
        )
        self.per_block_is_block_type_allowed = torch.ones_like(
            self.per_block_considered_block_types, dtype=torch.bool
        )
        # as we add conformer samplers to the task, we assign them an intex
        self.conformer_samplers = palette.default_conformer_samplers()
        # this will map from conformer sampler to its index in this task
        self.conformer_sampler_index = {
            id(sampler): i for i, sampler in enumerate(self.conformer_samplers)
        }
        self.per_block_conformer_sampler_allowed = torch.ones(
            (systems.n_poses, systems.max_n_blocks, len(self.conformer_samplers)),
            dtype=torch.bool,
            device=systems.device,
        )
        self.per_block_chi_expansion = torch.zeros(
            (
                systems.n_poses,
                systems.max_n_blocks,
                self.per_block_considered_block_types.shape[2],
                4,
            ),
            dtype=torch.int32,
            device=systems.device,
        )

    def restrict_to_repacking(self):
        # old way of doing things; soon to be removed
        # for one_pose_blts in self.blts:
        #     for blt in one_pose_blts:
        #         blt.restrict_to_repacking()

        # new way of doing things:
        # Use the pre-calculated masks to disable packing for
        # all block types that are not allowed except those which
        # meet the definition of "repacking" for the original
        # packer palette (i.e. same name3)
        self.per_block_is_block_type_allowed = torch.logical_and(
            self.per_block_is_block_type_allowed, self.restrict_to_repacking_masks
        )

    def restrict_absent_name3s(self, name3s):
        """Disallow all block types at all positions except those with the given name3s.

        This is somewhat slow and does not cache the relationship between name3s and
        permitted block types, so consider writing your own version of this function
        if you call with the same list of name3s many times.
        """
        bt_name3_matches = torch.tensor(
            [bt.name3 in name3s for bt in self.pbt.active_block_types],
            dtype=torch.bool,
            device=self.device,
        )
        is_real_considered_block_type = self.per_block_considered_block_types != -1
        (
            nz_real_cons_bt_pose,
            nz_real_cons_bt_block,
            nz_real_cons_bt_which_block_type,
        ) = torch.nonzero(is_real_considered_block_type, as_tuple=True)
        self.per_block_is_block_type_allowed[
            nz_real_cons_bt_pose,
            nz_real_cons_bt_block,
            nz_real_cons_bt_which_block_type,
        ] = torch.logical_and(
            self.per_block_is_block_type_allowed[
                nz_real_cons_bt_pose,
                nz_real_cons_bt_block,
                nz_real_cons_bt_which_block_type,
            ],
            bt_name3_matches[
                self.per_block_considered_block_types[
                    nz_real_cons_bt_pose,
                    nz_real_cons_bt_block,
                    nz_real_cons_bt_which_block_type,
                ]
            ],
        )

    def add_conformer_sampler(self, sampler: ConformerSampler):
        # old way of doing things; soon to be removed
        # for one_pose_blts in self.blts:
        #     for blt in one_pose_blts:
        #         blt.add_conformer_sampler(sampler)

        # new way of doing things:
        self.conformer_samplers.append(sampler)
        self.conformer_sampler_index[id(sampler)] = len(self.conformer_samplers) - 1
        self.per_block_conformer_sampler_allowed = torch.cat(
            [
                self.per_block_conformer_sampler_allowed,
                torch.ones(
                    (
                        self.per_block_conformer_sampler_allowed.shape[0],
                        self.per_block_conformer_sampler_allowed.shape[1],
                        1,
                    ),
                    dtype=torch.bool,
                    device=self.device,
                ),
            ],
            dim=-1,
        )

    def add_conformer_sampler_by_block_mask(
        self, sampler: ConformerSampler, block_type_mask: Tensor[torch.bool][:, :]
    ):
        assert (
            block_type_mask.shape == self.per_block_n_considered_block_types.shape[:2]
        )
        assert block_type_mask.device == self.per_block_conformer_sampler_allowed.device

        # new way of doing things:
        self.conformer_samplers.append(sampler)
        self.conformer_sampler_index[id(sampler)] = len(self.conformer_samplers) - 1
        self.per_block_conformer_sampler_allowed = torch.cat(
            [
                self.per_block_conformer_sampler_allowed,
                block_type_mask.unsqueeze(-1),
            ],
            dim=-1,
        )

    def or_expand_chi(self, chi_ind: int):
        # old way of doing things; soon to be removed
        # for one_pose_blts in self.blts:
        #     for blt in one_pose_blts:
        #         blt.or_expand_chi(chi_ind)

        # new way of doing things
        self.per_block_chi_expansion[:, :, :, chi_ind] = 1

    def or_expand_chi_to(self, chi_ind: int, sample_level: int):
        # old way of doing things; soon to be removed
        # for one_pose_blts in self.blts:
        #     for blt in one_pose_blts:
        #         blt.or_expand_chi_to(chi_ind, sample_level)

        # new way of doing things: max over the current sample level
        # and the new sample level.
        self.per_block_chi_expansion[:, :, :, chi_ind] = torch.max(
            self.per_block_chi_expansion[:, :, :, chi_ind], sample_level
        )

    def disable_packing_by_block_mask(self, block_type_mask: Tensor[torch.bool][:, :]):
        assert block_type_mask.device == self.device
        assert block_type_mask.shape == self.per_block_is_block_type_allowed.shape[:2]
        # old way of doing things; soon to be removed
        # for one_pose_blts in self.blts:
        #     for blt in one_pose_blts:
        #         blt.disable_packing()

        # new way of doing things:
        self.per_block_is_block_type_allowed = torch.logical_and(
            self.per_block_is_block_type_allowed, ~block_type_mask.unsqueeze(-1)
        )


class SetPackerTask:
    """Set as in concrete. Once everything wrt the desired packing
    task has been determined, pack_rotamers will construct this
    object to create and hold the many mappings that the various
    members of the packer need.
    """

    @classmethod
    def from_packer_task(cls, task: PackerTask):
        set_task = cls()
        set_task.pbt = task.pbt
        set_task.device = task.device
        set_task.is_real_block = task.is_real_block
        set_task.real_block_pose, set_task.real_block_block = torch.nonzero(
            set_task.is_real_block, as_tuple=True
        )

        set_task.per_block_orig_block_type = task.per_block_orig_block_type
        set_task.per_block_considered_block_types = (
            task.per_block_considered_block_types
        )
        set_task.per_block_n_considered_block_types = (
            task.per_block_n_considered_block_types
        )
        set_task.per_block_considered_block_types_is_orig = (
            task.per_block_considered_block_types_is_orig
        )
        set_task.restrict_to_repacking_masks = task.restrict_to_repacking_masks
        set_task.per_block_is_block_type_allowed = task.per_block_is_block_type_allowed
        set_task.conformer_samplers = task.conformer_samplers
        set_task.conformer_sampler_index = task.conformer_sampler_index
        set_task.per_block_conformer_sampler_allowed = (
            task.per_block_conformer_sampler_allowed
        )
        set_task.per_block_chi_expansion = task.per_block_chi_expansion

        max_n_blocks = task.per_block_considered_block_types.shape[1]
        is_real_cbt = set_task.per_block_considered_block_types != -1
        cbt_pose, cbt_block, cbt_which_block_type = torch.nonzero(
            is_real_cbt, as_tuple=True
        )
        cbt_block_type = set_task.per_block_considered_block_types[
            cbt_pose, cbt_block, cbt_which_block_type
        ]

        # these are the "global block types" that the conformer samplers
        # will be asked to build rotamers for; note that sometimes
        # a conformer sampler must build a rotamer for a block type that
        # is not "allowed." This happens when the user disables all allowed block
        # types for a particular block. The way tmol's packer works, residues
        # that are part of the "background" must still have one rotamer for energy
        # evaluation purposes. The upshot is that if we index on the set of allowed
        # block types, we would miss the disallowed block types that we must still
        # build rotamers for. Thus we index on all the _considered_ block types,
        # and then refine that set by the allowed subset for most conformer samplers.
        set_task.cons_bt_pose = cbt_pose
        set_task.cons_bt_block = cbt_block
        set_task.cons_bt_block_type = cbt_block_type
        set_task.cons_bt_which_block_type = cbt_which_block_type

        set_task.global_block_ind_for_considered_block_types = (
            max_n_blocks * cbt_pose + cbt_block
        )

        # these non-zeros are indices in the n-poses x n-blocks x max-n-considered
        # tensor.
        allowed_pose, allowed_block, allowed_which_bt = torch.nonzero(
            task.per_block_is_block_type_allowed, as_tuple=True
        )
        allowed_bt = task.per_block_considered_block_types[
            allowed_pose, allowed_block, allowed_which_bt
        ]
        set_task.allowed_bt_pose = allowed_pose
        set_task.allowed_bt_block = allowed_block
        set_task.allowed_bt_which_block_type = allowed_which_bt
        set_task.allowed_bt_block_type = allowed_bt

        # these non-zeros are indices in a one-dimensional tensor of
        # all (real) considered block types.
        set_task.is_cons_bt_allowed = task.per_block_is_block_type_allowed[
            cbt_pose, cbt_block, cbt_which_block_type
        ]
        set_task.allowed_cons_bt = torch.nonzero(
            set_task.is_cons_bt_allowed,
            as_tuple=True,
        )[0]

        return set_task
