import torch
import logging

from tmol.types.torch import Tensor
from tmol.pose.pose_stack import PoseStack
from tmol.score.score_function import ScoreFunction
from tmol.pack.packer_task import PackerTask, PackerPalette
from tmol.pack.rotamer.dunbrack.dunbrack_chi_sampler import DunbrackChiSampler
from tmol.pack.rotamer.fixed_aa_chi_sampler import FixedAAChiSampler

# from tmol.pack.rotamer.include_current_sampler import IncludeCurrentSampler
from tmol.pack.pack_rotamers import pack_rotamers
from tmol.score.dunbrack.params import DunbrackParamResolver
from tmol.database import ParameterDatabase

logger = logging.getLogger(__name__)


def build_missing_sidechains_with_missing_atoms(
    pose_stack: PoseStack,
    sfxn: ScoreFunction,
    dunbrack_sampler: DunbrackChiSampler,
    block_has_missing_atoms: Tensor[torch.bool][:, :],
    rts,
) -> PoseStack:
    """Build missing sidechains using the packer with explicit missing atoms information.

    This function examines the block_has_missing_atoms tensor to determine
    which blocks have missing sidechain atoms. For blocks with missing atoms,
    it adds the DunbrackChiSampler and FixedAAChiSampler to the PackerTask.
    For blocks without missing atoms, it adds the IncludeCurrentSampler to
    preserve existing sidechains. Then it calls pack_rotamers to build the
    missing sidechains.

    Args:
        pose_stack: The pose stack containing the structures to process
        sfxn: The score function to use for packing (typically beta2016)
        dunbrack_sampler: The DunbrackChiSampler configured with the default database
        block_has_missing_atoms: Boolean tensor indicating which blocks have missing atoms
                               Shape: [n_poses, max_n_blocks]

    Returns:
        PoseStack: A new pose stack with missing sidechains built
    """

    restype_set = pose_stack.packed_block_types.restype_set

    # Create a PackerPalette and PackerTask
    palette = PackerPalette(restype_set)
    task = PackerTask(pose_stack, palette)
    task.restrict_to_repacking()  # no design

    def _is_dunbrack_rebuild_eligible(restype) -> bool:
        polymer = restype.properties.polymer
        return (
            polymer.is_polymer
            and polymer.polymer_type == "amino_acid"
            and polymer.backbone_type == "alpha"
        )

    def _collect_dunbrack_rebuild_targets():
        targets = set()
        skipped_noneligible_missing = {}
        for pose_ind, one_pose_blts in enumerate(task.blts):
            for blt in one_pose_blts:
                block_ind = blt.seqpos
                has_missing = bool(block_has_missing_atoms[pose_ind, block_ind])
                if not has_missing:
                    continue

                restype = pose_stack.block_type(pose_ind, block_ind)
                if _is_dunbrack_rebuild_eligible(restype):
                    targets.add((pose_ind, block_ind))
                else:
                    skipped_noneligible_missing[restype.name] = (
                        skipped_noneligible_missing.get(restype.name, 0) + 1
                    )
        return targets, skipped_noneligible_missing

    rebuild_targets, skipped_noneligible_missing = _collect_dunbrack_rebuild_targets()

    if skipped_noneligible_missing:
        logger.info(
            "Skipping missing-sidechain rebuild for %d non-Dunbrack blocks: %s",
            sum(skipped_noneligible_missing.values()),
            ", ".join(
                f"{name}:{count}"
                for name, count in sorted(skipped_noneligible_missing.items())
            ),
        )

    # If only non-eligible blocks (e.g. ligands) are missing, there is nothing
    # Dunbrack can build. Return the original pose stack to avoid invoking the
    # sampler with an empty buildable set.
    if not rebuild_targets:
        return pose_stack

    # Configure per-block behavior:
    # - targeted blocks get Dunbrack/FixedAA sidechain rebuilding.
    # - all others are fixed to their current coordinates.
    fixed_sampler = FixedAAChiSampler()
    for pose_ind, one_pose_blts in enumerate(task.blts):
        for blt in one_pose_blts:
            block_key = (pose_ind, blt.seqpos)
            if block_key in rebuild_targets:
                blt.include_current = False
                blt.add_conformer_sampler(dunbrack_sampler)
                blt.add_conformer_sampler(fixed_sampler)
            else:
                blt.disable_packing()

    # Call pack_rotamers to build the missing sidechains
    return pack_rotamers(pose_stack, sfxn, task, verbose=True)


def create_dunbrack_sampler_from_database(
    param_db: ParameterDatabase, device: torch.device
) -> DunbrackChiSampler:
    """Create a DunbrackChiSampler from the default database.

    Args:
        param_db: The parameter database containing Dunbrack parameters
        device: The device to use for the sampler

    Returns:
        DunbrackChiSampler: Configured sampler for rotamer building
    """
    param_resolver = DunbrackParamResolver.from_database(param_db.scoring.dun, device)
    return DunbrackChiSampler.from_database(param_resolver)
