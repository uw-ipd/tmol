import torch

from tmol.types.torch import Tensor
from tmol.pose.pose_stack import PoseStack
from tmol.score.score_function import ScoreFunction
from tmol.pack.packer_task import PackerTask, PackerPalette
from tmol.pack.rotamer.dunbrack.dunbrack_chi_sampler import DunbrackChiSampler
from tmol.pack.rotamer.fixed_aa_chi_sampler import FixedAAChiSampler
from tmol.pack.rotamer.include_current_sampler import IncludeCurrentSampler
from tmol.pack.pack_rotamers import pack_rotamers
from tmol.score.dunbrack.params import DunbrackParamResolver
from tmol.database import ParameterDatabase


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
    n_poses = pose_stack.block_type_ind.shape[0]
    max_n_blocks = pose_stack.block_type_ind.shape[1]

    # Create a PackerPalette and PackerTask
    palette = PackerPalette(rts)
    task = PackerTask(pose_stack, palette)

    # Add conformer samplers based on missing atoms status
    for pose_ind in range(n_poses):
        for block_ind in range(max_n_blocks):
            # Check if this is a real block (not padding)
            if pose_stack.is_real_block(pose_ind, block_ind):
                has_missing = block_has_missing_atoms[pose_ind, block_ind]

                # Get the block-level task for this position
                # Note: task.blts is organized as [pose][block], but we need to find
                # the correct index since not all blocks are real
                blt = None
                blt_index = 0
                for i, one_pose_blts in enumerate(task.blts):
                    if i == pose_ind:
                        for j, candidate_blt in enumerate(one_pose_blts):
                            if j == block_ind:
                                blt = candidate_blt
                                break
                        break

                if blt is None:
                    continue

                if has_missing:
                    # Block has missing atoms - add Dunbrack and FixedAA samplers
                    # to build new sidechains
                    blt.add_conformer_sampler(dunbrack_sampler)
                    blt.add_conformer_sampler(FixedAAChiSampler())
                else:
                    # Block has no missing atoms - add IncludeCurrent sampler
                    # to preserve existing sidechains
                    blt.add_conformer_sampler(IncludeCurrentSampler())
    task = PackerTask(pose_stack, palette)
    task.restrict_to_repacking()
    task.set_include_current()

    fixed_sampler = FixedAAChiSampler()
    task.add_conformer_sampler(dunbrack_sampler)
    task.add_conformer_sampler(fixed_sampler)

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
