import torch

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
    task.set_include_current()
    task.restrict_to_repacking()  # no design

    # Iterate through the block level tasks and either disable packing if the sidechain already
    # exists, or else make sure we dont try and load the current sidechain with missing atoms
    for pose_ind in range(block_has_missing_atoms.size(0)):
        for block_ind in range(block_has_missing_atoms.size(1)):
            if pose_stack.is_real_block(pose_ind, block_ind):
                has_missing = block_has_missing_atoms[pose_ind, block_ind]
                if has_missing:
                    task.blts[pose_ind][block_ind].include_current = False
                else:
                    task.blts[pose_ind][block_ind].disable_packing()

    # Add the samplers
    fixed_sampler = FixedAAChiSampler()
    task.add_conformer_sampler(dunbrack_sampler)
    task.add_conformer_sampler(fixed_sampler)

    # Call pack_rotamers to build the missing sidechains
    return pack_rotamers(pose_stack, sfxn, task, verbose=True)
