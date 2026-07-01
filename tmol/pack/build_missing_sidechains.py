import torch

from tmol.types.torch import Tensor
from tmol.pose.pose_stack import PoseStack
from tmol.score.score_function import ScoreFunction
from tmol.pack.packer_task import PackerTask, PackerPalette
from tmol.pack.rotamer.dunbrack.dunbrack_chi_sampler import DunbrackChiSampler
from tmol.pack.rotamer.fixed_aa_chi_sampler import FixedAAChiSampler
from tmol.pack.pack_rotamers import pack_rotamers


def build_missing_sidechains(
    pose_stack: PoseStack,
    sfxn: ScoreFunction,
    dunbrack_sampler: DunbrackChiSampler,
    block_has_missing_atoms: Tensor[torch.bool][:, :],
    rts,
    no_optH: bool = False,
) -> PoseStack:
    """Build missing sidechains and place hydrogens using per-block sampler assignment.

    Assigns samplers on a per-block basis in a single packing run:

    - Blocks with missing non-leaf (heavy) atoms: DunbrackChiSampler +
      FixedAAChiSampler.  The input conformation is not included as a rotamer
      because the sidechain is incomplete.
    - All other real blocks (leaf-only or no missing atoms): OptHSampler, which
      keeps heavy atoms fixed and samples proton chi angles and NHQ flips.
      FallbackSampler (always present by default) covers residue types that
      OptH does not handle (ALA, GLY, etc.).

    When no_optH=True the old behavior is preserved: only Dunbrack runs for
    blocks with missing heavy atoms; all other blocks are frozen.

    Note: IncludeCurrentSampler is intentionally not used.  For Dunbrack
    blocks the native conformation is broken and must not appear as a rotamer.
    For OptH blocks, OptH includes native as rotamer-0 for NHQ residues and
    FallbackSampler covers the rest.

    Args:
        pose_stack: The pose stack to process.
        sfxn: Score function used for packing.
        dunbrack_sampler: DunbrackChiSampler configured from the parameter DB.
        block_has_missing_atoms: Boolean tensor [n_poses, max_n_blocks]; True
            for blocks that have missing non-leaf (heavy) atoms.
        rts: ResidueTypeSet (unused directly; kept for API compatibility).
        no_optH: When True, skip OptH and preserve old Dunbrack-only behavior.

    Returns:
        PoseStack with missing sidechains built and (by default) hydrogens
        placed and optimized.
    """
    from tmol.pack.rotamer.opth_sampler import OptHSampler

    assert block_has_missing_atoms.device == pose_stack.device

    palette = PackerPalette()
    task = PackerTask(pose_stack, palette)
    task.restrict_to_repacking()

    fixed_sampler = FixedAAChiSampler()
    opth_sampler = None if no_optH else OptHSampler()

    task.add_conformer_sampler_by_block_mask(dunbrack_sampler, block_has_missing_atoms)
    task.add_conformer_sampler_by_block_mask(fixed_sampler, block_has_missing_atoms)
    block_does_not_have_missing_atoms = torch.logical_not(block_has_missing_atoms)
    if not no_optH:
        task.add_conformer_sampler_by_block_mask(
            opth_sampler, block_does_not_have_missing_atoms
        )
    else:
        task.disable_packing_by_block_mask(block_does_not_have_missing_atoms)

    return pack_rotamers(pose_stack, sfxn, task, verbose=False)
