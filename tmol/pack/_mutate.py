import torch

from tmol.chemical import one2three
from tmol.database import ParameterDatabase
from tmol.pose import PoseStack
from tmol.score import ScoreFunction, beta2016_score_function
from tmol.pack import PackerTask, PackerPalette, pack_rotamers
from tmol.pack.rotamer import FixedAAChiSampler, IncludeCurrentSampler
from tmol.pack.rotamer.dunbrack import create_dunbrack_sampler_from_database


def _mutate_blocks(
    pose_stack: PoseStack,
    pose_index: int,
    name3_by_block: dict,
    score_function: ScoreFunction | None,
    param_db: ParameterDatabase | None,
    seed: int | None,
) -> PoseStack:
    """Design the given blocks of one pose to the given three-letter identities.

    Every other block, and every other pose in the stack, is frozen at its
    current conformation. The input PoseStack itself is not modified.
    """
    if score_function is None:
        score_function = beta2016_score_function(pose_stack.device)
    if param_db is None:
        param_db = getattr(score_function, "_param_db", None)

    target_mask = torch.zeros(
        (pose_stack.n_poses, pose_stack.max_n_blocks),
        dtype=torch.bool,
        device=pose_stack.device,
    )
    for block in name3_by_block:
        target_mask[pose_index, block] = True

    task = PackerTask(pose_stack, PackerPalette())
    task.disable_packing_by_block_mask(~target_mask)
    for block, name3 in name3_by_block.items():
        single_mask = torch.zeros_like(target_mask)
        single_mask[pose_index, block] = True
        task.restrict_absent_name3s({name3}, single_mask)
    task.add_conformer_sampler(
        create_dunbrack_sampler_from_database(param_db, pose_stack.device)
    )
    task.add_conformer_sampler(FixedAAChiSampler())
    task.add_conformer_sampler(IncludeCurrentSampler())

    return pack_rotamers(pose_stack, score_function, task, seed=seed)


def mutate_residue(
    pose_stack: PoseStack,
    pose_index: int,
    block: int,
    new_identity: str,
    score_function: ScoreFunction | None = None,
    *,
    param_db: ParameterDatabase | None = None,
    seed: int | None = None,
) -> PoseStack:
    """Change one block's residue identity via the packer.

    Builds a PackerTask restricted to repacking just the target block, with
    its allowed identities narrowed to ``new_identity``, and returns a new
    PoseStack with that block's type and conformation chosen by the Dunbrack
    rotamer library. Every other block, and every other pose in the stack, is
    left untouched. The input PoseStack itself is not modified.

    Args:
        pose_stack: The pose stack to mutate.
        pose_index: Which pose in the stack to mutate.
        block: Block index to change.
        new_identity: Three-letter code of the requested residue type.
        score_function: Score function used for packing; defaults to
            beta2016 on the pose stack's device.
        param_db: Parameter database used to build the Dunbrack sampler;
            defaults to the score function's own database.
        seed: Packer seed, as for :func:`~tmol.pack.pack_rotamers`.

    Returns:
        A new PoseStack with the requested block mutated.
    """
    return _mutate_blocks(
        pose_stack, pose_index, {block: new_identity}, score_function, param_db, seed
    )


def mutate_to_sequence(
    pose_stack: PoseStack,
    pose_index: int,
    sequence: str,
    score_function: ScoreFunction | None = None,
    *,
    param_db: ParameterDatabase | None = None,
    seed: int | None = None,
) -> PoseStack:
    """Mutate every real block of one pose whose identity doesn't match ``sequence``.

    ``sequence`` is a one-letter sequence, as returned by
    :func:`~tmol.pose.get_sequence`, for the real (non-padding) blocks of one
    pose, in block order. Positions whose current three-letter code already
    matches their requested letter are left untouched; every differing
    position is redesigned to the requested identity in a single packing run.

    Args:
        pose_stack: The pose stack to mutate.
        pose_index: Which pose in the stack to mutate.
        sequence: One letter per real block of ``pose_index``, in block order.
        score_function: Score function used for packing; defaults to
            beta2016 on the pose stack's device.
        param_db: Parameter database used to build the Dunbrack sampler;
            defaults to the score function's own database.
        seed: Packer seed, as for :func:`~tmol.pack.pack_rotamers`.

    Returns:
        A new PoseStack with every differing position mutated to match.
    """
    pbt = pose_stack.packed_block_types
    block_type_ind = pose_stack.block_type_ind64[pose_index]
    real_blocks = torch.nonzero(block_type_ind >= 0).flatten().tolist()
    if len(sequence) != len(real_blocks):
        raise ValueError(
            f"sequence has {len(sequence)} residues but pose {pose_index} has "
            f"{len(real_blocks)} real blocks"
        )

    name3_by_block = {}
    for block, letter in zip(real_blocks, sequence):
        new_name3 = one2three(letter)
        if new_name3 is None:
            raise ValueError(f"{letter!r} is not a valid one-letter amino acid code")
        current_name3 = pbt.active_block_types[int(block_type_ind[block])].name3
        if new_name3 != current_name3:
            name3_by_block[block] = new_name3

    if not name3_by_block:
        return pose_stack

    return _mutate_blocks(
        pose_stack, pose_index, name3_by_block, score_function, param_db, seed
    )
