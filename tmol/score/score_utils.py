def calculate_block_pair_ddg(pose_stack, mask, mask2=None, sfxn=None, sum_terms=True):
    """Calculate DDG score between two subsets of blocks within each pose, defined by 2 masks.
    If only one mask is provided, it will use the inverse of the first mask for the second.

    Args:
        pose_stack: The pose stack to score
        mask: Boolean tensor of shape [n_poses, n_blocks]. True values indicate masked indices.
        mask2: Boolean tensor of shape [n_poses, n_blocks]. If not provided, it will use the inverse
            of the first mask as the second mask.
        sfxn: Optional score function to use. If not provided, will default to beta2016

    Returns:
        Tensor of shape [n_poses] or [n_terms, n_poses] containing the ddg score for each pose,
        separated by terms if requested.
    """
    torch_device = pose_stack.device

    if sfxn is None:
        from tmol.score import beta2016_score_function

        sfxn = beta2016_score_function(torch_device)

    scorer = sfxn.render_block_pair_scoring_module(pose_stack)
    block_pair_scores = scorer(pose_stack.coords, sum_terms=False)

    # block_pair_scores shape: [n_terms, n_poses, n_blocks, n_blocks]
    n_terms, n_poses, n_blocks, _ = block_pair_scores.shape

    # mask shape: [n_poses, n_blocks]
    # Use mask2 if provided, otherwise use ~mask for the second set of indices
    other_mask = mask2 if mask2 is not None else ~mask

    # Create masks for both sides of the diagonal for each pose
    # Side 1: i is in mask, j is in other_mask
    mask_i = mask.unsqueeze(2)  # Shape: [n_poses, n_blocks, 1]
    other_j = other_mask.unsqueeze(1)  # Shape: [n_poses, 1, n_blocks]
    cross_mask_1 = mask_i & other_j

    # Side 2: i is in other_mask, j is in mask
    other_i = other_mask.unsqueeze(2)  # Shape: [n_poses, n_blocks, 1]
    mask_j = mask.unsqueeze(1)  # Shape: [n_poses, 1, n_blocks]
    cross_mask_2 = other_i & mask_j

    # Combine both sides
    cross_mask = cross_mask_1 | cross_mask_2  # Shape: [n_poses, n_blocks, n_blocks]

    # Expand mask to cover each score term
    cross_mask = cross_mask.unsqueeze(0).expand(n_terms, -1, -1, -1)

    # Apply mask and sum all of the block pair energies
    ddg_scores = block_pair_scores[cross_mask].view(n_terms, n_poses, -1).sum(dim=2)

    if sum_terms:
        ddg_scores = ddg_scores.sum(dim=0)

    return ddg_scores
