def calculate_block_pair_ddg(pose_stack, mask, mask2=None, sfxn=None):
    """Calculate DDG score between two subsets of blocks within each pose, defined by 2 masks.
    If only one mask is provided, it will use the inverse of the first mask for the second.

    Args:
        pose_stack: The pose stack to score
        mask: Boolean tensor of shape [n_poses, n_blocks]. True values indicate masked indices.
        mask2: Boolean tensor of shape [n_poses, n_blocks]. If not provided, it will use the inverse
            of the first mask as the second mask.
        sfxn: Optional score function to use. If not provided, will default to beta2016

    Returns:
        Tensor of shape [n_poses] containing the ddg score for each pose
    """
    torch_device = pose_stack.device

    if sfxn is None:
        from tmol.score import beta2016_score_function

        sfxn = beta2016_score_function(torch_device)

    scorer = sfxn.render_block_pair_scoring_module(pose_stack)
    block_pair_scores = scorer(pose_stack.coords)

    # block_pair_scores shape: [B, N, N] where B is batch size
    B, N, _ = block_pair_scores.shape

    # mask shape: [B, N]
    # Use mask2 if provided, otherwise use ~mask for the second set of indices
    other_mask = mask2 if mask2 is not None else ~mask

    # Create masks for both sides of the diagonal for each pose
    # Side 1: i is in mask, j is in other_mask
    mask_i = mask.unsqueeze(2)  # Shape: [B, N, 1]
    other_j = other_mask.unsqueeze(1)  # Shape: [B, 1, N]
    cross_mask_1 = mask_i & other_j

    # Side 2: i is in other_mask, j is in mask
    other_i = other_mask.unsqueeze(2)  # Shape: [B, N, 1]
    mask_j = mask.unsqueeze(1)  # Shape: [B, 1, N]
    cross_mask_2 = other_i & mask_j

    # Combine both sides
    cross_mask = cross_mask_1 | cross_mask_2  # Shape: [B, N, N]

    # Apply mask and sum over the last two dimensions (N, N) for each pose
    ddg_scores = block_pair_scores[cross_mask].view(B, -1).sum(dim=1)

    return ddg_scores
