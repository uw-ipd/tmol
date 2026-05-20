import torch

from tmol import run_cart_min


def calculate_block_pair_ddg(
    pose_stack, mask, mask2=None, sfxn=None, sum_terms=True, minimize=True
):
    """Calculate DDG score between two subsets of blocks within each pose, defined by 2 masks.
    If only one mask is provided, it will use the inverse of the first mask for the second.

    Args:
        pose_stack: The pose stack to score
        mask: Boolean tensor of shape [n_poses, n_blocks]. True values indicate masked indices.
        mask2: Boolean tensor of shape [n_poses, n_blocks]. If not provided, it will use the inverse
            of the first mask as the second mask.
        sfxn: Optional score function to use. If not provided, will default to beta2016
        sum_terms: If True, sum all score terms into a single score per pose. If False,
            return per-term scores.
        minimize: If True (default), run cartesian minimization on the masked atoms before
            computing the DDG score.

    Returns:
        Tensor of shape [n_poses] or [n_terms, n_poses] containing the ddg score for each pose,
        separated by terms if requested.
    """
    torch_device = pose_stack.device

    if sfxn is None:
        from tmol.score import beta2016_score_function

        sfxn = beta2016_score_function(torch_device)

    if minimize:
        coord_mask = build_coord_mask_for_mask_and_interacting_atoms(pose_stack, mask)
        print("coord_mask True:", coord_mask.count_nonzero())
        print("coord_mask False:", (~coord_mask).count_nonzero())
        pose_stack = run_cart_min(pose_stack, sfxn, coord_mask)

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


def build_coord_mask_for_mask_and_interacting_atoms(pose_stack, mask):

    # Build coord_mask: True for atoms in masked blocks AND sidechain atoms within 3.0 Angstroms
    n_poses, max_n_atoms, _ = pose_stack.coords.shape
    n_blocks = pose_stack.max_n_blocks
    max_n_block_atoms = pose_stack.max_n_block_atoms
    pbt = pose_stack.packed_block_types

    # Expand coords to [n_poses, n_blocks, max_n_block_atoms, 3]
    expanded_coords, real_expanded_pose_ats = pose_stack.expand_coords()

    # Get number of atoms per block [n_poses, n_blocks]
    # n_ats_per_block = pose_stack.n_ats_per_block

    # Create atom-level mask for atoms in masked blocks
    # block_atom_mask[i, j, k] = True if block j is masked for pose i and atom k is real
    block_mask_expanded = mask.unsqueeze(2).expand(
        -1, -1, max_n_block_atoms
    )  # [n_poses, n_blocks, max_n_block_atoms]
    atom_in_masked_block = (
        block_mask_expanded & real_expanded_pose_ats
    )  # [n_poses, n_blocks, max_n_block_atoms]

    # Flatten to per-atom mask [n_poses, n_blocks * max_n_block_atoms]
    atom_in_masked_block_flat = atom_in_masked_block.reshape(n_poses, -1)

    # Create a mask for real atoms in the pose [n_poses, max_n_atoms]
    real_atoms = pose_stack.real_atoms

    # Map expanded atom indices to flat coords indices
    # For each pose, the atoms are laid out contiguously in coords
    # We need to figure out which flat index each (block, atom_in_block) pair maps to
    block_coord_offset64 = pose_stack.block_coord_offset64  # [n_poses, n_blocks]

    # Create flat indices for all atoms in expanded view
    # atom_local_idx[i, j, k] = k (atom index within block)
    atom_local_idx = (
        torch.arange(max_n_block_atoms, device=pose_stack.device)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(n_poses, n_blocks, -1)
    )

    # flat_atom_idx[i, j, k] = block_coord_offset64[i, j] + k
    flat_atom_idx = (
        block_coord_offset64.unsqueeze(2) + atom_local_idx
    )  # [n_poses, n_blocks, max_n_block_atoms]
    flat_atom_idx_flat = flat_atom_idx.reshape(
        n_poses, -1
    )  # [n_poses, n_blocks * max_n_block_atoms]

    # Create coord_mask initialized to False
    coord_mask = torch.zeros(
        (n_poses, max_n_atoms), dtype=torch.bool, device=pose_stack.device
    )

    # Set True for atoms in masked blocks
    # Use scatter to set the appropriate indices
    # Only include valid flat indices (some may exceed max_n_atoms
    # due to padding in the expanded block view).
    valid_flat_idx = flat_atom_idx_flat < max_n_atoms
    # Keep 2D shape; for invalid positions set src to False (no-op) and index to 0 (safe dummy)
    atom_in_masked_block_flat_safe = atom_in_masked_block_flat.clone()
    atom_in_masked_block_flat_safe[~valid_flat_idx] = False
    flat_atom_idx_flat_safe = flat_atom_idx_flat.clone()
    flat_atom_idx_flat_safe[~valid_flat_idx] = 0
    coord_mask.scatter_(1, flat_atom_idx_flat_safe, atom_in_masked_block_flat_safe)

    # Build sidechain atom mask: True for atoms that are sidechain atoms
    # An atom is a sidechain atom if it's real and NOT a mainchain atom
    # For non-polymeric residues, all atoms are considered sidechain
    block_type_ind64 = pose_stack.block_type_ind64  # [n_poses, n_blocks]

    # Create expanded sidechain mask [n_poses, n_blocks, max_n_block_atoms]
    is_sidechain_expanded = torch.zeros(
        (n_poses, n_blocks, max_n_block_atoms),
        dtype=torch.bool,
        device=pose_stack.device,
    )

    for bt_idx in range(pbt.n_types):
        rt = pbt.active_block_types[bt_idx]
        mc_atoms = (
            rt.properties.polymer.mainchain_atoms if rt.properties.polymer else None
        )

        # Get positions of this block type in pose_stack
        bt_positions = block_type_ind64 == bt_idx  # [n_poses, n_blocks]
        if not bt_positions.any():
            continue

        # Create mask for real atoms of this block type
        bt_real_atoms = real_expanded_pose_ats & bt_positions.unsqueeze(
            2
        )  # [n_poses, n_blocks, max_n_block_atoms]

        if mc_atoms is not None and len(mc_atoms) > 0:
            # Get mainchain atom indices
            mc_atom_indices = torch.tensor(
                [rt.atom_to_idx[at] for at in mc_atoms], device=pose_stack.device
            )
            # Create mask for mainchain atoms
            mc_mask = torch.zeros(
                max_n_block_atoms, dtype=torch.bool, device=pose_stack.device
            )
            mc_mask[mc_atom_indices] = True
            # Sidechain = real atoms that are NOT mainchain
            is_sidechain_expanded[bt_positions] = bt_real_atoms[bt_positions] & ~mc_mask
        else:
            # Non-polymeric: all real atoms are sidechain
            is_sidechain_expanded[bt_positions] = bt_real_atoms[bt_positions]

    # Now find atoms within 5.0 Angstroms of any masked atom
    # Get coordinates of masked atoms and all real atoms
    coords = pose_stack.coords  # [n_poses, max_n_atoms, 3]

    # Create a per-pose distance computation
    # For each pose, compute distance from each real atom to nearest masked atom
    for p in range(n_poses):
        # Get masked atom indices for this pose (flat indices into coords)
        masked_flat_idx = flat_atom_idx_flat[
            p, atom_in_masked_block_flat[p]
        ]  # [n_masked_atoms]

        if masked_flat_idx.numel() == 0:
            continue

        # Get masked atom coordinates [n_masked_atoms, 3]
        masked_atom_coords = coords[p, masked_flat_idx, :]  # [n_masked_atoms, 3]

        # Get all real atom coordinates for this pose [n_real_atoms, 3]
        real_atom_indices = torch.nonzero(real_atoms[p], as_tuple=True)[
            0
        ]  # [n_real_atoms]
        all_real_coords = coords[p, real_atom_indices, :]  # [n_real_atoms, 3]

        # Compute pairwise distances [n_real_atoms, n_masked_atoms]
        # Using broadcasting: all_real_coords[:, None, :] - masked_atom_coords[None, :, :]
        diff = all_real_coords.unsqueeze(1) - masked_atom_coords.unsqueeze(
            0
        )  # [n_real_atoms, n_masked_atoms, 3]
        distances = torch.sqrt((diff**2).sum(dim=2))  # [n_real_atoms, n_masked_atoms]

        # Find atoms within 3.0 Angstroms of any masked atom
        min_distances = distances.min(dim=1)[0]  # [n_real_atoms]
        nearby_atoms = min_distances <= 5.0  # [n_real_atoms]

        # For nearby atoms NOT in the original mask, only include sidechain atoms
        # First, get the expanded sidechain mask for this pose, flattened
        is_sidechain_flat = is_sidechain_expanded[p].reshape(
            -1
        )  # [n_blocks * max_n_block_atoms]

        # Create a per-atom sidechain mask for flat atom indices
        # Some flat indices may exceed max_n_atoms due to padding in expanded view
        is_sidechain_per_atom = torch.zeros(
            max_n_atoms, dtype=torch.bool, device=pose_stack.device
        )
        valid_flat_idx_p = flat_atom_idx_flat[p] < max_n_atoms
        flat_atom_idx_flat_p_safe = flat_atom_idx_flat[p][valid_flat_idx_p]
        is_sidechain_flat_safe = is_sidechain_flat[valid_flat_idx_p]
        is_sidechain_per_atom.scatter_(
            0, flat_atom_idx_flat_p_safe, is_sidechain_flat_safe
        )

        # Check which nearby atoms are sidechain atoms
        nearby_is_sidechain = is_sidechain_per_atom[real_atom_indices[nearby_atoms]]

        # Determine which nearby atoms to include:
        # - Atoms in masked blocks are always included (already in coord_mask)
        # - Atoms NOT in masked blocks are only included if they are sidechain atoms
        atoms_in_masked_blocks = coord_mask[p, real_atom_indices[nearby_atoms]]
        atoms_to_add = nearby_atoms.clone()
        atoms_to_add[nearby_atoms] = atoms_in_masked_blocks | nearby_is_sidechain

        # Update coord_mask for this pose
        coord_mask[p, real_atom_indices[atoms_to_add]] = True

    return coord_mask
