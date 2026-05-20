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


def res_mask_to_coord_mask(pose_stack, mask):
    """Convert a block-level (residue) boolean mask to an atom-level coordinate mask.

    For each pose, atoms belonging to blocks where the mask is True are marked as True
    in the output. The output mask can be used as a ``coord_mask`` argument to
    functions like ``run_cart_min``.

    Args:
        pose_stack: The pose stack. Must have attributes ``coords``, ``max_n_blocks``,
            ``max_n_block_atoms``, ``block_coord_offset64``, and ``real_atoms``.
        mask: Boolean tensor of shape ``[n_poses, n_blocks]``. ``True`` at
            ``mask[i, j]`` indicates that all atoms of block ``j`` in pose ``i``
            should be marked.

    Returns:
        Boolean tensor of shape ``[n_poses, max_n_atoms_per_pose]``, where
        ``max_n_atoms_per_pose = pose_stack.coords.shape[1]``.
    """
    n_poses, max_n_atoms, _ = pose_stack.coords.shape
    n_blocks = pose_stack.max_n_blocks
    max_n_block_atoms = pose_stack.max_n_block_atoms

    # Expand the block mask to per-atom indices
    block_coord_offset64 = pose_stack.block_coord_offset64  # [n_poses, n_blocks]

    # For each block, we need to know which flat coord indices are real atoms.
    # block_coord_offset64[i, j] is the starting flat index of block j in pose i.
    # The k-th atom (0 <= k < max_n_block_atoms) of block j has flat index
    #   flat_idx[i, j, k] = block_coord_offset64[i, j] + k
    #
    # Not every k corresponds to a real atom (due to padding); we only mark
    # those where real_expanded_pose_ats[i, j, k] is True.

    # atom_local_idx[k] = k
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

    # Keep an auxiliary index for scatter: [n_poses, -1]  (in case of binning we need 2D)
    flat_atom_idx_flat = flat_atom_idx.reshape(n_poses, -1)

    # Real-atom expanded mask: [n_poses, n_blocks, max_n_block_atoms]
    _, real_expanded_pose_ats = pose_stack.expand_coords()

    # Block mask expanded to atom granularity: [n_poses, n_blocks, max_n_block_atoms]
    block_mask_expanded = mask.unsqueeze(2).expand(-1, -1, max_n_block_atoms)

    # Atom is masked if its block is masked AND it is a real atom
    atom_is_masked = block_mask_expanded & real_expanded_pose_ats

    # Flatten for scatter
    atom_is_masked_flat = atom_is_masked.reshape(n_poses, -1)  # [n_poses, ...]

    # Build output coord_mask
    coord_mask = torch.zeros(
        (n_poses, max_n_atoms), dtype=torch.bool, device=pose_stack.device
    )

    # Guard against out-of-range flat indices (padding in expanded view can exceed max_n_atoms)
    valid_flat_idx = flat_atom_idx_flat < max_n_atoms

    atom_is_masked_flat_safe = atom_is_masked_flat.clone()
    atom_is_masked_flat_safe[~valid_flat_idx] = False

    flat_atom_idx_flat_safe = flat_atom_idx_flat.clone()
    flat_atom_idx_flat_safe[~valid_flat_idx] = 0

    coord_mask.scatter_(1, flat_atom_idx_flat_safe, atom_is_masked_flat_safe)

    return coord_mask


def build_sidechain_coord_mask(pose_stack):
    """Build a coord_mask that selects only atoms belonging to sidechains.

    For polymeric residues, sidechain atoms are defined as real atoms that are
    NOT mainchain atoms. For non-polymeric residues, all real atoms are
    considered sidechain atoms. The output mask can be used as a ``coord_mask``
    argument to functions like ``run_cart_min``.

    Args:
        pose_stack: The pose stack. Must have attributes ``coords``,
            ``max_n_blocks``, ``max_n_block_atoms``, ``block_coord_offset64``,
            ``real_atoms``, ``block_type_ind64``, and ``packed_block_types``.

    Returns:
        Boolean tensor of shape ``[n_poses, max_n_atoms_per_pose]``, where
        ``max_n_atoms_per_pose = pose_stack.coords.shape[1]``. True at
        ``coord_mask[i, j]`` indicates that atom ``j`` of pose ``i`` is a
        sidechain atom.
    """
    n_poses, max_n_atoms, _ = pose_stack.coords.shape
    n_blocks = pose_stack.max_n_blocks
    max_n_block_atoms = pose_stack.max_n_block_atoms
    pbt = pose_stack.packed_block_types

    # Expand coords to get real atom mask [n_poses, n_blocks, max_n_block_atoms]
    _, real_expanded_pose_ats = pose_stack.expand_coords()

    # block_coord_offset64[i, j] = starting flat coord index of block j in pose i
    block_coord_offset64 = pose_stack.block_coord_offset64  # [n_poses, n_blocks]

    # atom_local_idx[i, j, k] = k  (atom index within block)
    atom_local_idx = (
        torch.arange(max_n_block_atoms, device=pose_stack.device)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(n_poses, n_blocks, -1)
    )

    # flat_atom_idx[i, j, k] = block_coord_offset64[i, j] + k
    flat_atom_idx = block_coord_offset64.unsqueeze(2) + atom_local_idx
    flat_atom_idx_flat = flat_atom_idx.reshape(n_poses, -1)

    # Build expanded sidechain mask [n_poses, n_blocks, max_n_block_atoms]
    block_type_ind64 = pose_stack.block_type_ind64  # [n_poses, n_blocks]

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

        # Get positions of this block type in the pose stack
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

    # Flatten sidechain mask for scatter operation
    is_sidechain_flat = is_sidechain_expanded.reshape(n_poses, -1)

    # Build output coord_mask
    coord_mask = torch.zeros(
        (n_poses, max_n_atoms), dtype=torch.bool, device=pose_stack.device
    )

    # Guard against out-of-range flat indices (padding can exceed max_n_atoms)
    valid_flat_idx = flat_atom_idx_flat < max_n_atoms
    is_sidechain_flat_safe = is_sidechain_flat.clone()
    is_sidechain_flat_safe[~valid_flat_idx] = False
    flat_atom_idx_flat_safe = flat_atom_idx_flat.clone()
    flat_atom_idx_flat_safe[~valid_flat_idx] = 0

    coord_mask.scatter_(1, flat_atom_idx_flat_safe, is_sidechain_flat_safe)

    return coord_mask


def compute_block_centroids_and_furthest_dist(pose_stack):
    """For each block in a pose stack, compute the average coordinate (centroid)
    of all of its atoms, as well as the distance of the furthest atom from this
    average.

    Args:
        pose_stack: A PoseStack object.

    Returns:
        block_centroids: Tensor of shape [n_poses, n_blocks, 3] containing
            the average coordinate (centroid) of all real atoms in each block.
            Padding blocks and blocks with no atoms will have NaN centroids.
        block_furthest_dist: Tensor of shape [n_poses, n_blocks] containing
            the distance of the furthest atom from the centroid for each block.
            Padding blocks and blocks with no atoms will have NaN values.
    """
    # Expand coords to [n_poses, n_blocks, max_n_block_atoms, 3]
    expanded_coords, real_expanded_pose_ats = pose_stack.expand_coords()

    # Count real atoms per block: [n_poses, n_blocks, 1]
    n_real_atoms = real_expanded_pose_ats.sum(dim=2, keepdim=True)

    # Compute sum of coordinates for real atoms in each block
    masked_coords = expanded_coords * real_expanded_pose_ats.unsqueeze(3).float()
    coord_sum = masked_coords.sum(dim=2)  # [n_poses, n_blocks, 3]

    # Average (centroid) for each block; avoid division by zero for padding blocks
    has_atoms = n_real_atoms > 0  # [n_poses, n_blocks, 1]
    block_centroids = torch.where(
        has_atoms,
        coord_sum / n_real_atoms.float(),
        torch.tensor(float("nan"), device=pose_stack.device, dtype=torch.float32),
    )

    # Compute distances from each atom to its block centroid
    center_coords = expanded_coords - block_centroids.unsqueeze(2)
    atom_dists = torch.sqrt(
        (center_coords**2).sum(dim=3)
    )  # [n_poses, n_blocks, max_n_block_atoms]

    # Max distance per block (zero out padding atoms first)
    atom_dists_masked = atom_dists * real_expanded_pose_ats.float()
    block_max_dist = atom_dists_masked.amax(dim=2)  # [n_poses, n_blocks]

    # Set padding blocks to NaN
    is_real_block = pose_stack.block_type_ind != -1  # [n_poses, n_blocks]
    block_furthest_dist = torch.where(
        is_real_block & has_atoms.squeeze(2),
        block_max_dist,
        torch.tensor(float("nan"), device=pose_stack.device, dtype=torch.float32),
    )

    # Also set centroids for padding blocks to NaN
    block_centroids = torch.where(
        is_real_block.unsqueeze(2),
        block_centroids,
        torch.tensor(float("nan"), device=pose_stack.device, dtype=torch.float32),
    )

    return block_centroids, block_furthest_dist


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


def build_coord_mask_for_mask_and_nearby_blocks(pose_stack, mask):
    """Build a coord mask starting from a per-block mask, extending to
    sidechain atoms of blocks whose centroid is within dynamic range of
    any masked block centroid.

    All atoms from blocks in ``mask`` are unconditionally included.
    Additionally, sidechain atoms from any *unmasked* block are included
    if the distance between its centroid and the centroid of **any** masked
    block is **less than the sum of their respective furthest-atom-from-centroid
    distances**.

    Args:
        pose_stack: The pose stack. Must have attributes ``coords``,
            ``max_n_blocks``, ``max_n_block_atoms``, ``block_coord_offset64``,
            ``real_atoms``, ``block_type_ind64``, ``block_type_ind``, and
            ``packed_block_types``.
        mask: Boolean tensor of shape ``[n_poses, n_blocks]``.

    Returns:
        Boolean tensor of shape ``[n_poses, max_n_atoms]`` suitable for
        use as a ``coord_mask`` argument to ``run_cart_min``.
    """
    n_poses, max_n_atoms, _ = pose_stack.coords.shape
    n_blocks = pose_stack.max_n_blocks
    max_n_block_atoms = pose_stack.max_n_block_atoms

    # ---------------------------------------------------------------
    # 1.  All atoms from the masked blocks themselves.
    # ---------------------------------------------------------------
    coord_mask = res_mask_to_coord_mask(pose_stack, mask)

    # ---------------------------------------------------------------
    # 2.  Per-atom sidechain mask.
    # ---------------------------------------------------------------
    sidechain_mask = build_sidechain_coord_mask(pose_stack)  # [n_poses, max_n_atoms]

    # ---------------------------------------------------------------
    # 3.  Block centroids and furthest-atom-from-centroid distances.
    # ---------------------------------------------------------------
    block_centroids, block_furthest_dist = compute_block_centroids_and_furthest_dist(
        pose_stack
    )

    # ---------------------------------------------------------------
    # 4.  Block-block adjacency matrix.
    # ---------------------------------------------------------------
    adjacency = compute_block_adjacency(
        block_centroids, block_furthest_dist
    )  # [n_poses, n_blocks, n_blocks]

    # ---------------------------------------------------------------
    # 5.  Find unmasked blocks adjacent to any masked block.
    # ---------------------------------------------------------------
    # adjacency[mask] -> [n_poses, n_masked, n_blocks]
    # any masked block adjacent to block j means block j should contribute sidechains
    # Expand mask for broadcasting: mask[:, :, None] & adjacency
    nearby_mask = (mask.unsqueeze(2) & adjacency).any(dim=1)  # [n_poses, n_blocks]

    # ---------------------------------------------------------------
    # 6.  Sidechain atoms from nearby blocks.
    # ---------------------------------------------------------------
    # all atoms from nearby blocks
    coord_mask_nearby = res_mask_to_coord_mask(pose_stack, nearby_mask)
    # keep only the sidechain atoms among those
    sidechain_nearby_mask = coord_mask_nearby & sidechain_mask

    # ---------------------------------------------------------------
    # 7.  Combine: all atoms from original masked blocks + sidechain
    #     atoms from nearby blocks.  Avoid double-counting the masked
    #     blocks themselves (their full atoms are already in coord_mask).
    # ---------------------------------------------------------------
    coord_mask = coord_mask | sidechain_nearby_mask

    return coord_mask


def compute_block_adjacency(block_centroids, block_furthest_dist, constant=5.0):
    """Compute a boolean block-level adjacency matrix.

    Two blocks *i* and *j* (in the same pose) are considered adjacent when
    the distance between their centroids is **less than the sum of their
    furthest-atom-from-centroid distances plus a constant**.

    .. math::

        \\|\\mathbf{c}_i - \\mathbf{c}_j\\|
        < d_i + d_j + \\text{constant}

    Args:
        block_centroids: Tensor of shape ``[n_poses, n_blocks, 3]``
            containing the centroid coordinate of each block (e.g. as
            returned by :func:`compute_block_centroids_and_furthest_dist`).
        block_furthest_dist: Tensor of shape ``[n_poses, n_blocks]``
            containing the distance of the atom furthest from the centroid
            for each block.
        constant: A scalar added to the sum of furthest distances.  Default
            is ``5.0``.

    Returns:
        Boolean tensor of shape ``[n_poses, n_blocks, n_blocks]`` where
        ``adjacency[p, i, j]`` is ``True`` when the two blocks *i* and *j*
        in pose *p* are adjacent.

        The diagonal is always ``False`` (a block is not adjacent to itself).
        Padding / NaN-containing blocks are treated as not adjacent to any
        block.
    """
    n_poses, n_blocks, _ = block_centroids.shape

    # Pairwise centroid distance [n_poses, n_blocks, n_blocks]
    diff = block_centroids.unsqueeze(2) - block_centroids.unsqueeze(1)
    centroid_dists = torch.sqrt((diff**2).sum(dim=3))

    # Sum of furthest distances [n_poses, n_blocks, n_blocks]
    dist_sum = (
        block_furthest_dist.unsqueeze(2) + block_furthest_dist.unsqueeze(1)
    )

    # Adjacent if centroid distance < furthest distance sum + constant
    adjacency = centroid_dists < (dist_sum + constant)

    # Exclude self (diagonal)
    adjacency = adjacency & ~torch.eye(n_blocks, dtype=torch.bool, device=block_centroids.device).unsqueeze(0)

    # Exclude NaN blocks (padding / zero-atom blocks)
    has_nan = torch.isnan(block_furthest_dist)  # [n_poses, n_blocks]
    adjacency = adjacency & ~has_nan.unsqueeze(2) & ~has_nan.unsqueeze(1)

    return adjacency
