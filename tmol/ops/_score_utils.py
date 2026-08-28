import torch

from tmol.database import ParameterDatabase
from tmol.pose import PoseStack
from tmol.score import ScoreFunction, beta2016_score_function
from tmol.pack import pack_rotamers, PackerTask, PackerPalette
from tmol.pack.rotamer import IncludeCurrentSampler, FixedAAChiSampler
from tmol.pack.rotamer.dunbrack import create_dunbrack_sampler_from_database
from tmol.optimization import run_cart_min
from tmol.types import Tensor

_EINSUM_MIN_BYTES = 512 * 1024 * 1024
_INTERACTION_DISTANCE_WORKSPACE_BYTES = 256 * 1024 * 1024


def _sum_cross_block_scores(
    block_pair_scores: Tensor[torch.float32][:, :, :, :],
    mask: Tensor[torch.bool][:, :],
    other_mask: Tensor[torch.bool][:, :],
    *,
    memory_efficient: bool = False,
) -> Tensor[torch.float32][:, :]:
    """Sum both orientations selected by two block masks.

    Args:
        block_pair_scores: Scores shaped ``[n_terms, n_poses, n_blocks, n_blocks]``.
        mask: First block set shaped ``[n_poses, n_blocks]``.
        other_mask: Second block set with the same shape as ``mask``.
        memory_efficient: Avoid a term-expanded selection for large inputs.

    Returns:
        Per-term, per-pose sums shaped ``[n_terms, n_poses]``.
    """
    cross_mask = (mask.unsqueeze(2) & other_mask.unsqueeze(1)) | (
        other_mask.unsqueeze(2) & mask.unsqueeze(1)
    )
    n_terms, n_poses = block_pair_scores.shape[:2]

    if not memory_efficient:
        expanded_mask = cross_mask.unsqueeze(0).expand(n_terms, -1, -1, -1)
        return block_pair_scores[expanded_mask].view(n_terms, n_poses, -1).sum(dim=2)

    # These reductions avoid the large term-expanded selection above and also
    # support batches in which poses select different numbers of pairs. Their
    # floating-point reduction order can differ from the default path.
    score_bytes = block_pair_scores.numel() * block_pair_scores.element_size()
    if score_bytes >= _EINSUM_MIN_BYTES:
        return torch.einsum(
            "tpij,pij->tp", block_pair_scores, cross_mask.to(block_pair_scores.dtype)
        )
    return block_pair_scores.masked_fill(~cross_mask.unsqueeze(0), 0).sum(dim=(2, 3))


def _sum_single_block_cross_scores(
    block_pair_scores: Tensor[torch.float32][:, :, :, :],
    block_indices: Tensor[torch.int64][:],
    other_mask: Tensor[torch.bool][:, :],
    *,
    sets_are_disjoint: bool = False,
) -> Tensor[torch.float32][:, :]:
    """Sum interactions between one block per pose and selected partners.

    Args:
        block_pair_scores: Scores shaped ``[n_terms, n_poses, n_blocks, n_blocks]``.
        block_indices: One selected block index per pose, shaped ``[n_poses]``.
        other_mask: Partner blocks shaped ``[n_poses, n_blocks]``.
        sets_are_disjoint: Skip diagonal correction when the sets cannot overlap.

    Returns:
        Per-term, per-pose sums shaped ``[n_terms, n_poses]``.
    """
    n_terms, n_poses, n_blocks, _ = block_pair_scores.shape
    gather_rows = block_indices.reshape(1, n_poses, 1, 1).expand(
        n_terms, -1, 1, n_blocks
    )
    gather_columns = block_indices.reshape(1, n_poses, 1, 1).expand(
        n_terms, -1, n_blocks, 1
    )
    rows = block_pair_scores.gather(2, gather_rows).squeeze(2)
    columns = block_pair_scores.gather(3, gather_columns).squeeze(3)
    if sets_are_disjoint:
        return rows.add_(columns).mul_(other_mask.unsqueeze(0)).sum(dim=2)

    diagonal = rows.gather(2, block_indices[None, :, None].expand(n_terms, -1, 1))
    cross_scores = rows.add_(columns).mul_(other_mask.unsqueeze(0)).sum(dim=2)
    diagonal_selected = other_mask.gather(1, block_indices[:, None]).squeeze(1)
    # A block present in both sets contributes its diagonal score once; the row
    # plus column reduction includes it twice, so subtract one copy.
    return cross_scores - (diagonal.squeeze(2) * diagonal_selected.unsqueeze(0))


def calculate_block_pair_ddg(
    pose_stack: PoseStack,
    mask: Tensor[torch.bool][:, :] | Tensor[torch.int64][:],
    mask2: Tensor[torch.bool][:, :] | None = None,
    sfxn: ScoreFunction | None = None,
    sum_terms: bool = True,
    minimize: bool = True,
    pack: bool = False,
    database: ParameterDatabase | None = None,
    return_pose_stack: bool = False,
    *,
    memory_efficient: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, PoseStack]:
    """Score interactions between two block sets in each pose.

    Args:
        pose_stack: Poses to score.
        mask: First set shaped ``[n_poses, n_blocks]``. For single-site scans,
            integer indices shaped ``[n_poses]`` avoid a dense pair-mask reduction.
        mask2: Optional second block set. The complement of ``mask`` is used by
            default.
        sfxn: Score function; defaults to beta2016 on the pose device.
        sum_terms: Sum score terms into one value per pose.
        minimize: Minimize selected and nearby side-chain atoms before scoring.
        pack: Repack selected and adjacent blocks before minimization.
        database: Parameter database used to construct the Dunbrack sampler.
        return_pose_stack: Return the packed/minimized poses with their scores.
        memory_efficient: Use a lower-memory Boolean-mask reduction. Its floating-
            point summation order may differ from the default path.

    Returns:
        Scores shaped ``[n_poses]`` or ``[n_terms, n_poses]``. When requested,
        returns ``(scores, scored_pose_stack)``.
    """
    torch_device = pose_stack.device

    single_block_indices = None
    if mask.ndim == 1 and mask.dtype != torch.bool:
        if torch.is_floating_point(mask) or torch.is_complex(mask):
            raise TypeError("single-block mask indices must have an integer dtype")
        if mask.shape[0] != pose_stack.n_poses:
            raise ValueError("mask must contain one block index per pose")
        single_block_indices = mask.to(device=torch_device, dtype=torch.long)
        mask = torch.zeros_like(pose_stack.block_type_ind, dtype=torch.bool)
        mask.scatter_(1, single_block_indices[:, None], True)

    if sfxn is None:
        sfxn = beta2016_score_function(torch_device)

    if pack:
        # Compute block-level centroids and furthest-atom distances.
        block_centroids, block_furthest_dist = (
            compute_block_centroids_and_furthest_dist(pose_stack)
        )

        # Compute adjacency matrix: blocks i and j are adjacent when the
        # distance between their centroids is less than the sum of their
        # furthest-atom distances plus a constant (default 5.0 A).
        adjacency = compute_block_adjacency(
            block_centroids, block_furthest_dist
        )  # [n_poses, n_blocks, n_blocks]

        # Find blocks adjacent to any masked block.
        # adjacency[i, j, k] True -> block j (masked) and block k are adjacent.
        # nearby_mask[i, k] = any masked block j is adjacent to block k.
        nearby_mask = (mask.unsqueeze(2) & adjacency).any(dim=1)  # [n_poses, n_blocks]

        # Combine: pack residues in the original mask AND adjacent residues.
        pack_mask = mask | nearby_mask  # [n_poses, n_blocks]

        # Build a PackerTask restricted to repacking only the selected residues.
        palette = PackerPalette()

        if database is None:
            database = getattr(sfxn, "_param_db", None)
        dun_sampler = create_dunbrack_sampler_from_database(database, pose_stack.device)

        task = PackerTask(pose_stack, palette)
        fixed_sampler = FixedAAChiSampler()
        task.add_conformer_sampler(dun_sampler)
        task.add_conformer_sampler(fixed_sampler)
        task.add_conformer_sampler(IncludeCurrentSampler())
        task.restrict_to_repacking()

        # Disable packing for blocks that are not in the pack_mask.
        task.disable_packing_by_block_mask(~pack_mask)

        pose_stack = pack_rotamers(pose_stack, sfxn, task)

    if minimize:
        coord_mask = build_coord_mask_for_mask_and_interacting_atoms(pose_stack, mask)
        pose_stack = run_cart_min(pose_stack, sfxn, coord_mask)

    scorer = sfxn.render_block_pair_scoring_module(pose_stack)
    defer_weights = single_block_indices is not None
    block_pair_scores = scorer(
        pose_stack.coords, sum_terms=False, apply_weights=not defer_weights
    )

    other_mask = mask2 if mask2 is not None else ~mask
    if single_block_indices is None:
        ddg_scores = _sum_cross_block_scores(
            block_pair_scores, mask, other_mask, memory_efficient=memory_efficient
        )
    else:
        ddg_scores = _sum_single_block_cross_scores(
            block_pair_scores,
            single_block_indices,
            other_mask,
            sets_are_disjoint=mask2 is None,
        )
        term_weights = scorer.weights[:, 0, 0, 0].unsqueeze(1)
        ddg_scores = ddg_scores * term_weights

    if sum_terms:
        ddg_scores = ddg_scores.sum(dim=0)

    if return_pose_stack:
        return ddg_scores, pose_stack

    return ddg_scores


def res_mask_to_coord_mask(
    pose_stack: PoseStack, mask: Tensor[torch.bool][:, :]
) -> Tensor[torch.bool][:, :]:
    """Expand a block-level selection into an atom coordinate mask.

    Args:
        pose_stack: Poses supplying the block-to-coordinate layout.
        mask: Selected blocks shaped ``[n_poses, n_blocks]``.

    Returns:
        Selected real atoms shaped ``[n_poses, max_n_atoms]``.
    """
    block_for_atom, _, real_atoms = _compact_atom_layout(pose_stack)
    return mask.gather(1, block_for_atom) & real_atoms


def _compact_atom_layout(
    pose_stack: PoseStack,
) -> tuple[
    Tensor[torch.int64][:, :],
    Tensor[torch.int64][:, :],
    Tensor[torch.bool][:, :],
]:
    """Map compact pose coordinates to their blocks and local atom indices.

    Args:
        pose_stack: Poses whose immutable topology defines the mapping.

    Returns:
        Block indices, within-block atom indices, and real-atom flags, each
        shaped ``[n_poses, max_n_atoms]``. The mapping is cached on the pose.
    """
    cache_name = "_score_utils_compact_atom_layout"
    if hasattr(pose_stack, cache_name):
        return getattr(pose_stack, cache_name)

    n_poses, max_n_atoms, _ = pose_stack.coords.shape
    atom_index = (
        torch.arange(max_n_atoms, device=pose_stack.device)
        .expand(n_poses, -1)
        .contiguous()
    )
    # PoseStackBuilder leaves padded block offsets at zero. Replace them with
    # an end sentinel so every row is sorted before the batched search.
    block_starts = pose_stack.block_coord_offset64.masked_fill(
        pose_stack.block_type_ind64 < 0, max_n_atoms
    )
    block_for_atom = torch.searchsorted(
        block_starts.contiguous(), atom_index, right=True
    ).sub_(1)
    block_for_atom.clamp_(0, pose_stack.max_n_blocks - 1)
    block_offset_for_atom = pose_stack.block_coord_offset64.gather(1, block_for_atom)
    real_atoms = pose_stack.real_atoms
    atom_in_block = atom_index.sub(block_offset_for_atom)
    atom_in_block.masked_fill_(~real_atoms, 0)

    layout = (block_for_atom, atom_in_block, real_atoms)
    object.__setattr__(pose_stack, cache_name, layout)
    return layout


def _sidechain_atom_mask_for_block_type(
    pose_stack: PoseStack,
) -> Tensor[torch.bool][:, :]:
    """Return a cached ``[n_types, max_n_block_atoms]`` side-chain mask."""
    pbt = pose_stack.packed_block_types
    cache_name = "_sidechain_atom_mask"
    if hasattr(pbt, cache_name):
        return getattr(pbt, cache_name)

    sidechain_atom_mask = pbt.atom_is_real.clone()
    for block_type_index, residue_type in enumerate(pbt.active_block_types):
        polymer = residue_type.properties.polymer
        if polymer is None:
            continue
        mainchain_atom_indices = [
            residue_type.atom_to_idx[atom] for atom in polymer.mainchain_atoms
        ]
        sidechain_atom_mask[block_type_index, mainchain_atom_indices] = False

    object.__setattr__(pbt, cache_name, sidechain_atom_mask)
    return sidechain_atom_mask


def build_sidechain_coord_mask(
    pose_stack: PoseStack,
) -> Tensor[torch.bool][:, :]:
    """Select side-chain atoms in each pose.

    Args:
        pose_stack: Poses supplying block types and coordinate layout.

    Returns:
        Mask shaped ``[n_poses, max_n_atoms]``. Non-polymers contribute all
        real atoms; polymers exclude their declared main-chain atoms.
    """
    block_for_atom, atom_in_block, real_atoms = _compact_atom_layout(pose_stack)
    block_type_for_atom = pose_stack.block_type_ind64.gather(1, block_for_atom)
    sidechain_atom_mask = _sidechain_atom_mask_for_block_type(pose_stack)
    return (
        sidechain_atom_mask[block_type_for_atom.clamp_min(0), atom_in_block]
        & real_atoms
    )


def compute_block_centroids_and_furthest_dist(
    pose_stack: PoseStack,
) -> tuple[Tensor[torch.float32][:, :, 3], Tensor[torch.float32][:, :]]:
    """Compute each block's centroid and enclosing radius.

    Args:
        pose_stack: Poses to summarize.

    Returns:
        Centroids shaped ``[n_poses, n_blocks, 3]`` and maximum distances
        shaped ``[n_poses, n_blocks]``. Padding blocks contain NaNs.
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


def build_coord_mask_for_mask_and_interacting_atoms(
    pose_stack: PoseStack,
    mask: Tensor[torch.bool][:, :],
    interaction_distance: float = 5.0,
) -> Tensor[torch.bool][:, :]:
    """Select masked blocks and nearby side-chain atoms.

    Args:
        pose_stack: Poses containing coordinates shaped ``[n_poses, n_atoms, 3]``.
        mask: Selected blocks shaped ``[n_poses, n_blocks]``.
        interaction_distance: Maximum atom-to-selected-atom distance in Angstroms.

    Returns:
        Coordinate mask shaped ``[n_poses, n_atoms]``. All atoms in selected
        blocks are included; only side-chain atoms are added from other blocks.
    """
    coord_mask = res_mask_to_coord_mask(pose_stack, mask)
    sidechain_mask = build_sidechain_coord_mask(pose_stack)
    n_masked_atoms = coord_mask.sum(dim=1)
    max_n_masked_atoms = int(n_masked_atoms.max().item())
    if max_n_masked_atoms == 0:
        return coord_mask

    # Pack each pose's selected coordinates into a padded dense tensor so
    # mutants share distance kernels instead of launching once per pose.
    n_poses, max_n_atoms, coordinate_dim = pose_stack.coords.shape
    masked_coords = torch.zeros(
        (n_poses, max_n_masked_atoms, coordinate_dim),
        dtype=pose_stack.coords.dtype,
        device=pose_stack.device,
    )
    masked_coords_are_real = torch.zeros(
        (n_poses, max_n_masked_atoms), dtype=torch.bool, device=pose_stack.device
    )
    masked_pose, masked_atom = torch.nonzero(coord_mask, as_tuple=True)
    first_masked_atom = torch.cumsum(n_masked_atoms, dim=0) - n_masked_atoms
    masked_slot = torch.arange(masked_pose.shape[0], device=pose_stack.device)
    masked_slot -= first_masked_atom[masked_pose]
    masked_coords[masked_pose, masked_slot] = pose_stack.coords[
        masked_pose, masked_atom
    ]
    masked_coords_are_real[masked_pose, masked_slot] = True

    # Bound the temporary [chunk, n_atoms, n_masked_atoms, 3] difference tensor.
    bytes_per_pose = (
        max_n_atoms
        * max_n_masked_atoms
        * (coordinate_dim + 1)
        * pose_stack.coords.element_size()
    )
    poses_per_chunk = max(1, _INTERACTION_DISTANCE_WORKSPACE_BYTES // bytes_per_pose)
    nearby_atoms = torch.empty_like(coord_mask)
    for first_pose in range(0, n_poses, poses_per_chunk):
        last_pose = min(first_pose + poses_per_chunk, n_poses)
        differences = (
            pose_stack.coords[first_pose:last_pose, :, None, :]
            - masked_coords[first_pose:last_pose, None, :, :]
        )
        differences.square_()
        distances = differences.sum(dim=3).sqrt_()
        distances.masked_fill_(
            ~masked_coords_are_real[first_pose:last_pose, None, :], float("inf")
        )
        nearby_atoms[first_pose:last_pose] = distances.amin(dim=2).le(
            interaction_distance
        )

    return coord_mask | (nearby_atoms & pose_stack.real_atoms & sidechain_mask)


def build_coord_mask_for_mask_and_nearby_blocks(
    pose_stack: PoseStack, mask: Tensor[torch.bool][:, :]
) -> Tensor[torch.bool][:, :]:
    """Select masked blocks and side chains of centroid-adjacent blocks.

    Args:
        pose_stack: Poses to select from.
        mask: Selected blocks shaped ``[n_poses, n_blocks]``.

    Returns:
        Coordinate mask shaped ``[n_poses, max_n_atoms]``.
    """
    coord_mask = res_mask_to_coord_mask(pose_stack, mask)
    sidechain_mask = build_sidechain_coord_mask(pose_stack)  # [n_poses, max_n_atoms]

    block_centroids, block_furthest_dist = compute_block_centroids_and_furthest_dist(
        pose_stack
    )
    adjacency = compute_block_adjacency(
        block_centroids, block_furthest_dist
    )  # [n_poses, n_blocks, n_blocks]

    # A block is nearby when any selected block is adjacent to it.
    nearby_mask = (mask.unsqueeze(2) & adjacency).any(dim=1)  # [n_poses, n_blocks]

    coord_mask_nearby = res_mask_to_coord_mask(pose_stack, nearby_mask)
    sidechain_nearby_mask = coord_mask_nearby & sidechain_mask
    return coord_mask | sidechain_nearby_mask


def compute_block_adjacency(
    block_centroids: Tensor[torch.float32][:, :, 3],
    block_furthest_dist: Tensor[torch.float32][:, :],
    constant: float = 5.0,
) -> Tensor[torch.bool][:, :, :]:
    """Find blocks whose enclosing spheres are within a fixed gap.

    Args:
        block_centroids: Centroids shaped ``[n_poses, n_blocks, 3]``.
        block_furthest_dist: Enclosing radii shaped ``[n_poses, n_blocks]``.
        constant: Maximum gap between two enclosing spheres.

    Returns:
        Adjacency shaped ``[n_poses, n_blocks, n_blocks]`` with a false
        diagonal and false entries for padding blocks.
    """
    n_poses, n_blocks, _ = block_centroids.shape

    # Pairwise centroid distance [n_poses, n_blocks, n_blocks]
    diff = block_centroids.unsqueeze(2) - block_centroids.unsqueeze(1)
    centroid_dists = torch.sqrt((diff**2).sum(dim=3))

    # Sum of furthest distances [n_poses, n_blocks, n_blocks]
    dist_sum = block_furthest_dist.unsqueeze(2) + block_furthest_dist.unsqueeze(1)

    # Adjacent if centroid distance < furthest distance sum + constant
    adjacency = centroid_dists < (dist_sum + constant)

    # Exclude self (diagonal)
    adjacency = adjacency & ~torch.eye(
        n_blocks, dtype=torch.bool, device=block_centroids.device
    ).unsqueeze(0)

    # Exclude NaN blocks (padding / zero-atom blocks)
    has_nan = torch.isnan(block_furthest_dist)  # [n_poses, n_blocks]
    adjacency = adjacency & ~has_nan.unsqueeze(2) & ~has_nan.unsqueeze(1)

    return adjacency
