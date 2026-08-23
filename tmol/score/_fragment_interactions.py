from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FragmentInteractionScores:
    """Per-fragment interactions with an explicitly selected partner."""

    scores: torch.Tensor
    mapping: tuple


def calculate_fragment_interactions(  # noqa: C901
    pose_stack,
    partner_mask,
    *,
    sfxn,
    mapping=None,
    sum_terms=False,
):
    """Return each ligand fragment's interaction with ``partner_mask``.

    The connected multi-block pose is scored once. Fragment-fragment entries
    remain in the block-pair matrix and are not silently assigned to either
    fragment.

    ``sfxn`` is required and must be built from the same ligand-extended
    parameter database used to construct ``pose_stack``.

    Returns:
        :class:`FragmentInteractionScores`. ``scores`` has shape
        ``[n_terms, n_poses, n_fragments]`` or ``[n_poses, n_fragments]`` when
        ``sum_terms`` is true.
    """

    from tmol.pose._split_block_mapping import SplitBlockMapping

    if mapping is None:
        sbm = getattr(pose_stack, "split_block_mapping", None)
    elif isinstance(mapping, SplitBlockMapping):
        sbm = mapping
        mapping = None  # use new path
    else:
        sbm = None  # use legacy path with the provided FragmentedLigandPoseMapping

    if sbm is None and mapping is None:
        raise ValueError("No split-block mapping was supplied or attached to the pose")
    if partner_mask.shape != pose_stack.block_type_ind.shape:
        raise ValueError(
            "partner_mask must have shape [n_poses, max_n_blocks]; got "
            f"{tuple(partner_mask.shape)}"
        )
    if partner_mask.dtype != torch.bool:
        raise TypeError("partner_mask must be a boolean tensor")
    if partner_mask.device != pose_stack.device:
        raise ValueError("partner_mask must be on the same device as pose_stack")
    if sfxn is None:
        raise ValueError(
            "sfxn is required and must use the ligand-extended parameter database"
        )

    # Use split_block_mapping when no explicit mapping was provided.
    # Fragment block indices are the same across all poses (topology is canonical),
    # so pose-0 entries define the column layout for every pose.
    if sbm is not None:
        pose0_entries = sorted(
            (e for e in sbm.entries if e.pose_ind == 0),
            key=lambda e: e.block_ind,
        )
        if not pose0_entries:
            raise ValueError("split_block_mapping has no entries for pose 0")
        expected_block_inds = {e.block_ind for e in pose0_entries}
        for pose_index in range(pose_stack.n_poses):
            pose_block_inds = {
                e.block_ind for e in sbm.entries if e.pose_ind == pose_index
            }
            if pose_block_inds != expected_block_inds:
                raise ValueError(
                    "Fragment mappings must use identical block topology in every pose"
                )
        for entry in sbm.entries:
            if bool(partner_mask[entry.pose_ind, entry.block_ind]):
                raise ValueError("partner_mask must not include ligand fragment blocks")

        scorer = sfxn.render_block_pair_scoring_module(pose_stack)
        block_pair_scores = scorer(pose_stack.coords, sum_terms=False)
        n_terms, n_poses, _, _ = block_pair_scores.shape
        result = torch.zeros(
            (n_terms, n_poses, len(pose0_entries)),
            dtype=block_pair_scores.dtype,
            device=block_pair_scores.device,
        )
        for frag_idx, entry in enumerate(pose0_entries):
            b = entry.block_ind
            result[:, :, frag_idx] = (
                block_pair_scores[:, :, b, :] * partner_mask.unsqueeze(0)
            ).sum(dim=2) + (
                block_pair_scores[:, :, :, b] * partner_mask.unsqueeze(0)
            ).sum(
                dim=2
            )
        if sum_terms:
            result = result.sum(dim=0)
        return FragmentInteractionScores(scores=result, mapping=tuple(pose0_entries))

    # Legacy path: explicit FragmentedLigandPoseMapping passed as mapping=
    fragment_records = tuple(
        sorted(
            (record for record in mapping.blocks if record.pose_index == 0),
            key=lambda record: record.block_index,
        )
    )
    expected_columns = {
        (record.pose_residue_label, record.block_index) for record in fragment_records
    }
    for pose_index in range(pose_stack.n_poses):
        pose_columns = {
            (record.pose_residue_label, record.block_index)
            for record in mapping.blocks
            if record.pose_index == pose_index
        }
        if pose_columns != expected_columns:
            raise ValueError(
                "Fragment mappings must use identical block topology in every pose"
            )
    for record in mapping.blocks:
        if bool(partner_mask[record.pose_index, record.block_index]):
            raise ValueError("partner_mask must not include ligand fragment blocks")
    scorer = sfxn.render_block_pair_scoring_module(pose_stack)
    block_pair_scores = scorer(pose_stack.coords, sum_terms=False)
    n_terms, n_poses, _, _ = block_pair_scores.shape
    result = torch.zeros(
        (n_terms, n_poses, len(fragment_records)),
        dtype=block_pair_scores.dtype,
        device=block_pair_scores.device,
    )
    for fragment_index, record in enumerate(fragment_records):
        block_index = record.block_index
        result[:, :, fragment_index] = (
            block_pair_scores[:, :, block_index, :] * partner_mask.unsqueeze(0)
        ).sum(dim=2) + (
            block_pair_scores[:, :, :, block_index] * partner_mask.unsqueeze(0)
        ).sum(
            dim=2
        )
    if sum_terms:
        result = result.sum(dim=0)
    return FragmentInteractionScores(scores=result, mapping=fragment_records)
