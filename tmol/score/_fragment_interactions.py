from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TYPE_CHECKING

import torch

from tmol.pose._split_block_mapping import SplitBlockEntry, SplitBlockMapping
from tmol.types import Tensor

if TYPE_CHECKING:
    from tmol.pose import PoseStack
    from tmol.score._score_function import ScoreFunction


class _LegacyFragmentRecord(Protocol):
    """Fields consumed from the compatibility ligand mapping."""

    pose_index: int
    block_index: int
    pose_residue_label: int


class _LegacyFragmentMapping(Protocol):
    """Structural type for the compatibility fragmented-ligand mapping."""

    blocks: tuple[_LegacyFragmentRecord, ...]


FragmentRecord = SplitBlockEntry | _LegacyFragmentRecord


_FRAGMENT_REDUCTION_WORKSPACE_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class FragmentInteractionScores:
    """Per-fragment interactions with an explicitly selected partner."""

    scores: torch.Tensor
    mapping: tuple[FragmentRecord, ...]


@dataclass(frozen=True)
class _NormalizedFragmentRecord:
    """Common indexing fields for current and compatibility mappings."""

    pose_index: int
    block_index: int
    column_key: tuple[int, ...]
    source: FragmentRecord


@dataclass(frozen=True)
class _FragmentMappingData:
    """Validated mapping indices cached on one pose topology."""

    source_mapping: SplitBlockMapping | _LegacyFragmentMapping
    fragment_records: tuple[FragmentRecord, ...]
    record_linear_indices: Tensor[torch.int64][:]
    fragment_block_indices: Tensor[torch.int64][:]


def _fragment_mapping_data(
    pose_stack: PoseStack,
    mapping: SplitBlockMapping | _LegacyFragmentMapping,
) -> _FragmentMappingData:
    """Normalize and cache current or compatibility mapping records."""
    cached = getattr(pose_stack, "_fragment_interaction_mapping_data", None)
    tensor_device = pose_stack.block_type_ind.device
    if (
        isinstance(cached, _FragmentMappingData)
        and cached.source_mapping is mapping
        and cached.record_linear_indices.device == tensor_device
    ):
        return cached

    if isinstance(mapping, SplitBlockMapping):
        records = tuple(
            _NormalizedFragmentRecord(
                pose_index=entry.pose_ind,
                block_index=entry.block_ind,
                column_key=(entry.block_ind,),
                source=entry,
            )
            for entry in mapping.entries
        )
        require_pose_zero_record = True
    else:
        records = tuple(
            _NormalizedFragmentRecord(
                pose_index=record.pose_index,
                block_index=record.block_index,
                column_key=(record.pose_residue_label, record.block_index),
                source=record,
            )
            for record in mapping.blocks
        )
        require_pose_zero_record = False

    records_by_pose: list[list[_NormalizedFragmentRecord]] = [
        [] for _ in range(pose_stack.n_poses)
    ]
    for record in records:
        if not (
            0 <= record.pose_index < pose_stack.n_poses
            and 0 <= record.block_index < pose_stack.max_n_blocks
        ):
            raise ValueError(
                "Fragment mapping entry is outside the pose stack: "
                f"pose {record.pose_index}, block {record.block_index}"
            )
        records_by_pose[record.pose_index].append(record)

    pose_zero_records = tuple(
        sorted(records_by_pose[0], key=lambda record: record.block_index)
    )
    if require_pose_zero_record and not pose_zero_records:
        raise ValueError("split_block_mapping has no entries for pose 0")

    expected_columns = {record.column_key for record in pose_zero_records}
    for pose_records in records_by_pose:
        pose_columns = {record.column_key for record in pose_records}
        if len(pose_columns) != len(pose_records):
            raise ValueError("Fragment mapping contains duplicate columns in one pose")
        if pose_columns != expected_columns:
            raise ValueError(
                "Fragment mappings must use identical block topology in every pose"
            )

    data = _FragmentMappingData(
        source_mapping=mapping,
        fragment_records=tuple(record.source for record in pose_zero_records),
        record_linear_indices=torch.tensor(
            [
                record.pose_index * pose_stack.max_n_blocks + record.block_index
                for record in records
            ],
            dtype=torch.int64,
            device=tensor_device,
        ),
        fragment_block_indices=torch.tensor(
            [record.block_index for record in pose_zero_records],
            dtype=torch.int64,
            device=tensor_device,
        ),
    )
    pose_stack._fragment_interaction_mapping_data = data
    return data


def _sum_fragment_partner_scores(
    block_pair_scores: torch.Tensor,
    partner_mask: Tensor[torch.bool][:, :],
    fragment_block_indices: Tensor[torch.int64][:],
    *,
    max_workspace_bytes: int = _FRAGMENT_REDUCTION_WORKSPACE_BYTES,
) -> torch.Tensor:
    """Reduce both block-matrix orientations for many fragments at once.

    The gathered rows and columns are chunked so that their combined temporary
    storage stays near ``max_workspace_bytes``. This keeps the number of CUDA
    launches independent of the number of fragments within each chunk.

    Args:
        block_pair_scores: Weighted scores shaped
            ``[n_terms, n_poses, n_blocks, n_blocks]``.
        partner_mask: Selected partner blocks shaped ``[n_poses, n_blocks]``.
        fragment_block_indices: Fragment block columns shared by every pose,
            shaped ``[n_fragments]``.
        max_workspace_bytes: Target upper bound for the two gathered score
            tensors in one reduction chunk.

    Returns:
        Scores shaped ``[n_terms, n_poses, n_fragments]``.
    """
    n_terms, n_poses, n_blocks, _ = block_pair_scores.shape
    n_fragments = fragment_block_indices.numel()
    if n_fragments == 0 or n_terms == 0 or n_poses == 0 or n_blocks == 0:
        zero_scores = block_pair_scores.sum(dim=(-2, -1)).unsqueeze(-1)
        return zero_scores.expand(n_terms, n_poses, n_fragments)

    bytes_per_fragment = (
        2 * n_terms * n_poses * n_blocks * block_pair_scores.element_size()
    )
    fragments_per_chunk = max(1, max_workspace_bytes // bytes_per_fragment)
    partner_weights = partner_mask.to(dtype=block_pair_scores.dtype)

    partials = []
    for fragment_indices in fragment_block_indices.split(fragments_per_chunk):
        selected_scores = torch.index_select(block_pair_scores, 2, fragment_indices)
        selected_scores.add_(
            torch.index_select(block_pair_scores, 3, fragment_indices).transpose(-1, -2)
        )
        partials.append(torch.einsum("tpfb,pb->tpf", selected_scores, partner_weights))
    return partials[0] if len(partials) == 1 else torch.cat(partials, dim=2)


def calculate_fragment_interactions(
    pose_stack: PoseStack,
    partner_mask: Tensor[torch.bool][:, :],
    *,
    sfxn: ScoreFunction,
    mapping: SplitBlockMapping | _LegacyFragmentMapping | None = None,
    sum_terms: bool = False,
) -> FragmentInteractionScores:
    """Return each ligand fragment's interaction with ``partner_mask``.

    The connected multi-block pose is scored once. Fragment-fragment entries
    remain in the block-pair matrix and are not silently assigned to either
    fragment. ``sfxn`` must use the same ligand-extended parameter database as
    ``pose_stack``.

    Args:
        pose_stack: Connected poses with identical fragment block layouts.
        partner_mask: Partner blocks shaped ``[n_poses, max_n_blocks]``. The
            mask must exclude every ligand fragment block.
        sfxn: Score function built from the pose's parameter database.
        mapping: Optional split-block mapping. By default the mapping attached
            to ``pose_stack`` is used. Legacy fragmented-ligand mappings remain
            accepted for compatibility.
        sum_terms: Sum the score-term dimension when true.

    Returns:
        Fragment scores shaped ``[n_terms, n_poses, n_fragments]``, or
        ``[n_poses, n_fragments]`` when ``sum_terms`` is true, plus the pose-zero
        mapping records that define the fragment columns.

    Raises:
        TypeError: If ``partner_mask`` is not Boolean.
        ValueError: If the mask, score function, or mapping is incompatible
            with the poses.
    """
    if mapping is None:
        mapping = getattr(pose_stack, "split_block_mapping", None)
    if mapping is None:
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

    mapping_data = _fragment_mapping_data(pose_stack, mapping)
    if bool(partner_mask.flatten()[mapping_data.record_linear_indices].any()):
        raise ValueError("partner_mask must not include ligand fragment blocks")

    scorer = sfxn.render_block_pair_scoring_module(pose_stack)
    block_pair_scores = scorer(pose_stack.coords, sum_terms=False)
    result = _sum_fragment_partner_scores(
        block_pair_scores,
        partner_mask,
        mapping_data.fragment_block_indices,
    )
    if sum_terms:
        result = result.sum(dim=0)
    return FragmentInteractionScores(
        scores=result,
        mapping=mapping_data.fragment_records,
    )
