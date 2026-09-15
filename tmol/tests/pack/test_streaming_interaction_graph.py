import itertools

import pytest
import torch

from tmol.pack.compiled import (
    accumulate_interaction_graph_entries,
    build_interaction_graph,
    finalize_interaction_graph_topology,
    initialize_interaction_graph_topology,
    note_interaction_graph_topology,
    pack_anneal,
)
from tmol.pack._pack_rotamers import _build_streaming_interaction_graph
from tmol.tests import requires_cuda


def graph_metadata(counts, device):
    n_rots_for_block = torch.tensor(counts, dtype=torch.int64, device=device)
    n_rots_for_pose = n_rots_for_block.sum(dim=1)
    rot_offset_for_pose = torch.cat(
        (
            torch.zeros(1, dtype=torch.int64, device=device),
            n_rots_for_pose.cumsum(0)[:-1],
        )
    )
    rot_offset_for_block = torch.zeros_like(n_rots_for_block)
    pose_for_rot = []
    block_ind_for_rot = []
    offset = 0
    for pose, pose_counts in enumerate(counts):
        for block, count in enumerate(pose_counts):
            rot_offset_for_block[pose, block] = offset
            pose_for_rot.extend([pose] * count)
            block_ind_for_rot.extend([block] * count)
            offset += count
    return (
        n_rots_for_pose,
        rot_offset_for_pose,
        n_rots_for_block,
        rot_offset_for_block,
        torch.tensor(pose_for_rot, dtype=torch.int64, device=device),
        torch.zeros(offset, dtype=torch.int64, device=device),
        torch.tensor(block_ind_for_rot, dtype=torch.int32, device=device),
    )


def score_entries(metadata, edges, *, duplicate=False):
    (
        _,
        _,
        n_rots_for_block,
        rot_offset_for_block,
        _,
        _,
        _,
    ) = metadata
    entries = []
    values = []
    for pose, pose_counts in enumerate(n_rots_for_block.tolist()):
        for block, count in enumerate(pose_counts):
            offset = int(rot_offset_for_block[pose, block])
            for local in range(count):
                entries.append((pose, offset + local, offset + local))
                values.append(0.5 + pose + block * 0.0625 + local * 0.0078125)
    for pose, block1, block2 in edges:
        offset1 = int(rot_offset_for_block[pose, block1])
        offset2 = int(rot_offset_for_block[pose, block2])
        count1 = int(n_rots_for_block[pose, block1])
        count2 = int(n_rots_for_block[pose, block2])
        for local1, local2 in itertools.product(range(count1), range(count2)):
            entries.append((pose, offset1 + local1, offset2 + local2))
            values.append(
                1.0
                + pose
                + block1 * 0.25
                + block2 * 0.125
                + local1 * 0.03125
                + local2 * 0.015625
            )
    if duplicate and entries:
        entries.append(entries[0])
        values.append(2.0)
    if entries:
        indices = torch.tensor(entries, dtype=torch.int32).T.contiguous()
    else:
        indices = torch.empty((3, 0), dtype=torch.int32)
    return indices.to(n_rots_for_block.device), torch.tensor(
        values, dtype=torch.float32, device=n_rots_for_block.device
    )


def staged_graph(metadata, term_entries, chunk_size):
    device = metadata[0].device
    empty_indices = torch.empty((3, 0), dtype=torch.int32, device=device)
    empty_values = torch.empty(0, dtype=torch.float32, device=device)
    base = list(
        build_interaction_graph(
            False,
            chunk_size,
            1,
            *metadata,
            empty_indices,
            empty_values,
            False,
        )
    )
    topology = list(
        initialize_interaction_graph_topology(
            chunk_size, metadata[2], base[4], empty_values
        )
    )
    for indices, values in term_entries:
        topology[0], topology[1] = note_interaction_graph_topology(
            chunk_size,
            metadata[2],
            metadata[3],
            metadata[6],
            topology[2],
            topology[3],
            topology[4],
            topology[5],
            topology[0],
            topology[1],
            indices,
            values,
        )
    base[11:16] = finalize_interaction_graph_topology(
        chunk_size,
        base[4],
        topology[3],
        topology[4],
        topology[5],
        topology[0],
        topology[1],
        empty_values,
    )
    for indices, values in term_entries:
        base[9], base[10], base[15] = accumulate_interaction_graph_entries(
            chunk_size,
            metadata[2],
            metadata[3],
            metadata[6],
            topology[2],
            base[7],
            base[4],
            base[5],
            base[11],
            base[12],
            base[13],
            base[14],
            base[9],
            base[10],
            base[15],
            indices,
            values,
        )
    return tuple(base)


@pytest.mark.parametrize(
    "counts,edges",
    [
        ([[3, 2, 1, 1]], [(0, 0, 1), (0, 0, 2), (0, 1, 3), (0, 2, 3)]),
        (
            [[2, 3, 2, 1]],
            [
                (0, first, second)
                for first in range(4)
                for second in range(first + 1, 4)
            ],
        ),
        (
            [[2, 4, 1, 0], [3, 2, 0, 1]],
            [(0, 0, 1), (0, 0, 2), (1, 0, 1), (1, 1, 3)],
        ),
        ([[2, 3, 1, 0]], []),
    ],
    ids=["sparse", "dense", "jagged", "empty"],
)
@pytest.mark.parametrize("duplicate", [False, True], ids=["unique", "duplicate"])
def test_streaming_graph_matches_existing(torch_device, counts, edges, duplicate):
    metadata = graph_metadata(counts, torch_device)
    indices, values = score_entries(metadata, edges, duplicate=duplicate)
    split = indices.shape[1] // 2
    term_entries = [
        (indices[:, :split], values[:split]),
        (indices[:, split:], values[split:]),
    ]
    streamed = staged_graph(metadata, term_entries, chunk_size=2)

    coalesced = torch.sparse_coo_tensor(
        indices.to(torch.int64),
        values,
        size=(
            len(counts),
            int(metadata[0].sum()),
            int(metadata[0].sum()),
        ),
    ).coalesce()
    existing = build_interaction_graph(
        False,
        2,
        1,
        *metadata,
        coalesced.indices().to(torch.int32),
        coalesced.values(),
        False,
    )

    for index, (actual, expected) in enumerate(zip(streamed, existing)):
        if index in (9, 10, 15):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        else:
            assert torch.equal(actual, expected)

    # Every CSR edge has its reverse and rows retain ascending index order.
    row_offsets, neighbors = streamed[11], streamed[12]
    max_blocks = streamed[4].shape[1]
    for pose in range(len(counts)):
        for block in range(max_blocks):
            row = pose * max_blocks + block
            row_neighbors = neighbors[row_offsets[row] : row_offsets[row + 1]]
            assert torch.equal(row_neighbors, row_neighbors.sort().values)
            for neighbor in row_neighbors.tolist():
                reverse_row = pose * max_blocks + neighbor
                assert (
                    block
                    in neighbors[
                        row_offsets[reverse_row] : row_offsets[reverse_row + 1]
                    ].tolist()
                )


def annealer_inputs(graph, chunk_size):
    return (
        graph[0].item(),
        graph[1].to(torch.int32),
        graph[2].to(torch.int32),
        graph[3].to(torch.int32),
        graph[4].to(torch.int32),
        graph[5].to(torch.int32),
        graph[6].to(torch.int32),
        chunk_size,
        graph[11],
        graph[12],
        graph[13],
        graph[14],
        graph[10],
        graph[15],
    )


@requires_cuda
def test_streaming_graph_preserves_assignments_and_rng_advancement():
    device = torch.device("cuda")
    counts = [[3, 2, 2, 1]]
    edges = [(0, first, second) for first in range(4) for second in range(first + 1, 4)]
    metadata = graph_metadata(counts, device)
    indices, values = score_entries(metadata, edges, duplicate=True)
    split = indices.shape[1] // 2
    streamed = staged_graph(
        metadata,
        [(indices[:, :split], values[:split]), (indices[:, split:], values[split:])],
        chunk_size=2,
    )
    coalesced = torch.sparse_coo_tensor(
        indices.to(torch.int64),
        values,
        size=(1, int(metadata[0].sum()), int(metadata[0].sum())),
    ).coalesce()
    existing = build_interaction_graph(
        False,
        2,
        1,
        *metadata,
        coalesced.indices().to(torch.int32),
        coalesced.values(),
        False,
    )

    torch.manual_seed(520)
    expected_scores, expected_assignments = pack_anneal(*annealer_inputs(existing, 2))
    expected_rng_state = torch.cuda.get_rng_state()
    torch.manual_seed(520)
    actual_scores, actual_assignments = pack_anneal(*annealer_inputs(streamed, 2))
    actual_rng_state = torch.cuda.get_rng_state()

    assert torch.equal(actual_scores, expected_scores)
    assert torch.equal(actual_assignments, expected_assignments)
    assert torch.equal(actual_rng_state, expected_rng_state)


@requires_cuda
def test_streaming_graph_memory_is_bounded_by_one_layout():
    """Do not retain every large duplicate term layout across either pass."""
    device = torch.device("cuda")
    counts = [[2] * 128]
    metadata = graph_metadata(counts, device)
    edge_entries, edge_values = score_entries(
        metadata, [(0, block, block + 1) for block in range(127)]
    )
    repeats = 4000
    n_terms = 6
    layout_bytes = repeats * (
        edge_entries.numel() * edge_entries.element_size()
        + edge_values.numel() * edge_values.element_size()
    )

    class SyntheticScorer:
        def _iter_weighted_sparse_entries(self, coords, *, retain_shared_dispatch=True):
            assert not retain_shared_dispatch
            for term in range(n_terms):
                yield (
                    term,
                    edge_entries.repeat(1, repeats),
                    edge_values.repeat(repeats),
                )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    graph = _build_streaming_interaction_graph(
        SyntheticScorer(),
        torch.empty(0, device=device),
        32,
        (1, *metadata),
        False,
    )
    torch.cuda.synchronize()
    peak_delta = torch.cuda.max_memory_allocated() - allocated_before

    assert graph[15].numel() > 0
    assert peak_delta < 4 * layout_bytes
