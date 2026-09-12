import pytest
import torch
from tmol.score.ljlk.potentials import build_compact_block_neighbors


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "shape", [(1, 1), (3, 17), (4, 127), (5, 128), (2, 129), (2, 256)]
)
def test_cpu_compact_neighbors_preserve_chunk_and_pose_boundaries(dtype, shape):
    n_poses, n_blocks = shape
    atom = torch.arange(n_blocks)
    one_pose = torch.stack(
        (2 * (atom % 17), 2 * ((atom // 17) % 5), atom % 3), dim=-1
    ).to(dtype)
    coords = one_pose.expand(n_poses, -1, -1).contiguous()
    types = torch.zeros(shape, dtype=torch.int32)
    types.reshape(-1)[::13] = -1
    offsets = torch.arange(n_poses * n_blocks, dtype=torch.int32)
    block_ids = torch.arange(n_blocks, dtype=torch.int32).repeat(n_poses)
    pose_ids = torch.arange(n_poses, dtype=torch.int32).repeat_interleave(n_blocks)
    rotamer_types = torch.zeros(n_poses * n_blocks, dtype=torch.int32)
    atom_counts = torch.ones(1, dtype=torch.int32)
    left, right = torch.triu_indices(n_blocks, n_blocks)
    within_reach = (coords[:, left] - coords[:, right]).square().sum(-1) < 36
    valid = (types[:, left] >= 0) & (types[:, right] >= 0) & within_reach
    expected = torch.arange(valid.numel(), dtype=torch.int32)[valid.reshape(-1)]
    original_threads = torch.get_num_threads()
    try:
        for threads in (1, 4):
            torch.set_num_threads(threads)
            (neighbors,) = build_compact_block_neighbors(
                coords.reshape(-1, 3),
                offsets,
                types,
                block_ids,
                pose_ids,
                rotamer_types,
                atom_counts,
                6.0,
            )
            assert int(neighbors[0]) == expected.numel()
            torch.testing.assert_close(
                neighbors[1 : expected.numel() + 1], expected, rtol=0, atol=0
            )
    finally:
        torch.set_num_threads(original_threads)
