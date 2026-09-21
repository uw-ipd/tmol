import pytest
import torch


@pytest.fixture
def ext():
    from tmol._load_ext import load_module

    return load_module(
        __name__,
        __file__,
        ["test_cpu.cpp", "test.pybind.cpp", "test_cuda.cu"],
        "tmol.tests.score.common.device_operations._ext",
    )


# ---------------------------------------------------------------------------
# forall: dst[i] = src[i] + i
# ---------------------------------------------------------------------------


def test_forall(ext, torch_device):
    src = torch.full((6,), 5, dtype=torch.int32, device=torch_device)
    result = ext.test_forall(src)
    expected = torch.tensor([5, 6, 7, 8, 9, 10], dtype=torch.int32)
    assert torch.equal(result.cpu(), expected)


def test_forall_large(ext, torch_device):
    # N=1000 spans multiple CTAs (launch_t has NT=128, VT=2 -> 256 per CTA).
    # src=zeros, so dst[i] = 0 + i = i.
    N = 1000
    src = torch.zeros(N, dtype=torch.int32, device=torch_device)
    result = ext.test_forall(src)
    expected = torch.arange(N, dtype=torch.int32)
    assert torch.equal(result.cpu(), expected)


def test_forall_independent(ext, torch_device):
    src = torch.zeros(1000, dtype=torch.int32, device=torch_device)
    result = ext.test_forall_independent(src)
    assert torch.equal(result.cpu(), torch.arange(1000, dtype=torch.int32))


def test_forall_independent_large(ext, torch_device):
    src = torch.zeros(100_000, dtype=torch.int32, device=torch_device)
    result = ext.test_forall_independent(src)
    assert torch.equal(result.cpu(), torch.arange(100_000, dtype=torch.int32))


def test_forall_grouped(ext, torch_device):
    src = torch.zeros((8, 17), dtype=torch.int32, device=torch_device)
    result = ext.test_forall_grouped(src)
    expected = torch.arange(src.numel(), dtype=torch.int32).reshape_as(src.cpu())
    assert torch.equal(result.cpu(), expected)


# ---------------------------------------------------------------------------
# foreach_combination_triple: dst[i][j][k] = src[i][j][k] + 1
# ---------------------------------------------------------------------------


def test_foreach_combination_triple(ext, torch_device):
    src = torch.zeros(2, 3, 4, dtype=torch.int32, device=torch_device)
    result = ext.test_foreach_combination_triple(src)
    expected = torch.ones(2, 3, 4, dtype=torch.int32)
    assert torch.equal(result.cpu(), expected)


def test_foreach_combination_triple_large(ext, torch_device):
    # 5 x 6 x 40 = 1200 total, spanning multiple CTAs.
    # src=zeros, so dst[i][j][k] = 0 + 1 = 1 everywhere.
    src = torch.zeros(5, 6, 40, dtype=torch.int32, device=torch_device)
    result = ext.test_foreach_combination_triple(src)
    expected = torch.ones(5, 6, 40, dtype=torch.int32)
    assert torch.equal(result.cpu(), expected)


# ---------------------------------------------------------------------------
# foreach_workgroup: dst[wg] = src[wg] + wg
# ---------------------------------------------------------------------------


def test_foreach_workgroup(ext, torch_device):
    src = torch.full((5,), 10, dtype=torch.int32, device=torch_device)
    result = ext.test_foreach_workgroup(src)
    expected = torch.tensor([10, 11, 12, 13, 14], dtype=torch.int32)
    assert torch.equal(result.cpu(), expected)


def test_foreach_workgroup_large(ext, torch_device):
    # 500 workgroups = 500 CTAs, each writing one element.
    # src=zeros, so dst[wg] = 0 + wg = wg.
    N = 500
    src = torch.zeros(N, dtype=torch.int32, device=torch_device)
    result = ext.test_foreach_workgroup(src)
    expected = torch.arange(N, dtype=torch.int32)
    assert torch.equal(result.cpu(), expected)


def test_foreach_independent_workgroup(ext, torch_device):
    src = torch.zeros(1000, dtype=torch.int32, device=torch_device)
    result = ext.test_foreach_independent_workgroup(src)
    assert torch.equal(result.cpu(), torch.arange(1000, dtype=torch.int32))


def test_foreach_pose_workgroup(ext, torch_device):
    src = torch.zeros(8, 17, dtype=torch.int32, device=torch_device)
    result = ext.test_foreach_pose_workgroup(src)
    expected = torch.arange(8 * 17, dtype=torch.int32).reshape(8, 17)
    assert torch.equal(result.cpu(), expected)


# ---------------------------------------------------------------------------
# checked dispatch size (no allocation)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("count", [2**31 - 2, 2**31 - 1])
def test_checked_dispatch_size_accepts_int32_boundary(ext, count):
    assert ext.test_checked_dispatch_size(count) == count


@pytest.mark.parametrize("count", [-1, 2**31])
def test_checked_dispatch_size_rejects_out_of_range(ext, count):
    with pytest.raises(OverflowError, match="signed 32-bit native dispatch limit"):
        ext.test_checked_dispatch_size(count)


@pytest.mark.parametrize(
    "lhs,rhs,expected",
    [(1, 2**31 - 1, 2**31 - 1), (46340, 46340, 2147395600)],
)
def test_checked_dispatch_product_accepts_int32_boundary(ext, lhs, rhs, expected):
    assert ext.test_checked_dispatch_product(lhs, rhs) == expected


@pytest.mark.parametrize("lhs,rhs", [(46341, 46341), (2**63 - 1, 2)])
def test_checked_dispatch_product_rejects_overflow(ext, lhs, rhs):
    with pytest.raises(OverflowError, match="(signed 32-bit|signed 64-bit)"):
        ext.test_checked_dispatch_product(lhs, rhs)


@pytest.mark.parametrize(
    "count,include_diagonal,expected",
    [
        (0, True, 0),
        (0, False, 0),
        (1, True, 1),
        (1, False, 0),
        (2, True, 3),
        (2, False, 1),
        (65535, True, 2147450880),
        (65536, False, 2147450880),
    ],
)
def test_checked_triangular_size_accepts_int32_boundary(
    ext, count, include_diagonal, expected
):
    assert ext.test_checked_triangular_size(count, include_diagonal) == expected


@pytest.mark.parametrize(
    "count,include_diagonal",
    [
        (-1, True),
        (65536, True),
        (65537, False),
        (2**63 - 1, True),
        (2**63 - 1, False),
    ],
)
def test_checked_triangular_size_rejects_overflow(ext, count, include_diagonal):
    with pytest.raises(OverflowError, match="(negative|signed (32|64)-bit)"):
        ext.test_checked_triangular_size(count, include_diagonal)


@pytest.mark.parametrize(
    "linear_index,dimension,expected",
    [
        (0, 4, (0, 1)),
        (2, 4, (0, 3)),
        (3, 4, (1, 2)),
        (5, 4, (2, 3)),
        (0, 65536, (0, 1)),
        (65534, 65536, (0, 65535)),
        (65535, 65536, (1, 2)),
        (2147450879, 65536, (65534, 65535)),
    ],
)
def test_upper_triangle_indices_avoid_intermediate_overflow(
    ext, linear_index, dimension, expected
):
    assert ext.test_upper_triangle_indices(linear_index, dimension) == expected


def test_dispatch_ceiling_division_does_not_overflow(ext):
    assert ext.test_safe_div_up(2**31 - 1, 32) == 2**26


# ---------------------------------------------------------------------------
# scan inclusive: cumulative prefix sum (inclusive)
# ---------------------------------------------------------------------------


def test_scan_inclusive(ext, torch_device):
    src = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32, device=torch_device)
    result = ext.test_scan_inclusive(src)
    expected = torch.tensor([1, 3, 6, 10, 15], dtype=torch.int32)
    assert torch.equal(result.cpu(), expected)


def test_scan_inclusive_large(ext, torch_device):
    # N=1000 ones; inclusive cumsum = [1, 2, ..., 1000].
    N = 1000
    src = torch.ones(N, dtype=torch.int32, device=torch_device)
    result = ext.test_scan_inclusive(src)
    expected = torch.arange(1, N + 1, dtype=torch.int32)
    assert torch.equal(result.cpu(), expected)


# ---------------------------------------------------------------------------
# scan exclusive: cumulative prefix sum (exclusive, identity=0)
# ---------------------------------------------------------------------------


def test_scan_exclusive(ext, torch_device):
    # dst[0] is pre-initialized to 0 (identity) inside the test function.
    src = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32, device=torch_device)
    result = ext.test_scan_exclusive(src)
    expected = torch.tensor([0, 1, 3, 6, 10], dtype=torch.int32)
    assert torch.equal(result.cpu(), expected)


def test_scan_exclusive_large(ext, torch_device):
    # N=1000 ones; exclusive cumsum = [0, 1, 2, ..., 999].
    N = 1000
    src = torch.ones(N, dtype=torch.int32, device=torch_device)
    result = ext.test_scan_exclusive(src)
    expected = torch.arange(N, dtype=torch.int32)
    assert torch.equal(result.cpu(), expected)


# ---------------------------------------------------------------------------
# scan_and_return_total inclusive
# ---------------------------------------------------------------------------


def test_scan_and_return_total_inclusive(ext, torch_device):
    src = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device=torch_device)
    dst, total = ext.test_scan_and_return_total_inclusive(src)
    assert torch.equal(dst.cpu(), torch.tensor([1, 3, 6, 10], dtype=torch.int32))
    assert total == 10


def test_scan_and_return_total_inclusive_large(ext, torch_device):
    N = 1000
    src = torch.ones(N, dtype=torch.int32, device=torch_device)
    dst, total = ext.test_scan_and_return_total_inclusive(src)
    assert torch.equal(dst.cpu(), torch.arange(1, N + 1, dtype=torch.int32))
    assert total == N


# ---------------------------------------------------------------------------
# scan_and_return_total exclusive
# ---------------------------------------------------------------------------


def test_scan_and_return_total_exclusive(ext, torch_device):
    # dst[0] is pre-initialized to 0 (identity) inside the test function.
    src = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device=torch_device)
    dst, total = ext.test_scan_and_return_total_exclusive(src)
    assert torch.equal(dst.cpu(), torch.tensor([0, 1, 3, 6], dtype=torch.int32))
    assert total == 10


def test_scan_and_return_total_exclusive_large(ext, torch_device):
    N = 1000
    src = torch.ones(N, dtype=torch.int32, device=torch_device)
    dst, total = ext.test_scan_and_return_total_exclusive(src)
    assert torch.equal(dst.cpu(), torch.arange(N, dtype=torch.int32))
    assert total == N


# ---------------------------------------------------------------------------
# reduce
# ---------------------------------------------------------------------------


def test_reduce(ext, torch_device):
    src = torch.tensor([3, 7, 2, 1, 5], dtype=torch.int32, device=torch_device)
    assert ext.test_reduce(src) == 18


def test_reduce_large(ext, torch_device):
    N = 1000
    src = torch.ones(N, dtype=torch.int32, device=torch_device)
    assert ext.test_reduce(src) == N


def _reference_rot_neighbor_indices(
    pose_stack_block_type,
    block_spheres,
    n_rots_for_block,
    rot_offset_for_block,
    rot_spheres,
    lockstep_group_for_block,
    reach,
):
    """Reproduce the former dense block-neighbor/count/offset ordering."""
    result = []
    n_poses, max_n_blocks = pose_stack_block_type.shape
    for pose in range(n_poses):
        for block1 in range(max_n_blocks):
            for block2 in range(block1, max_n_blocks):
                if (
                    pose_stack_block_type[pose, block1] < 0
                    or pose_stack_block_type[pose, block2] < 0
                ):
                    continue
                delta = (
                    block_spheres[pose, block1, :3] - block_spheres[pose, block2, :3]
                )
                threshold = (
                    block_spheres[pose, block1, 3]
                    + block_spheres[pose, block2, 3]
                    + reach
                )
                if not delta.square().sum() < threshold * threshold:
                    continue

                nrot1 = int(n_rots_for_block[pose, block1])
                nrot2 = int(n_rots_for_block[pose, block2])
                offset1 = int(rot_offset_for_block[pose, block1])
                offset2 = int(rot_offset_for_block[pose, block2])
                if nrot1 <= 0 or nrot2 <= 0 or offset1 < 0 or offset2 < 0:
                    continue
                group1 = int(lockstep_group_for_block[pose, block1])
                group2 = int(lockstep_group_for_block[pose, block2])
                if block1 == block2:
                    result.extend(
                        (pose, offset1 + i, offset1 + i) for i in range(nrot1)
                    )
                elif group1 >= 0 and group1 == group2:
                    result.extend(
                        (pose, offset1 + i, offset2 + i)
                        for i in range(min(nrot1, nrot2))
                    )
                else:
                    for i in range(nrot1):
                        for j in range(nrot2):
                            rot_delta = (
                                rot_spheres[offset1 + i, :3]
                                - rot_spheres[offset2 + j, :3]
                            )
                            rot_threshold = (
                                rot_spheres[offset1 + i, 3]
                                + rot_spheres[offset2 + j, 3]
                                + reach
                            )
                            if rot_delta.square().sum() < rot_threshold * rot_threshold:
                                result.append((pose, offset1 + i, offset2 + j))
    if not result:
        return torch.empty((3, 0), dtype=torch.int32)
    return torch.tensor(result, dtype=torch.int32).T.contiguous()


@pytest.mark.parametrize("lockstep", [False, True])
def test_rot_neighbor_indices_match_dense_reference(ext, torch_device, lockstep):
    """Keep canonical dispatch bytes across sparse, distant, and lockstep pairs."""
    device = torch_device
    block_types = torch.tensor(
        [[0, 1, 2, -1, 3], [0, 1, 2, 3, -1]],
        dtype=torch.int32,
        device=device,
    )
    n_rots = torch.tensor(
        [[2, 3, 0, 0, 2], [2, 2, 1, 1, 0]],
        dtype=torch.int32,
        device=device,
    )
    offsets = torch.tensor(
        [[0, 2, -1, -1, 5], [7, 9, 11, 12, -1]],
        dtype=torch.int32,
        device=device,
    )
    block_spheres = torch.tensor(
        [
            [[0, 0, 0, 1], [1, 0, 0, 1], [4, 0, 0, 1], [0, 0, 0, 0], [20, 0, 0, 1]],
            [[0, 0, 0, 1], [1, 0, 0, 1], [5, 0, 0, 1], [6, 0, 0, 1], [0, 0, 0, 0]],
        ],
        dtype=torch.float32,
        device=device,
    )
    rot_spheres = torch.tensor(
        [
            [0, 0, 0, 0.4],
            [8, 0, 0, 0.4],
            [0.5, 0, 0, 0.4],
            [8.5, 0, 0, 0.4],
            [30, 0, 0, 0.4],
            [20, 0, 0, 0.4],
            [22, 0, 0, 0.4],
            [0, 0, 0, 0.4],
            [9, 0, 0, 0.4],
            [0.5, 0, 0, 0.4],
            [9.5, 0, 0, 0.4],
            [100, 0, 0, 0.1],
            [-100, 0, 0, 0.1],
        ],
        dtype=torch.float32,
        device=device,
    )
    groups = torch.full_like(n_rots, -1)
    if lockstep:
        groups[0, :2] = 7
        groups[1, 2:4] = 9
    reach = 0.5

    actual = ext.test_rot_neighbor_indices_from_block_spheres(
        block_types,
        block_spheres,
        n_rots,
        offsets,
        rot_spheres,
        groups,
        reach,
    )
    expected = _reference_rot_neighbor_indices(
        block_types.cpu(),
        block_spheres.cpu(),
        n_rots.cpu(),
        offsets.cpu(),
        rot_spheres.cpu(),
        groups.cpu(),
        reach,
    )
    assert actual.cpu().numpy().tobytes() == expected.numpy().tobytes()


# ---------------------------------------------------------------------------
# load_balancing_search
# ---------------------------------------------------------------------------


def test_load_balancing_search(ext, torch_device):
    # Three generators producing [2, 3, 1] work units (total = 6).
    # Exclusive prefix sum of [2, 3, 1] is [0, 2, 5].
    # Each work unit maps to its generator: [0, 0, 1, 1, 1, 2].
    exc_offsets = torch.tensor([0, 2, 5], dtype=torch.int32, device=torch_device)
    result = ext.test_load_balancing_search(exc_offsets, 6)
    expected = torch.tensor([0, 0, 1, 1, 1, 2], dtype=torch.int32)
    assert torch.equal(result.cpu(), expected)


def test_load_balancing_search_large(ext, torch_device):
    # 200 generators each producing 5 work units; total = 1000.
    # Exclusive prefix sum: [0, 5, 10, ..., 995].
    # Each generator i maps to 5 consecutive work units, so expected[j] = j//5.
    n_generators = 200
    units_per_generator = 5
    n_total = n_generators * units_per_generator
    exc_offsets = torch.arange(
        0, n_total, units_per_generator, dtype=torch.int32, device=torch_device
    )
    result = ext.test_load_balancing_search(exc_offsets, n_total)
    expected = torch.arange(n_generators, dtype=torch.int32).repeat_interleave(
        units_per_generator
    )
    assert torch.equal(result.cpu(), expected)


# ---------------------------------------------------------------------------
# segmented scan inclusive
# ---------------------------------------------------------------------------


def test_segmented_scan_inclusive(ext, torch_device):
    # Three segments of length 2 each; inclusive prefix sum within each.
    # seg0: [1,2] -> [1,3],  seg1: [3,4] -> [3,7],  seg2: [5,6] -> [5,11].
    src = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int32, device=torch_device)
    seg_starts = torch.tensor([0, 2, 4], dtype=torch.int32, device=torch_device)
    result = ext.test_segmented_scan_inclusive(src, seg_starts)
    expected = torch.tensor([1, 3, 3, 7, 5, 11], dtype=torch.int32)
    assert torch.equal(result.cpu(), expected)


def test_segmented_scan_inclusive_large(ext, torch_device):
    # 50 segments x 20 ones = 1000 elements total.
    # Inclusive cumsum within each segment of ones: [1, 2, ..., 20].
    n_segs = 50
    seg_len = 20
    n = n_segs * seg_len
    src = torch.ones(n, dtype=torch.int32, device=torch_device)
    seg_starts = torch.arange(0, n, seg_len, dtype=torch.int32, device=torch_device)
    result = ext.test_segmented_scan_inclusive(src, seg_starts)
    expected = torch.arange(1, seg_len + 1, dtype=torch.int32).repeat(n_segs)
    assert torch.equal(result.cpu(), expected)


# ---------------------------------------------------------------------------
# segmented scan exclusive
# ---------------------------------------------------------------------------


def test_segmented_scan_exclusive(ext, torch_device):
    # Same segments as above; exclusive prefix sum (identity=0) within each.
    # seg0: [1,2] -> [0,1],  seg1: [3,4] -> [0,3],  seg2: [5,6] -> [0,5].
    src = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int32, device=torch_device)
    seg_starts = torch.tensor([0, 2, 4], dtype=torch.int32, device=torch_device)
    result = ext.test_segmented_scan_exclusive(src, seg_starts)
    expected = torch.tensor([0, 1, 0, 3, 0, 5], dtype=torch.int32)
    assert torch.equal(result.cpu(), expected)


def test_segmented_scan_exclusive_large(ext, torch_device):
    # 50 segments x 20 ones = 1000 elements total.
    # Exclusive cumsum within each segment of ones: [0, 1, ..., 19].
    n_segs = 50
    seg_len = 20
    n = n_segs * seg_len
    src = torch.ones(n, dtype=torch.int32, device=torch_device)
    seg_starts = torch.arange(0, n, seg_len, dtype=torch.int32, device=torch_device)
    result = ext.test_segmented_scan_exclusive(src, seg_starts)
    expected = torch.arange(0, seg_len, dtype=torch.int32).repeat(n_segs)
    assert torch.equal(result.cpu(), expected)
