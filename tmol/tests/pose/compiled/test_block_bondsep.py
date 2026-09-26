import torch
import pytest

from tmol.pose.compiled import block_bondsep

MAX_SIG_BOND_SEPARATION = 6


def gather_with_torch(pconn_matrix, pconn_offsets, block_n_conn, max_n_conn):
    """Index the shortest-path matrix with plain torch ops.

    The kernel replaces this; keeping it here is what makes the two comparable.
    """
    n_poses, max_n_blocks = block_n_conn.shape
    max_n_pconn = pconn_matrix.shape[1]

    bconn_ind = torch.arange(max_n_conn, dtype=torch.int64, device=pconn_matrix.device)
    real_bconn = bconn_ind[None, None, :] < block_n_conn[:, :, None]
    pconn_for_bconn = torch.where(
        real_bconn,
        pconn_offsets[:, :, None] + bconn_ind,
        0,
    ).flatten(1)

    n_padded_bconn = pconn_for_bconn.shape[1]
    pconn_rows = torch.gather(
        pconn_matrix,
        1,
        pconn_for_bconn[:, :, None].expand(n_poses, n_padded_bconn, max_n_pconn),
    )
    out = torch.gather(
        pconn_rows,
        2,
        pconn_for_bconn[:, None, :].expand(n_poses, n_padded_bconn, n_padded_bconn),
    ).reshape(n_poses, max_n_blocks, max_n_conn, max_n_blocks, max_n_conn)
    out = out.permute(0, 1, 3, 2, 4)
    out.masked_fill_(~real_bconn[:, :, None, :, None], MAX_SIG_BOND_SEPARATION)
    out.masked_fill_(~real_bconn[:, None, :, None, :], MAX_SIG_BOND_SEPARATION)
    return out.contiguous()


def random_case(n_poses, max_n_blocks, max_n_conn, device, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    block_n_conn = torch.randint(
        0, max_n_conn + 1, (n_poses, max_n_blocks), generator=generator
    ).to(torch.int32)
    # Each pose lays its own blocks' connections end to end, so the offsets that
    # index its matrix restart at zero for every pose.
    counts = block_n_conn.to(torch.int64)
    offsets = torch.zeros_like(counts)
    offsets[:, 1:] = counts.cumsum(1)[:, :-1]
    max_n_pconn = max(int(counts.sum(1).max()), 1)
    pconn_matrix = torch.randint(
        0,
        MAX_SIG_BOND_SEPARATION + 1,
        (n_poses, max_n_pconn, max_n_pconn),
        generator=generator,
    ).to(torch.int32)
    return (
        pconn_matrix.to(device).contiguous(),
        offsets.to(device).contiguous(),
        block_n_conn.to(device).contiguous(),
    )


@pytest.mark.parametrize(
    "n_poses,max_n_blocks,max_n_conn",
    [(1, 4, 2), (3, 7, 2), (2, 16, 3), (5, 9, 4), (1, 1, 1)],
)
def test_the_kernel_indexes_what_the_torch_ops_indexed(
    n_poses, max_n_blocks, max_n_conn, torch_device
):
    pconn_matrix, offsets, block_n_conn = random_case(
        n_poses,
        max_n_blocks,
        max_n_conn,
        torch_device,
        seed=n_poses * 100 + max_n_blocks,
    )
    expected = gather_with_torch(
        pconn_matrix.clone(), offsets, block_n_conn, max_n_conn
    )
    got = block_bondsep(
        pconn_matrix, offsets, block_n_conn, max_n_conn, MAX_SIG_BOND_SEPARATION
    )
    torch.testing.assert_close(got, expected)


def test_a_block_with_no_connections_is_all_sentinel(torch_device):
    pconn_matrix = torch.zeros((1, 4, 4), dtype=torch.int32, device=torch_device)
    offsets = torch.zeros((1, 3), dtype=torch.int64, device=torch_device)
    block_n_conn = torch.zeros((1, 3), dtype=torch.int32, device=torch_device)
    got = block_bondsep(pconn_matrix, offsets, block_n_conn, 2, MAX_SIG_BOND_SEPARATION)
    assert got.shape == (1, 3, 3, 2, 2)
    assert (got == MAX_SIG_BOND_SEPARATION).all()


def test_no_connections_at_all_needs_no_output(torch_device):
    pconn_matrix = torch.zeros((1, 2, 2), dtype=torch.int32, device=torch_device)
    offsets = torch.zeros((1, 2), dtype=torch.int64, device=torch_device)
    block_n_conn = torch.zeros((1, 2), dtype=torch.int32, device=torch_device)
    got = block_bondsep(pconn_matrix, offsets, block_n_conn, 0, MAX_SIG_BOND_SEPARATION)
    assert got.numel() == 0
