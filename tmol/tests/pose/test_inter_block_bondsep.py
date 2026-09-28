import pytest
import torch

from tmol.chemical import MAX_SIG_BOND_SEPARATION
from tmol.io import pose_stack_from_pdb
from tmol.pose import InterBlockBondsep, PoseStackBuilder
from tmol.pose.compiled import block_bondsep
from tmol.tests.pose.compiled.test_block_bondsep import random_case


def near_block_slot(row, block2):
    """``near_block_slot`` of ``tmol/score/common/count_pair.hh``."""
    last = row.shape[0] - 1
    slot = 0
    while slot < last and row[slot, 0] != block2 and row[slot, 0] >= 0:
        slot += 1
    return slot


def assert_well_formed(ibb):
    near_blocks = ibb.near_blocks.cpu()
    bondsep = ibb.bondsep.cpu()
    block2 = near_blocks[..., 0]
    stored = block2 >= 0
    assert (~stored[..., -1]).all(), "every row ends in an empty slot"
    assert (stored[..., :-1] | ~stored[..., 1:]).all(), "empty slots come last"
    ascending = (block2[..., 1:] > block2[..., :-1]) | ~stored[..., 1:]
    assert ascending.all()
    assert (block2 < ibb.max_n_blocks).all()
    assert (bondsep[~stored] == MAX_SIG_BOND_SEPARATION).all()
    assert (near_blocks[..., 1][~stored] == MAX_SIG_BOND_SEPARATION).all()
    if ibb.max_n_conn > 0:
        slab_min = torch.amin(bondsep, dim=(3, 4)).to(torch.int32)
        torch.testing.assert_close(near_blocks[..., 1], slab_min)
        assert (slab_min[stored] < MAX_SIG_BOND_SEPARATION).all()


def assert_lookup_reads_dense(ibb, dense):
    near_blocks, bondsep, dense = ibb.near_blocks.cpu(), ibb.bondsep.cpu(), dense.cpu()
    for pose in range(ibb.n_poses):
        for block1 in range(ibb.max_n_blocks):
            row = near_blocks[pose, block1]
            for block2 in range(ibb.max_n_blocks):
                slot = near_block_slot(row, block2)
                expected = dense[pose, block1, block2]
                assert torch.equal(bondsep[pose, block1, slot], expected)
                assert (
                    int(row[slot, 1]) == int(expected.amin()) or expected.numel() == 0
                )


@pytest.mark.parametrize(
    "n_poses,max_n_blocks,max_n_conn",
    [(1, 4, 2), (3, 7, 2), (2, 16, 3), (5, 9, 4), (1, 1, 1)],
)
def test_the_sparse_table_holds_the_dense_ops_separations(
    n_poses, max_n_blocks, max_n_conn, torch_device
):
    pconn_matrix, offsets, block_n_conn = random_case(
        n_poses, max_n_blocks, max_n_conn, torch_device, seed=n_poses + max_n_blocks
    )
    dense = block_bondsep(
        pconn_matrix, offsets, block_n_conn, max_n_conn, MAX_SIG_BOND_SEPARATION
    )
    ibb = InterBlockBondsep.from_connectivity(
        pconn_matrix, offsets, block_n_conn, max_n_conn
    )

    assert ibb.device == torch_device
    assert ibb.shape == dense.shape
    assert_well_formed(ibb)
    torch.testing.assert_close(ibb.to_dense(), dense)
    assert_lookup_reads_dense(ibb, dense)
    torch.testing.assert_close(
        InterBlockBondsep.from_dense(dense).to_dense(), ibb.to_dense()
    )


def test_a_graph_without_short_paths_stores_only_empty_slots(torch_device):
    pconn_matrix = torch.full(
        (2, 4, 4), MAX_SIG_BOND_SEPARATION, dtype=torch.int32, device=torch_device
    )
    offsets = torch.tensor([[0, 2], [0, 1]], dtype=torch.int64, device=torch_device)
    block_n_conn = torch.tensor(
        [[2, 2], [1, 3]], dtype=torch.int32, device=torch_device
    )

    ibb = InterBlockBondsep.from_connectivity(pconn_matrix, offsets, block_n_conn, 3)

    assert ibb.n_slots == 1
    assert_well_formed(ibb)
    assert (ibb.to_dense() == MAX_SIG_BOND_SEPARATION).all()


def test_concatenate_pads_and_select_poses_recovers_each_part(torch_device):
    parts = []
    for seed, (n_poses, max_n_blocks, max_n_conn) in enumerate(
        [(2, 5, 2), (1, 9, 3), (3, 3, 1)]
    ):
        pconn_matrix, offsets, block_n_conn = random_case(
            n_poses, max_n_blocks, max_n_conn, torch_device, seed=seed
        )
        parts.append(
            InterBlockBondsep.from_connectivity(
                pconn_matrix, offsets, block_n_conn, max_n_conn
            )
        )

    stacked = InterBlockBondsep.concatenate(parts, max_n_blocks=9, max_n_conn=3)

    assert stacked.shape == (6, 9, 9, 3, 3)
    assert stacked.n_slots == max(part.n_slots for part in parts)
    assert_well_formed(stacked)
    first = 0
    for part in parts:
        n, b, m = part.n_poses, part.max_n_blocks, part.max_n_conn
        selected = stacked.select_poses(first, first + n).to_dense()
        torch.testing.assert_close(selected[:, :b, :b, :m, :m], part.to_dense())
        padding = torch.ones_like(selected, dtype=torch.bool)
        padding[:, :b, :b, :m, :m] = False
        assert (selected[padding] == MAX_SIG_BOND_SEPARATION).all()
        first += n


def test_a_chain_stores_only_residues_two_apart(ubq_pdb, torch_device):
    pose_stack = pose_stack_from_pdb(ubq_pdb, torch_device)
    ibb = pose_stack.inter_block_bondsep
    n_blocks = pose_stack.max_n_blocks

    assert_well_formed(ibb)
    block2 = ibb.near_blocks[0, :, :, 0].cpu()
    for block1 in range(n_blocks):
        near = block2[block1][block2[block1] >= 0].tolist()
        assert near == list(range(max(block1 - 2, 0), min(block1 + 3, n_blocks)))
    assert ibb.n_slots == 6
    assert ibb.bondsep.nbytes == n_blocks * 6 * ibb.max_n_conn**2


def test_pose_stacks_keep_their_separations_when_stacked(ubq_pdb, torch_device):
    poses = [
        pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=n) for n in (12, 30)
    ]
    stacked = PoseStackBuilder.from_poses(poses, torch_device)

    assert_well_formed(stacked.inter_block_bondsep)
    dense = stacked.inter_block_bondsep.to_dense()
    for index, pose in enumerate(poses):
        n, m = pose.max_n_blocks, pose.packed_block_types.max_n_conn
        torch.testing.assert_close(
            dense[index, :n, :n, :m, :m], pose.inter_block_bondsep.to_dense()[0]
        )
