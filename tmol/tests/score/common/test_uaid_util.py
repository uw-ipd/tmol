import numpy
import torch
import pytest

from tmol.tests.score.common.uaid_util import resolve_uaids


@pytest.fixture
def uaid_pose_stack():
    n_blocks = 6

    pose_stack_block_type = torch.zeros((1, n_blocks), dtype=torch.int32)

    pose_stack_block_coord_offset = torch.zeros((1, n_blocks), dtype=torch.int32)
    pose_stack_block_coord_offset[0, 0] = 0
    pose_stack_block_coord_offset[0, 1] = 20
    pose_stack_block_coord_offset[0, 2] = 40
    pose_stack_block_coord_offset[0, 3] = 60
    pose_stack_block_coord_offset[0, 4] = 80
    pose_stack_block_coord_offset[0, 5] = 100

    pose_stack_inter_block_connections = torch.zeros(
        (1, n_blocks, 2, 2), dtype=torch.int32
    )
    # first residue
    pose_stack_inter_block_connections[0, 0, 0, 0] = -1
    pose_stack_inter_block_connections[0, 0, 0, 1] = -1
    pose_stack_inter_block_connections[0, 0, 1, 0] = 1
    pose_stack_inter_block_connections[0, 0, 1, 1] = 0
    # 2nd residue
    pose_stack_inter_block_connections[0, 1, 0, 0] = 0
    pose_stack_inter_block_connections[0, 1, 0, 1] = 1
    pose_stack_inter_block_connections[0, 1, 1, 0] = 2
    pose_stack_inter_block_connections[0, 1, 1, 1] = 0
    # 3rd residue
    pose_stack_inter_block_connections[0, 2, 0, 0] = 1
    pose_stack_inter_block_connections[0, 2, 0, 1] = 1
    pose_stack_inter_block_connections[0, 2, 1, 0] = 3
    pose_stack_inter_block_connections[0, 2, 1, 1] = 0
    # 4th residue
    pose_stack_inter_block_connections[0, 3, 0, 0] = 2
    pose_stack_inter_block_connections[0, 3, 0, 1] = 1
    pose_stack_inter_block_connections[0, 3, 1, 0] = 4
    pose_stack_inter_block_connections[0, 3, 1, 1] = 0
    # 5th residue
    pose_stack_inter_block_connections[0, 4, 0, 0] = 3
    pose_stack_inter_block_connections[0, 4, 0, 1] = 1
    pose_stack_inter_block_connections[0, 4, 1, 0] = 5
    pose_stack_inter_block_connections[0, 4, 1, 1] = 0
    # 6th residue
    pose_stack_inter_block_connections[0, 5, 0, 0] = 4
    pose_stack_inter_block_connections[0, 5, 0, 1] = 1
    pose_stack_inter_block_connections[0, 5, 1, 0] = -1
    pose_stack_inter_block_connections[0, 5, 1, 1] = -1

    block_type_atom_downstream_of_conn = torch.zeros((1, 2, 20), dtype=torch.int32)
    # consider backbone to be 0-2
    block_type_atom_downstream_of_conn[0, 0, 0] = 0
    block_type_atom_downstream_of_conn[0, 0, 1] = 1
    block_type_atom_downstream_of_conn[0, 0, 2] = 2
    block_type_atom_downstream_of_conn[0, 1, 0] = 2
    block_type_atom_downstream_of_conn[0, 1, 1] = 1
    block_type_atom_downstream_of_conn[0, 1, 2] = 0

    return (
        pose_stack_block_type,
        pose_stack_block_coord_offset,
        pose_stack_inter_block_connections,
        block_type_atom_downstream_of_conn,
    )


def test_resolve_uaids_smoke(uaid_pose_stack):
    uaids = torch.zeros((1, 3), dtype=torch.int32)
    block_inds = torch.zeros((1,), dtype=torch.int32)
    pose_inds = torch.zeros((1,), dtype=torch.int32)

    (
        pose_stack_block_type,
        pose_stack_block_coord_offset,
        pose_stack_inter_block_connections,
        block_type_atom_downstream_of_conn,
    ) = uaid_pose_stack

    resolved = resolve_uaids(
        uaids,
        block_inds,
        pose_inds,
        pose_stack_block_coord_offset,
        pose_stack_block_type,
        pose_stack_inter_block_connections,
        block_type_atom_downstream_of_conn,
    )
    assert resolved.device == torch.device("cpu")
    assert resolved.shape == (1,)
    assert resolved[0] == 0


def test_resolve_uaids_intra_res(uaid_pose_stack):
    n_uaids = 4
    uaids = torch.zeros((n_uaids, 3), dtype=torch.int32)
    block_inds = torch.zeros((n_uaids,), dtype=torch.int32)
    pose_inds = torch.zeros((n_uaids,), dtype=torch.int32)

    (
        pose_stack_block_type,
        pose_stack_block_coord_offset,
        pose_stack_inter_block_connections,
        block_type_atom_downstream_of_conn,
    ) = uaid_pose_stack

    uaids[0, 0] = 3
    uaids[1, 0] = 4
    uaids[2, 0] = 5
    uaids[3, 0] = 6

    block_inds[0] = 1
    block_inds[1] = 2
    block_inds[2] = 3
    block_inds[3] = 4

    resolved = resolve_uaids(
        uaids,
        block_inds,
        pose_inds,
        pose_stack_block_coord_offset,
        pose_stack_block_type,
        pose_stack_inter_block_connections,
        block_type_atom_downstream_of_conn,
    )

    block_inds64 = block_inds.to(torch.int64)
    pose_inds64 = pose_inds.to(torch.int64)
    resolved_gold = (
        pose_stack_block_coord_offset[pose_inds64, block_inds64] + uaids[:, 0]
    )
    numpy.testing.assert_equal(resolved_gold.numpy(), resolved.numpy())


def test_resolve_uaids_inter_res(uaid_pose_stack):
    n_uaids = 4
    uaids = torch.full((n_uaids, 3), -1, dtype=torch.int32)
    block_inds = torch.zeros((n_uaids,), dtype=torch.int32)
    pose_inds = torch.zeros((n_uaids,), dtype=torch.int32)

    (
        pose_stack_block_type,
        pose_stack_block_coord_offset,
        pose_stack_inter_block_connections,
        block_type_atom_downstream_of_conn,
    ) = uaid_pose_stack

    uaids[:] = torch.tensor([-1, 1, 0], dtype=torch.int32)

    block_inds[0] = 1
    block_inds[1] = 2
    block_inds[2] = 3
    block_inds[3] = 4

    resolved = resolve_uaids(
        uaids,
        block_inds,
        pose_inds,
        pose_stack_block_coord_offset,
        pose_stack_block_type,
        pose_stack_inter_block_connections,
        block_type_atom_downstream_of_conn,
    )

    resolved_gold = torch.tensor([40, 60, 80, 100], dtype=torch.int32)
    numpy.testing.assert_equal(resolved_gold.numpy(), resolved.numpy())


def test_resolve_uaids_inter_res2(uaid_pose_stack):
    n_uaids = 4
    uaids = torch.full((n_uaids, 3), -1, dtype=torch.int32)
    block_inds = torch.zeros((n_uaids,), dtype=torch.int32)
    pose_inds = torch.zeros((n_uaids,), dtype=torch.int32)

    (
        pose_stack_block_type,
        pose_stack_block_coord_offset,
        pose_stack_inter_block_connections,
        block_type_atom_downstream_of_conn,
    ) = uaid_pose_stack

    uaids[0, 1] = 0
    uaids[0, 2] = 0
    uaids[1, 1] = 0
    uaids[1, 2] = 1
    uaids[2, 1] = 1
    uaids[2, 2] = 0
    uaids[3, 1] = 1
    uaids[3, 2] = 1

    block_inds[0] = 1
    block_inds[1] = 2
    block_inds[2] = 3
    block_inds[3] = 4

    resolved = resolve_uaids(
        uaids,
        block_inds,
        pose_inds,
        pose_stack_block_coord_offset,
        pose_stack_block_type,
        pose_stack_inter_block_connections,
        block_type_atom_downstream_of_conn,
    )

    resolved_gold = torch.tensor([2, 21, 80, 101], dtype=torch.int32)
    numpy.testing.assert_equal(resolved_gold.numpy(), resolved.numpy())


def test_resolve_uaids_unresolved_connection(uaid_pose_stack):
    n_uaids = 2
    uaids = torch.full((n_uaids, 3), -1, dtype=torch.int32)
    block_inds = torch.zeros((n_uaids,), dtype=torch.int32)
    pose_inds = torch.zeros((n_uaids,), dtype=torch.int32)

    uaids[0, 1] = 0
    uaids[1, 1] = 1
    uaids[0, 2] = 0
    uaids[1, 2] = 0

    block_inds[0] = 0
    block_inds[1] = 5

    (
        pose_stack_block_coord_offset,
        pose_stack_block_type,
        pose_stack_inter_block_connections,
        block_type_atom_downstream_of_conn,
    ) = uaid_pose_stack

    resolved = resolve_uaids(
        uaids,
        block_inds,
        pose_inds,
        pose_stack_block_coord_offset,
        pose_stack_block_type,
        pose_stack_inter_block_connections,
        block_type_atom_downstream_of_conn,
    )

    resolved_gold = torch.tensor([-1, -1], dtype=torch.int32)
    numpy.testing.assert_equal(resolved_gold.numpy(), resolved.numpy())


def test_resolve_unspecified_uaids(uaid_pose_stack):
    n_uaids = 2
    uaids = torch.full((n_uaids, 3), -1, dtype=torch.int32)
    block_inds = torch.zeros((n_uaids,), dtype=torch.int32)
    pose_inds = torch.zeros((n_uaids,), dtype=torch.int32)

    (
        pose_stack_block_type,
        pose_stack_block_coord_offset,
        pose_stack_inter_block_connections,
        block_type_atom_downstream_of_conn,
    ) = uaid_pose_stack

    block_inds[0] = 1
    block_inds[1] = 2

    resolved = resolve_uaids(
        uaids,
        block_inds,
        pose_inds,
        pose_stack_block_coord_offset,
        pose_stack_block_type,
        pose_stack_inter_block_connections,
        block_type_atom_downstream_of_conn,
    )

    resolved_gold = torch.tensor([-1, -1], dtype=torch.int32)
    numpy.testing.assert_equal(resolved_gold.numpy(), resolved.numpy())
