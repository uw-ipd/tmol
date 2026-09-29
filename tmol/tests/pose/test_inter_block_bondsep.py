from pathlib import Path

import numpy as np
import pytest
import torch

from tmol.chemical import MAX_SIG_BOND_SEPARATION
from tmol.io import (
    atom_array_from_cif,
    build_context_from_biotite,
    pose_stack_from_biotite,
    pose_stack_from_pdb,
)
from tmol.pose import InterBlockBondsep, PoseStackBuilder
from tmol.pose.compiled import stacked_apsp

DATA = Path(__file__).parents[1] / "data"


def gather_with_torch(pconn_matrix, pconn_offsets, block_n_conn, max_n_conn):
    """The dense ``[pose, block1, block2, conn1, conn2]`` table, by plain indexing."""
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
    out = out.permute(0, 1, 3, 2, 4).clamp(max=MAX_SIG_BOND_SEPARATION)
    out.masked_fill_(~real_bconn[:, :, None, :, None], MAX_SIG_BOND_SEPARATION)
    out.masked_fill_(~real_bconn[:, None, :, None, :], MAX_SIG_BOND_SEPARATION)
    return out.to(torch.int8).contiguous()


def random_case(n_poses, max_n_blocks, max_n_conn, device, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    block_n_conn = torch.randint(
        0, max_n_conn + 1, (n_poses, max_n_blocks), generator=generator
    ).to(torch.int32)
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
    dense = gather_with_torch(pconn_matrix, offsets, block_n_conn, max_n_conn)
    ibb = InterBlockBondsep.from_connectivity(
        pconn_matrix, offsets, block_n_conn, max_n_conn
    )

    assert ibb.device == torch_device
    assert ibb.shape == dense.shape
    assert_well_formed(ibb)
    torch.testing.assert_close(ibb.to_dense(), dense)
    assert_lookup_reads_dense(ibb, dense)


@pytest.mark.parametrize(
    "distance,offsets,block_n_conn,max_n_conn",
    [
        (MAX_SIG_BOND_SEPARATION, [[0, 2], [0, 1]], [[2, 2], [1, 3]], 3),
        (0, [[0, 0, 0]], [[0, 0, 0]], 2),
        (0, [[0, 0]], [[0, 0]], 0),
    ],
)
def test_a_graph_without_short_paths_stores_only_empty_slots(
    distance, offsets, block_n_conn, max_n_conn, torch_device
):
    offsets = torch.tensor(offsets, dtype=torch.int64, device=torch_device)
    block_n_conn = torch.tensor(block_n_conn, dtype=torch.int32, device=torch_device)
    pconn_matrix = torch.full(
        (offsets.shape[0], 4, 4), distance, dtype=torch.int32, device=torch_device
    )

    ibb = InterBlockBondsep.from_connectivity(
        pconn_matrix, offsets, block_n_conn, max_n_conn
    )

    n_poses, n_blocks = offsets.shape
    assert ibb.shape == (n_poses, n_blocks, n_blocks, max_n_conn, max_n_conn)
    assert ibb.n_slots == 1
    assert_well_formed(ibb)
    assert (ibb.to_dense() == MAX_SIG_BOND_SEPARATION).all()


def test_separations_are_one_byte_and_saturate_at_the_cap(torch_device):
    pconn_matrix = torch.tensor(
        [[[0, 9], [300, 3]]], dtype=torch.int32, device=torch_device
    )
    offsets = torch.zeros((1, 1), dtype=torch.int64, device=torch_device)
    block_n_conn = torch.full((1, 1), 2, dtype=torch.int32, device=torch_device)

    ibb = InterBlockBondsep.from_connectivity(pconn_matrix, offsets, block_n_conn, 3)

    assert ibb.bondsep.dtype == torch.int8
    assert ibb.to_dense().tolist() == [[[[[0, 6, 6], [6, 3, 6], [6, 6, 6]]]]]


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


def random_bonded_graph(seed, device):
    """Random blocks with symmetric intra-block separations and random bonds."""
    generator = torch.Generator().manual_seed(seed)
    n_poses, n_blocks, max_n_conn = (
        int(torch.randint(1, hi, (1,), generator=generator)) for hi in (4, 12, 5)
    )
    counts = torch.randint(0, max_n_conn + 1, (n_poses, n_blocks), generator=generator)
    intra = torch.randint(
        0, 9, (n_poses, n_blocks, max_n_conn, max_n_conn), generator=generator
    )
    intra = torch.minimum(intra, intra.transpose(2, 3))
    intra.diagonal(dim1=2, dim2=3).zero_()
    connections = torch.full((n_poses, n_blocks, max_n_conn, 2), -1, dtype=torch.int64)
    for pose in range(n_poses):
        ends = torch.nonzero(torch.arange(max_n_conn) < counts[pose, :, None])
        ends = ends[torch.randperm(ends.shape[0], generator=generator)]
        n_bonds = int(
            torch.randint(0, ends.shape[0] // 2 + 1, (1,), generator=generator)
        )
        first, second = ends[:n_bonds], ends[n_bonds : 2 * n_bonds]
        connections[pose, first[:, 0], first[:, 1]] = second
        connections[pose, second[:, 0], second[:, 1]] = first
    return (
        counts.to(torch.int32).to(device),
        intra.to(torch.int32).to(device),
        connections.to(device),
    )


@pytest.mark.parametrize("seed", range(12))
def test_the_bounded_search_matches_all_pairs_shortest_paths(seed, torch_device):
    counts, intra, connections = random_bonded_graph(seed, torch_device)
    offsets = torch.cumsum(counts.to(torch.int64), 1) - counts.to(torch.int64)
    n_nodes = max(int(counts.sum(1).max()), 1)
    real = torch.arange(intra.shape[2], device=torch_device) < counts[..., None]
    pose, block, conn1, conn2 = torch.nonzero(
        real[..., :, None] & real[..., None, :], as_tuple=True
    )
    distances = torch.full(
        (counts.shape[0], n_nodes, n_nodes),
        MAX_SIG_BOND_SEPARATION,
        dtype=torch.int32,
        device=torch_device,
    )
    distances[pose, offsets[pose, block] + conn1, offsets[pose, block] + conn2] = intra[
        pose, block, conn1, conn2
    ]
    pose, block, conn = torch.nonzero(connections[..., 0] >= 0, as_tuple=True)
    partner = connections[pose, block, conn]
    distances[
        pose, offsets[pose, block] + conn, offsets[pose, partner[:, 0]] + partner[:, 1]
    ] = 1
    stacked_apsp(distances, MAX_SIG_BOND_SEPARATION)

    ibb = InterBlockBondsep.from_bonded_graph(counts, intra, connections)

    assert_well_formed(ibb)
    torch.testing.assert_close(
        ibb.to_dense(), gather_with_torch(distances, offsets, counts, intra.shape[2])
    )


def heavy_pose_stack(name, device):
    array = atom_array_from_cif(DATA / "metal_fixtures" / f"{name}.cif.gz")
    array = array[np.char.upper(array.element.astype(str)) != "H"]
    context = build_context_from_biotite(array, device)
    return pose_stack_from_biotite(array, device, context=context, no_optH=True)


def dense_apsp_bondsep(pose_stack):
    """The builder's all-pairs-shortest-path construction, as a reference."""
    pbt, block_types = pose_stack.packed_block_types, pose_stack.block_type_ind64
    pconn, offsets, counts, _ = PoseStackBuilder._take_real_conn_conn_intrablock_pairs(
        pbt, block_types, block_types >= 0
    )
    PoseStackBuilder._incorporate_inter_residue_connections_into_connectivity_graph(
        pose_stack.inter_residue_connections64, offsets, pconn
    )
    return PoseStackBuilder._calculate_interblock_bondsep_from_connectivity_graph(
        pbt, offsets, counts, pconn
    ).to_dense()


@pytest.mark.parametrize(
    "structure",
    [
        "ubq_stack",
        "clf_nitrogenase_7adr",
        "sf4_ferredoxin_1fdn",
        "zn_tetrahedral_3ks3",
        "nco_zdna_1dn8",
        "heme_myoglobin_5yce",
    ],
)
def test_the_bounded_search_finds_every_short_path(structure, ubq_pdb, torch_device):
    if structure == "ubq_stack":
        pose_stack = PoseStackBuilder.from_poses(
            [
                pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=n)
                for n in (9, 40)
            ],
            torch_device,
        )
    else:
        pose_stack = heavy_pose_stack(structure, torch_device)

    assert_well_formed(pose_stack.inter_block_bondsep)
    torch.testing.assert_close(
        pose_stack.inter_block_bondsep.to_dense(), dense_apsp_bondsep(pose_stack)
    )


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
