import numpy
import torch
import attr
import os
import pickle
import pytest

from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.pack.datatypes import PackerEnergyTables
from tmol.pack.simulated_annealing import run_simulated_annealing
from tmol.pack.compiled.compiled import validate_energies, build_interaction_graph
from tmol.pack.rotamer.build_rotamers import RotamerSet
from tmol.utility.cumsum import exclusive_cumsum, exclusive_cumsum1d
from tmol.io import pose_stack_from_pdb


@pytest.fixture
def ubq_ig():
    """This fixture returns two dictionaries of 1-body and a 2-body energies."""
    with open("tmol/tests/data/pack/1ubq_ig", "rb") as f:
        return pickle.load(f)


def test_load_ig(ubq_ig):
    oneb, twob = ubq_ig
    assert len(oneb) == 76
    nrots = numpy.zeros((76,), dtype=int)
    for i in range(76):
        arrname = "%d" % (i + 1)
        nrots[i] = oneb[arrname].shape[0]
    for i in range(76):
        for j in range(i + 1, 76):
            arrname = "%d-%d" % (i + 1, j + 1)
            if arrname in twob:
                assert nrots[i] == twob[arrname].shape[0]
                assert nrots[j] == twob[arrname].shape[1]


def construct_faux_rotamer_set_and_sparse_energies_table_from_ig(ig, pdb_fname, device):
    pose_stack = pose_stack_from_pdb(pdb_fname, device)
    oneb, twob = ig

    n_rots = torch.zeros((76,), dtype=torch.int64)
    for i in range(76):
        arrname = f"{i+1}"
        n_rots[i] = oneb[arrname].shape[0]

    def _ti64(x):
        return torch.tensor(x, dtype=torch.int64)

    def _tf32(x):
        return torch.tensor(x, dtype=torch.float32)

    def _d(x):
        return x.to(device)

    n_rots_total = torch.sum(n_rots).item()
    n_rots_for_pose = torch.full((1,), n_rots_total, dtype=torch.int64)
    rot_offset_for_pose = torch.zeros((1,), dtype=torch.int64)
    n_rots_for_block = n_rots[None, :]
    rot_offset_for_block = _ti64(exclusive_cumsum1d(n_rots)[None, :])
    pose_for_rot = torch.zeros((n_rots_total,), dtype=torch.int64)

    rot_is_start_for_block = torch.zeros((n_rots_total), dtype=torch.int64)
    rot_is_start_for_block[rot_offset_for_block[0]] = 1
    block_ind_for_rot = torch.cumsum(rot_is_start_for_block, dim=0) - 1
    block_ind_for_rot32 = block_ind_for_rot.to(torch.int32)
    block_type_ind_for_rot = pose_stack.block_type_ind64[
        pose_for_rot, block_ind_for_rot
    ]

    coords = torch.zeros((1, pose_stack.max_n_block_atoms, 3), dtype=torch.float32)

    rotamer_set = RotamerSet(
        n_rots_for_pose=_d(n_rots_for_pose),
        rot_offset_for_pose=_d(rot_offset_for_pose),
        n_rots_for_block=_d(n_rots_for_block),
        rot_offset_for_block=_d(rot_offset_for_block),
        pose_for_rot=_d(pose_for_rot),
        block_type_ind_for_rot=_d(block_type_ind_for_rot),
        block_ind_for_rot=_d(block_ind_for_rot32),
        coords=_d(coords),
    )

    energy1b = torch.zeros((n_rots_total), dtype=torch.float32)

    nonzero_inds = []
    ordered_energy_tables = []
    entry_offset = []
    cumm_offset = 0
    for i in range(76):
        i_n_rots = n_rots_for_block[0, i]
        i_offset = rot_offset_for_block[0, i]
        energy1b[i_offset : (i_offset + i_n_rots)] = _tf32(oneb[f"{i+1}"])
        for j in range(i + 1, 76):
            j_n_rots = n_rots_for_block[0, j]
            j_offset = rot_offset_for_block[0, j]
            table_name = f"{i+1}-{j+1}"
            if table_name in twob:
                # let's say all energies here will be listed as non-zero
                entry_offset.append(cumm_offset)
                cumm_offset += int(i_n_rots * j_n_rots)
                indices = torch.arange(i_n_rots * j_n_rots, dtype=torch.int64)
                inds_i = torch.div(indices, j_n_rots) + i_offset
                inds_j = torch.remainder(indices, j_n_rots) + j_offset
                nonzero_inds.append((inds_i, inds_j))
                ordered_energy_tables.append(
                    torch.tensor(twob[table_name], dtype=torch.float32)
                )
    inds = torch.zeros((3, cumm_offset), dtype=torch.int32)
    energies = torch.zeros((cumm_offset), dtype=torch.float32)
    for pair in range(len(nonzero_inds)):
        i_inds, j_inds = nonzero_inds[pair]
        ij_offset = entry_offset[pair]
        ij_n_pairs = i_inds.shape[0]
        inds[1, ij_offset : (ij_offset + ij_n_pairs)] = i_inds
        inds[2, ij_offset : (ij_offset + ij_n_pairs)] = j_inds
        energies[ij_offset : (ij_offset + ij_n_pairs)] = ordered_energy_tables[
            pair
        ].view(-1)

    return pose_stack, rotamer_set, _d(energy1b), _d(inds), _d(energies)


def construct_stacked_faux_rotamer_set_and_sparse_energies_table_from_ig(
    ig, pdb_fname, device
):
    pose_stack = pose_stack_from_pdb(pdb_fname, device)
    oneb, twob = ig
    n_res = len(oneb)

    n_poses = 2
    n_rots = torch.zeros((n_poses, 76), dtype=int)
    for i in range(76):
        arrname = f"{i+1}"
        n_rots[:, i] = oneb[arrname].shape[0]

    def _ti64(x):
        return torch.tensor(x, dtype=torch.int64)

    def _tf32(x):
        return torch.tensor(x, dtype=torch.float32)

    def _d(x):
        return x.to(device)

    n_rots_per_pose = torch.sum(n_rots, dim=0)
    n_rots_total = torch.sum(n_rots_per_pose).item()
    n_rots_for_pose = torch.full((2,), n_rots_total // 2, dtype=torch.int64)
    rot_offset_for_pose = torch.zeros((2,), dtype=torch.int64)
    rot_offset_for_pose[1] = n_rots_for_pose[0]
    n_rots_for_block = n_rots
    rot_offset_for_block = _ti64(exclusive_cumsum1d(n_rots.ravel())).reshape(
        n_rots.shape
    )
    pose_for_rot = torch.zeros((n_rots_total,), dtype=torch.int64)
    pose_for_rot[n_rots_per_pose[0] :] = 1

    rot_is_start_for_block = torch.zeros((n_rots_total), dtype=torch.int64)
    rot_is_start_for_block[rot_offset_for_block.ravel()] = 1
    block_ind_for_rot = torch.remainder(
        torch.cumsum(rot_is_start_for_block, dim=0) - 1, n_res
    )
    block_ind_for_rot32 = block_ind_for_rot.to(torch.int32)
    block_type_ind_for_rot = pose_stack.block_type_ind64[0, block_ind_for_rot]

    # obviously a bogus size for this tensor
    coords = torch.zeros((1, pose_stack.max_n_block_atoms, 3), dtype=torch.float32)

    rotamer_set = RotamerSet(
        n_rots_for_pose=_d(n_rots_for_pose),
        rot_offset_for_pose=_d(rot_offset_for_pose),
        n_rots_for_block=_d(n_rots_for_block),
        rot_offset_for_block=_d(rot_offset_for_block),
        pose_for_rot=_d(pose_for_rot),
        block_type_ind_for_rot=_d(block_type_ind_for_rot),
        block_ind_for_rot=_d(block_ind_for_rot32),
        coords=_d(coords),
    )

    energy1b = torch.zeros(n_rots_total, dtype=torch.float32)

    nonzero_inds = []
    ordered_energy_tables = []
    entry_offset = []
    cumm_offset = 0
    for i in range(76):
        i_n_rots = n_rots_for_block[0, i]
        i_offset = rot_offset_for_block[0, i]
        energy1b[i_offset : (i_offset + i_n_rots)] = _tf32(oneb[f"{i+1}"])
        for j in range(i + 1, 76):
            j_n_rots = n_rots_for_block[0, j]
            j_offset = rot_offset_for_block[0, j]
            table_name = f"{i+1}-{j+1}"
            if table_name in twob:
                # let's say all energies here will be listed as non-zero
                entry_offset.append(cumm_offset)
                cumm_offset += int(i_n_rots * j_n_rots)
                indices = torch.arange(i_n_rots * j_n_rots, dtype=torch.int64)
                pose_inds = torch.zeros(i_n_rots * j_n_rots, dtype=torch.int64)
                inds_i = torch.div(indices, j_n_rots) + i_offset
                inds_j = torch.remainder(indices, j_n_rots) + j_offset
                nonzero_inds.append((pose_inds, inds_i, inds_j))
                ordered_energy_tables.append(
                    torch.tensor(twob[table_name], dtype=torch.float32)
                )
    # Now insert the indices for the second pose
    for i in range(76):
        i_n_rots = n_rots_for_block[1, i]
        i_offset = rot_offset_for_block[1, i]
        energy1b[i_offset : (i_offset + i_n_rots)] = _tf32(oneb[f"{i+1}"])
        for j in range(i + 1, 76):
            j_n_rots = n_rots_for_block[1, j]
            j_offset = rot_offset_for_block[1, j]
            table_name = f"{i+1}-{j+1}"
            if table_name in twob:
                # let's say all energies here will be listed as non-zero
                entry_offset.append(cumm_offset)
                cumm_offset += int(i_n_rots * j_n_rots)
                indices = torch.arange(i_n_rots * j_n_rots, dtype=torch.int64)
                pose_inds = torch.full((i_n_rots * j_n_rots,), 1, dtype=torch.int64)
                inds_i = torch.div(indices, j_n_rots) + i_offset
                inds_j = torch.remainder(indices, j_n_rots) + j_offset
                nonzero_inds.append((pose_inds, inds_i, inds_j))
                ordered_energy_tables.append(
                    torch.tensor(twob[table_name], dtype=torch.float32)
                )
    inds = torch.zeros((3, cumm_offset), dtype=torch.int32)
    energies = torch.zeros((cumm_offset), dtype=torch.float32)
    for pair in range(len(nonzero_inds)):
        pose_inds, i_inds, j_inds = nonzero_inds[pair]
        ij_offset = entry_offset[pair]
        ij_n_pairs = i_inds.shape[0]
        inds[0, ij_offset : (ij_offset + ij_n_pairs)] = pose_inds
        inds[1, ij_offset : (ij_offset + ij_n_pairs)] = i_inds
        inds[2, ij_offset : (ij_offset + ij_n_pairs)] = j_inds
        energies[ij_offset : (ij_offset + ij_n_pairs)] = ordered_energy_tables[
            pair % len(twob)
        ].view(-1)

    pose_stack = PoseStackBuilder.from_poses([pose_stack, pose_stack], device=device)
    return pose_stack, rotamer_set, _d(energy1b), _d(inds), _d(energies)


def create_packer_energy_tables(
    pose_stack,
    rotamer_set,
    chunk_size,
    chunk_offset_offsets,
    chunk_offsets,
    energy1b,
    energy2b,
):
    packer_energy_tables = PackerEnergyTables(
        max_n_rotamers_per_pose=torch.max(rotamer_set.n_rots_for_pose).cpu().item(),
        pose_n_res=pose_stack.n_res_per_pose,
        pose_n_rotamers=rotamer_set.n_rots_for_pose,
        pose_rotamer_offset=rotamer_set.rot_offset_for_pose,
        nrotamers_for_res=rotamer_set.n_rots_for_block,
        oneb_offsets=rotamer_set.rot_offset_for_block,
        res_for_rot=rotamer_set.block_ind_for_rot,
        chunk_size=chunk_size,
        chunk_offset_offsets=chunk_offset_offsets,
        chunk_offsets=chunk_offsets,
        energy1b=energy1b,
        energy2b=energy2b,
    )
    return packer_energy_tables


def test_construct_rotamer_set_and_sparse_energies_table_from_ig(ubq_ig, torch_device):
    pdb_fname = "tmol/tests/data/pdb/1ubq.pdb"

    # Step 1: convert the IG that we're getting from disk
    # into the format that we expect from the score function
    ps, rotamer_set, energy1b, sparse_indices, energies = (
        construct_faux_rotamer_set_and_sparse_energies_table_from_ig(
            ubq_ig, pdb_fname, torch_device
        )
    )
    assert rotamer_set.n_rots_for_pose.dtype == torch.int64
    assert rotamer_set.rot_offset_for_pose.dtype == torch.int64
    assert rotamer_set.n_rots_for_block.dtype == torch.int64
    assert rotamer_set.rot_offset_for_block.dtype == torch.int64
    assert rotamer_set.pose_for_rot.dtype == torch.int64
    assert rotamer_set.block_type_ind_for_rot.dtype == torch.int64
    assert rotamer_set.block_ind_for_rot.dtype == torch.int32
    assert sparse_indices.dtype == torch.int32
    assert energies.dtype == torch.float32

    assert rotamer_set.n_rots_for_pose.shape == tuple([1])
    assert rotamer_set.rot_offset_for_pose.shape == tuple([1])
    assert rotamer_set.n_rots_for_block.shape == tuple([1, 76])
    assert rotamer_set.rot_offset_for_block.shape == tuple([1, 76])
    assert rotamer_set.pose_for_rot.shape == tuple([1518])
    assert rotamer_set.block_type_ind_for_rot.shape == tuple([1518])
    assert rotamer_set.block_ind_for_rot.shape == tuple([1518])
    assert sparse_indices.shape == tuple([3, 304426])
    assert energies.shape == tuple([304426])

    assert rotamer_set.n_rots_for_pose.device == torch_device
    assert rotamer_set.rot_offset_for_pose.device == torch_device
    assert rotamer_set.n_rots_for_block.device == torch_device
    assert rotamer_set.rot_offset_for_block.device == torch_device
    assert rotamer_set.pose_for_rot.device == torch_device
    assert rotamer_set.block_type_ind_for_rot.device == torch_device
    assert rotamer_set.block_ind_for_rot.device == torch_device
    assert sparse_indices.device == torch_device
    assert energies.device == torch_device

    ps2, rotamer_set2, energy1b, sparse_indices2, energies2 = (
        construct_stacked_faux_rotamer_set_and_sparse_energies_table_from_ig(
            ubq_ig, pdb_fname, torch_device
        )
    )
    assert rotamer_set2.n_rots_for_pose.dtype == torch.int64
    assert rotamer_set2.rot_offset_for_pose.dtype == torch.int64
    assert rotamer_set2.n_rots_for_block.dtype == torch.int64
    assert rotamer_set2.rot_offset_for_block.dtype == torch.int64
    assert rotamer_set2.pose_for_rot.dtype == torch.int64
    assert rotamer_set2.block_type_ind_for_rot.dtype == torch.int64
    assert rotamer_set2.block_ind_for_rot.dtype == torch.int32
    assert sparse_indices2.dtype == torch.int32
    assert energies2.dtype == torch.float32

    assert rotamer_set2.n_rots_for_pose.shape == tuple([2])
    assert rotamer_set2.rot_offset_for_pose.shape == tuple([2])
    assert rotamer_set2.n_rots_for_block.shape == tuple([2, 76])
    assert rotamer_set2.rot_offset_for_block.shape == tuple([2, 76])
    assert rotamer_set2.pose_for_rot.shape == tuple([3036])
    assert rotamer_set2.block_type_ind_for_rot.shape == tuple([3036])
    assert rotamer_set2.block_ind_for_rot.shape == tuple([3036])
    assert sparse_indices2.shape == tuple([3, 608852])
    assert energies2.shape == tuple([608852])

    assert rotamer_set2.n_rots_for_pose.device == torch_device
    assert rotamer_set2.rot_offset_for_pose.device == torch_device
    assert rotamer_set2.n_rots_for_block.device == torch_device
    assert rotamer_set2.rot_offset_for_block.device == torch_device
    assert rotamer_set2.pose_for_rot.device == torch_device
    assert rotamer_set2.block_type_ind_for_rot.device == torch_device
    assert rotamer_set2.block_ind_for_rot.device == torch_device
    assert sparse_indices2.device == torch_device
    assert energies2.device == torch_device


def test_build_interaction_graph(ubq_ig, torch_device):
    pdb_fname = "tmol/tests/data/pdb/1ubq.pdb"

    # Step 1: convert the IG that we're getting from disk
    # into the format that we expect from the score function
    ps, rotamer_set, energy1b, sparse_indices, energies = (
        construct_faux_rotamer_set_and_sparse_energies_table_from_ig(
            ubq_ig, pdb_fname, torch_device
        )
    )

    chunk_size = 16

    (chunk_pair_offset_for_block_pair, chunk_pair_offset, energy2b) = (
        build_interaction_graph(
            chunk_size,
            rotamer_set.n_rots_for_pose,
            rotamer_set.rot_offset_for_pose,
            rotamer_set.n_rots_for_block,
            rotamer_set.rot_offset_for_block,
            rotamer_set.pose_for_rot,
            rotamer_set.block_type_ind_for_rot,
            rotamer_set.block_ind_for_rot,
            sparse_indices,
            energies,
        )
    )

    assert chunk_pair_offset_for_block_pair.shape == (1, 76, 76)
    assert chunk_pair_offset_for_block_pair.dtype == torch.int64
    assert chunk_pair_offset_for_block_pair.device == torch_device
    assert chunk_pair_offset.shape == (5016,)
    assert chunk_pair_offset.dtype == torch.int64
    assert chunk_pair_offset.device == torch_device
    assert energy2b.shape == (608852,)
    assert energy2b.dtype == torch.float32
    assert energy2b.device == torch_device

    for i in range(sparse_indices[:, :1000].shape[1]):
        i_energy = energies[i].item()
        pose = sparse_indices[0, i].item()
        rot1 = sparse_indices[1, i].item()
        rot2 = sparse_indices[2, i].item()
        block1 = rotamer_set.block_ind_for_rot[rot1].item()
        block2 = rotamer_set.block_ind_for_rot[rot2].item()
        chunk_offset_for_blocks_ij = chunk_pair_offset_for_block_pair[
            pose, block1, block2
        ].item()
        chunk_offset_for_blocks_ji = chunk_pair_offset_for_block_pair[
            pose, block2, block1
        ].item()
        if chunk_offset_for_blocks_ij == -1:
            if i_energy != 0:
                print(
                    i_energy,
                    pose,
                    rot1,
                    rot2,
                    block1,
                    block2,
                    chunk_offset_for_blocks_ij,
                    chunk_offset_for_blocks_jj,
                )
            assert i_energy == 0
        else:
            n_rots_for_block1 = rotamer_set.n_rots_for_block[pose, block1].item()
            n_rots_for_block2 = rotamer_set.n_rots_for_block[pose, block2].item()

            n_chunks_for_block1 = (n_rots_for_block1 - 1) // chunk_size + 1
            n_chunks_for_block2 = (n_rots_for_block2 - 1) // chunk_size + 1

            rot_on_block1 = rot1 - rotamer_set.rot_offset_for_block[pose, block1].item()
            rot_on_block2 = rot2 - rotamer_set.rot_offset_for_block[pose, block2].item()
            chunk1 = rot_on_block1 // chunk_size
            chunk2 = rot_on_block2 // chunk_size

            overhang1 = n_rots_for_block1 - chunk1 * chunk_size
            overhang2 = n_rots_for_block2 - chunk2 * chunk_size
            chunk1_size = chunk_size if overhang1 > chunk_size else overhang1
            chunk2_size = chunk_size if overhang2 > chunk_size else overhang2
            rot_ind_wi_chunk1 = rot_on_block1 - chunk1 * chunk_size
            rot_ind_wi_chunk2 = rot_on_block2 - chunk2 * chunk_size

            chunk_offset_ij = chunk_pair_offset[
                chunk_offset_for_blocks_ij + chunk1 * n_chunks_for_block2 + chunk2
            ].item()
            chunk_offset_ji = chunk_pair_offset[
                chunk_offset_for_blocks_ji + chunk2 * n_chunks_for_block1 + chunk1
            ].item()

            if chunk_offset_ij == -1:
                assert i_energy == 0
            else:
                e2b_ij = energy2b[
                    chunk_offset_ij
                    + rot_ind_wi_chunk1 * chunk2_size
                    + rot_ind_wi_chunk2
                ]
                assert e2b_ij == i_energy

            if chunk_offset_ji == -1:
                assert i_energy == 0
            else:
                e2b_ji = energy2b[
                    chunk_offset_ji
                    + rot_ind_wi_chunk2 * chunk1_size
                    + rot_ind_wi_chunk1
                ]
                assert e2b_ji == i_energy


def test_build_multi_pose_interaction_graph(ubq_ig, torch_device):
    pdb_fname = "tmol/tests/data/pdb/1ubq.pdb"

    # Step 1: convert the IG that we're getting from disk
    # into the format that we expect from the score function
    pose_stack, rotamer_set, energyb1, sparse_indices, energies = (
        construct_stacked_faux_rotamer_set_and_sparse_energies_table_from_ig(
            ubq_ig, pdb_fname, torch_device
        )
    )

    n_energies_per_pose = 608852 // 2
    chunk_size = 16

    (chunk_pair_offset_for_block_pair, chunk_pair_offset, energy2b) = (
        build_interaction_graph(
            chunk_size,
            rotamer_set.n_rots_for_pose,
            rotamer_set.rot_offset_for_pose,
            rotamer_set.n_rots_for_block,
            rotamer_set.rot_offset_for_block,
            rotamer_set.pose_for_rot,
            rotamer_set.block_type_ind_for_rot,
            rotamer_set.block_ind_for_rot,
            sparse_indices,
            energies,
        )
    )
    assert chunk_pair_offset_for_block_pair.shape == (2, 76, 76)
    assert chunk_pair_offset_for_block_pair.dtype == torch.int64
    assert chunk_pair_offset_for_block_pair.device == torch_device
    assert chunk_pair_offset.shape == (10032,)
    assert chunk_pair_offset.dtype == torch.int64
    assert chunk_pair_offset.device == torch_device
    assert energy2b.shape == (2 * 608852,)
    assert energy2b.dtype == torch.float32
    assert energy2b.device == torch_device

    # print("testing energy2b accuracy")

    for i in range(sparse_indices[:, -1000:].shape[1]):
        i_energy = energies[i].item()
        pose = sparse_indices[0, i].item()
        rot1 = sparse_indices[1, i].item()
        rot2 = sparse_indices[2, i].item()
        block1 = rotamer_set.block_ind_for_rot[rot1].item()
        block2 = rotamer_set.block_ind_for_rot[rot2].item()
        chunk_offset_for_blocks_ij = chunk_pair_offset_for_block_pair[
            pose, block1, block2
        ].item()
        chunk_offset_for_blocks_ji = chunk_pair_offset_for_block_pair[
            pose, block2, block1
        ].item()
        if chunk_offset_for_blocks_ij == -1:
            if i_energy != 0:
                print(
                    i_energy,
                    pose,
                    rot1,
                    rot2,
                    block1,
                    block2,
                    chunk_offset_for_blocks_ij,
                    chunk_offset_for_blocks_jj,
                )
            assert i_energy == 0
        else:
            n_rots_for_block1 = rotamer_set.n_rots_for_block[pose, block1].item()
            n_rots_for_block2 = rotamer_set.n_rots_for_block[pose, block2].item()

            n_chunks_for_block1 = (n_rots_for_block1 - 1) // chunk_size + 1
            n_chunks_for_block2 = (n_rots_for_block2 - 1) // chunk_size + 1

            rot_on_block1 = rot1 - rotamer_set.rot_offset_for_block[pose, block1].item()
            rot_on_block2 = rot2 - rotamer_set.rot_offset_for_block[pose, block2].item()
            chunk1 = rot_on_block1 // chunk_size
            chunk2 = rot_on_block2 // chunk_size

            overhang1 = n_rots_for_block1 - chunk1 * chunk_size
            overhang2 = n_rots_for_block2 - chunk2 * chunk_size
            chunk1_size = chunk_size if overhang1 > chunk_size else overhang1
            chunk2_size = chunk_size if overhang2 > chunk_size else overhang2
            rot_ind_wi_chunk1 = rot_on_block1 - chunk1 * chunk_size
            rot_ind_wi_chunk2 = rot_on_block2 - chunk2 * chunk_size

            chunk_offset_ij = chunk_pair_offset[
                chunk_offset_for_blocks_ij + chunk1 * n_chunks_for_block2 + chunk2
            ].item()
            chunk_offset_ji = chunk_pair_offset[
                chunk_offset_for_blocks_ji + chunk2 * n_chunks_for_block1 + chunk1
            ].item()

            if chunk_offset_ij == -1:
                assert i_energy == 0
            else:
                e2b_ij = energy2b[
                    chunk_offset_ij
                    + rot_ind_wi_chunk1 * chunk2_size
                    + rot_ind_wi_chunk2
                ]
                assert e2b_ij == i_energy

            if chunk_offset_ji == -1:
                assert i_energy == 0
            else:
                e2b_ji = energy2b[
                    chunk_offset_ji
                    + rot_ind_wi_chunk2 * chunk1_size
                    + rot_ind_wi_chunk1
                ]
                assert e2b_ji == i_energy


def test_run_single_pose_simA(ubq_ig, torch_device):
    pdb_fname = "tmol/tests/data/pdb/1ubq.pdb"

    # Step 1: convert the IG that we're getting from disk
    # into the format that we expect from the score function
    ps, rotamer_set, energy1b, sparse_indices, energies = (
        construct_faux_rotamer_set_and_sparse_energies_table_from_ig(
            ubq_ig, pdb_fname, torch_device
        )
    )

    chunk_size = 16

    (chunk_pair_offset_for_block_pair, chunk_pair_offset, energy2b) = (
        build_interaction_graph(
            chunk_size,
            rotamer_set.n_rots_for_pose,
            rotamer_set.rot_offset_for_pose,
            rotamer_set.n_rots_for_block,
            rotamer_set.rot_offset_for_block,
            rotamer_set.pose_for_rot,
            rotamer_set.block_type_ind_for_rot,
            rotamer_set.block_ind_for_rot,
            sparse_indices,
            energies,
        )
    )

    packer_energy_tables = create_packer_energy_tables(
        ps,
        rotamer_set,
        chunk_size,
        chunk_pair_offset_for_block_pair,
        chunk_pair_offset,
        energy1b,
        energy2b,
    )

    scores, rotamer_assignments = run_simulated_annealing(packer_energy_tables)

    n_traj = 1 if torch_device == torch.device("cpu") else 1250
    assert scores.shape == (1, n_traj)
    scores_cpu = scores.cpu()
    # make sure that scores are in ascending order
    score_delta = scores_cpu[:, :-1] - scores_cpu[:, 1:]
    assert torch.all(score_delta <= 0.0)

    assert rotamer_assignments.shape == (1, n_traj, 76)


def test_run_two_poses_simA(ubq_ig, torch_device):
    pdb_fname = "tmol/tests/data/pdb/1ubq.pdb"

    ps, rotamer_set, energy1b, sparse_indices, energies = (
        construct_stacked_faux_rotamer_set_and_sparse_energies_table_from_ig(
            ubq_ig, pdb_fname, torch_device
        )
    )

    chunk_size = 16

    (chunk_pair_offset_for_block_pair, chunk_pair_offset, energy2b) = (
        build_interaction_graph(
            chunk_size,
            rotamer_set.n_rots_for_pose,
            rotamer_set.rot_offset_for_pose,
            rotamer_set.n_rots_for_block,
            rotamer_set.rot_offset_for_block,
            rotamer_set.pose_for_rot,
            rotamer_set.block_type_ind_for_rot,
            rotamer_set.block_ind_for_rot,
            sparse_indices,
            energies,
        )
    )

    packer_energy_tables = create_packer_energy_tables(
        ps,
        rotamer_set,
        chunk_size,
        chunk_pair_offset_for_block_pair,
        chunk_pair_offset,
        energy1b,
        energy2b,
    )

    scores, rotamer_assignments = run_simulated_annealing(packer_energy_tables)

    n_traj = 1 if torch_device == torch.device("cpu") else 1250
    assert scores.shape == (2, n_traj)
    assert scores.device == torch_device
    scores_cpu = scores.cpu()
    # make sure that scores are in ascending order
    score_delta = scores_cpu[:, :-1] - scores_cpu[:, 1:]
    assert torch.all(score_delta <= 0.0)

    assert rotamer_assignments.device == torch_device
    assert rotamer_assignments.shape == (2, n_traj, 76)
