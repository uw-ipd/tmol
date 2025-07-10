import numpy
import torch
import attr
import os
import pickle

from tmol.pack.datatypes import PackerEnergyTables
from tmol.pack.simulated_annealing import run_simulated_annealing
from tmol.pack.compiled.compiled import validate_energies, build_interaction_graph
from tmol.pack.rotamer.build_rotamers import RotamerSet
from tmol.utility.cumsum import exclusive_cumsum, exclusive_cumsum1d
from tmol.io import pose_stack_from_pdb


def load_ig_from_file(fname):
    with open("1ubq_ig", "rb") as f:
        return pickle.load(f)
    store = zarr.ZipStore(fname)
    zgroup = zarr.group(store=store)
    nres = zgroup.attrs["nres"]
    oneb_energies = {}
    # restype_groups = {}
    twob_energies = {}
    for i in range(1, nres + 1):
        oneb_arrname = "%d" % i
        restype_group_arrname = "%d_rtgroups" % i
        oneb_energies[oneb_arrname] = numpy.array(zgroup[oneb_arrname], dtype=float)
        # restype_groups[oneb_arrname] = numpy.array(
        #     zgroup[restype_group_arrname], dtype=int
        # )
        for j in range(i + 1, nres + 1):
            twob_arrname = "%d-%d" % (i, j)
            if twob_arrname in zgroup:
                twob_energies[twob_arrname] = numpy.array(
                    zgroup[twob_arrname], dtype=float
                )
    return oneb_energies, twob_energies


def test_load_ig():
    fname = "1ubq_ig"
    oneb, twob = load_ig_from_file(fname)
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


def construct_faux_rotamer_set_and_sparse_energies_table_from_ig(
    ig_fname, pdb_fname, device
):
    pose_stack = pose_stack_from_pdb(pdb_fname, device)
    oneb, twob = load_ig_from_file(ig_fname)

    # packer_energy_tables = create_chunk_twobody_energy_table(oneb, twob)

    n_rots = torch.zeros((76,), dtype=int)
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
    # print("rot_offset_for_block", rot_offset_for_block[0, :20])
    pose_for_rot = torch.zeros((n_rots_total,), dtype=torch.int64)

    rot_is_start_for_block = torch.zeros((n_rots_total), dtype=torch.int64)
    rot_is_start_for_block[rot_offset_for_block[0]] = 1
    block_ind_for_rot = torch.cumsum(rot_is_start_for_block, dim=0) - 1
    # print("block_ind_for_rot", block_ind_for_rot[:30])
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
                # print(f"i j are neighbors {i+1}, {j+1}, {i_n_rots} * {j_n_rots} = {i_n_rots * j_n_rots} offset {cumm_offset}")
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
        # print(i_inds.shape, j_inds.shape, ij_offset, ij_n_pairs)
        inds[1, ij_offset : (ij_offset + ij_n_pairs)] = i_inds
        inds[2, ij_offset : (ij_offset + ij_n_pairs)] = j_inds
        energies[ij_offset : (ij_offset + ij_n_pairs)] = ordered_energy_tables[
            pair
        ].view(-1)

    return pose_stack, rotamer_set, _d(energy1b), _d(inds), _d(energies)


def construct_stacked_faux_rotamer_set_and_sparse_energies_table_from_ig(
    ig_fname, pdb_fname, device
):
    pose_stack = pose_stack_from_pdb(pdb_fname, device)
    oneb, twob = load_ig_from_file(ig_fname)
    n_res = len(oneb)

    # packer_energy_tables = create_chunk_twobody_energy_table(oneb, twob)

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
    # print("rot_offset_for_block", rot_offset_for_block[0, :20])
    pose_for_rot = torch.zeros((n_rots_total,), dtype=torch.int64)
    pose_for_rot[n_rots_per_pose[0] :] = 1

    rot_is_start_for_block = torch.zeros((n_rots_total), dtype=torch.int64)
    rot_is_start_for_block[rot_offset_for_block.ravel()] = 1
    block_ind_for_rot = torch.remainder(
        torch.cumsum(rot_is_start_for_block, dim=0) - 1, n_res
    )
    # print("block_ind_for_rot", block_ind_for_rot[:30])
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
                # print(f"i j are neighbors {i+1}, {j+1}, {i_n_rots} * {j_n_rots} = {i_n_rots * j_n_rots} offset {cumm_offset}")
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
                # print(f"i j are neighbors {i+1}, {j+1}, {i_n_rots} * {j_n_rots} = {i_n_rots * j_n_rots} offset {cumm_offset}")
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
        # print(i_inds.shape, j_inds.shape, ij_offset, ij_n_pairs)
        inds[0, ij_offset : (ij_offset + ij_n_pairs)] = pose_inds
        inds[1, ij_offset : (ij_offset + ij_n_pairs)] = i_inds
        inds[2, ij_offset : (ij_offset + ij_n_pairs)] = j_inds
        energies[ij_offset : (ij_offset + ij_n_pairs)] = ordered_energy_tables[
            pair % len(twob)
        ].view(-1)

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


def test_construct_rotamer_set_and_sparse_energies_table_from_ig(torch_device):
    ig_fname = "1ubq_ig"
    pdb_fname = "tmol/tests/data/pdb/1ubq.pdb"

    # Step 1: convert the IG that we're getting from disk
    # into the format that we expect from the score function
    ps, rotamer_set, energy1b, sparse_indices, energies = (
        construct_faux_rotamer_set_and_sparse_energies_table_from_ig(
            ig_fname, pdb_fname, torch_device
        )
    )

    ps2, rotamer_set2, energy1b, sparse_indices2, energies2 = (
        construct_stacked_faux_rotamer_set_and_sparse_energies_table_from_ig(
            ig_fname, pdb_fname, torch_device
        )
    )

    # let's assert something??

    print(rotamer_set.n_rots_for_pose.shape)
    print(sparse_indices.shape)
    print(energies.shape)

    print(rotamer_set2.n_rots_for_pose.shape)
    print(sparse_indices2.shape)
    print(energies2.shape)


def test_build_interaction_graph(torch_device):
    # torch_device = torch.device("cpu")
    ig_fname = "1ubq_ig"
    pdb_fname = "tmol/tests/data/pdb/1ubq.pdb"

    # Step 1: convert the IG that we're getting from disk
    # into the format that we expect from the score function
    ps, rotamer_set, energy1b, sparse_indices, energies = (
        construct_faux_rotamer_set_and_sparse_energies_table_from_ig(
            ig_fname, pdb_fname, torch_device
        )
    )

    # print("rotamer_set.n_rots_for_pose", rotamer_set.n_rots_for_pose.dtype)
    # print("rotamer_set.rot_offset_for_pose",   rotamer_set.rot_offset_for_pose.dtype)
    # print("rotamer_set.n_rots_for_block",      rotamer_set.n_rots_for_block.dtype)
    # print("rotamer_set.rot_offset_for_block",  rotamer_set.rot_offset_for_block.dtype)
    # print("rotamer_set.pose_for_rot",          rotamer_set.pose_for_rot.dtype)
    # print("rotamer_set.block_type_ind_for_rot",rotamer_set.block_type_ind_for_rot.dtype)
    # print("rotamer_set.block_ind_for_rot",     rotamer_set.block_ind_for_rot.dtype)
    # print("sparse_indices",                    sparse_indices.dtype)
    # print("energies",                          energies.dtype)

    # print(rotamer_set.n_rots_for_pose)
    # print(sparse_indices.shape)
    # print(energies.shape)

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

    # for i in range(76):
    #     i_n_rots = rotamer_set.n_rots_for_block[0, i].item()
    #     i_n_chunks = (i_n_rots - 1) // chunk_size + 1;
    #     print(f"i: {i} {i_n_rots} {i_n_chunks}")
    #     for j in range(76):
    #         j_n_rots = rotamer_set.n_rots_for_block[0, j].item()
    #         j_n_chunks = (j_n_rots - 1) // chunk_size + 1;
    #         print(f" j: {j} {j_n_rots} {j_n_chunks}")
    #         ij_chunk_offset_offset = chunk_pair_offset_for_block_pair[0, i, j].item()
    #         print(f"{i} w {j}: chunk_offset_offset {chunk_pair_offset_for_block_pair[0, i, j]}")
    #         if ij_chunk_offset_offset != -1:
    #             ij_chunk_offsets = chunk_pair_offset[ij_chunk_offset_offset:(ij_chunk_offset_offset + i_n_chunks*j_n_chunks)].view(i_n_chunks, j_n_chunks)
    #             print(f"{i} {j} chunk_offsets:")
    #             print(ij_chunk_offsets)
    # return

    # print(chunk_pair_offset_for_block_pair.shape)
    # print(chunk_pair_offset.shape)
    # print(energy2b.shape)

    assert chunk_pair_offset_for_block_pair.device == torch_device
    assert chunk_pair_offset.device == torch_device
    assert energy2b.device == torch_device

    # print(chunk_pair_offset_for_block_pair[0,:10, :10])
    # print(chunk_pair_offset[:20])

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

            # print(
            #     "pose                ", pose  ,
            #     "rot1                ", rot1  ,
            #     "rot2                ", rot2  ,
            #     "block1              ", block1,
            #     "block2              ", block2,
            #     "chunk_offset_for_blocks_ij", chunk_offset_for_blocks_ij,
            #     "chunk_offset_for_blocks_ji", chunk_offset_for_blocks_ji,
            #     "n_rots_for_block1   ", n_rots_for_block1   ,
            #     "n_rots_for_block2   ", n_rots_for_block2   ,
            #     "n_chunks_for_block1 ", n_chunks_for_block1 ,
            #     "n_chunks_for_block2 ", n_chunks_for_block2 ,
            #     "rot_on_block1       ", rot_on_block1       ,
            #     "rot_on_block2       ", rot_on_block2       ,
            #     "chunk1              ", chunk1              ,
            #     "chunk2              ", chunk2              ,
            #     "overhang1           ", overhang1           ,
            #     "overhang2           ", overhang2           ,
            #     "chunk1_size         ", chunk1_size         ,
            #     "chunk2_size         ", chunk2_size         ,
            #     "rot_ind_wi_chunk1   ", rot_ind_wi_chunk1   ,
            #     "rot_ind_wi_chunk2   ", rot_ind_wi_chunk2   ,
            #     "chunk_offset_ij     ", chunk_offset_ij     ,
            #     "chunk_offset_ji     ", chunk_offset_ji     ,
            # )

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


def test_build_multi_pose_interaction_graph(torch_device):
    # torch_device = torch.device("cpu")
    ig_fname = "1ubq_ig"
    pdb_fname = "tmol/tests/data/pdb/1ubq.pdb"

    # Step 1: convert the IG that we're getting from disk
    # into the format that we expect from the score function
    rotamer_set, energyb1, sparse_indices, energies = (
        construct_stacked_faux_rotamer_set_and_sparse_energies_table_from_ig(
            ig_fname, pdb_fname, torch_device
        )
    )

    n_energies_per_pose = 608852 // 2
    print("n_energies_per_pose", n_energies_per_pose)

    print("rotamer_set.n_rots_for_pose", rotamer_set.n_rots_for_pose.dtype)
    print("rotamer_set.rot_offset_for_pose", rotamer_set.rot_offset_for_pose.dtype)
    print("rotamer_set.n_rots_for_block", rotamer_set.n_rots_for_block.dtype)
    print("rotamer_set.rot_offset_for_block", rotamer_set.rot_offset_for_block.dtype)
    print("rotamer_set.pose_for_rot", rotamer_set.pose_for_rot.dtype)
    print(
        "rotamer_set.block_type_ind_for_rot", rotamer_set.block_type_ind_for_rot.dtype
    )
    print("rotamer_set.block_ind_for_rot", rotamer_set.block_ind_for_rot.dtype)
    print("sparse_indices", sparse_indices.dtype)
    print("energies", energies.dtype)

    print("rotamer_set.n_rots_for_pose", rotamer_set.n_rots_for_pose.shape)
    print("rotamer_set.rot_offset_for_pose", rotamer_set.rot_offset_for_pose.shape)
    print("rotamer_set.n_rots_for_block", rotamer_set.n_rots_for_block.shape)
    print("rotamer_set.rot_offset_for_block", rotamer_set.rot_offset_for_block.shape)
    print("rotamer_set.pose_for_rot", rotamer_set.pose_for_rot.shape)
    print(
        "rotamer_set.block_type_ind_for_rot", rotamer_set.block_type_ind_for_rot.shape
    )
    print("rotamer_set.block_ind_for_rot", rotamer_set.block_ind_for_rot.shape)
    print("sparse_indices", sparse_indices.shape)
    print("energies", energies.shape)

    print("rotamer_set.n_rots_for_pose", rotamer_set.n_rots_for_pose)
    print("rotamer_set.rot_offset_for_pose", rotamer_set.rot_offset_for_pose)
    print("rotamer_set.n_rots_for_block", rotamer_set.n_rots_for_block[:, :5])
    print("rotamer_set.rot_offset_for_block", rotamer_set.rot_offset_for_block[:, :5])
    print(
        "rotamer_set.pose_for_rot",
        rotamer_set.pose_for_rot[:5],
        rotamer_set.pose_for_rot[1518:1523],
    )
    print(
        "rotamer_set.block_type_ind_for_rot",
        rotamer_set.block_type_ind_for_rot[:5],
        rotamer_set.block_type_ind_for_rot[1518:1523],
    )
    print(
        "rotamer_set.block_ind_for_rot",
        rotamer_set.block_ind_for_rot[:5],
        rotamer_set.block_ind_for_rot[1518:1523],
    )
    print(
        "sparse_indices",
        sparse_indices[:, :5],
        sparse_indices[:, n_energies_per_pose : (n_energies_per_pose + 5)],
    )
    print(
        "energies",
        energies[:5],
        energies[n_energies_per_pose : (n_energies_per_pose + 5)],
    )

    print(rotamer_set.n_rots_for_pose)
    print(sparse_indices.shape)
    print(energies.shape)

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

    # print(chunk_pair_offset_for_block_pair.shape)
    # print(chunk_pair_offset.shape)
    # print(energy2b.shape)

    assert chunk_pair_offset_for_block_pair.device == torch_device
    assert chunk_pair_offset.device == torch_device
    assert energy2b.device == torch_device

    # print(chunk_pair_offset_for_block_pair[0,:10, :10])
    # print(chunk_pair_offset[:20])

    print("testing energy2b accuracy")

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

            # print(
            #     "pose                ", pose  ,
            #     "rot1                ", rot1  ,
            #     "rot2                ", rot2  ,
            #     "block1              ", block1,
            #     "block2              ", block2,
            #     "chunk_offset_for_blocks_ij", chunk_offset_for_blocks_ij,
            #     "chunk_offset_for_blocks_ji", chunk_offset_for_blocks_ji,
            #     "n_rots_for_block1   ", n_rots_for_block1   ,
            #     "n_rots_for_block2   ", n_rots_for_block2   ,
            #     "n_chunks_for_block1 ", n_chunks_for_block1 ,
            #     "n_chunks_for_block2 ", n_chunks_for_block2 ,
            #     "rot_on_block1       ", rot_on_block1       ,
            #     "rot_on_block2       ", rot_on_block2       ,
            #     "chunk1              ", chunk1              ,
            #     "chunk2              ", chunk2              ,
            #     "overhang1           ", overhang1           ,
            #     "overhang2           ", overhang2           ,
            #     "chunk1_size         ", chunk1_size         ,
            #     "chunk2_size         ", chunk2_size         ,
            #     "rot_ind_wi_chunk1   ", rot_ind_wi_chunk1   ,
            #     "rot_ind_wi_chunk2   ", rot_ind_wi_chunk2   ,
            #     "chunk_offset_ij     ", chunk_offset_ij     ,
            #     "chunk_offset_ji     ", chunk_offset_ji     ,
            # )

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


def test_build_packer_energy_tables(torch_device):
    # torch_device = torch.device("cpu")
    ig_fname = "1ubq_ig"
    pdb_fname = "tmol/tests/data/pdb/1ubq.pdb"

    # Step 1: convert the IG that we're getting from disk
    # into the format that we expect from the score function
    ps, rotamer_set, energy1b, sparse_indices, energies = (
        construct_faux_rotamer_set_and_sparse_energies_table_from_ig(
            ig_fname, pdb_fname, torch_device
        )
    )

    # print("rotamer_set.n_rots_for_pose", rotamer_set.n_rots_for_pose.dtype)
    # print("rotamer_set.rot_offset_for_pose",   rotamer_set.rot_offset_for_pose.dtype)
    # print("rotamer_set.n_rots_for_block",      rotamer_set.n_rots_for_block.dtype)
    # print("rotamer_set.rot_offset_for_block",  rotamer_set.rot_offset_for_block.dtype)
    # print("rotamer_set.pose_for_rot",          rotamer_set.pose_for_rot.dtype)
    # print("rotamer_set.block_type_ind_for_rot",rotamer_set.block_type_ind_for_rot.dtype)
    # print("rotamer_set.block_ind_for_rot",     rotamer_set.block_ind_for_rot.dtype)
    # print("sparse_indices",                    sparse_indices.dtype)
    # print("energies",                          energies.dtype)

    # print(rotamer_set.n_rots_for_pose)
    # print(sparse_indices.shape)
    # print(energies.shape)

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
    print(scores.shape)
    print(scores[:, 0])
    print(rotamer_assignments.shape)


def aa_neighb_nonzero_submatrix(twob, rtg1, rtg2):
    # rtg1 = exclusive_cumsum(rtg1_start)
    # rtg2 = exclusive_cumsum(rtg2_start)

    rtg1_start = numpy.concatenate((numpy.ones(1, dtype=int), rtg1[1:] - rtg1[:-1]))
    rtg2_start = numpy.concatenate((numpy.ones(1, dtype=int), rtg2[1:] - rtg2[:-1]))

    n_rtg1 = numpy.sum(rtg1_start)
    n_rtg2 = numpy.sum(rtg2_start)

    rtg1_offsets = numpy.nonzero(rtg1_start)[0]
    rtg2_offsets = numpy.nonzero(rtg2_start)[0]

    rtg_nrots1 = numpy.concatenate(
        (
            rtg1_offsets[1:] - rtg1_offsets[:-1],
            numpy.full((1,), rtg1_start.shape[0] - rtg1_offsets[-1], dtype=int),
        )
    )
    rtg_nrots2 = numpy.concatenate(
        (
            rtg2_offsets[1:] - rtg2_offsets[:-1],
            numpy.full((1,), rtg2_start.shape[0] - rtg2_offsets[-1], dtype=int),
        )
    )
    # print(rtg_nrots1)
    # print(rtg_nrots2)

    fine_offsets = numpy.full((n_rtg1, n_rtg2), -1, dtype=int)
    count = 0
    for i in range(n_rtg1):
        i_slice = slice(rtg1_offsets[i], (rtg1_offsets[i] + rtg_nrots1[i]))
        for j in range(n_rtg2):
            j_slice = slice(rtg2_offsets[j], (rtg2_offsets[j] + rtg_nrots2[j]))
            e2b_slice = twob[i_slice, j_slice]
            # print(i, rtg_nrots1[i], j, rtg_nrots2[j], e2b_slice.shape)
            assert (rtg_nrots1[i], rtg_nrots2[j]) == e2b_slice.shape
            if numpy.any(e2b_slice != 0):
                fine_offsets[i, j] = count
                count += rtg_nrots1[i] * rtg_nrots2[j]
    rtg_sparse_matrix = numpy.zeros((count,), dtype=float)
    for i in range(n_rtg1):
        i_slice = slice(rtg1_offsets[i], (rtg1_offsets[i] + rtg_nrots1[i]))
        for j in range(n_rtg2):
            j_slice = slice(rtg2_offsets[j], (rtg2_offsets[j] + rtg_nrots2[j]))
            ij_offset = fine_offsets[i, j]
            if ij_offset >= 0:
                e2b_slice = twob[i_slice, j_slice].reshape(-1)
                insert_slice = slice(
                    ij_offset, (ij_offset + rtg_nrots1[i] * rtg_nrots2[j])
                )
                rtg_sparse_matrix[insert_slice] = e2b_slice
    return fine_offsets, rtg_sparse_matrix


def count_aa_sparse_memory_usage(oneb, restype_groups, twob):
    nres = len(oneb)

    count_sparse = 0
    count_dense = 0
    count_nonzero = 0
    for i in range(nres):
        for j in range(i + 1, nres):
            one_name = "{}".format(i + 1)
            two_name = "{}".format(j + 1)
            onetwo_name = "{}-{}".format(i + 1, j + 1)
            if onetwo_name in twob:
                onetwo_twob = twob[onetwo_name]
                fine_offsets, rtg_sparse_matrix = aa_neighb_nonzero_submatrix(
                    onetwo_twob, restype_groups[one_name], restype_groups[two_name]
                )
                count_dense += onetwo_twob.shape[0] * onetwo_twob.shape[1]
                count_sparse += rtg_sparse_matrix.shape[0]
                count_nonzero += numpy.nonzero(rtg_sparse_matrix)[0].shape[0]
    return count_dense, count_sparse, count_nonzero


def dont_test_nonzero_submatrix():
    fname = "1ubq_redes_noex.zarr"
    oneb, restype_groups, twob = load_ig_from_file(fname)

    dense, sparse, nonzero = count_aa_sparse_memory_usage(oneb, restype_groups, twob)
    print(dense, sparse, nonzero)


def dont_test_aasparse_mat_repack():
    fnames = [
        "1wzbFHA",
        "1qtxFHB",
        "1kd8FHB",
        "1ojhFHA",
        "1ff4FHA",
        "1vmgFHA",
        "1u36FHA",
        "1w0nFHA",
    ]
    for fname in fnames:
        path_to_zarr_file = "zarr_igs2/repack/" + fname + "_repack_noex.zarr"
        assert os.path.isfile(path_to_zarr_file)
        oneb, restype_groups, twob = load_ig_from_file(path_to_zarr_file)
        dense, sparse, nonzero = count_aa_sparse_memory_usage(
            oneb, restype_groups, twob
        )
        print(dense, sparse, nonzero, nonzero / dense, sparse / dense, nonzero / sparse)


def dont_test_aasparse_mat_redes_ex1ex2():
    fnames = [
        "1wzbFHA",
        "1qtxFHB",
        "1kd8FHB",
        "1ojhFHA",
        "1ff4FHA",
        "1vmgFHA",
        "1u36FHA",
        "1w0nFHA",
    ]
    for fname in fnames:
        path_to_zarr_file = "zarr_igs2/redes_ex1ex2/" + fname + "_redes_ex1ex2.zarr"
        assert os.path.isfile(path_to_zarr_file)
        oneb, restype_groups, twob = load_ig_from_file(path_to_zarr_file)
        dense, sparse, nonzero = count_aa_sparse_memory_usage(
            oneb, restype_groups, twob
        )
        print(dense, sparse, nonzero, nonzero / dense, sparse / dense, nonzero / sparse)


def find_nonzero_submatrix_chunks(twob, chunk_size):

    n_rots1 = twob.shape[0]
    n_rots2 = twob.shape[1]
    n_chunks1 = int((n_rots1 - 1) // chunk_size + 1)
    n_chunks2 = int((n_rots2 - 1) // chunk_size + 1)

    # fine_offsets = numpy.full((n_chunks1, n_chunks2), -1, dtype=int)
    chunk_pair_nenergies = numpy.full((n_chunks1, n_chunks2), 0, dtype=int)
    for i in range(n_chunks1):
        i_nrots = min(n_rots1 - i * chunk_size, chunk_size)
        i_slice = slice(i * chunk_size, i * chunk_size + i_nrots)
        for j in range(n_chunks2):
            j_nrots = min(n_rots2 - j * chunk_size, chunk_size)
            j_slice = slice(j * chunk_size, j * chunk_size + j_nrots)
            e2b_slice = twob[i_slice, j_slice]
            # print(i, rtg_nrots1[i], j, rtg_nrots2[j], e2b_slice.shape)
            assert (i_nrots, j_nrots) == e2b_slice.shape
            if numpy.any(e2b_slice != 0):
                chunk_pair_nenergies[i, j] = i_nrots * j_nrots
    return chunk_pair_nenergies


def chunk_nonzero_submatrix(twob, chunk_pair_nenergies, chunk_size):
    n_rots1 = twob.shape[0]
    n_rots2 = twob.shape[1]
    n_chunks1 = int((n_rots1 - 1) // chunk_size + 1)
    n_chunks2 = int((n_rots2 - 1) // chunk_size + 1)

    nenergies = numpy.sum(chunk_pair_nenergies)
    fine_offsets = exclusive_cumsum(chunk_pair_nenergies.reshape(-1)).reshape(
        chunk_pair_nenergies.shape
    )
    fine_offsets[chunk_pair_nenergies == 0] = -1
    rtg_sparse_matrix = numpy.zeros((nenergies,), dtype=float)
    for i in range(n_chunks1):
        i_nrots = min(n_rots1 - i * chunk_size, chunk_size)
        i_slice = slice(i * chunk_size, i * chunk_size + i_nrots)
        for j in range(n_chunks2):
            j_nrots = min(n_rots2 - j * chunk_size, chunk_size)
            j_slice = slice(j * chunk_size, j * chunk_size + j_nrots)
            ij_offset = fine_offsets[i, j]
            if ij_offset >= 0:
                e2b_slice = twob[i_slice, j_slice].reshape(-1)
                insert_slice = slice(ij_offset, (ij_offset + i_nrots * j_nrots))
                rtg_sparse_matrix[insert_slice] = e2b_slice
    return fine_offsets, rtg_sparse_matrix


def count_chunk_sparse_memory_usage(oneb, twob, chunk_size):
    nres = len(oneb)

    count_sparse = 0
    count_dense = 0
    count_nonzero = 0
    for i in range(nres):
        for j in range(i + 1, nres):
            one_name = "{}".format(i + 1)
            two_name = "{}".format(j + 1)
            onetwo_name = "{}-{}".format(i + 1, j + 1)
            if onetwo_name in twob:
                onetwo_twob = twob[onetwo_name]
                chunk_pair_nenergies = find_nonzero_submatrix_chunks(twob, chunk_size)

                # fine_offsets, rtg_sparse_matrix = chunk_nonzero_submatrix(
                #     onetwo_twob, chunk_pair_nenergies, chunk_size
                # )
                count_dense += onetwo_twob.shape[0] * onetwo_twob.shape[1]
                count_sparse += numpy.sum(
                    chunk_pair_nenergies
                )  # rtg_sparse_matrix.shape[0]
                count_nonzero += numpy.nonzero(rtg_sparse_matrix)[0].shape[0]
    return count_dense, count_sparse, count_nonzero


def dont_test_aasparse_mat_repack():
    fnames = [
        "1wzbFHA",
        "1qtxFHB",
        "1kd8FHB",
        "1ojhFHA",
        "1ff4FHA",
        "1vmgFHA",
        "1u36FHA",
        "1w0nFHA",
    ]
    for fname in fnames:
        path_to_zarr_file = "zarr_igs2/repack/" + fname + "_repack_noex.zarr"
        assert os.path.isfile(path_to_zarr_file)
        oneb, restype_groups, twob = load_ig_from_file(path_to_zarr_file)
        dense, sparse, nonzero = count_aa_sparse_memory_usage(
            oneb, restype_groups, twob
        )
        print(dense, sparse, nonzero, nonzero / dense, sparse / dense, nonzero / sparse)


def dont_test_aasparse_mat_redes_ex1ex2():
    fnames = [
        "1wzbFHA",
        "1qtxFHB",
        "1kd8FHB",
        "1ojhFHA",
        "1ff4FHA",
        "1vmgFHA",
        "1u36FHA",
        "1w0nFHA",
    ]
    results = {}
    for fname in fnames:
        print(fname)
        path_to_zarr_file = "zarr_igs2/redes_ex1ex2/" + fname + "_redes_ex1ex2.zarr"
        assert os.path.isfile(path_to_zarr_file)
        oneb, twob = load_ig_from_file(path_to_zarr_file)
        for chunk in [8, 16, 32, 64]:
            results[(fname, chunk)] = count_chunk_sparse_memory_usage(oneb, twob, chunk)
            print(results[(fname, chunk)])
    print()
    print()
    for chunk in [8, 16, 32, 64]:
        print(chunk)
        for fname in fnames:
            dense, sparse, nonzero = results[(fname, chunk)]
            print(
                dense,
                sparse,
                nonzero,
                nonzero / dense,
                sparse / dense,
                nonzero / sparse,
            )


def count_table_size(twob, restype_groups):
    rtg_start = [1] + restype
    count = 0
    for tabname in twob:
        shape = twob[tabname].shape
        count += shape[0] * shape[1]
    return count


def create_twobody_energy_table(oneb, twob):
    nres = len(oneb)
    offsets = numpy.zeros((nres, nres), dtype=numpy.int64)
    nenergies = numpy.zeros((nres, nres), dtype=int)
    nrotamers_for_res = numpy.array(
        [oneb["{}".format(i + 1)].shape[0] for i in range(nres)], dtype=int
    )
    nrots_total = numpy.sum(nrotamers_for_res)
    oneb_offsets = exclusive_cumsum(nrotamers_for_res)

    energy1b = numpy.zeros(nrots_total, dtype=float)
    res_for_rot = numpy.zeros(nrots_total, dtype=int)
    for i in range(nres):
        tablename = "{}".format(i + 1)
        table = oneb[tablename]
        start = oneb_offsets[i]
        energy1b[(start) : (start + table.shape[0])] = table
        res_for_rot[(start) : (start + table.shape[0])] = i

    for i in range(nres):
        for j in range(i + 1, nres):
            tabname = "{}-{}".format(i + 1, j + 1)
            if tabname in twob:
                nenergies[i, j] = nrotamers_for_res[i] * nrotamers_for_res[j]
                nenergies[j, i] = nrotamers_for_res[i] * nrotamers_for_res[j]

    twob_offsets = exclusive_cumsum(nenergies.reshape(-1)).reshape(nenergies.shape)
    n_rpes_total = numpy.sum(nenergies)
    energy2b = numpy.zeros(n_rpes_total, dtype=float)
    for i in range(nres):
        for j in range(i + 1, nres):
            if nenergies[i, j] == 0:
                continue
            tabname = "{}-{}".format(i + 1, j + 1)
            table = twob[tabname]
            start_ij = twob_offsets[i, j]
            extent = nenergies[i, j]
            energy2b[start_ij : (start_ij + extent)] = table.reshape(-1)
            start_ji = twob_offsets[j, i]
            energy2b[start_ji : (start_ji + extent)] = table.T.reshape(-1)

    return PackerEnergyTables(
        nrotamers_for_res=nrotamers_for_res,
        oneb_offsets=oneb_offsets,
        res_for_rot=res_for_rot,
        nenergies=nenergies,
        twob_offsets=twob_offsets,
        energy1b=energy1b,
        energy2b=energy2b,
    )


def create_chunk_twobody_energy_table(oneb, twob, chunk_size):
    nres = len(oneb)
    offsets = numpy.zeros((nres, nres), dtype=numpy.int64)
    nrotamers_for_res = numpy.array(
        [oneb["{}".format(i + 1)].shape[0] for i in range(nres)], dtype=int
    )
    nrots_total = numpy.sum(nrotamers_for_res)
    oneb_offsets = exclusive_cumsum(nrotamers_for_res)

    energy1b = numpy.zeros(nrots_total, dtype=float)
    res_for_rot = numpy.zeros(nrots_total, dtype=int)
    for i in range(nres):
        tablename = "{}".format(i + 1)
        table = oneb[tablename]
        start = oneb_offsets[i]
        energy1b[(start) : (start + table.shape[0])] = table
        res_for_rot[(start) : (start + table.shape[0])] = i

    # sparse_tables = {}
    # fine_offsets = {}
    chunk_pair_nenergies = {}
    respair_nenergies = numpy.zeros((nres, nres), dtype=int)
    respair_nchunkpairs = numpy.zeros((nres, nres), dtype=int)
    for i in range(nres):
        for j in range(i + 1, nres):
            tabname = "{}-{}".format(i + 1, j + 1)
            if tabname in twob:
                ij_chunk_pair_nenergies = find_nonzero_submatrix_chunks(
                    twob[tabname], chunk_size
                )
                chunk_pair_nenergies[(i, j)] = ij_chunk_pair_nenergies
                ij_nenergies = numpy.sum(ij_chunk_pair_nenergies)
                respair_nenergies[i, j] = ij_nenergies
                respair_nenergies[j, i] = ij_nenergies
                ij_n_sparse_pairs = (
                    ij_chunk_pair_nenergies.shape[0] * ij_chunk_pair_nenergies.shape[1]
                )
                respair_nchunkpairs[i, j] = ij_n_sparse_pairs
                respair_nchunkpairs[j, i] = ij_n_sparse_pairs

    twob_offsets = exclusive_cumsum(respair_nenergies.reshape(-1)).reshape(
        respair_nenergies.shape
    )
    chunk_offset_offsets = exclusive_cumsum(respair_nchunkpairs.reshape(-1)).reshape(
        respair_nchunkpairs.shape
    )

    n_rpes_total = numpy.sum(respair_nenergies)
    n_chunk_offsets_total = numpy.sum(respair_nchunkpairs)

    energy2b = numpy.zeros(n_rpes_total, dtype=float)
    fine_chunk_offsets = numpy.zeros(n_chunk_offsets_total, dtype=int)

    for i in range(nres):
        i_nrotamers = nrotamers_for_res[i]
        i_nchunks = int((i_nrotamers - 1) // chunk_size + 1)
        for j in range(i + 1, nres):
            if respair_nenergies[i, j] == 0:
                continue
            tabname = "{}-{}".format(i + 1, j + 1)
            ij_twob = twob[tabname]

            j_nrotamers = nrotamers_for_res[j]
            j_nchunks = int((j_nrotamers - 1) // chunk_size + 1)

            start_ij = twob_offsets[i, j]
            start_ji = twob_offsets[j, i]

            ij_chunk_pair_nenergies = chunk_pair_nenergies[(i, j)]

            assert (i_nchunks, j_nchunks) == ij_chunk_pair_nenergies.shape

            ij_fine_offsets, ij_sparse_matrix = chunk_nonzero_submatrix(
                ij_twob, ij_chunk_pair_nenergies, chunk_size
            )

            ji_fine_offsets, ji_sparse_matrix = chunk_nonzero_submatrix(
                ij_twob.T, ij_chunk_pair_nenergies.T, chunk_size
            )

            ij_n_chunk_pairs = i_nchunks * j_nchunks

            ij_chunk_offset_offset = chunk_offset_offsets[i, j]
            ij_chunk_offset_slice = slice(
                ij_chunk_offset_offset, ij_chunk_offset_offset + ij_n_chunk_pairs
            )
            fine_chunk_offsets[ij_chunk_offset_slice] = ij_fine_offsets.reshape(-1)

            ji_chunk_offset_offset = chunk_offset_offsets[j, i]
            ji_chunk_offset_slice = slice(
                ji_chunk_offset_offset, ji_chunk_offset_offset + ij_n_chunk_pairs
            )
            fine_chunk_offsets[ji_chunk_offset_slice] = ji_fine_offsets.reshape(-1)

            ij_e2b_offset = twob_offsets[i, j]
            ij_e2b_slice = slice(
                ij_e2b_offset, ij_e2b_offset + ij_sparse_matrix.shape[0]
            )
            energy2b[ij_e2b_slice] = ij_sparse_matrix

            ji_e2b_offset = twob_offsets[j, i]
            ji_e2b_slice = slice(
                ji_e2b_offset, ji_e2b_offset + ji_sparse_matrix.shape[0]
            )
            energy2b[ji_e2b_slice] = ji_sparse_matrix

    chunk_size = numpy.full((1,), chunk_size, dtype=int)

    return PackerEnergyTables(
        nrotamers_for_res=nrotamers_for_res,
        oneb_offsets=oneb_offsets,
        res_for_rot=res_for_rot,
        respair_nenergies=respair_nenergies,
        chunk_size=chunk_size,
        chunk_offset_offsets=chunk_offset_offsets,
        twob_offsets=twob_offsets,
        fine_chunk_offsets=fine_chunk_offsets,
        energy1b=energy1b,
        energy2b=energy2b,
    )


def test_energy_table_construction():
    fname = "1ubq_redes_noex.zarr"
    fname = "1ubq_ig"
    oneb, twob = load_ig_from_file(fname)
    chunk_size = 16
    energy_tables = create_chunk_twobody_energy_table(oneb, twob, chunk_size)
    et = energy_tables

    nrots_total = et.res_for_rot.shape[0]
    # pick two residues, 12 and 14
    assert "12-14" in twob

    i_res_nrots = et.nrotamers_for_res[11]
    j_res_nrots = et.nrotamers_for_res[13]

    i_nchunks = (i_res_nrots - 1) // chunk_size + 1
    j_nchunks = (j_res_nrots - 1) // chunk_size + 1

    for i in range(et.oneb_offsets[11], et.oneb_offsets[11] + et.nrotamers_for_res[11]):
        ires = et.res_for_rot[i]
        assert ires == 11
        i_rot_on_res = i - et.oneb_offsets[ires]
        for j in range(
            et.oneb_offsets[13], et.oneb_offsets[13] + et.nrotamers_for_res[13]
        ):
            jres = et.res_for_rot[j]
            assert jres == 13
            j_rot_on_res = j - et.oneb_offsets[jres]
            if et.respair_nenergies[ires, jres] == 0:
                continue

            i_chunk = i_rot_on_res // chunk_size
            j_chunk = j_rot_on_res // chunk_size
            i_rot_in_chunk = i_rot_on_res - chunk_size * i_chunk
            j_rot_in_chunk = j_rot_on_res - chunk_size * j_chunk
            i_chunk_size = min(
                chunk_size, et.nrotamers_for_res[11] - chunk_size * i_chunk
            )
            j_chunk_size = min(
                chunk_size, et.nrotamers_for_res[13] - chunk_size * j_chunk
            )

            ij_chunk_offset = et.chunk_offset_offsets[ires, jres]
            ji_chunk_offset = et.chunk_offset_offsets[jres, ires]

            ij_chunk_start = et.fine_chunk_offsets[
                ij_chunk_offset + i_chunk * j_nchunks + j_chunk
            ]
            ji_chunk_start = et.fine_chunk_offsets[
                ji_chunk_offset + j_chunk * i_nchunks + i_chunk
            ]

            ij_energy = et.energy2b[
                et.twob_offsets[ires, jres]
                + ij_chunk_start
                + i_rot_in_chunk * j_chunk_size
                + j_rot_in_chunk
            ]
            ji_energy = et.energy2b[
                et.twob_offsets[jres, ires]
                + ji_chunk_start
                + j_rot_in_chunk * i_chunk_size
                + i_rot_in_chunk
            ]

            assert ij_energy == ji_energy  # exact equality ok since they are copies


def test_run_sim_annealing():
    torch_device = torch.device("cuda")

    fname = "1ubq_redes_noex.zarr"
    fname = "1ubq_ig"
    oneb, twob = load_ig_from_file(fname)
    chunk_size = 16
    et = create_chunk_twobody_energy_table(oneb, twob, chunk_size)

    print("nrotamers", et.res_for_rot.shape[0])
    et_dev = et.to(torch_device)

    scores, rotamer_assignments = run_simulated_annealing(et_dev)

    sort_scores, sort_inds = scores[0, :].sort()
    nkeep = min(scores.shape[0], 20)
    best_scores = sort_scores[0:nkeep].cpu()
    best_score_inds = sort_inds[0:nkeep]
    best_rot_assignments = rotamer_assignments[best_score_inds, :].cpu()

    scores = best_scores.cpu()
    rotamer_assignments = best_rot_assignments.cpu()

    # scores = scores[0:nkeep].cpu()
    # rotamer_assignments = rotamer_assignments[0:nkeep, :].cpu()

    # print("scores", scores, best_scores)
    # print("rotamer_assignments", rotamer_assignments.shape)
    # print("assignment 0", rotamer_assignments[0,0:20])
    # print("sorted assignment 0", best_rot_assignments[0,0:20])

    validated_scores = validate_energies(
        et.nrotamers_for_res,
        et.oneb_offsets,
        et.res_for_rot,
        et.respair_nenergies,
        et.chunk_size,
        et.chunk_offset_offsets,
        et.twob_offsets,
        et.fine_chunk_offsets,
        et.energy1b,
        et.energy2b,
        rotamer_assignments,
    )

    print("validated scores?", validated_scores)
    torch.testing.assert_allclose(scores, validated_scores)


def test_run_sim_annealing_on_repacking_jobs():
    chunk_size = 16
    torch_device = torch.device("cuda")
    fnames = [
        "1wzbFHA",
        "1qtxFHB",
        "1kd8FHB",
        "1ojhFHA",
        "1ff4FHA",
        "1vmgFHA",
        "1u36FHA",
        "1w0nFHA",
    ]
    # fnames = ["1u36FHA"]
    for fname in fnames:
        path_to_zarr_file = "zarr_igs/repack/" + fname + "_repack.zarr"
        oneb, twob = load_ig_from_file(path_to_zarr_file)
        et = create_chunk_twobody_energy_table(oneb, twob, chunk_size)
        # print("nrotamers", et.res_for_rot.shape[0])
        # print("table size", count_table_size(twob))
        et_dev = et.to(torch_device)

        # print("running sim annealing on", fname)
        scores, rotamer_assignments = run_simulated_annealing(et_dev)
        print("scores", scores)

        scores_temp = scores
        # scores = scores.cpu().numpy()
        print("scores again", scores)
        numpy.set_printoptions(threshold=1e5)
        # print("scores", fname)
        # for i in range(scores.shape[1]):
        #    print(scores[0, i], scores[1, i])

        # scores = scores_temp[1, :]
        scores = scores[0, :]
        sort_scores, sort_inds = scores.sort()
        nkeep = min(scores.shape[0], 5)
        best_scores = sort_scores[0:nkeep]
        best_score_inds = sort_inds[0:nkeep]
        best_rot_assignments = rotamer_assignments[best_score_inds, :]

        scores = best_scores.cpu()

        rotamer_assignments = best_rot_assignments.cpu()
        # print("scores", " ".join([str(scores[i].item()) for i in range(scores.shape[0])]))

        validated_scores = validate_energies(
            et.nrotamers_for_res,
            et.oneb_offsets,
            et.res_for_rot,
            et.respair_nenergies,
            et.chunk_size,
            et.chunk_offset_offsets,
            et.twob_offsets,
            et.fine_chunk_offsets,
            et.energy1b,
            et.energy2b,
            rotamer_assignments,
        )

        # print("validated scores?", validated_scores)
        torch.testing.assert_allclose(scores, validated_scores)


def test_run_sim_annealing_on_redes_ex1ex2_jobs():
    chunk_size = 16
    torch_device = torch.device("cuda")
    fnames = [
        "1wzbFHA",
        "1qtxFHB",
        "1kd8FHB",
        "1ojhFHA",
        "1ff4FHA",
        "1vmgFHA",
        "1u36FHA",
        "1w0nFHA",
    ]
    # fnames = ["1w0nFHA"]
    # fnames = ["1u36FHA", "1w0nFHA"]
    for fname in fnames:
        path_to_zarr_file = "zarr_igs/redes_ex1ex2/" + fname + "_redes_ex1ex2.zarr"
        oneb, twob = load_ig_from_file(path_to_zarr_file)
        # print("table size", count_table_size(twob))
        et = create_chunk_twobody_energy_table(oneb, twob, chunk_size)
        # print("energy2b", et.energy2b.shape[0])
        # print("nrotamers", et.res_for_rot.shape[0])
        # nz = torch.nonzero(et.energy2b)
        # big = torch.nonzero(et.energy2b > 5)
        # print(fname, "number non-zero enties in energy2b:", nz.shape[0] / 2, "big", big.shape[0] / 2, "vs",
        #       et.energy2b.shape[0] / 2
        # )
        et_dev = et.to(torch_device)

        print("running sim annealing on", fname)
        scores, rotamer_assignments = run_simulated_annealing(et_dev)

        scores_temp = scores
        scores = scores.cpu().numpy()
        numpy.set_printoptions(threshold=1e5)
        # print("scores", fname)
        # for i in range(scores.shape[1]):
        #     print(" ".join([str(val) for val in scores[:, i]]))

        scores = scores_temp[0, :]
        sort_scores, sort_inds = scores.sort()
        nkeep = min(scores.shape[0], 5)
        best_scores = sort_scores[0:nkeep]
        best_score_inds = sort_inds[0:nkeep]
        best_rot_assignments = rotamer_assignments[best_score_inds, :]

        scores = best_scores.cpu()

        rotamer_assignments = best_rot_assignments.cpu()
        print(
            "scores", " ".join([str(scores[i].item()) for i in range(scores.shape[0])])
        )

        validated_scores = validate_energies(
            et.nrotamers_for_res,
            et.oneb_offsets,
            et.res_for_rot,
            et.respair_nenergies,
            et.chunk_size,
            et.chunk_offset_offsets,
            et.twob_offsets,
            et.fine_chunk_offsets,
            et.energy1b,
            et.energy2b,
            rotamer_assignments,
        )

        # print("validated scores?", validated_scores)
        torch.testing.assert_allclose(scores, validated_scores)
