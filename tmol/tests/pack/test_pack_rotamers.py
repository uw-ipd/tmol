import torch
import time

from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.score.score_function import ScoreFunction
from tmol.score.score_types import ScoreType

from tmol.pack.compiled.compiled import build_interaction_graph
from tmol.pack.packer_task import PackerTask, PackerPalette
from tmol.pack.rotamer.build_rotamers import build_rotamers
from tmol.pack.rotamer.fixed_aa_chi_sampler import (
    FixedAAChiSampler,
)
from tmol.pack.datatypes import PackerEnergyTables
from tmol.pack.simulated_annealing import run_simulated_annealing
from tmol.pack.impose_rotamers import impose_top_rotamer_assignments

from tmol.io import pose_stack_from_pdb
from tmol.io.write_pose_stack_pdb import write_pose_stack_pdb

from tmol.pack.pack_rotamers import pack_rotamers


def test_pack_rotamers(default_database, ubq_pdb, dun_sampler, torch_device):
    n_poses = 4

    p = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=76)

    pose_stack = PoseStackBuilder.from_poses([p] * n_poses, torch_device)
    restype_set = pose_stack.packed_block_types.restype_set

    palette = PackerPalette(restype_set)
    task = PackerTask(pose_stack, palette)
    task.restrict_to_repacking()
    task.set_include_current()

    fixed_sampler = FixedAAChiSampler()
    task.add_conformer_sampler(dun_sampler)
    task.add_conformer_sampler(fixed_sampler)

    sfxn = ScoreFunction(param_db=default_database, device=torch_device)
    sfxn.set_weight(ScoreType.fa_ljatr, 1.0)
    sfxn.set_weight(ScoreType.fa_ljrep, 0.55)
    sfxn.set_weight(ScoreType.fa_lk, 1.0)
    sfxn.set_weight(ScoreType.hbond, 1.0)

    pbt = pose_stack.packed_block_types

    print("starting packer steps")
    start_time = time.perf_counter()

    pose_stack, rotamer_set = build_rotamers(pose_stack, task, pbt.chem_db)
    print("Built", rotamer_set.n_rotamers_total, "rotamers")

    rotamer_scoring_module = sfxn.render_rotamer_scoring_module(pose_stack, rotamer_set)

    print("starting energy calculation")
    energies = rotamer_scoring_module(rotamer_set.coords)
    energies = energies.coalesce()
    print("done")

    # n_rots_total = rotamer_set.n_rotamers_total
    # energy1b = torch.zeros((n_rots_total), dtype=torch.float32, device=torch_device)

    chunk_size = 16

    (energy1b, chunk_pair_offset_for_block_pair, chunk_pair_offset, energy2b) = (
        build_interaction_graph(
            chunk_size,
            rotamer_set.n_rots_for_pose,
            rotamer_set.rot_offset_for_pose,
            rotamer_set.n_rots_for_block,
            rotamer_set.rot_offset_for_block,
            rotamer_set.pose_for_rot,
            rotamer_set.block_type_ind_for_rot,
            rotamer_set.block_ind_for_rot,
            energies.indices().to(torch.int32),
            energies.values(),
        )
    )

    packer_energy_tables = PackerEnergyTables(
        max_n_rotamers_per_pose=rotamer_set.max_n_rots_per_pose,
        pose_n_res=pose_stack.n_res_per_pose,
        pose_n_rotamers=rotamer_set.n_rots_for_pose,
        pose_rotamer_offset=rotamer_set.rot_offset_for_pose,
        nrotamers_for_res=rotamer_set.n_rots_for_block,
        oneb_offsets=rotamer_set.rot_offset_for_block,
        res_for_rot=rotamer_set.block_ind_for_rot,
        chunk_size=chunk_size,
        chunk_offset_offsets=chunk_pair_offset_for_block_pair,
        chunk_offsets=chunk_pair_offset,
        energy1b=energy1b,
        energy2b=energy2b,
    )

    scores, rotamer_assignments = run_simulated_annealing(packer_energy_tables)

    # print("top scores", scores[:, 0])
    # for i in range(n_poses):
    #     for j in range(scores.shape[1]):
    #         print("score", i, j, scores[i,j])

    new_pose_stack = impose_top_rotamer_assignments(
        pose_stack, rotamer_set, rotamer_assignments
    )
    if torch_device == torch.device("cuda"):
        torch.cuda.synchronize()
    stop_time = time.perf_counter()

    elapsed_time = stop_time - start_time

    print(f"Execution time: {elapsed_time:.6f} seconds")

    write_pose_stack_pdb(new_pose_stack, "pack_rotamers_1ubq_ex1ex2.pdb")

    wpsm = sfxn.render_whole_pose_scoring_module(new_pose_stack)
    new_scores = wpsm(new_pose_stack.coords)
    print("confirm new scores", new_scores)
    torch.testing.assert_close(scores[:, 0], new_scores, atol=1e-3, rtol=1e-5)


def test_pack_rotamers2(default_database, ubq_pdb, dun_sampler, torch_device):

    if torch_device == torch.device("cpu"):
        return
    n_poses = 100
    # print("Device!", torch_device)

    p = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=76)

    pose_stack = PoseStackBuilder.from_poses([p] * n_poses, torch_device)
    # from tmol.io.chain_deduction import chain_inds_for_pose_stack
    # chain_ind_for_block = chain_inds_for_pose_stack(pose_stack)
    # print("chain_ind_for_block", chain_ind_for_block[:,0])
    # return

    restype_set = pose_stack.packed_block_types.restype_set

    palette = PackerPalette(restype_set)
    task = PackerTask(pose_stack, palette)
    task.restrict_to_repacking()
    task.set_include_current()
    task.or_expand_chi(1)
    task.or_expand_chi(2)

    fixed_sampler = FixedAAChiSampler()
    task.add_conformer_sampler(dun_sampler)
    task.add_conformer_sampler(fixed_sampler)

    sfxn = ScoreFunction(param_db=default_database, device=torch_device)
    sfxn.set_weight(ScoreType.fa_ljatr, 1.0)
    sfxn.set_weight(ScoreType.fa_ljrep, 0.55)
    sfxn.set_weight(ScoreType.fa_lk, 1.0)
    sfxn.set_weight(ScoreType.hbond, 1.0)

    # warmup:
    print("starting warmup packer run")
    start_time = time.perf_counter()
    new_pose_stack = pack_rotamers(pose_stack, sfxn, task)
    if torch_device == torch.device("cuda"):
        torch.cuda.synchronize()
    stop_time = time.perf_counter()

    elapsed_time = stop_time - start_time

    print(f"warmup execution time: {elapsed_time:.6f} seconds")

    print("starting packer steps")
    start_time = time.perf_counter()

    new_pose_stack = pack_rotamers(pose_stack, sfxn, task)

    if torch_device == torch.device("cuda"):
        torch.cuda.synchronize()
    stop_time = time.perf_counter()

    elapsed_time = stop_time - start_time

    print(f"Execution time: {elapsed_time:.6f} seconds")

    write_pose_stack_pdb(new_pose_stack, "pack_rotamers_1ubq_ex1ex2.pdb")


def test_pack_rotamers_irregular_sized_poses(
    default_database, ubq_pdb, dun_sampler, torch_device
):

    if torch_device == torch.device("cpu"):
        return
    n_poses = 20
    # print("Device!", torch_device)

    pose_stack = PoseStackBuilder.from_poses(
        [
            pose_stack_from_pdb(
                ubq_pdb, torch_device, residue_start=0, residue_end=20 + i
            )
            for i in range(n_poses)
        ],
        torch_device,
    )
    # from tmol.io.chain_deduction import chain_inds_for_pose_stack
    # chain_ind_for_block = chain_inds_for_pose_stack(pose_stack)
    # print("chain_ind_for_block", chain_ind_for_block[:,0])
    # return

    restype_set = pose_stack.packed_block_types.restype_set

    palette = PackerPalette(restype_set)
    task = PackerTask(pose_stack, palette)
    task.restrict_to_repacking()
    task.set_include_current()
    task.or_expand_chi(1)
    task.or_expand_chi(2)

    fixed_sampler = FixedAAChiSampler()
    task.add_conformer_sampler(dun_sampler)
    task.add_conformer_sampler(fixed_sampler)

    sfxn = ScoreFunction(param_db=default_database, device=torch_device)
    sfxn.set_weight(ScoreType.fa_ljatr, 1.0)
    sfxn.set_weight(ScoreType.fa_ljrep, 0.55)
    sfxn.set_weight(ScoreType.fa_lk, 1.0)
    sfxn.set_weight(ScoreType.hbond, 1.0)

    # warmup:
    print("starting warmup packer run")
    start_time = time.perf_counter()
    new_pose_stack = pack_rotamers(pose_stack, sfxn, task)
    if torch_device == torch.device("cuda"):
        torch.cuda.synchronize()
    stop_time = time.perf_counter()

    elapsed_time = stop_time - start_time

    print(f"warmup execution time: {elapsed_time:.6f} seconds")

    print("starting packer steps")
    start_time = time.perf_counter()

    new_pose_stack = pack_rotamers(pose_stack, sfxn, task)

    if torch_device == torch.device("cuda"):
        torch.cuda.synchronize()
    stop_time = time.perf_counter()

    elapsed_time = stop_time - start_time

    print(f"Execution time: {elapsed_time:.6f} seconds")

    write_pose_stack_pdb(new_pose_stack, "pack_rotamers_1ubq_ex1ex2.pdb")
