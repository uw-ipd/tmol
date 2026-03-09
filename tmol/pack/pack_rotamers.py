import torch
import time

from tmol.pose.pose_stack import PoseStack
from tmol.score.score_function import ScoreFunction

from tmol.pack.compiled.compiled import build_interaction_graph
from tmol.pack.packer_task import PackerTask
from tmol.pack.rotamer.build_rotamers import build_rotamers
from tmol.pack.datatypes import PackerEnergyTables
from tmol.pack.simulated_annealing import run_simulated_annealing
from tmol.pack.impose_rotamers import impose_top_rotamer_assignments


def pack_rotamers(
    pose_stack: PoseStack, sfxn: ScoreFunction, task: PackerTask, verbose=False
):
    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    pbt = pose_stack.packed_block_types

    pose_stack, rotamer_set = build_rotamers(pose_stack, task, pbt.chem_db)
    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time1 = time.perf_counter()

    rotamer_scoring_module = sfxn.render_rotamer_scoring_module(pose_stack, rotamer_set)

    energies = rotamer_scoring_module(rotamer_set.coords)
    energies = energies.coalesce()

    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time2 = time.perf_counter()

    chunk_size = 16

    energy1b, chunk_pair_offset_for_block_pair, chunk_pair_offset, energy2b = (
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
    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time3 = time.perf_counter()

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
    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time4 = time.perf_counter()

    scores, rotamer_assignments = run_simulated_annealing(packer_energy_tables)
    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()

    print(scores)
    print(torch.argmin(scores, 1))
    rotamer_assignments[0][0] = rotamer_assignments[0][torch.argmin(scores, 1)]
    print(rotamer_assignments)
    print(pose_stack.n_res_per_pose)
    torch.set_printoptions(profile="full")
    print(rotamer_set.n_rots_for_block)
    torch.set_printoptions(profile="default")
    end_time5 = time.perf_counter()
    new_pose_stack = impose_top_rotamer_assignments(
        pose_stack, rotamer_set, rotamer_assignments
    )
    if verbose and torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time6 = time.perf_counter()

    if verbose:
        print(
            f"pack_rotamers {end_time6 - start_time: .2f}"
            + f" build rots: {end_time1-start_time: .2f} calcRPEs: {end_time2 - end_time1: .2f}"
            + f" build IG: {end_time3-end_time2: .2f} build IG part2: {end_time4 - end_time3: .2f}"
            + f" run SA: {end_time5-end_time4: .2f} pose ctor: {end_time6 - end_time5: .2f}"
        )

    return new_pose_stack
