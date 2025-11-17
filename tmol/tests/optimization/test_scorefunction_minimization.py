import torch
import numpy
import pytest

from torch.utils import benchmark

import time
from contextlib import contextmanager
import pathlib


# from tmol.pose.pose_stack import PoseStack
from tmol.io import pose_stack_from_pdb
from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.optimization.lbfgs_armijo import LBFGS_Armijo
from tmol.score.score_function import ScoreFunction
from tmol.score.score_types import ScoreType
from tmol.score import beta2016_score_function
from tmol.kinematics.fold_forest import EdgeType, FoldForest
from tmol.kinematics.script_modules import PoseStackKinematicsModule

from tmol.optimization.sfxn_modules import CartesianSfxnNetwork, KinForestSfxnNetwork
from tmol.kinematics.move_map import MoveMap
from tmol.optimization.kin_min import build_kinforest_network


def test_cart_minimize_w_pose_and_sfxn_smoke(ubq_pdb, default_database, torch_device):
    pose_stack1 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=4)
    pose_stack5 = PoseStackBuilder.from_poses([pose_stack1] * 5, torch_device)

    sfxn = ScoreFunction(default_database, torch_device)
    sfxn.set_weight(ScoreType.fa_ljatr, 1.0)
    sfxn.set_weight(ScoreType.fa_ljrep, 0.55)
    sfxn.set_weight(ScoreType.fa_lk, 0.8)

    cart_sfxn_network = CartesianSfxnNetwork(sfxn, pose_stack5)
    optimizer = LBFGS_Armijo(cart_sfxn_network.parameters(), lr=0.1, max_iter=20)

    E0 = cart_sfxn_network.whole_pose_scoring_module(
        cart_sfxn_network.full_coords
    ).sum()

    def closure():
        optimizer.zero_grad()
        E = cart_sfxn_network().sum()
        E.backward()
        return E

    optimizer.step(closure)

    E1 = cart_sfxn_network.whole_pose_scoring_module(
        cart_sfxn_network.full_coords
    ).sum()
    assert E1 < E0


def test_kin_minimize_w_pose_and_sfxn_smoke(ubq_pdb, default_database, torch_device):
    pose_stack1 = pose_stack_from_pdb(ubq_pdb, torch_device)
    pose_stack5 = PoseStackBuilder.from_poses([pose_stack1] * 5, torch_device)

    sfxn = ScoreFunction(default_database, torch_device)
    sfxn.set_weight(ScoreType.fa_ljatr, 1.0)
    sfxn.set_weight(ScoreType.fa_ljrep, 0.55)
    sfxn.set_weight(ScoreType.fa_lk, 0.8)

    n_res = pose_stack5.max_n_blocks
    kin_module = PoseStackKinematicsModule(
        pose_stack5, FoldForest.polymeric_forest(numpy.full(5, n_res))
    )

    assert kin_module.kmd.forest.id.device == torch_device
    assert kin_module.kmd.scan_data_fw.nodes.device == torch_device
    assert kin_module.kmd.scan_data_fw.scans.device == torch_device
    assert kin_module.kmd.scan_data_fw.gens.device == torch.device("cpu")
    assert kin_module.kmd.scan_data_bw.nodes.device == torch_device
    assert kin_module.kmd.scan_data_bw.scans.device == torch_device
    assert kin_module.kmd.scan_data_bw.gens.device == torch.device("cpu")

    kin_sfxn_network = KinForestSfxnNetwork(sfxn, pose_stack5, kin_module)
    assert kin_sfxn_network.full_dofs.device == torch_device
    assert kin_sfxn_network.masked_dofs.device == torch_device
    assert kin_sfxn_network.full_coords.device == torch_device
    assert kin_sfxn_network.flat_coords.device == torch_device

    optimizer = LBFGS_Armijo(kin_sfxn_network.parameters(), lr=0.1, max_iter=20)

    E0 = kin_sfxn_network.whole_pose_scoring_module(kin_sfxn_network.full_coords).sum()

    def closure():
        optimizer.zero_grad()
        E = kin_sfxn_network().sum()
        E.backward()
        return E

    optimizer.step(closure)

    E1 = kin_sfxn_network.whole_pose_scoring_module(kin_sfxn_network.full_coords).sum()
    assert E1 < E0


@pytest.mark.parametrize("n_poses", [1, 3, 10, 30])
@pytest.mark.benchmark(group=["minimize_pose_stack"])
def test_minimize_w_pose_and_sfxn_benchmark(
    benchmark, ubq_pdb, default_database, torch_device, n_poses
):
    if torch_device == torch.device("cpu"):
        return

    pose_stack1 = pose_stack_from_pdb(ubq_pdb, torch_device)
    pose_stack = PoseStackBuilder.from_poses([pose_stack1] * n_poses, torch_device)
    start_coords = pose_stack.coords.clone()

    sfxn = ScoreFunction(default_database, torch_device)
    sfxn.set_weight(ScoreType.fa_ljatr, 1.0)
    sfxn.set_weight(ScoreType.fa_ljrep, 0.55)
    sfxn.set_weight(ScoreType.fa_lk, 0.8)

    @benchmark
    def run():
        pose_stack.coords[:] = start_coords

        cart_sfxn_network = CartesianSfxnNetwork(sfxn, pose_stack)
        optimizer = LBFGS_Armijo(cart_sfxn_network.parameters(), lr=0.1, max_iter=20)

        cart_sfxn_network.whole_pose_scoring_module(cart_sfxn_network.full_coords)

        def closure():
            optimizer.zero_grad()
            E = cart_sfxn_network().sum()
            E.backward()
            return E

        optimizer.step(closure)

        cart_sfxn_network.whole_pose_scoring_module(cart_sfxn_network.full_coords)

    run


def test_minimizer(ubq_pdb, torch_device):
    pose_stack = pose_stack_from_pdb(ubq_pdb, torch_device)
    sfxn = beta2016_score_function(torch_device)

    wpsm = sfxn.render_whole_pose_scoring_module(pose_stack)
    wpsm(pose_stack.coords)

    network = CartesianSfxnNetwork(sfxn, pose_stack)
    optimizer = LBFGS_Armijo(network.parameters(), lr=0.1, max_iter=200)

    def closure():
        optimizer.zero_grad()
        E = network().sum()
        E.backward()
        return E

    Estart = network().sum()
    optimizer.step(closure)
    Estop = network().sum()
    assert Estop < Estart


def make_benchmark_timer(
    f,
    closure,
):
    timer = benchmark.Timer(
        stmt="f(closure)",
        globals={"f": f, "closure": closure},
        label=f"{f.__name__=}, {closure=}",
    )

    return timer


@contextmanager
def mark_and_time(name: str):
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.profiler.record_function(name):
        try:
            yield
        finally:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            print(f"{name}: {1000 * elapsed:0.2f}ms")


def test_profile_minimizer(ubq_pdb, torch_device):
    N_poses = 10
    if torch_device == torch.device("cpu"):
        return
    pose_stack1 = pose_stack_from_pdb(ubq_pdb, torch_device)
    pose_stackN = PoseStackBuilder.from_poses([pose_stack1] * N_poses, torch_device)

    sfxn = beta2016_score_function(torch_device)
    ff = FoldForest.polymeric_forest(
        numpy.full(N_poses, pose_stackN.max_n_blocks, dtype=numpy.int32)
    )
    mm = MoveMap.from_pose_stack(pose_stackN)
    mm.move_all_jumps = True
    mm.move_all_named_torsions = True

    network = build_kinforest_network(pose_stackN, sfxn, ff, mm)
    wpsm = sfxn.render_whole_pose_scoring_module(pose_stackN)
    wpsm(pose_stackN.coords)

    optimizer = LBFGS_Armijo(network.parameters(), lr=0.1, max_iter=200)

    def closure():
        optimizer.zero_grad()
        E = network().sum()
        E.backward()
        return E

    t1 = make_benchmark_timer(optimizer.step, closure)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
        execution_trace_observer=None,
    ) as prof:
        for _ in range(4):
            with mark_and_time("kin_min"):
                t1.timeit(10)
            prof.step()

    (pathlib.Path(".").resolve() / "output").mkdir(exist_ok=True)

    prof.export_chrome_trace("output/kinmin.json")


def test_profile_minimizer2(erbb2_and_pertuzumab_pdb, torch_device):
    N_poses = 3
    # we are missing residues 102-110 in the PDB
    res_not_connected = torch.zeros(
        (1, 564 - 9 + 214 + 216 + 6, 2), dtype=torch.bool, device=torch_device
    )
    res_not_connected[0, 100, 1] = True
    res_not_connected[0, 101, 0] = True
    if torch_device == torch.device("cpu"):
        return
    pose_stack1 = pose_stack_from_pdb(
        erbb2_and_pertuzumab_pdb, torch_device, res_not_connected=res_not_connected
    )
    pose_stackN = PoseStackBuilder.from_poses([pose_stack1] * N_poses, torch_device)

    pose_stackN_orig_coords = pose_stackN.coords.detach().clone()

    edges = numpy.array(
        [
            [EdgeType.root_jump, -1, 0, 0],
            [EdgeType.polymer, 0, 100, 0],
            [EdgeType.jump, 0, 101, 0],
            [EdgeType.polymer, 101, 563 - 9, 0],
            [EdgeType.jump, 0, 563 - 9 + 1, 1],
            [EdgeType.jump, 0, 563 - 9 + 214 + 1, 2],
            [EdgeType.polymer, 563 - 9 + 1, 563 - 9 + 214, 0],
            [EdgeType.polymer, 563 - 9 + 214 + 1, 563 - 9 + 214 + 216 + 6, 0],
        ],
        dtype=numpy.int32,
    )
    edges = numpy.tile(edges, (N_poses, 1, 1)).reshape(N_poses, -1, 4)
    print("Edges", edges)

    sfxn = beta2016_score_function(torch_device)
    ff = FoldForest.from_edges(edges)
    mm = MoveMap.from_pose_stack(pose_stackN)
    # keep Jump 0 fixed, as this is the jump connecting the ends of the missing loop and we don't want that moving
    mm.set_move_all_jump_dofs_for_jump(
        torch.arange(N_poses, dtype=torch.int64, device=torch_device), 1
    )
    mm.set_move_all_jump_dofs_for_jump(
        torch.arange(N_poses, dtype=torch.int64, device=torch_device), 2
    )
    mm.move_all_named_torsions = True

    network = build_kinforest_network(pose_stackN, sfxn, ff, mm)
    wpsm = sfxn.render_whole_pose_scoring_module(pose_stackN)
    wpsm(pose_stackN.coords)

    optimizer = LBFGS_Armijo(network.parameters(), lr=0.1, max_iter=200)

    def closure():
        optimizer.zero_grad()
        E = network().sum()
        E.backward()
        return E

    t1 = make_benchmark_timer(optimizer.step, closure)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
        execution_trace_observer=None,
    ) as prof:
        for _ in range(4):
            # pose_stackN.coords.copy_(pose_stackN_orig_coords)
            with mark_and_time("kin_min"):
                t1.timeit(10)
            prof.step()

    (pathlib.Path(".").resolve() / "output").mkdir(exist_ok=True)

    prof.export_chrome_trace("output/kinmin_erbb2_and_pertuzumab.json")
