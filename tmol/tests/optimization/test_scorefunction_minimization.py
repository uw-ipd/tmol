import torch
import pytest
import time
import attrs

# from tmol.pose.pose_stack import PoseStack
from tmol.io import pose_stack_from_pdb
from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.optimization.lbfgs_armijo import LBFGS_Armijo
from tmol.score.score_function import ScoreFunction
from tmol.score.score_types import ScoreType
from tmol.score import beta2016_score_function

# from tmol.optimization.modules import DOFMaskingFunc
from tmol.optimization.sfxn_modules import CartesianSfxnNetwork


def test_minimize_w_pose_and_sfxn_smoke(rts_ubq_res, default_database, torch_device):
    pose_stack1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, rts_ubq_res[:4], torch_device
    )
    pose_stack5 = PoseStackBuilder.from_poses([pose_stack1] * 5, torch_device)

    sfxn = ScoreFunction(default_database, torch_device)
    sfxn.set_weight(ScoreType.fa_lj, 1.0)
    sfxn.set_weight(ScoreType.fa_lk, 0.8)

    cart_sfxn_network = CartesianSfxnNetwork(sfxn, pose_stack5)
    optimizer = LBFGS_Armijo(cart_sfxn_network.parameters(), lr=0.1, max_iter=20)

    E0 = cart_sfxn_network.whole_pose_scoring_module(
        cart_sfxn_network.full_coords
    ).sum()
    # print("E0", E0)

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


@pytest.mark.parametrize("n_poses", [1, 3, 10, 30])
@pytest.mark.benchmark(group=["minimize_pose_stack"])
def test_minimize_w_pose_and_sfxn_benchmark(
    benchmark, rts_ubq_res, default_database, torch_device, n_poses
):
    if torch_device == torch.device("cpu"):
        return

    pose_stack1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, rts_ubq_res, torch_device
    )
    pose_stack = PoseStackBuilder.from_poses([pose_stack1] * n_poses, torch_device)
    start_coords = pose_stack.coords.clone()

    sfxn = ScoreFunction(default_database, torch_device)
    sfxn.set_weight(ScoreType.fa_lj, 1.0)
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


# @requires_cuda
def test_minimizer_vs_just_score(ubq_pdb):
    torch_device = torch.device("cuda")
    pose_stack = pose_stack_from_pdb(ubq_pdb, torch_device)
    sfxn = beta2016_score_function(torch_device)

    wpsm = sfxn.render_whole_pose_scoring_module(pose_stack)
    wpsm(pose_stack.coords)

    network = CartesianSfxnNetwork(sfxn, pose_stack)
    optimizer = LBFGS_Armijo(network.parameters(), lr=0.1, max_iter=20)

    def closure1():
        optimizer.zero_grad()
        E = network().sum()
        E.backward()
        return E

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(400):
        closure1()
    torch.cuda.synchronize()
    stop = time.time()
    optimizer_time = stop - start

    pose_stack = attrs.evolve(
        pose_stack, coords=pose_stack.coords.clone().detach().requires_grad_(True)
    )
    wpsm = sfxn.render_whole_pose_scoring_module(pose_stack)
    wpsm(pose_stack.coords)

    def closure2():
        # optimizer.zero_grad() # just score+derivs
        # E = network().sum()
        # E.backward()
        E = wpsm(pose_stack.coords).sum()
        E.backward()

        return E

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(400):
        closure2()
    torch.cuda.synchronize()
    stop = time.time()
    just_score_time = stop - start

    E = network().sum().cpu()
    print(
        "finished",
        E.item(),
        "optimize:",
        optimizer_time,
        "just score:",
        just_score_time,
    )
