import torch
import pytest

# from tmol.pose.pose_stack import PoseStack
from tmol.io import pose_stack_from_pdb
from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.optimization.lbfgs_armijo import LBFGS_Armijo
from tmol.score.score_function import ScoreFunction
from tmol.score.score_types import ScoreType
from tmol.score import beta2016_score_function

# from tmol.optimization.modules import DOFMaskingFunc
from tmol.optimization.sfxn_modules import CartesianSfxnNetwork, KinematicSfxnNetwork


def test_minimize_w_pose_and_sfxn_smoke(rts_ubq_res, default_database, torch_device):
    pose_stack1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, rts_ubq_res[:4], torch_device
    )
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


@pytest.mark.benchmark(group="pose_50step_minimization")
def test_cart_minimizer(benchmark, ubq_pdb, torch_device):
    pose_stack = pose_stack_from_pdb(ubq_pdb, torch_device)
    sfxn = beta2016_score_function(torch_device)

    wpsm = sfxn.render_whole_pose_scoring_module(pose_stack)
    wpsm(pose_stack.coords)

    @benchmark
    def do_minimize():
        network = CartesianSfxnNetwork(sfxn, pose_stack)
        optimizer = LBFGS_Armijo(network.parameters(), lr=0.001, max_iter=50)

        def closure():
            optimizer.zero_grad()
            E = network().sum()
            E.backward()
            return E

        Estart = network().sum()
        optimizer.step(closure)
        Estop = network().sum()
        return Estart, Estop

    Estart, Estop = do_minimize
    assert Estop < Estart


@pytest.mark.parametrize("nonideal", [False, True])
@pytest.mark.benchmark(group="pose_50step_minimization")
def test_dof_minimizer(benchmark, ubq_pdb, torch_device, nonideal):
    pose_stack = pose_stack_from_pdb(ubq_pdb, torch_device)
    sfxn = beta2016_score_function(torch_device)

    wpsm = sfxn.render_whole_pose_scoring_module(pose_stack)
    wpsm(pose_stack.coords)

    @benchmark
    def do_minimize():
        network = KinematicSfxnNetwork(sfxn, pose_stack, nonideal=nonideal)
        optimizer = LBFGS_Armijo(network.parameters(), lr=0.001, max_iter=50)

        def closure():
            optimizer.zero_grad()
            E = network().sum()
            E.backward()
            return E

        Estart = network().sum()
        optimizer.step(closure)
        Estop = network().sum()
        return Estart, Estop

    Estart, Estop = do_minimize
    assert Estop < Estart
