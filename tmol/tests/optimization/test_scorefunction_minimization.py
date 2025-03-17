import torch
import numpy
import pytest

# from tmol.pose.pose_stack import PoseStack
from tmol.io import pose_stack_from_pdb
from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.optimization.lbfgs_armijo import LBFGS_Armijo
from tmol.score.score_function import ScoreFunction
from tmol.score.score_types import ScoreType
from tmol.score import beta2016_score_function
from tmol.kinematics.fold_forest import FoldForest
from tmol.kinematics.script_modules import PoseStackKinematicsModule

from tmol.optimization.sfxn_modules import CartesianSfxnNetwork, KinForestSfxnNetwork


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
