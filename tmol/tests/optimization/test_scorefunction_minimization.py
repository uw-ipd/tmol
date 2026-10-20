import attr
import pytest
import torch

from tmol.io import pose_stack_from_file
from tmol.pose import PoseStackBuilder
from tmol.optimization import (
    LBFGS_Armijo,
    CartesianSfxnNetwork,
    KinForestSfxnNetwork,
    build_kinforest_network,
)
from tmol.score import (
    ScoreFunction,
    ScoreType,
    beta2016_score_function,
)
from tmol.kinematics import (
    FoldForest,
    MoveMap,
    PoseStackKinematicsModule,
)


def test_cart_minimize_w_pose_and_sfxn_smoke(ubq_pdb, default_database, torch_device):
    pose_stack1 = pose_stack_from_file(
        ubq_pdb,
        torch_device,
        residue_end=4,
        no_optH=True,
        missing_density_distance_threshold=0.0,
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


def test_cart_network_reset_reuses_topology_and_updates_weights(
    ubq_pdb, default_database, torch_device
):
    pose_stack = pose_stack_from_file(
        ubq_pdb,
        torch_device,
        residue_end=4,
        no_optH=True,
        missing_density_distance_threshold=0.0,
    )
    sfxn = ScoreFunction(default_database, torch_device)
    sfxn.set_weight(ScoreType.fa_ljatr, 1.0)
    network = CartesianSfxnNetwork(
        sfxn,
        pose_stack,
        cuda_graph=("forward_backward" if torch_device.type == "cuda" else False),
    )

    # A cloned pose has independent connectivity storage; FastRelax packing
    # preserves these tensors within a ramp, so only that case is reusable.
    assert not network._reusable_for(sfxn, pose_stack.clone())

    updated_pose = attr.evolve(pose_stack, coords=pose_stack.coords + 0.01)
    assert network._reusable_for(sfxn, updated_pose)

    sfxn.set_weight(ScoreType.fa_ljatr, 0.5)
    assert network._reset(sfxn, updated_pose)
    actual = network()
    expected_network = CartesianSfxnNetwork(sfxn, updated_pose)
    expected = expected_network()

    torch.testing.assert_close(network.full_coords, updated_pose.coords)
    torch.testing.assert_close(
        network.whole_pose_scoring_module.weights[:, 0], sfxn.weights_tensor()
    )
    torch.testing.assert_close(actual, expected)

    # Adding or removing a term changes the rendered module layout and must
    # force the caller to build a fresh network.
    sfxn.set_weight(ScoreType.fa_ljatr, 0.0)
    assert not network._reset(sfxn, updated_pose)


def _kin_network_inputs(ubq_pdb, default_database, torch_device):
    pose_stack = pose_stack_from_file(
        ubq_pdb,
        torch_device,
        residue_end=6,
        no_optH=True,
        missing_density_distance_threshold=0.0,
        ligand_hydrogens="input",
    )
    sfxn = ScoreFunction(default_database, torch_device)
    sfxn.set_weight(ScoreType.fa_ljatr, 1.0)
    sfxn.set_weight(ScoreType.fa_ljrep, 0.55)
    sfxn.set_weight(ScoreType.fa_lk, 0.8)
    ff = FoldForest.reasonable_fold_forest(pose_stack)
    mm = MoveMap.from_pose_stack(pose_stack)
    mm.move_all_jumps = True
    mm.move_all_named_torsions = True
    return pose_stack, sfxn, ff, mm


def _score_and_gradient(network):
    network.masked_dofs.grad = None
    scores = network()
    scores.sum().backward()
    return scores.detach(), network.masked_dofs.grad.clone()


def test_kin_network_reset_reuses_topology_and_updates_weights(
    ubq_pdb, default_database, torch_device
):
    pose_stack, sfxn, ff, mm = _kin_network_inputs(
        ubq_pdb, default_database, torch_device
    )
    network = build_kinforest_network(
        pose_stack, sfxn, ff, mm, cuda_graph=torch_device.type == "cuda"
    )
    assert [p for p in network.parameters()] == [network.masked_dofs]

    updated_pose = attr.evolve(pose_stack, coords=pose_stack.coords + 0.01)
    assert network._reusable_for(sfxn, updated_pose, ff, mm)

    sfxn.set_weight(ScoreType.fa_ljatr, 0.5)
    assert network._reset(sfxn, updated_pose, ff, mm)
    actual = _score_and_gradient(network)
    expected = _score_and_gradient(build_kinforest_network(updated_pose, sfxn, ff, mm))

    torch.testing.assert_close(network.full_coords, updated_pose.coords)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])

    # A move-map edit changes the DOF mask and must force a rebuild.
    changed_mm = MoveMap.from_pose_stack(pose_stack)
    changed_mm.move_all_named_torsions = True
    assert not network._reset(sfxn, updated_pose, ff, changed_mm)


def test_kin_network_cuda_graph_matches_eager(ubq_pdb, default_database, torch_device):
    if torch_device.type != "cuda":
        pytest.skip("CUDA graph test")

    device = torch_device
    pose_stack, sfxn, ff, mm = _kin_network_inputs(ubq_pdb, default_database, device)
    graphed = build_kinforest_network(pose_stack, sfxn, ff, mm, cuda_graph="both")
    eager = build_kinforest_network(pose_stack, sfxn, ff, mm)
    assert set(graphed._cuda_graphs) == {"forward", "autograd"}

    generator = torch.Generator(device=device).manual_seed(0)
    for _ in range(3):
        step = 0.02 * torch.randn(
            graphed.masked_dofs.shape, device=device, generator=generator
        )
        with torch.no_grad():
            graphed.masked_dofs.add_(step)
            eager.masked_dofs.add_(step)
            # The graph always takes the weighted fused reduction, which
            # small eager calls skip; allow for its summation order.
            torch.testing.assert_close(graphed(), eager(), rtol=1e-5, atol=1e-3)

        graphed_scores, graphed_grad = _score_and_gradient(graphed)
        eager_scores, eager_grad = _score_and_gradient(eager)
        torch.testing.assert_close(graphed_scores, eager_scores, rtol=1e-5, atol=1e-3)
        torch.testing.assert_close(graphed_grad, eager_grad, rtol=1e-4, atol=1e-3)
        torch.testing.assert_close(graphed.full_coords, eager.full_coords)


def test_kin_minimize_w_pose_and_sfxn_smoke(ubq_pdb, default_database, torch_device):
    pose_stack1 = pose_stack_from_file(
        ubq_pdb, torch_device, no_optH=True, missing_density_distance_threshold=0.0
    )
    pose_stack5 = PoseStackBuilder.from_poses([pose_stack1] * 5, torch_device)

    sfxn = ScoreFunction(default_database, torch_device)
    sfxn.set_weight(ScoreType.fa_ljatr, 1.0)
    sfxn.set_weight(ScoreType.fa_ljrep, 0.55)
    sfxn.set_weight(ScoreType.fa_lk, 0.8)

    kin_module = PoseStackKinematicsModule(
        pose_stack5, FoldForest.reasonable_fold_forest(pose_stack5)
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

    pose_stack1 = pose_stack_from_file(
        ubq_pdb, torch_device, no_optH=True, missing_density_distance_threshold=0.0
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


def test_minimizer(ubq_pdb, torch_device):
    pose_stack = pose_stack_from_file(
        ubq_pdb, torch_device, no_optH=True, missing_density_distance_threshold=0.0
    )
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
