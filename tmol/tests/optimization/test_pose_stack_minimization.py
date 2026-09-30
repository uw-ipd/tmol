"""Minimizing a stack of distinct poses should match minimizing them one-by-one."""

import attr
import pytest
import torch

from tmol import (
    PoseStack,
    beta2016_score_function,
    run_cart_min,
    run_kin_min,
    FoldForest,
    MoveMap,
)
from tmol.io import atom_array_from_cif, pose_stack_from_biotite
from tmol.pose import PoseStackBuilder
from tmol.optimization import CartesianSfxnNetwork, KinForestSfxnNetwork
from tmol.kinematics import PoseStackKinematicsModule
from tmol.tests.data import data_path


def _score_per_pose(pose_stack: PoseStack, sfxn):
    wpsm = sfxn.render_whole_pose_scoring_module(pose_stack)
    return wpsm(pose_stack.coords).detach()


def _cart_min_per_pose(pose_stack: PoseStack, sfxn):
    return _score_per_pose(run_cart_min(pose_stack, sfxn), sfxn)


def _kin_min_per_pose(pose_stack: PoseStack, sfxn):
    ff = FoldForest.reasonable_fold_forest(pose_stack)
    mm = MoveMap.from_pose_stack(pose_stack)
    mm.move_all_jumps = True
    mm.move_all_named_torsions = True
    return _score_per_pose(run_kin_min(pose_stack, sfxn, ff, mm), sfxn)


def _report(label, start, one_by_one, stacked):
    lines = [
        f"{label}: per-pose energies",
        f"{'pose':>4} {'start':>12} {'one-by-one':>12} {'stacked':>12} {'delta':>10}",
    ]
    for i in range(len(stacked)):
        lines.append(
            f"{i:>4} {start[i]:>12.3f} {one_by_one[i]:>12.3f}"
            f" {stacked[i]:>12.3f} {stacked[i] - one_by_one[i]:>10.3f}"
        )
    return "\n".join(lines)


def _compare(label, poses, stack, sfxn, min_per_pose, tol=5.0):
    start = _score_per_pose(stack, sfxn)
    one_by_one = torch.cat([min_per_pose(p, sfxn) for p in poses])
    stacked = min_per_pose(stack, sfxn)

    report = _report(label, start, one_by_one, stacked)
    assert torch.all(stacked < start), report
    # FP32 reduction order differs between a heterogeneous stack and separate
    # runs, so nonlinear line-search trajectories need a loose energy bound.
    assert torch.all(torch.abs(stacked - one_by_one) < tol), report


def test_score_stack_of_distinct_poses_matches_individual(
    distinct_pose_stacks, stack_of_distinct_poses, torch_device
):
    """Baseline: scoring is stack-invariant, so any difference is the minimizer's."""
    sfxn = beta2016_score_function(torch_device)
    individual = torch.cat([_score_per_pose(p, sfxn) for p in distinct_pose_stacks])
    stacked = _score_per_pose(stack_of_distinct_poses, sfxn)
    torch.testing.assert_close(stacked, individual, rtol=1e-4, atol=1e-3)


def test_cart_min_stack_of_distinct_poses(
    distinct_pose_stacks, stack_of_distinct_poses, torch_device
):
    sfxn = beta2016_score_function(torch_device)
    _compare(
        "cart, distinct poses",
        distinct_pose_stacks,
        stack_of_distinct_poses,
        sfxn,
        _cart_min_per_pose,
    )


@pytest.mark.xfail
def test_kin_min_stack_of_distinct_poses(
    distinct_pose_stacks, stack_of_distinct_poses, torch_device
):
    sfxn = beta2016_score_function(torch_device)
    _compare(
        "kin, distinct poses",
        distinct_pose_stacks,
        stack_of_distinct_poses,
        sfxn,
        _kin_min_per_pose,
    )


def test_cart_network_segment_ids(
    distinct_pose_stacks, stack_of_distinct_poses, torch_device
):
    """Each pose's coordinate DOFs must be labelled with that pose."""
    sfxn = beta2016_score_function(torch_device)
    network = CartesianSfxnNetwork(sfxn, stack_of_distinct_poses)

    segment_ids = network.segment_ids
    assert segment_ids.shape == (network.masked_coords.numel(),)
    counts = torch.bincount(segment_ids)
    assert len(counts) == len(distinct_pose_stacks)
    solo_counts = torch.tensor(
        [
            CartesianSfxnNetwork(sfxn, pose).masked_coords.numel()
            for pose in distinct_pose_stacks
        ],
        device=counts.device,
    )
    # Padding in a heterogeneous stack must not create optimizer variables.
    assert torch.all(counts == solo_counts), f"{counts} != {solo_counts}"
    # coords are laid out pose-major, so the labels come in contiguous runs
    assert torch.all(segment_ids[1:] >= segment_ids[:-1])


def test_cart_network_all_coords_fast_path(distinct_pose_stacks, torch_device):
    """An all-true mask keeps coordinates as a view and reconstructs a copy."""
    pose_stack = distinct_pose_stacks[0]
    original_coords = pose_stack.coords.clone()
    sfxn = beta2016_score_function(torch_device)
    coord_mask = torch.ones_like(pose_stack.real_atoms)
    network = CartesianSfxnNetwork(sfxn, pose_stack, coord_mask)

    assert network._all_coords_movable
    energy = network().sum()
    energy.backward()

    assert network.full_coords.data_ptr() == network.masked_coords.data_ptr()
    assert network.masked_coords.grad is not None

    reconstructed = network.pose_stack_from_dofs()
    torch.testing.assert_close(reconstructed.coords, network.full_coords)
    assert reconstructed.coords.data_ptr() != network.masked_coords.data_ptr()
    torch.testing.assert_close(pose_stack.coords, original_coords)


def test_kin_network_segment_ids(
    distinct_pose_stacks, stack_of_distinct_poses, torch_device
):
    """Each pose's torsion DOFs must be labelled with that pose."""
    sfxn = beta2016_score_function(torch_device)

    def network_for(pose_stack):
        kin_module = PoseStackKinematicsModule(
            pose_stack, FoldForest.reasonable_fold_forest(pose_stack)
        )
        return KinForestSfxnNetwork(sfxn, pose_stack, kin_module)

    network = network_for(stack_of_distinct_poses)
    segment_ids = network.segment_ids
    assert segment_ids.shape == (network.masked_dofs.numel(),)

    counts = torch.bincount(segment_ids, minlength=len(distinct_pose_stacks))
    solo_counts = torch.tensor(
        [network_for(p).masked_dofs.numel() for p in distinct_pose_stacks],
        device=counts.device,
    )
    # a pose contributes the same dofs whether it is minimized alone or in a stack
    assert torch.all(counts == solo_counts), f"{counts} != {solo_counts}"


def test_cart_min_stack_of_identical_poses(distinct_pose_stacks, torch_device):
    """Control: a stack of copies of one pose should match minimizing it alone."""
    sfxn = beta2016_score_function(torch_device)
    poses = [distinct_pose_stacks[0]] * 3
    stack = PoseStackBuilder.from_poses(poses, torch_device)
    _compare("cart, identical poses", poses, stack, sfxn, _cart_min_per_pose)


def _sweep_regression_poses(device):
    """2LNY holds a HIS_POS; in 1MBO the O2 on the haem iron is an acceptor
    without a base, and lk_ball bases cross its metal connection; 1COI is a
    plain capped peptide. Built on one database lineage so they stack."""
    database = None
    poses = []
    for name in (
        "his_pos_nmr_2lny.cif.zst",
        "oxygen_acceptor_1mbo.cif.zst",
        "capped_peptide_1coi.pdb.zst",
    ):
        pose, context = pose_stack_from_biotite(
            atom_array_from_cif(data_path("sweep_regressions", name)),
            device,
            prepare_ligands=True,
            ligand_seed=0,
            param_db=database,
            return_context=True,
        )
        database = context.parameter_database
        # metal donor forms extend the database per pose
        if pose.packed_block_types.chem_db is not database.chemical:
            database = attr.evolve(database, chemical=pose.packed_block_types.chem_db)
        poses.append(pose)
    return poses, beta2016_score_function(device, param_db=database)


def _score_and_grad(pose_stack, sfxn, coords=None):
    coords = pose_stack.coords if coords is None else coords
    coords = coords.detach().clone().requires_grad_(True)
    energies = sfxn.render_whole_pose_scoring_module(pose_stack)(coords)
    (grad,) = torch.autograd.grad(energies.sum(), coords)
    return energies.detach(), grad


def _assert_scored_as_alone(poses, sfxn, energies, grads, which):
    for i in which:
        energy, grad = _score_and_grad(poses[i], sfxn)
        n_atoms = poses[i].coords.shape[1]
        assert torch.isfinite(grads[i, :n_atoms]).all(), f"pose {i}"
        torch.testing.assert_close(
            energies[i], energy[0], rtol=1e-4, atol=1e-3, msg=f"pose {i}"
        )
        torch.testing.assert_close(
            grads[i, :n_atoms], grad[0], rtol=1e-4, atol=1e-3, msg=f"pose {i}"
        )


def test_stack_of_sweep_regressions_matches_individual(torch_device):
    """Each pose of a heterogeneous stack has the energy and gradient it has
    alone. lk_ball sent the gradient of an acceptor base across a connection to
    the first pose's copy of that block, and an acceptor without a base gave NaN
    gradients."""
    poses, sfxn = _sweep_regression_poses(torch_device)
    stack = PoseStackBuilder.from_poses(poses, torch_device)
    energies, grads = _score_and_grad(stack, sfxn)
    _assert_scored_as_alone(poses, sfxn, energies, grads, range(len(poses)))
    if torch_device.type == "cpu":
        _compare("cart, sweep regressions", poses, stack, sfxn, _cart_min_per_pose)
    else:
        # CUDA reductions are not deterministic, and repeated minimizations of
        # 1MBO alone end in its haem pocket up to 7 REU apart.
        minimized = _cart_min_per_pose(stack, sfxn)
        assert torch.isfinite(minimized).all()
        assert torch.all(minimized < energies)


def test_nan_in_one_pose_stays_in_that_pose(torch_device):
    """A pose with a NaN coordinate scores NaN; its neighbours in the stack score
    and minimize as they would without it."""
    poses, sfxn = _sweep_regression_poses(torch_device)
    stack = PoseStackBuilder.from_poses(poses, torch_device)
    coords = stack.coords.clone()
    coords[1, 0] = float("nan")
    stack = attr.evolve(stack, coords=coords)
    energies, grads = _score_and_grad(stack, sfxn)
    assert torch.isnan(energies[1])
    _assert_scored_as_alone(poses, sfxn, energies, grads, (0, 2))

    minimized = _cart_min_per_pose(stack, sfxn)
    one_by_one = torch.cat([_cart_min_per_pose(poses[i], sfxn) for i in (0, 2)])
    assert torch.isfinite(minimized[[0, 2]]).all()
    assert torch.all(torch.abs(minimized[[0, 2]] - one_by_one) < 5.0)
