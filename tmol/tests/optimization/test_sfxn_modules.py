import torch

from tmol.io import pose_stack_from_pdb
from tmol.kinematics.fold_forest import FoldForest
from tmol.kinematics.script_modules import PoseStackKinematicsModule
from tmol.score.score_function import ScoreFunction
from tmol.score.score_types import ScoreType

from tmol.optimization.sfxn_modules import CartesianSfxnNetwork, KinForestSfxnNetwork


def _sfxn(default_database, torch_device):
    sfxn = ScoreFunction(default_database, torch_device)
    sfxn.set_weight(ScoreType.fa_ljatr, 1.0)
    sfxn.set_weight(ScoreType.fa_ljrep, 0.55)
    return sfxn


def _kin_net(sfxn, pose_stack):
    ff = FoldForest.reasonable_fold_forest(pose_stack)
    kin_module = PoseStackKinematicsModule(pose_stack, ff)
    return KinForestSfxnNetwork(sfxn, pose_stack, kin_module)


def test_cart_dof_pose_assignment_single_pose(ubq_pdb, default_database, torch_device):
    """Single pose: all DOFs map to pose index 0."""
    pose_stack = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=4)
    net = CartesianSfxnNetwork(_sfxn(default_database, torch_device), pose_stack)

    dpa = net.dof_pose_assignment()

    assert dpa.dtype == torch.int64
    assert dpa.shape == (net.masked_coords.numel(),)
    assert dpa.min().item() == 0
    assert dpa.max().item() == 0, "Single pose: all DOFs must be assigned to pose 0"


def test_cart_dof_pose_assignment_equal_blocks(
    stack_of_two_six_res_ubqs, default_database, torch_device
):
    """Two poses with the same block count get identical DOF counts.

    The default coord_mask is all-True (including padding slots), so each pose
    contributes exactly max_n_pose_atoms * 3 DOFs in pose-index order.
    """
    pose_stack = stack_of_two_six_res_ubqs  # 2 poses, both 6 residues
    n_poses = pose_stack.n_poses
    net = CartesianSfxnNetwork(_sfxn(default_database, torch_device), pose_stack)

    dpa = net.dof_pose_assignment()
    max_n_atoms = pose_stack.max_n_pose_atoms

    assert dpa.dtype == torch.int64
    assert dpa.shape == (net.masked_coords.numel(),)
    assert dpa.min().item() == 0
    assert dpa.max().item() == n_poses - 1

    # With the default all-True mask every atom slot (real or padding) is
    # active, so each pose owns exactly max_n_pose_atoms * 3 DOF entries.
    dofs_per_pose = max_n_atoms * 3
    for i in range(n_poses):
        count = (dpa == i).sum().item()
        assert (
            count == dofs_per_pose
        ), f"Pose {i}: expected {dofs_per_pose} DOFs, got {count}"

    # Pose-0 DOFs must all precede pose-1 DOFs (row-major ordering).
    last_pose0 = (dpa == 0).nonzero(as_tuple=False).max().item()
    first_pose1 = (dpa == 1).nonzero(as_tuple=False).min().item()
    assert last_pose0 < first_pose1, "All pose-0 DOFs must precede pose-1 DOFs"


def test_cart_dof_pose_assignment_jagged_poses(
    jagged_stack_of_465_res_ubqs, default_database, torch_device
):
    """Jagged stack (4, 6, 5 residues) with real-atom mask: DOF counts match
    actual per-pose atom counts and are ordered pose-by-pose.
    """
    pose_stack = jagged_stack_of_465_res_ubqs  # 3 poses: 4, 6, 5 residues
    n_poses = pose_stack.n_poses  # 3
    coord_mask = pose_stack.real_atoms  # [n_poses, max_n_pose_atoms], bool

    net = CartesianSfxnNetwork(
        _sfxn(default_database, torch_device), pose_stack, coord_mask
    )
    dpa = net.dof_pose_assignment()

    assert dpa.dtype == torch.int64
    assert dpa.shape == (net.masked_coords.numel(),)
    assert dpa.min().item() == 0
    assert dpa.max().item() == n_poses - 1

    # DOF count for pose i must equal 3 * (real atoms in pose i).
    for i in range(n_poses):
        expected = 3 * coord_mask[i].sum().item()
        actual = (dpa == i).sum().item()
        assert actual == expected, (
            f"Pose {i}: expected {expected} DOFs (3 × {coord_mask[i].sum().item()} atoms),"
            f" got {actual}"
        )

    # Pose-0 (4 res) has fewer atoms than pose-1 (6 res).
    assert (dpa == 0).sum() < (
        dpa == 1
    ).sum(), "4-residue pose should have fewer DOFs than 6-residue pose"

    # DOFs are ordered by pose: all pose-i DOFs precede pose-(i+1) DOFs.
    for i in range(n_poses - 1):
        last_i = (dpa == i).nonzero(as_tuple=False).max().item()
        first_j = (dpa == i + 1).nonzero(as_tuple=False).min().item()
        assert last_i < first_j, f"All pose-{i} DOFs must precede pose-{i + 1} DOFs"


def test_kin_dof_pose_assignment_single_pose(ubq_pdb, default_database, torch_device):
    """Single pose: all DOFs map to pose index 0."""
    pose_stack = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=4)
    net = _kin_net(_sfxn(default_database, torch_device), pose_stack)

    dpa = net.dof_pose_assignment()

    assert dpa.dtype == torch.int64
    assert dpa.shape == (net.masked_dofs.numel(),)
    assert dpa.min().item() == 0
    assert dpa.max().item() == 0, "Single pose: all DOFs must be assigned to pose 0"


def test_kin_dof_pose_assignment_equal_blocks(
    stack_of_two_six_res_ubqs, default_database, torch_device
):
    """Two poses with the same block count: all DOFs in {0, 1} and each pose
    receives the same number of DOFs.
    """
    pose_stack = stack_of_two_six_res_ubqs  # 2 poses, both 6 residues
    n_poses = pose_stack.n_poses
    net = _kin_net(_sfxn(default_database, torch_device), pose_stack)

    dpa = net.dof_pose_assignment()

    assert dpa.dtype == torch.int64
    assert dpa.shape == (net.masked_dofs.numel(),)
    assert dpa.min().item() == 0
    assert dpa.max().item() == n_poses - 1

    # Every pose must have at least one DOF.
    for i in range(n_poses):
        assert (dpa == i).sum().item() > 0, f"Pose {i} has no DOFs assigned"

    # Identical structure → identical DOF count per pose.
    counts = [(dpa == i).sum().item() for i in range(n_poses)]
    assert (
        counts[0] == counts[1]
    ), f"Equal-structure poses must get equal DOF counts; got {counts}"


def test_kin_dof_pose_assignment_jagged_poses(
    jagged_stack_of_465_res_ubqs, default_database, torch_device
):
    """Jagged stack (4, 6, 5 residues): every pose index appears and DOF
    counts grow with residue count.
    """
    pose_stack = jagged_stack_of_465_res_ubqs  # 3 poses: 4, 6, 5 residues
    n_poses = pose_stack.n_poses  # 3
    net = _kin_net(_sfxn(default_database, torch_device), pose_stack)

    dpa = net.dof_pose_assignment()

    assert dpa.dtype == torch.int64
    assert dpa.shape == (net.masked_dofs.numel(),)
    assert dpa.min().item() == 0
    assert dpa.max().item() == n_poses - 1

    # Every pose must have at least one DOF.
    for i in range(n_poses):
        assert (dpa == i).sum().item() > 0, f"Pose {i} has no DOFs assigned"

    # The number of backbone phi_c DOFs scales with residue count, so:
    #   pose 0 (4 res) < pose 1 (6 res)  and  pose 2 (5 res) < pose 1 (6 res)
    counts = [(dpa == i).sum().item() for i in range(n_poses)]
    assert (
        counts[0] < counts[1]
    ), f"4-res pose should have fewer DOFs than 6-res pose; got {counts}"
    assert (
        counts[2] < counts[1]
    ), f"5-res pose should have fewer DOFs than 6-res pose; got {counts}"


def test_kin_dof_pose_assignment_matches_id_formula(
    jagged_stack_of_465_res_ubqs, default_database, torch_device
):
    """Each kinematic DOF's pose index equals id[kin_atom] // max_n_pose_atoms.

    This verifies the formula used in dof_pose_assignment() against an
    independent reconstruction from the kinematic forest data.
    """
    pose_stack = jagged_stack_of_465_res_ubqs
    net = _kin_net(_sfxn(default_database, torch_device), pose_stack)

    max_n_pose_atoms = pose_stack.max_n_pose_atoms
    id_int = net.id.to(torch.int64)

    # Reconstruct expected assignment independently.
    # kin atom 0 is a virtual root with no real DOFs; assign it to pose 0
    # so indexing is consistent.
    expected_per_kin_atom = torch.zeros(
        id_int.shape[0], dtype=torch.int64, device=id_int.device
    )
    expected_per_kin_atom[1:] = id_int[1:] // max_n_pose_atoms

    masked_kin_atoms = net.dof_mask.nonzero(as_tuple=False)[:, 0]
    expected = expected_per_kin_atom[masked_kin_atoms]

    torch.testing.assert_close(net.dof_pose_assignment(), expected)
