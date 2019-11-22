import torch
import numpy
import numpy.testing

import pytest

from tmol.kinematics.builder import KinematicBuilder
from tmol.kinematics.operations import inverseKin, forwardKin
from tmol.kinematics.script_modules import KinematicModule
from tmol.system.packed import PackedResidueSystem, PackedResidueSystemStack
from tmol.score.bonded_atom import BondedAtomScoreGraph

from tmol.kinematics.datatypes import NodeType, KinTree


def score(coords):
    """Dummy scorefunction for a conformation."""
    # assert coords.shape == (20, 3)
    dists = (coords.unsqueeze(1) - coords.unsqueeze(0)).norm(dim=-1)
    igraph = torch.triu(
        ~torch.eye(dists.shape[0], dtype=torch.uint8, device=coords.device)
    ) & (dists < 3.4)
    score = (3.4 - dists[igraph]) * (3.4 - dists[igraph])
    return torch.sum(score)


def dscore(coords):
    """Dummy scorefunction derivs for a conformation."""
    # assert coords.shape == (20, 3)
    natoms = coords.shape[0]
    dxs = coords.unsqueeze(1) - coords.unsqueeze(0)
    dists = dxs.norm(dim=-1)
    igraph = (
        torch.triu(~torch.eye(dists.shape[0], dtype=torch.uint8, device=coords.device))
        & (dists < 3.4)
    ).nonzero()

    dEdxs = torch.zeros([natoms, natoms, 3], dtype=torch.double, device=coords.device)
    dEdxs[igraph[:, 0], igraph[:, 1], :] = (
        -2
        * (3.4 - dists[igraph[:, 0], igraph[:, 1]].reshape(-1, 1))
        * (
            dxs[igraph[:, 0], igraph[:, 1], :]
            / dists[igraph[:, 0], igraph[:, 1]].reshape(-1, 1)
        )
    )

    dEdx = torch.zeros([natoms, 3], device=coords.device)
    dEdx = dEdxs.sum(dim=1) - dEdxs.sum(dim=0)

    return dEdx


@pytest.fixture
def kintree(torch_device):
    ROOT = NodeType.root
    JUMP = NodeType.jump
    BOND = NodeType.bond
    NATOMS = 21

    # kinematics definition
    kintree = KinTree.full(NATOMS, 0)
    kintree[0] = KinTree.node(0, ROOT, 0, 0, 0, 0)

    kintree[1] = KinTree.node(1, JUMP, 0, 2, 1, 3)
    kintree[2] = KinTree.node(1, BOND, 1, 2, 1, 3)
    kintree[3] = KinTree.node(1, BOND, 2, 3, 2, 1)
    kintree[4] = KinTree.node(1, BOND, 2, 4, 2, 1)
    kintree[5] = KinTree.node(1, BOND, 4, 5, 4, 2)

    kintree[6] = KinTree.node(2, JUMP, 1, 7, 6, 8)
    kintree[7] = KinTree.node(2, BOND, 6, 7, 6, 8)
    kintree[8] = KinTree.node(2, BOND, 7, 8, 7, 6)
    kintree[9] = KinTree.node(2, BOND, 7, 9, 7, 6)
    kintree[10] = KinTree.node(2, BOND, 9, 10, 9, 7)

    kintree[11] = KinTree.node(3, JUMP, 1, 12, 11, 13)
    kintree[12] = KinTree.node(3, BOND, 11, 12, 11, 13)
    kintree[13] = KinTree.node(3, BOND, 12, 13, 12, 11)
    kintree[14] = KinTree.node(3, BOND, 12, 14, 12, 11)
    kintree[15] = KinTree.node(3, BOND, 14, 15, 14, 12)

    kintree[16] = KinTree.node(4, JUMP, 1, 17, 16, 18)
    kintree[17] = KinTree.node(4, BOND, 16, 17, 16, 18)
    kintree[18] = KinTree.node(4, BOND, 17, 18, 17, 16)
    kintree[19] = KinTree.node(4, BOND, 17, 19, 17, 16)
    kintree[20] = KinTree.node(4, BOND, 19, 20, 19, 17)

    return kintree.to(device=torch_device)


@pytest.fixture
def coords(torch_device):
    NATOMS = 21
    coords = torch.empty([NATOMS, 3], dtype=torch.double)

    coords[0, :] = torch.Tensor([0.000, 0.000, 0.000])

    coords[1, :] = torch.Tensor([2.000, 2.000, 2.000])
    coords[2, :] = torch.Tensor([3.458, 2.000, 2.000])
    coords[3, :] = torch.Tensor([3.988, 1.222, 0.804])
    coords[4, :] = torch.Tensor([4.009, 3.420, 2.000])
    coords[5, :] = torch.Tensor([3.383, 4.339, 1.471])

    coords[6, :] = torch.Tensor([5.184, 3.594, 2.596])
    coords[7, :] = torch.Tensor([5.821, 4.903, 2.666])
    coords[8, :] = torch.Tensor([5.331, 5.667, 3.888])
    coords[9, :] = torch.Tensor([7.339, 4.776, 2.690])
    coords[10, :] = torch.Tensor([7.881, 3.789, 3.186])

    coords[11, :] = torch.Tensor([7.601, 2.968, 5.061])
    coords[12, :] = torch.Tensor([6.362, 2.242, 4.809])
    coords[13, :] = torch.Tensor([6.431, 0.849, 5.419])
    coords[14, :] = torch.Tensor([5.158, 3.003, 5.349])
    coords[15, :] = torch.Tensor([5.265, 3.736, 6.333])

    coords[16, :] = torch.Tensor([4.011, 2.824, 4.701])
    coords[17, :] = torch.Tensor([2.785, 3.494, 5.115])
    coords[18, :] = torch.Tensor([2.687, 4.869, 4.470])
    coords[19, :] = torch.Tensor([1.559, 2.657, 4.776])
    coords[20, :] = torch.Tensor([1.561, 1.900, 3.805])

    return coords.to(device=torch_device)


def test_score_smoketest(coords):
    score(coords[1:, :])


def test_forward_refold(kintree, coords, torch_device):
    # fd: with single precision 1e-9 is too strict for the assert_allclose calls
    dofs = inverseKin(kintree, coords)
    refold_kincoords = forwardKin(kintree, dofs)

    numpy.testing.assert_allclose(coords.cpu(), refold_kincoords.cpu(), atol=1e-6)


def test_perturb(kintree, coords, torch_device):
    dofs = inverseKin(kintree, coords)
    pcoords = forwardKin(kintree, dofs)

    assert numpy.allclose(coords.cpu(), pcoords.cpu())

    def coord_changed(a, b, atol=1e-3):
        return numpy.abs(a - b) > atol

    # Translate jump dof
    # fd: with single precision 1e-7 is too strict
    t_dofs = dofs.clone()
    t_dofs.jump.RBx[6] += 0.2
    t_dofs.jump.RBy[6] += 0.2
    t_dofs.jump.RBz[6] += 0.2
    pcoords = forwardKin(kintree, t_dofs)

    numpy.testing.assert_allclose(pcoords[1:6].cpu(), coords[1:6].cpu(), atol=1e-6)
    assert numpy.all(coord_changed(pcoords[6:11].cpu(), coords[6:11].cpu()))
    numpy.testing.assert_allclose(pcoords[11:16].cpu(), coords[11:16].cpu(), atol=1e-6)
    numpy.testing.assert_allclose(pcoords[16:21].cpu(), coords[16:21].cpu(), atol=1e-6)

    # Rotate jump dof "delta"
    rd_dofs = dofs.clone()

    assert rd_dofs.jump.RBdel_alpha[6] == 0
    assert rd_dofs.jump.RBdel_beta[6] == 0
    assert rd_dofs.jump.RBdel_gamma[6] == 0

    rd_dofs.jump.RBdel_alpha[6] += 0.1
    rd_dofs.jump.RBdel_beta[6] += 0.2
    rd_dofs.jump.RBdel_gamma[6] += 0.3

    pcoords = forwardKin(kintree, rd_dofs)
    numpy.testing.assert_allclose(pcoords[1:6].cpu(), coords[1:6].cpu(), atol=1e-6)
    numpy.testing.assert_allclose(pcoords[6].cpu(), coords[6].cpu(), atol=1e-6)
    assert numpy.all(
        numpy.any(coord_changed(pcoords[7:11].cpu(), coords[7:11].cpu()), axis=-1)
    )
    numpy.testing.assert_allclose(pcoords[11:16].cpu(), coords[11:16].cpu(), atol=1e-6)
    numpy.testing.assert_allclose(pcoords[16:21].cpu(), coords[16:21].cpu(), atol=1e-6)

    # Rotate jump dof
    r_dofs = dofs.clone()

    r_dofs.jump.RBalpha[6] += 0.1
    r_dofs.jump.RBbeta[6] += 0.2
    r_dofs.jump.RBgamma[6] += 0.3
    pcoords = forwardKin(kintree, r_dofs)
    numpy.testing.assert_allclose(pcoords[1:6].cpu(), coords[1:6].cpu(), atol=1e-6)
    numpy.testing.assert_allclose(pcoords[6].cpu(), coords[6].cpu(), atol=1e-6)
    assert numpy.all(
        numpy.any(coord_changed(pcoords[7:11].cpu(), coords[7:11].cpu()), axis=-1)
    )
    numpy.testing.assert_allclose(pcoords[11:16].cpu(), coords[11:16].cpu(), atol=1e-6)
    numpy.testing.assert_allclose(pcoords[16:21].cpu(), coords[16:21].cpu(), atol=1e-6)


def test_root_sibling_derivs(torch_device):
    """Verify derivatives in post-jump bonded siblings."""
    NATOMS = 6

    ROOT = NodeType.root
    JUMP = NodeType.jump
    BOND = NodeType.bond

    # kinematics definition
    kintree = KinTree.full(NATOMS, 0)
    kintree[0] = KinTree.node(0, ROOT, 0, 0, 0, 0)

    kintree[1] = KinTree.node(1, JUMP, 0, 2, 1, 3)
    kintree[2] = KinTree.node(1, BOND, 1, 2, 1, 3)
    kintree[3] = KinTree.node(1, BOND, 1, 3, 1, 2)
    kintree[4] = KinTree.node(1, BOND, 1, 4, 1, 2)
    kintree[5] = KinTree.node(1, BOND, 4, 5, 4, 1)  # fd: 2->1

    coords = torch.tensor(
        [
            [0.000, 0.000, 0.000],
            [2.000, 2.000, 2.000],
            [3.458, 2.000, 2.000],
            [3.988, 1.222, 0.804],
            [4.009, 3.420, 2.000],
            [3.383, 4.339, 1.471],
        ]
    ).to(torch.double)

    compute_verify_derivs(kintree, coords)


def test_derivs(kintree, coords, torch_device):
    compute_verify_derivs(kintree, coords)


# use torch autograd machinery
# note that the "zeroing output" machinery fails on CUDA
#   instead, use an explicit check to make sure derivatives match
def compute_verify_derivs(kintree, coords):
    dofs = inverseKin(kintree, coords)
    op = KinematicModule(kintree, coords.device)

    # we only minimize the "rbdel" dofs
    minimizable_dofs = dofs.raw[:, :6].requires_grad_(True)

    def eval_kin(dofs_x):
        dofsfull = dofs.raw.clone()
        dofsfull[:, :6] = dofs_x
        return op(dofsfull)

    result = eval_kin(minimizable_dofs)

    # Extract results from torch/autograd/gradcheck.py
    from torch.autograd.gradcheck import get_numerical_jacobian, get_analytical_jacobian

    (analytical,), reentrant, correct_grad_sizes = get_analytical_jacobian(
        (minimizable_dofs,), result
    )
    numerical = get_numerical_jacobian(
        eval_kin, minimizable_dofs, minimizable_dofs, 2e-3
    )

    torch.testing.assert_allclose(analytical, numerical)


def test_forward_refold_two_systems(ubq_system, torch_device):
    twoubq = PackedResidueSystemStack((ubq_system, ubq_system))
    bonds = BondedAtomScoreGraph.build_for(twoubq, device=torch_device)
    tworoots = numpy.array((0, twoubq.systems[0].system_size), dtype=int)

    coords_raw = torch.tensor(
        numpy.concatenate((twoubq.systems[0].coords, twoubq.systems[1].coords), axis=0),
        dtype=torch.float64,
        device=torch_device,
    )

    kintree = (
        KinematicBuilder()
        .append_connected_components(
            tworoots,
            *KinematicBuilder.bonds_to_connected_component(
                roots=tworoots,
                bonds=bonds.bonds,
                system_size=int(twoubq.systems[0].system_size),
            ),
        )
        .kintree.to(torch_device)
    )

    ids = kintree.id.type(torch.long)
    coords = coords_raw[ids]
    coords[0, :] = 0

    dofs = inverseKin(kintree, coords)
    refold_kincoords = forwardKin(kintree, dofs)
    refold_kincoords[0, :] = 0

    numpy.testing.assert_allclose(coords.cpu(), refold_kincoords.cpu(), atol=1e-6)


def test_forward_refold_w_jagged_system(ubq_res, torch_device):
    # torch_device = torch.device("cpu")
    ubq40 = PackedResidueSystem.from_residues(ubq_res[:40])
    ubq60 = PackedResidueSystem.from_residues(ubq_res[:60])
    system_size = max(ubq40.system_size, ubq60.system_size)

    twoubq = PackedResidueSystemStack((ubq40, ubq60))
    bonds = BondedAtomScoreGraph.build_for(twoubq, device=torch_device)
    tworoots = numpy.array((0, twoubq.systems[1].system_size), dtype=int)

    coords_raw = torch.zeros(
        (2, system_size, 3), dtype=torch.float64, device=torch_device
    )
    coords_raw[0, 0 : ubq40.system_size, :] = torch.tensor(
        ubq40.coords, dtype=torch.float64, device=torch_device
    )
    coords_raw[1, 0 : ubq60.system_size, :] = torch.tensor(
        ubq60.coords, dtype=torch.float64, device=torch_device
    )
    coords_raw = coords_raw.reshape(2 * system_size, 3)

    kintree = (
        KinematicBuilder()
        .append_connected_components(
            tworoots,
            *KinematicBuilder.bonds_to_connected_component(
                roots=tworoots, bonds=bonds.bonds, system_size=int(system_size)
            ),
        )
        .kintree.to(torch_device)
    )

    ids = kintree.id.type(torch.long)
    coords = coords_raw[ids, :]
    coords[0, :] = 0

    dofs = inverseKin(kintree, coords)
    refold_kincoords = forwardKin(kintree, dofs)
    refold_kincoords[0, :] = 0

    numpy.testing.assert_allclose(coords.cpu(), refold_kincoords.cpu(), atol=1e-6)
