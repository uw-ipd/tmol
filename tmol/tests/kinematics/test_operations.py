import torch
import numpy
import numpy.testing

from math import nan

import pytest

from tmol.kinematics.operations import (
    ExecutionStrategy,
    backwardKin,
    forwardKin,
    resolveDerivs,
)

from tmol.kinematics.datatypes import (KinDOF, NodeType, KinTree)


def score(coords):
    """Dummy scorefunction for a conformation."""
    #assert coords.shape == (20, 3)
    dists = (coords.unsqueeze(1) - coords.unsqueeze(0)).norm(dim=-1)
    igraph = (
        torch.triu(
            ~torch.eye(
                dists.shape[0],
                dtype=torch.uint8,
                device=coords.device,
            )
        ) & (dists < 3.4)
    )
    score = (3.4 - dists[igraph]) * (3.4 - dists[igraph])
    return torch.sum(score)


def dscore(coords):
    """Dummy scorefunction derivs for a conformation."""
    #assert coords.shape == (20, 3)
    natoms = coords.shape[0]
    dxs = coords.unsqueeze(1) - coords.unsqueeze(0)
    dists = dxs.norm(dim=-1)
    igraph = (
        torch.triu(
            ~torch.
            eye(dists.shape[0], dtype=torch.uint8, device=coords.device)
        ) & (dists < 3.4)
    ).nonzero()

    dEdxs = torch.zeros([natoms, natoms, 3],
                        dtype=torch.double,
                        device=coords.device)
    dEdxs[igraph[:, 0], igraph[:, 1], :
          ] = -2 * (3.4 - dists[igraph[:, 0], igraph[:, 1]].reshape(-1, 1)) * (
              dxs[igraph[:, 0], igraph[:, 1], :] /
              dists[igraph[:, 0], igraph[:, 1]].reshape(-1, 1)
          )

    dEdx = torch.zeros([natoms, 3], device=coords.device)
    dEdx = dEdxs.sum(dim=1) - dEdxs.sum(dim=0)

    return dEdx


@pytest.fixture
def expected_analytic_derivs():
    return torch.tensor(
        [[+0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          +0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          +0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
         [+2.66453526e-15,  1.77635684e-15, -8.88178420e-16,
          +5.32907052e-15,  3.55271368e-15,  4.44089210e-15,
          +0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
         [+1.13718660e+01, -1.06522516e+01,  8.61394482e-01,
          +1.13718660e+01,             nan,             nan,
                      nan,             nan,             nan],
         [+1.70214350e+00,  3.40681357e-01, -7.30414633e+00,
          -1.77635684e-15,             nan,             nan,
                      nan,             nan,             nan],
         [+9.66972247e+00, -1.04967293e+01, -5.80415867e+00,
          +7.68770907e-01,             nan,             nan,
                      nan,             nan,             nan],
         [+7.68770907e-01, -6.06030702e-02, -9.52873353e+00,
          +1.77635684e-15,             nan,             nan,
                      nan,             nan,             nan],
         [-1.25079554e+01,  1.57258359e+00, -8.89832569e+00,
           5.52111345e+00,  7.40141077e+00, -9.92561863e+00,
           0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
         [-8.72247305e+00,  8.50512385e+00, -1.47396223e+01,
          -8.72247305e+00,             nan,             nan,
                      nan,             nan,             nan],
         [+1.70112178e+00,  4.52266852e+00, -6.81301657e+00,
          -7.10542736e-15,             nan,             nan,
                      nan,             nan,             nan],
         [-1.04235948e+01,  7.57222441e+00, -1.08827823e+01,
          -1.93873880e+00,             nan,             nan,
                      nan,             nan,             nan],
         [-1.93873880e+00,  5.88968488e+00, -4.52952181e+00,
          +0.00000000e+00,             nan,             nan,
                      nan,             nan,             nan],
         [-9.66405135e+00,  7.51131337e+00,  9.49787678e+00,
           2.14775674e-01,  1.47298659e+01, -1.71810938e+01,
           0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
         [+1.15033592e+01, -1.05854011e+01,  1.16243389e+00,
          +1.15033592e+01,             nan,             nan,
                      nan,             nan,             nan],
         [+1.69990508e+00,  3.43791230e-01, -7.30134090e+00,
          -5.32907052e-15,             nan,             nan,
                      nan,             nan,             nan],
         [+9.80345414e+00, -1.05263343e+01, -5.85401170e+00,
          +7.74393479e-01,             nan,             nan,
                      nan,             nan,             nan],
         [+7.74393479e-01, -4.76323730e-02, -9.57540886e+00,
          -2.66453526e-15,             nan,             nan,
                      nan,             nan,             nan],
         [+1.45955969e+01,  3.92793100e+00,  3.79171849e+00,
          +2.46056151e+00,  1.02226395e+01, -9.80368607e+00,
           0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
         [-9.32753028e+00,  6.84468376e+00, -1.51385559e+01,
          -9.32753028e+00,             nan,             nan,
                      nan,             nan,             nan],
         [+1.62693321e+00,  3.87588970e+00, -6.91878652e+00,
          -3.55271368e-15,             nan,             nan,
                      nan,             nan,             nan],
         [-1.09544635e+01,  9.24780005e+00, -1.05711446e+01,
          -1.79443222e+00,             nan,             nan,
                      nan,             nan,             nan],
         [-1.79443222e+00,  6.15307006e+00, -3.75430706e+00,
          +1.77635684e-15,             nan,             nan,
                      nan,             nan,             nan]],
        dtype=torch.double
    ) # yapf: disable


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
    #fd: with single precision 1e-9 is too strict for the assert_allclose calls
    bkin = backwardKin(kintree, coords)

    for es in ExecutionStrategy:
        fkin = forwardKin(kintree, bkin.dofs, es)
        numpy.testing.assert_allclose(coords, fkin.coords, atol=1e-6)


def test_perturb(kintree, coords, torch_device):
    dofs = backwardKin(kintree, coords).dofs

    pcoords = forwardKin(kintree, dofs).coords
    assert numpy.allclose(coords, pcoords)

    def coord_changed(a, b, atol=1e-3):
        return numpy.abs(a - b) > atol

    # Translate jump dof
    #fd: with single precision 1e-7 is too strict
    t_dofs = dofs.clone()
    t_dofs.jump.RBx[6] += 0.2
    t_dofs.jump.RBy[6] += 0.2
    t_dofs.jump.RBz[6] += 0.2
    pcoords = forwardKin(kintree, t_dofs).coords

    numpy.testing.assert_allclose(pcoords[1:6], coords[1:6], atol=1e-6)
    assert numpy.all(coord_changed(pcoords[6:11], coords[6:11]))
    numpy.testing.assert_allclose(pcoords[11:16], coords[11:16], atol=1e-6)
    numpy.testing.assert_allclose(pcoords[16:21], coords[16:21], atol=1e-6)

    # Rotate jump dof "delta"
    rd_dofs = dofs.clone()

    assert rd_dofs.jump.RBdel_alpha[6] == 0
    assert rd_dofs.jump.RBdel_beta[6] == 0
    assert rd_dofs.jump.RBdel_gamma[6] == 0

    rd_dofs.jump.RBdel_alpha[6] += 0.1
    rd_dofs.jump.RBdel_beta[6] += 0.2
    rd_dofs.jump.RBdel_gamma[6] += 0.3

    pcoords = forwardKin(kintree, rd_dofs).coords
    numpy.testing.assert_allclose(pcoords[1:6], coords[1:6], atol=1e-6)
    numpy.testing.assert_allclose(pcoords[6], coords[6], atol=1e-6)
    assert numpy.all(
        numpy.any(coord_changed(pcoords[7:11], coords[7:11]), axis=-1)
    )
    numpy.testing.assert_allclose(pcoords[11:16], coords[11:16], atol=1e-6)
    numpy.testing.assert_allclose(pcoords[16:21], coords[16:21], atol=1e-6)

    # Rotate jump dof
    r_dofs = dofs.clone()

    r_dofs.jump.RBalpha[6] += 0.1
    r_dofs.jump.RBbeta[6] += 0.2
    r_dofs.jump.RBgamma[6] += 0.3
    pcoords = forwardKin(kintree, r_dofs).coords
    numpy.testing.assert_allclose(pcoords[1:6], coords[1:6], atol=1e-6)
    numpy.testing.assert_allclose(pcoords[6], coords[6], atol=1e-6)
    assert numpy.all(
        numpy.any(coord_changed(pcoords[7:11], coords[7:11]), axis=-1)
    )
    numpy.testing.assert_allclose(pcoords[11:16], coords[11:16], atol=1e-6)
    numpy.testing.assert_allclose(pcoords[16:21], coords[16:21], atol=1e-6)


@pytest.mark.parametrize("execution_strategy", [e for e in ExecutionStrategy])
def test_root_sibling_derivs(torch_device, execution_strategy):
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
    kintree[5] = KinTree.node(1, BOND, 4, 5, 4, 1)  #fd: 2->1

    coords = torch.tensor([
        [0.000, 0.000, 0.000],
        [2.000, 2.000, 2.000],
        [3.458, 2.000, 2.000],
        [3.988, 1.222, 0.804],
        [4.009, 3.420, 2.000],
        [3.383, 4.339, 1.471],
    ]).to(torch.double)

    compute_verify_derivs(kintree, coords, execution_strategy)


@pytest.mark.parametrize("execution_strategy", [e for e in ExecutionStrategy])
def test_derivs(
        kintree, coords, torch_device, execution_strategy,
        expected_analytic_derivs
):
    compute_verify_derivs(kintree, coords, execution_strategy)


def compute_verify_derivs(
        kintree,
        coords,
        execution_strategy,
        expected_analytic_derivs=None,
):
    NATOMS, _ = coords.shape
    bkin = backwardKin(kintree, coords)
    HTs, dofs = bkin.hts, bkin.dofs

    # Compute numeric derivs and store node indicies
    bonds = []
    jumps = []

    dsc_dtors_numeric = KinDOF.full(len(dofs), 0)
    for i in numpy.arange(0, NATOMS):
        if kintree.doftype[i] == NodeType.bond:
            ndof = 4
            bonds.append(i)
        elif kintree.doftype[i] == NodeType.jump:
            ndof = 6
            jumps.append(i)
        elif kintree.doftype[i] == NodeType.root:
            assert i == 0
            continue
        else:
            raise NotImplementedError

        for j in range(ndof):
            dofs.raw[i, j] += 0.0001
            coordsAlt = forwardKin(kintree, dofs).coords
            sc_p = score(coordsAlt[1:, :])
            dofs.raw[i, j] -= 0.0002
            coordsAlt = forwardKin(kintree, dofs).coords
            sc_m = score(coordsAlt[1:, :])
            dofs.raw[i, j] += 0.0001

            dsc_dtors_numeric.raw[i, j] = (sc_p - sc_m) / 0.0002

    # Compute analytic derivs for all available strategies
    dsc_dx = coords.new_zeros([NATOMS, 3], dtype=torch.double)
    dsc_dx[1:] = dscore(coords[1:, :])

    dsc_dtors_analytic = resolveDerivs(
        kintree,
        dofs,
        HTs,
        dsc_dx,
        execution_strategy,
    )

    # Verify numeric/analytic derivatives
    assert_jump_dof_allclose(
        dsc_dtors_analytic.jump[jumps],
        dsc_dtors_numeric.jump[jumps],
        atol=1e-7,
    )

    assert_bond_dof_allclose(
        dsc_dtors_analytic.bond[bonds],
        dsc_dtors_numeric.bond[bonds],
        atol=1e-7,
    )

    if expected_analytic_derivs is not None:
        # Verify against stored derivatives for regression
        #fd: reducing tolerance here to 1e-4
        # * numpy double->torch double leads to differences as high as 5e-6
        # * numeric v analytic comparison is still at 1e-7 so these changes
        #     are likely due to changes in the "dummy score"
        numpy.testing.assert_allclose(
            dsc_dtors_analytic.raw[1:],
            expected_analytic_derivs[1:],
            atol=1e-4,
        )


def assert_bond_dof_allclose(actual, expected, **kwargs):
    numpy.testing.assert_allclose(actual.phi_p, expected.phi_p, **kwargs)
    numpy.testing.assert_allclose(actual.theta, expected.theta, **kwargs)
    numpy.testing.assert_allclose(actual.d, expected.d, **kwargs)
    numpy.testing.assert_allclose(actual.phi_c, expected.phi_c, **kwargs)


def assert_jump_dof_allclose(actual, expected, **kwargs):
    numpy.testing.assert_allclose(actual.RBx, expected.RBx, **kwargs)
    numpy.testing.assert_allclose(actual.RBy, expected.RBy, **kwargs)
    numpy.testing.assert_allclose(actual.RBz, expected.RBz, **kwargs)

    numpy.testing.assert_allclose(
        actual.RBdel_alpha, expected.RBdel_alpha, **kwargs
    )
    numpy.testing.assert_allclose(
        actual.RBdel_beta, expected.RBdel_beta, **kwargs
    )
    numpy.testing.assert_allclose(
        actual.RBdel_gamma, expected.RBdel_gamma, **kwargs
    )

    numpy.testing.assert_allclose(actual.RBalpha, expected.RBalpha, **kwargs)
    numpy.testing.assert_allclose(actual.RBbeta, expected.RBbeta, **kwargs)
    numpy.testing.assert_allclose(actual.RBgamma, expected.RBgamma, **kwargs)
