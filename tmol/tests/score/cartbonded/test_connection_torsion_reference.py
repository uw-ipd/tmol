"""Independent Cartesian reference for the peptide-bond torsions."""

import pytest
import torch

from tmol.score.cartbonded import CartBondedEnergyTerm
from tmol.tests.score.common import pose_stack_from_pdb_and_resnums


@pytest.mark.parametrize("resnums", [[(0, 4)], [(17, 20)]])
def test_peptide_bond_torsion_energy_and_gradient(
    ubq_pdb, default_database, torch_device, resnums
):
    pose = pose_stack_from_pdb_and_resnums(ubq_pdb, torch_device, resnums)
    term = CartBondedEnergyTerm(default_database, torch_device)
    pbt = pose.packed_block_types
    for bt in pbt.active_block_types:
        term.setup_block_type(bt)
    term.setup_packed_block_types(pbt)
    term.setup_poses(pose)
    module = term.render_block_pair_scoring_module(pose)
    coords = pose.coords.double().clone()
    # Twist the peptide bonds away from planarity, where the energy and its
    # gradient would be near zero whether or not the terms are scored.
    coords += 0.17 * torch.sin(
        torch.arange(coords.numel(), device=torch_device)
    ).reshape_as(coords)
    coords.requires_grad_(True)
    n_blocks = pose.max_n_blocks
    weights = 0.3 + torch.arange(n_blocks * n_blocks, device=torch_device).reshape(
        n_blocks, n_blocks
    )
    weights.fill_diagonal_(0)
    actual = (module(coords)[2, 0] * weights).sum()

    def xyz(block, name):
        bt = pbt.active_block_types[int(pose.block_type_ind[0, block])]
        return coords[0, int(pose.block_coord_offset[0, block]) + bt.atom_to_idx[name]]

    def torsion(k2, a, b, c, d):
        # k2 * (cos(2 phi - pi) + 1), the wildcard rows' potential, is
        # 2 k2 sin(phi)^2. Plane normals avoid tmol's dihedral routines.
        first = torch.linalg.cross(b - a, c - b)
        second = torch.linalg.cross(c - b, d - c)
        cosine = first.dot(second) / (first.norm() * second.norm())
        return 2 * k2 * (1 - cosine.square())

    expected = coords.new_zeros(())
    for left in range(n_blocks - 1):
        right = left + 1
        bt = pbt.active_block_types[int(pose.block_type_ind[0, right])]
        c, n = xyz(left, "C"), xyz(right, "N")
        energy = torsion(9.667, xyz(left, "O"), c, n, xyz(right, "CA"))
        if bt.base_name != "PRO":
            energy = energy + torsion(10.458, xyz(left, "CA"), c, n, xyz(right, "H"))
            energy = energy + torsion(10.992, xyz(left, "O"), c, n, xyz(right, "H"))
        expected += weights[left, right] * energy
    assert float(expected.detach()) > 0.01
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    actual_grad = torch.autograd.grad(actual, coords, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, coords)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=2e-5, rtol=2e-5)
