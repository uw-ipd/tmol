"""Independent Cartesian reference for ordinary and proline peptide impropers."""

import pytest
import torch

from tmol.score.cartbonded import CartBondedEnergyTerm
from tmol.tests.score.common import pose_stack_from_pdb_and_resnums


@pytest.mark.parametrize("resnums", [[(0, 4)], [(17, 20)]])
def test_connection_improper_energy_and_gradient(
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
    # Move away from planarity, where both a missing term and a correct term
    # would have near-zero gradients.
    coords += 0.17 * torch.sin(
        torch.arange(coords.numel(), device=torch_device)
    ).reshape_as(coords)
    coords.requires_grad_(True)
    n_blocks = pose.max_n_blocks
    weights = 0.3 + torch.arange(n_blocks * n_blocks, device=torch_device).reshape(
        n_blocks, n_blocks
    )
    weights.fill_diagonal_(0)
    actual = (module(coords)[3, 0] * weights).sum()

    def xyz(block, name):
        bt = pbt.active_block_types[int(pose.block_type_ind[0, block])]
        return coords[0, int(pose.block_coord_offset[0, block]) + bt.atom_to_idx[name]]

    def improper(a, b, c, d):
        # The database's k2=20, phi2=pi potential is 40*sin(phi)^2.
        # Plane normals avoid using tmol's dihedral or derivative routines.
        first = torch.linalg.cross(b - a, c - b)
        second = torch.linalg.cross(c - b, d - c)
        cosine = first.dot(second) / (first.norm() * second.norm())
        return 40 * (1 - cosine.square())

    expected = coords.new_zeros(())
    for left in range(n_blocks - 1):
        right = left + 1
        bt = pbt.active_block_types[int(pose.block_type_ind[0, right])]
        nitrogen_quad = (
            (xyz(right, "CD"), xyz(left, "C"), xyz(right, "N"), xyz(right, "CA"))
            if bt.base_name == "PRO"
            else (xyz(right, "CA"), xyz(left, "C"), xyz(right, "N"), xyz(right, "H"))
        )
        carbon_quad = (xyz(left, "CA"), xyz(right, "N"), xyz(left, "C"), xyz(left, "O"))
        expected += weights[left, right] * (
            improper(*nitrogen_quad) + improper(*carbon_quad)
        )
    assert float(expected.detach()) > 0.01
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    actual_grad = torch.autograd.grad(actual, coords, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, coords)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=2e-5, rtol=2e-5)
