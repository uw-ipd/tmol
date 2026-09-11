"""Renamed gamma-peptide junctions retain the curated peptide harmonic forces."""

import torch

from tmol.io import pose_stack_from_cif
from tmol.tests.data import data_path
from tmol.tests.score.cartbonded.test_explicit_connection_parameters import scoring_term


def test_gamma_junction_length_angle_energy_and_gradient(torch_device):
    pose, context = pose_stack_from_cif(
        data_path("ncaa_fixtures") / "gamma_peptide_1gac.cif",
        torch_device,
        prepare_ligands=True,
        ligand_seed=20260909,
        no_optH=True,
        return_context=True,
    )
    pbt = pose.packed_block_types
    types = [pbt.active_block_types[i] for i in pose.block_type_ind[0].tolist()]
    left = next(i for i, bt in enumerate(types) if bt.base_name == "FGA")
    right = int(
        pose.inter_residue_connections[0, left, types[left].up_connection_ind, 0]
    )
    assert right > left
    assert types[right].base_name != "PRO"
    coords = pose.coords.double().clone()
    coords += 0.07 * torch.sin(
        torch.arange(coords.numel(), device=torch_device).reshape_as(coords)
    )
    coords.requires_grad_(True)
    module = scoring_term(
        pose, context.parameter_database
    ).render_block_pair_scoring_module(pose)
    actual = module(coords)[:2, 0, left, right]

    def xyz(block, name):
        return coords[
            0, int(pose.block_coord_offset[0, block]) + types[block].atom_to_idx[name]
        ]

    def constant(value):
        # Native parameter storage is float32 even when coordinates are double.
        return float(torch.tensor(value, dtype=torch.float32))

    c, n = xyz(left, "CD"), xyz(right, "N")
    # Independent Cartesian formulas with the database's peptide constants;
    # no generated row traversal or tmol geometry/derivative helper is used.
    length = 0.5 * constant(369.445) * ((c - n).norm() - constant(1.32868)).square()
    angle = coords.new_zeros(())
    for a, b, d, optimum, stiffness in (
        (xyz(left, "CG"), c, n, 2.02807, 160.0),
        (xyz(left, "OE1"), c, n, 2.14676, 170.864),
        (c, n, xyz(right, "CA"), 2.12407, 96.53),
        (c, n, xyz(right, "H"), 2.07956, 76.432),
    ):
        first, second = a - b, d - b
        theta = torch.acos(first.dot(second) / (first.norm() * second.norm()))
        angle += 0.5 * constant(stiffness) * (theta - constant(optimum)).square()
    expected = torch.stack((length, angle))
    assert float(expected.detach().sum()) > 0.01
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    weights = coords.new_tensor((0.7, 1.9))
    actual_grad = torch.autograd.grad(
        (actual * weights).sum(), coords, retain_graph=True
    )[0]
    expected_grad = torch.autograd.grad((expected * weights).sum(), coords)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=2e-5, rtol=2e-5)
