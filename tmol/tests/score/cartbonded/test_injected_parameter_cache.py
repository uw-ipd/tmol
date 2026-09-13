"""Injecting bonded parameters must change energies on an already annotated pose."""

import attr
import pytest
import torch

from tmol.database import inject_residue_params
from tmol.score.cartbonded import CartBondedEnergyTerm
from tmol.tests.score.common import pose_stack_from_pdb_and_resnums


@pytest.mark.parametrize("block_pairs", [False, True])
@pytest.mark.parametrize("changed_first", [False, True])
def test_injected_parameters_have_independent_scoring_annotations(
    ubq_pdb, default_database, torch_device, block_pairs, changed_first
):
    pose = pose_stack_from_pdb_and_resnums(ubq_pdb, torch_device, [(24, 30)])
    pbt = pose.packed_block_types
    original = default_database.scoring.cartbonded.residue_params["ALA"]
    old_bond = next(
        row
        for row in original.length_parameters
        if (row.atm1, row.atm2) == ("CA", "CB")
    )
    new_bond = attr.evolve(old_bond, x0=2.3, K=199.0)
    replacement = attr.evolve(
        original,
        length_parameters=tuple(
            new_bond if row is old_bond else row for row in original.length_parameters
        ),
    )
    changed = inject_residue_params(
        default_database, residue_types=[], cartbonded_params={"ALA": replacement}
    )
    databases = {"original": default_database, "changed": changed}
    modules = {}
    for name in (("changed", "original") if changed_first else ("original", "changed")):
        term = CartBondedEnergyTerm(databases[name], torch_device)
        for bt in pbt.active_block_types:
            term.setup_block_type(bt)
        term.setup_packed_block_types(pbt)
        term.setup_poses(pose)
        render = (
            term.render_block_pair_scoring_module
            if block_pairs
            else term.render_whole_pose_scoring_module
        )
        modules[name] = render(pose)
    coords = pose.coords.double().clone()
    coords += 0.1 * torch.sin(
        torch.arange(coords.numel(), device=torch_device)
    ).reshape_as(coords)
    coords.requires_grad_(True)
    weights = 0.3 + torch.arange(pose.max_n_blocks, device=torch_device).double()

    def energy(name):
        values = modules[name](coords)[0]
        return (values[0].diagonal() * weights).sum() if block_pairs else values.sum()

    delta = energy("changed") - energy("original")

    # The native table stores single-precision parameters even for double
    # coordinates. Use the same represented constants in an independent
    # harmonic distance expression, without tmol geometry/hash helpers.
    def fp32(value):
        return float(torch.tensor(value, dtype=torch.float32))

    reference = coords.new_zeros(())
    n_alanines = 0
    for block, type_index in enumerate(pose.block_type_ind[0].tolist()):
        bt = pbt.active_block_types[type_index]
        if bt.base_name != "ALA":
            continue
        n_alanines += 1
        start = int(pose.block_coord_offset[0, block])
        distance = (
            coords[0, start + bt.atom_to_idx["CA"]]
            - coords[0, start + bt.atom_to_idx["CB"]]
        ).norm()
        value = 0.5 * fp32(new_bond.K) * (distance - fp32(new_bond.x0)).square()
        value -= 0.5 * fp32(old_bond.K) * (distance - fp32(old_bond.x0)).square()
        reference += value * (weights[block] if block_pairs else 1)
    assert n_alanines > 0
    assert float(reference.detach().abs()) > 1
    torch.testing.assert_close(delta, reference, rtol=1e-7, atol=1e-7)
    actual_gradient = torch.autograd.grad(delta, coords, retain_graph=True)[0]
    reference_gradient = torch.autograd.grad(reference, coords)[0]
    torch.testing.assert_close(
        actual_gradient, reference_gradient, rtol=1e-7, atol=1e-7
    )
    assert default_database.scoring.cartbonded.residue_params["ALA"] is original
    assert changed.scoring.cartbonded.hash != default_database.scoring.cartbonded.hash
