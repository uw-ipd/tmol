"""Bounded Cartbonded annotations and ownership on a reused pose."""

import attr
import pytest
import torch

from tmol.database import inject_residue_params
from tmol.io import extended_pose_stack_from_sequences
from tmol.score.cartbonded import CartBondedEnergyTerm


def setup(term, pose):
    for bt in pose.packed_block_types.active_block_types:
        term.setup_block_type(bt)
    term.setup_packed_block_types(pose.packed_block_types)
    term.setup_poses(pose)


def test_unindexed_cuda_device_reuses_annotations(
    default_database, torch_device, monkeypatch
):
    if torch_device.type != "cuda":
        pytest.skip("Unindexed CUDA must compare against its resolved device")
    pose = extended_pose_stack_from_sequences(["AA"], device=torch_device)
    term = CartBondedEnergyTerm(default_database, torch.device("cuda"))
    first = term.setup_packed_block_types(pose.packed_block_types)

    def unexpected_lookup(*args):
        raise AssertionError("A warm annotation should not be rebuilt")

    monkeypatch.setattr(term, "get_params_for_res", unexpected_lookup)
    assert term.setup_packed_block_types(pose.packed_block_types) is first


@pytest.mark.parametrize("reverse", [False, True])
def test_connection_ownership_follows_term_and_module(
    default_database, torch_device, reverse
):
    pose = extended_pose_stack_from_sequences(["AA"], device=torch_device)
    empty = attr.evolve(
        default_database,
        scoring=attr.evolve(
            default_database.scoring,
            genbonded=attr.evolve(
                default_database.scoring.genbonded, rosetta_typed=frozenset()
            ),
        ),
    )
    terms = {
        "owned": CartBondedEnergyTerm(default_database, torch_device),
        "unowned": CartBondedEnergyTerm(empty, torch_device),
    }
    modules = {}
    for name in (("unowned", "owned") if reverse else ("owned", "unowned")):
        setup(terms[name], pose)
        modules[name] = terms[name].render_block_pair_scoring_module(pose)
    coords = pose.coords.double().clone()
    coords += 0.2 * torch.sin(
        torch.arange(coords.numel(), device=torch_device).reshape_as(coords)
    )
    coords.requires_grad_(True)
    energies = {name: module(coords)[3, 0, 0, 1] for name, module in modules.items()}
    assert energies["owned"].item() > 0.01
    assert energies["unowned"].item() == 0
    gradient = torch.autograd.grad(energies["unowned"], coords, retain_graph=True)[0]
    assert torch.count_nonzero(gradient) == 0
    for name, term in terms.items():
        assert (
            term.render_block_pair_scoring_module(pose)(coords)[3, 0, 0, 1].item()
            == energies[name].item()
        )
        assert modules[name](coords)[3, 0, 0, 1].item() == energies[name].item()


def test_eviction_bounds_annotations_and_preserves_rendered_modules(
    default_database, torch_device
):
    pose = extended_pose_stack_from_sequences(["AA"], device=torch_device)
    original = default_database.scoring.cartbonded.residue_params["ALA"]
    terms, modules, energies = [], [], []
    ownership_mask = None
    for i in range(5):
        row = attr.evolve(original.length_parameters[0], x0=1.0 + 0.2 * i)
        changed = inject_residue_params(
            default_database,
            [],
            cartbonded_params={
                "ALA": attr.evolve(
                    original, length_parameters=(row, *original.length_parameters[1:])
                )
            },
        )
        term = CartBondedEnergyTerm(changed, torch_device)
        setup(term, pose)
        current_mask = pose.packed_block_types.cartbonded_atom_is_rosetta
        if ownership_mask is not None:
            assert current_mask is ownership_mask
        ownership_mask = current_mask
        module = term.render_whole_pose_scoring_module(pose)
        terms.append(term)
        modules.append(module)
        energies.append(module(pose.coords))
    pbt = pose.packed_block_types
    assert len(pbt.cartbonded_annotations) <= 2
    assert all(len(bt.cartbonded_annotations) <= 2 for bt in pbt.active_block_types)
    assert terms[0].hash not in pbt.cartbonded_annotations
    for term, module, expected in zip(terms, modules, energies):
        torch.testing.assert_close(module(pose.coords), expected)
        torch.testing.assert_close(
            term.render_whole_pose_scoring_module(pose)(pose.coords), expected
        )
        assert len(pbt.cartbonded_annotations) <= 2
