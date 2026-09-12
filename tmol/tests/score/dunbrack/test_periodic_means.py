"""Changing a mean angle's full-turn representative cannot change its potential."""

import math

import attr
import pytest
import torch

from tmol.io import pose_stack_from_pdb
from tmol.score.dunbrack import DunbrackEnergyTerm, ScoringDunbrackDatabaseView
from tmol.tests.score.dunbrack.test_parameter_identity import setup, render


@pytest.mark.parametrize("block_pairs", [False, True])
@pytest.mark.parametrize("turns", [-2, 1, 3])
def test_mean_full_turns_preserve_energies_and_gradients(
    default_database, ubq_pdb, torch_device, block_pairs, turns
):
    pose = pose_stack_from_pdb(ubq_pdb, torch_device, residue_start=0, residue_end=12)
    original = DunbrackEnergyTerm(default_database, torch_device)
    shifted = DunbrackEnergyTerm(default_database, torch_device)
    index = [field.name for field in attr.fields(ScoringDunbrackDatabaseView)].index(
        "rotameric_mean_tables"
    )
    # Shift a private coefficient tensor, leaving the shared resolver intact.
    # A constant full turn in every spline coefficient shifts every evaluated
    # mean by that turn and leaves its backbone derivatives unchanged.
    shifted.dunbrack_db[index] = (
        original.dunbrack_db[index].double() + turns * 2 * math.pi
    )

    def evaluate(term):
        setup(term, pose)
        module = render(term, pose, block_pairs)
        coords = pose.coords.detach().double().requires_grad_(True)
        energies = module(coords)
        weights = torch.linspace(
            0.5, 1.5, energies.numel(), dtype=coords.dtype, device=torch_device
        ).reshape(energies.shape)
        gradient = torch.autograd.grad((energies * weights).sum(), coords)[0]
        assert torch.isfinite(energies).all() and torch.isfinite(gradient).all()
        return energies.detach(), gradient

    left, right = evaluate(original), evaluate(shifted)
    for actual, expected in zip(right, left):
        torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-8)
