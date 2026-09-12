"""Rotamer scoring avoids derivative storage when it is not needed."""

import gc

import pytest
import torch

from tmol import pose_stack_from_pdb
from tmol.tests import requires_cuda
from tmol.pack import PackerPalette, PackerTask, SetPackerTask
from tmol.pack.rotamer import IncludeCurrentSampler, build_rotamers
from tmol.score.cartbonded import CartBondedEnergyTerm
from tmol.score.genbonded import GenBondedEnergyTerm
from tmol.score.dunbrack import DunbrackEnergyTerm
from tmol.score.elec import ElecEnergyTerm
from tmol.score.hbond import HBondEnergyTerm
from tmol.score.ljlk import LJLKEnergyTerm

pytestmark = requires_cuda


@pytest.fixture
def torch_device():
    return torch.device("cuda", torch.cuda.current_device())


@pytest.mark.parametrize(
    "term_class",
    [
        CartBondedEnergyTerm,
        GenBondedEnergyTerm,
        DunbrackEnergyTerm,
        ElecEnergyTerm,
        HBondEnergyTerm,
        LJLKEnergyTerm,
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rotamer_forward_avoids_unused_gradient_storage(
    term_class, dtype, ubq_pdb, default_database, torch_device
):
    pose = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=15)
    task = PackerTask(pose, PackerPalette())
    task.restrict_to_repacking()
    task.add_conformer_sampler(IncludeCurrentSampler())
    task = SetPackerTask.from_packer_task(task)
    pose, rotamers = build_rotamers(pose, task, pose.packed_block_types.chem_db)
    term = term_class(param_db=default_database, device=torch_device)
    for block in pose.packed_block_types.active_block_types:
        term.setup_block_type(block)
    term.setup_packed_block_types(pose.packed_block_types)
    term.setup_poses(pose)
    scorer = term.render_rotamer_scoring_module(pose, rotamers)
    tracked = rotamers.coords.to(dtype).detach().requires_grad_(True)
    detached = tracked.detach()

    def allocation(coords):
        torch.cuda.synchronize(torch_device)
        before = torch.cuda.memory_allocated(torch_device)
        torch.cuda.reset_peak_memory_stats(torch_device)
        result = scorer(coords)
        torch.cuda.synchronize(torch_device)
        return result, torch.cuda.max_memory_allocated(torch_device) - before

    scorer(detached)
    scorer(tracked)
    gc.collect()
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        (inferred, inferred_indices), inference_peak = allocation(detached)
        (scored, scored_indices), gradient_peak = allocation(tracked)
    finally:
        if gc_enabled:
            gc.enable()
    torch.testing.assert_close(inferred, scored)
    torch.testing.assert_close(inferred_indices, scored_indices, rtol=0, atol=0)
    if term_class is HBondEnergyTerm:
        assert inference_peak < gradient_peak, (inference_peak, gradient_peak)
    else:
        assert gradient_peak <= inference_peak, (inference_peak, gradient_peak)
    (gradient,) = torch.autograd.grad(scored.sum(), tracked)
    assert torch.isfinite(gradient).all()
