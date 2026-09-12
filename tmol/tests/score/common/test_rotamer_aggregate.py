"""Aggregate scores must not share output slots across CPU workers."""

import torch

from tmol import pose_stack_from_pdb
from tmol.pack import PackerPalette, PackerTask, SetPackerTask
from tmol.pack.rotamer import FixedAAChiSampler, IncludeCurrentSampler, build_rotamers
from tmol.pack.rotamer.dunbrack import create_dunbrack_sampler_from_database
from tmol.pose import PoseStackBuilder
from tmol.score.cartbonded import CartBondedEnergyTerm
from tmol.score.cartbonded.potentials import cartbonded_rotamer_scores


def test_cpu_aggregate_rotamer_scores_match_serial(ubq_pdb, default_database):
    device = torch.device("cpu")
    poses = [pose_stack_from_pdb(ubq_pdb, device, residue_end=n) for n in (12, 17, 25)]
    pose = PoseStackBuilder.from_poses(poses, device)
    task = PackerTask(pose, PackerPalette())
    task.restrict_to_repacking()
    task.add_conformer_sampler(IncludeCurrentSampler())
    task.add_conformer_sampler(FixedAAChiSampler())
    task.add_conformer_sampler(
        create_dunbrack_sampler_from_database(default_database, device)
    )
    task = SetPackerTask.from_packer_task(task)
    pose, rotamers = build_rotamers(pose, task, pose.packed_block_types.chem_db)
    term = CartBondedEnergyTerm(param_db=default_database, device=device)
    for block in pose.packed_block_types.active_block_types:
        term.setup_block_type(block)
    term.setup_packed_block_types(pose.packed_block_types)
    term.setup_poses(pose)
    scorer = term.render_rotamer_scoring_module(pose, rotamers)
    coords = rotamers.coords.detach().double()
    tail = scorer._static_tail_for_coords(coords)
    assert tail[-1] is True
    tail = (*tail[:-1], False)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        expected, indices = cartbonded_rotamer_scores(coords, *tail)
        assert indices.shape[1] >= 256  # Exercise parallel-dispatch eligibility.
        torch.set_num_threads(4)
        for _ in range(4):
            actual, _ = cartbonded_rotamer_scores(coords, *tail)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous_threads)
