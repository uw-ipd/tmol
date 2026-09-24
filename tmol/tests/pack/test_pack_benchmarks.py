"""What the packer costs, measured where callers actually reach it."""

import pytest
import torch

from tmol.pack import build_missing_sidechains
from tmol.pose import PoseStackBuilder
from tmol.score import beta2016_score_function
from tmol.tests.score.common import pose_stack_from_pdb_and_resnums


@pytest.mark.parametrize("n_poses", [1, 3, 10])
@pytest.mark.parametrize("no_optH", [True, False], ids=["rotamers_only", "with_optH"])
@pytest.mark.benchmark(group=["build_missing_sidechains"])
def test_build_missing_sidechains_benchmark(
    benchmark, ubq_pdb, torch_device, dun_sampler, n_poses, no_optH
):
    """Rebuilding the sidechains a structure does not carry.

    This is the packer as a caller reaches it, and it is the one step of pose
    construction nothing measured: the existing benchmarks cover building a pose
    stack and minimizing one, but not filling it in. The two ids separate the
    rotamer search from the proton optimisation that follows it, which is the
    comparison anyone weighing ``no_optH`` needs.
    """
    if torch_device == torch.device("cpu"):
        return

    pose = pose_stack_from_pdb_and_resnums(ubq_pdb, torch_device)
    pose_stack = PoseStackBuilder.from_poses([pose] * n_poses, device=torch_device)
    start_coords = pose_stack.coords.clone()

    block_has_missing_atoms = torch.zeros(
        (pose_stack.n_poses, pose_stack.max_n_blocks),
        dtype=torch.bool,
        device=torch_device,
    )
    block_has_missing_atoms[:, 40:60] = True
    sfxn = beta2016_score_function(torch_device)

    @benchmark
    def run():
        pose_stack.coords[:] = start_coords
        build_missing_sidechains(
            pose_stack=pose_stack,
            sfxn=sfxn,
            dunbrack_sampler=dun_sampler,
            no_optH=no_optH,
            block_has_missing_atoms=block_has_missing_atoms,
        )

    run
