"""What the packer costs, measured where callers actually reach it."""

import pytest
import torch

from tmol.pack import build_missing_sidechains
from tmol.pose import PoseStackBuilder
from tmol.score import beta2016_score_function
from tmol.tests.score.common import pose_stack_from_pdb_and_resnums


@pytest.mark.parametrize("n_poses", [1, 3, 10])
@pytest.mark.parametrize("no_optH", [True, False], ids=["rotamers_only", "with_optH"])
@pytest.mark.benchmark(group="build_missing_sidechains")
def test_build_missing_sidechains_benchmark(
    benchmark, ubq_pdb, torch_device, dun_sampler, n_poses, no_optH
):
    """Rebuild sidechains on CPU/CUDA, with optional proton optimization.

    Coordinate/RNG reset is excluded; all device work completes before the
    timed call returns. One iteration per round keeps each input identical.
    """
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

    def setup():
        torch.manual_seed(0)
        pose_stack.coords[:] = start_coords
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)

    def run():
        result = build_missing_sidechains(
            pose_stack=pose_stack,
            sfxn=sfxn,
            dunbrack_sampler=dun_sampler,
            no_optH=no_optH,
            block_has_missing_atoms=block_has_missing_atoms,
        )
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        return result

    benchmark.pedantic(run, setup=setup, rounds=5, iterations=1, warmup_rounds=1)
