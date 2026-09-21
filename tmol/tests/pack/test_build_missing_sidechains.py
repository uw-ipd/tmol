import torch
import tmol.pack._build_missing_sidechains as build_missing_sidechains_module

from tmol.pack import build_missing_sidechains
from tmol.score import beta2016_score_function
from tmol.pose import PoseStackBuilder
from tmol.tests.score.common import pose_stack_from_pdb_and_resnums


def test_build_missing_sidechains_jagged_pose_stack(ubq_pdb, torch_device, dun_sampler):
    res_50 = [(0, 50)]
    res_30 = [(20, 50)]
    p1 = pose_stack_from_pdb_and_resnums(ubq_pdb, torch_device)
    p2 = pose_stack_from_pdb_and_resnums(ubq_pdb, torch_device, res_50)
    p3 = pose_stack_from_pdb_and_resnums(ubq_pdb, torch_device, res_30)
    pn = PoseStackBuilder.from_poses([p1, p2, p3], device=torch_device)

    block_has_missing_atoms = torch.zeros(
        (pn.n_poses, pn.max_n_blocks), dtype=torch.bool, device=torch_device
    )
    block_has_missing_atoms[0, 40:60] = True
    block_has_missing_atoms[2, 10:20] = True
    sfxn = beta2016_score_function(torch_device)
    build_missing_sidechains(
        pose_stack=pn,
        sfxn=sfxn,
        dunbrack_sampler=dun_sampler,
        no_optH=False,
        block_has_missing_atoms=block_has_missing_atoms,
    )


def test_build_missing_sidechains_no_optH(ubq_pdb, torch_device, dun_sampler):
    res_50 = [(0, 50)]
    res_30 = [(20, 50)]
    p1 = pose_stack_from_pdb_and_resnums(ubq_pdb, torch_device)
    p2 = pose_stack_from_pdb_and_resnums(ubq_pdb, torch_device, res_50)
    p3 = pose_stack_from_pdb_and_resnums(ubq_pdb, torch_device, res_30)
    pn = PoseStackBuilder.from_poses([p1, p2, p3], device=torch_device)

    block_has_missing_atoms = torch.zeros(
        (pn.n_poses, pn.max_n_blocks), dtype=torch.bool, device=torch_device
    )
    block_has_missing_atoms[0, 40:60] = True
    block_has_missing_atoms[2, 10:20] = True
    sfxn = beta2016_score_function(torch_device)
    # A fully frozen middle pose must remain safe across repeated anneals.
    for _ in range(2):
        build_missing_sidechains(
            pose_stack=pn,
            sfxn=sfxn,
            dunbrack_sampler=dun_sampler,
            no_optH=True,
            block_has_missing_atoms=block_has_missing_atoms,
        )


def test_build_missing_sidechains_skips_na_sampler_for_complete_pose(
    ubq_pdb, torch_device, dun_sampler, monkeypatch
):
    pose_stack = pose_stack_from_pdb_and_resnums(ubq_pdb, torch_device)
    block_has_missing_atoms = torch.zeros(
        (pose_stack.n_poses, pose_stack.max_n_blocks),
        dtype=torch.bool,
        device=torch_device,
    )

    class UnexpectedNASampler:
        def defines_rotamers_for_bts(self, *_args, **_kwargs):
            raise AssertionError(
                "NA sampler should not run for a complete protein pose"
            )

    monkeypatch.setattr(
        build_missing_sidechains_module,
        "pack_rotamers",
        lambda pose_stack, *_args, **_kwargs: pose_stack,
    )
    result = build_missing_sidechains(
        pose_stack=pose_stack,
        sfxn=beta2016_score_function(torch_device),
        dunbrack_sampler=dun_sampler,
        no_optH=True,
        block_has_missing_atoms=block_has_missing_atoms,
        na_sampler=UnexpectedNASampler(),
    )

    assert result is pose_stack


def test_build_missing_sidechains_freezes_complete_non_opth_blocks(
    ubq_pdb, torch_device, dun_sampler, monkeypatch
):
    """Do not send chemically immutable blocks through one-rotamer packing."""
    from tmol.pack.rotamer import FixedAAChiSampler, OptHSampler

    pose_stack = pose_stack_from_pdb_and_resnums(ubq_pdb, torch_device)
    missing = torch.zeros(
        (pose_stack.n_poses, pose_stack.max_n_blocks),
        dtype=torch.bool,
        device=torch_device,
    )
    captured = {}

    def capture_task(pose_stack, _sfxn, task, **_kwargs):
        captured["task"] = task
        return pose_stack

    monkeypatch.setattr(build_missing_sidechains_module, "pack_rotamers", capture_task)
    build_missing_sidechains(
        pose_stack=pose_stack,
        sfxn=beta2016_score_function(torch_device),
        dunbrack_sampler=dun_sampler,
        no_optH=False,
        block_has_missing_atoms=missing,
    )

    real = pose_stack.block_type_ind64 >= 0
    supported = real & OptHSampler().defines_rotamers_for_bts(
        pose_stack.packed_block_types,
        pose_stack.block_type_ind64.clamp_min(0),
    )
    task = captured["task"]
    assert dun_sampler not in task.conformer_samplers
    assert not any(
        isinstance(sampler, FixedAAChiSampler) for sampler in task.conformer_samplers
    )
    packable = task.per_block_is_block_type_allowed.any(dim=2) & real
    torch.testing.assert_close(packable, supported)
