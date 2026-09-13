from types import SimpleNamespace

import pytest
import torch

from tmol.pack.rotamer import IncludeCurrentSampler, FallbackSampler
from tmol.pack.rotamer._include_current_sampler import (
    create_full_dof_inds_to_copy_from_orig_to_rotamers_for_include_current_sampler as plan,
)


def copy_fixture(device):
    def tensor(values):
        return torch.tensor(values, dtype=torch.int64, device=device)

    poses = SimpleNamespace(
        device=device,
        block_type_ind=tensor([[0, 1, -1, -1], [2, 0, 1, -1]]).int(),
        packed_block_types=SimpleNamespace(
            n_atoms=tensor([2, 5, 9, 1024]).int(), max_n_atoms=1024
        ),
    )
    task = SimpleNamespace(
        global_block_ind_for_considered_block_types=tensor([5, 0, 6, 1, 4])
    )
    gbt = tensor([0, 1, 2, 3, 4, 0])
    selected = tensor([4, 0, 3])
    inputs = (
        poses,
        task,
        gbt,
        tensor([0, 0, 1, 1, 2, 0]),
        selected,
        tensor([1, 0, 0, 1, 1]).int(),
        gbt[selected].int(),
        tensor([3, 12, 23, 40, 53, 75]),
    )
    expected = (
        tensor([*range(53, 62), 3, 4, *range(40, 45)]),
        tensor([*range(7, 16), 16, 17, *range(2, 7)]),
    )
    return inputs, expected


def test_copy_plan_handles_padding_permutations_and_unused_large_types(torch_device):
    inputs, expected = copy_fixture(torch_device)
    snapshots = [x.clone() for x in inputs[2:]]
    actual = plan(*inputs)
    for result, wanted in zip(actual, expected):
        torch.testing.assert_close(result, wanted, rtol=0, atol=0)
    for original, snapshot in zip(inputs[2:], snapshots):
        torch.testing.assert_close(original, snapshot, rtol=0, atol=0)


def test_empty_copy_plan_needs_no_pose_annotations(torch_device):
    empty = torch.empty(0, dtype=torch.int64, device=torch_device)
    dst, src = plan(None, None, empty, empty, empty, empty, empty, empty)
    assert dst.numel() == src.numel() == 0
    assert dst.device == src.device == torch_device
    assert dst.dtype == src.dtype == torch.int64


@pytest.mark.parametrize("sampler_class", [IncludeCurrentSampler, FallbackSampler])
@pytest.mark.parametrize("custom_stream", [False, True])
def test_copy_preserves_dofs_without_global_cuda_barriers(
    torch_device, monkeypatch, custom_stream, sampler_class
):
    if custom_stream and torch_device.type != "cuda":
        pytest.skip("A custom CUDA stream requires CUDA")
    inputs, (dst, src) = copy_fixture(torch_device)
    poses, task, gbt, types, selected, counts, sampler_gbt, offsets = inputs
    source = torch.arange(24 * 9, dtype=torch.float32, device=torch_device).reshape(
        24, 9
    )
    target = torch.full((90, 9), -999.0, device=torch_device)
    expected = target.clone()
    expected[dst + 1] = source[src + 1]

    def fill():
        with monkeypatch.context() as patch:
            # Even a CPU task on a GPU host must not synchronize unrelated work.
            patch.setattr(torch.cuda, "is_available", lambda: True)
            patch.setattr(
                torch.cuda,
                "synchronize",
                lambda *args, **kwargs: pytest.fail("global CUDA barrier during copy"),
            )
            sampler_class().fill_dofs_for_samples(
                poses,
                task,
                None,
                source,
                gbt,
                types,
                offsets,
                None,
                selected,
                counts,
                sampler_gbt,
                {},
                target,
            )

    if custom_stream:
        stream = torch.cuda.Stream(device=torch_device)
        stream.wait_stream(torch.cuda.current_stream(torch_device))
        with torch.cuda.stream(stream):
            fill()
        torch.cuda.current_stream(torch_device).wait_stream(stream)
    else:
        fill()
    torch.testing.assert_close(target, expected, rtol=0, atol=0)


@pytest.mark.parametrize("sampler_class", [IncludeCurrentSampler, FallbackSampler])
def test_current_conformers_reconstruct_ragged_pose_coordinates(
    ubq_pdb, torch_device, monkeypatch, sampler_class
):
    from tmol.io import pose_stack_from_pdb
    from tmol.pack import PackerPalette, PackerTask, SetPackerTask
    from tmol.pack.rotamer import build_rotamers
    from tmol.pose import PoseStackBuilder

    poses = PoseStackBuilder.from_poses(
        [
            pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=3),
            pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=5),
        ],
        torch_device,
    )
    monkeypatch.setattr(PackerPalette, "default_conformer_samplers", lambda self: [])
    task = PackerTask(poses, PackerPalette())
    task.restrict_to_repacking()
    task.add_conformer_sampler(sampler_class())
    _, rotamers = build_rotamers(
        poses, SetPackerTask.from_packer_task(task), poses.packed_block_types.chem_db
    )
    torch.testing.assert_close(
        rotamers.n_rots_for_block, task.is_real_block.to(torch.int64), rtol=0, atol=0
    )
    for rot in range(len(rotamers.block_ind_for_rot)):
        pose = int(rotamers.pose_for_rot[rot])
        block = int(rotamers.block_ind_for_rot[rot])
        kind = int(poses.block_type_ind[pose, block])
        n_atoms = int(poses.packed_block_types.n_atoms[kind])
        original_offset = int(poses.block_coord_offset[pose, block])
        rotamer_offset = int(rotamers.coord_offset_for_rot[rot])
        torch.testing.assert_close(
            rotamers.coords[rotamer_offset : rotamer_offset + n_atoms],
            poses.coords[pose, original_offset : original_offset + n_atoms],
            rtol=0,
            atol=2e-5,
        )
