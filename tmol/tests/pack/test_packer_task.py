import torch

from tmol.pack.packer_task import PackerPalette, PackerTask, SetPackerTask
from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.io import pose_stack_from_pdb


def test_packer_palette_smoke():
    pp = PackerPalette()
    assert pp


def test_packer_palette_design_to_canonical_aas(
    fresh_default_restype_set, fresh_default_packed_block_types, torch_device
):
    pp = PackerPalette()
    pbt = fresh_default_packed_block_types

    arg_index, _ = next(
        (i, rt)
        for i, rt in enumerate(fresh_default_restype_set.residue_types)
        if rt.name == "ARG"
    )
    n_allowed, allowed_bts, allowed_is_orig = pp.block_types_from_original(
        pbt, torch.tensor([[arg_index]], dtype=torch.int64, device=torch_device)
    )
    assert n_allowed[0, 0] == 21
    assert allowed_bts.shape == (1, 1, 21)
    assert allowed_is_orig.shape == (1, 1, 21)


def test_packer_palette_design_to_canonical_aas2_backward_compat(
    fresh_default_restype_set, fresh_default_packed_block_types, torch_device
):
    pp = PackerPalette()
    pbt = fresh_default_packed_block_types
    gly_ind, gly_rt = next(
        (i, rt)
        for i, rt in enumerate(fresh_default_restype_set.residue_types)
        if rt.name == "GLY"
    )
    n_allowed, allowed_bts, allowed_is_orig = pp.block_types_from_original(
        pbt, torch.tensor([[gly_ind]], dtype=torch.int64, device=torch_device)
    )
    assert n_allowed[0, 0] == 21
    assert allowed_bts.shape == (1, 1, 21)
    assert allowed_is_orig.shape == (1, 1, 21)


def test_packer_task_smoke(ubq_pdb, torch_device):
    p1 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=5)
    p2 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=7)
    poses = PoseStackBuilder.from_poses([p1, p2], torch_device)
    palette = PackerPalette()
    task = PackerTask(poses, palette)
    assert task


def test_residue_level_task_his_restrict_to_repacking_backward_compat(
    ubq_pdb, torch_device
):
    palette = PackerPalette()

    pose_stack = pose_stack_from_pdb(ubq_pdb, torch_device)
    pbt = pose_stack.packed_block_types
    i, his_res = next(
        (i, res)
        for (i, res) in enumerate(pbt.active_block_types)
        if res.name in ["HIS", "HIS_D"]
    )
    assert his_res
    # find a residue that's his:
    is_his = pose_stack.block_type_ind == i
    his_pose_ind, his_block_ind = torch.nonzero(is_his, as_tuple=True)
    one_his_pose_ind = his_pose_ind[0].item()
    one_his_block_ind = his_block_ind[0].item()

    # blt = BlockLevelTask(i, his_res, palette)
    # assert len(blt.considered_block_types) == 21
    # assert len(blt.block_type_allowed) == 21
    # assert sum(blt.block_type_allowed) == 21
    task = PackerTask(pose_stack, palette)
    assert (
        torch.sum(
            task.per_block_considered_block_types[
                one_his_pose_ind, one_his_block_ind, :
            ]
            != -1
        )
        == 21
    )
    assert (
        torch.sum(
            task.per_block_is_block_type_allowed[one_his_pose_ind, one_his_block_ind, :]
        )
        == 21
    )
    task.restrict_to_repacking()
    assert (
        torch.sum(
            task.per_block_is_block_type_allowed[one_his_pose_ind, one_his_block_ind, :]
        )
        == 2
    )

    # blt.restrict_to_repacking()
    # assert sum(blt.block_type_allowed) == 2
    n_allowed, allowed_bts, allowed_is_orig = palette.block_types_from_original(
        pbt, torch.tensor([[i]], dtype=torch.int64, device=torch_device)
    )
    restrict_to_repacking_masks = palette.create_restrict_to_repacking_mask(
        pbt, torch.tensor([[i]], dtype=torch.int64)
    )
    assert n_allowed[0, 0] == 21
    assert sum(restrict_to_repacking_masks[0, 0].to(torch.int64)) == 2


def test_packer_task_ctor(ubq_pdb, default_restype_set, torch_device):
    palette = PackerPalette()
    p1 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=5)
    p2 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=7)
    poses = PoseStackBuilder.from_poses([p1, p2], torch_device)

    task = PackerTask(poses, palette)
    assert task.per_block_n_considered_block_types.shape == (2, 7)
    assert task.per_block_n_considered_block_types.device == torch_device
    assert task.per_block_considered_block_types.shape == (2, 7, 21)
    assert task.per_block_considered_block_types.device == torch_device
    assert task.per_block_is_block_type_allowed.shape == (2, 7, 21)
    assert task.per_block_is_block_type_allowed.device == torch_device
    assert task.per_block_orig_block_type.shape == (2, 7)
    assert task.per_block_orig_block_type.device == torch_device
    assert task.restrict_to_repacking_masks.shape == (2, 7, 21)
    assert task.restrict_to_repacking_masks.device == torch_device
    assert len(task.conformer_samplers) == len(palette.default_conformer_samplers())
    for sampler in task.conformer_samplers:
        assert id(sampler) in task.conformer_sampler_index
        ind = task.conformer_sampler_index[id(sampler)]
        assert task.conformer_samplers[ind] is sampler
    assert task.per_block_conformer_sampler_allowed.shape == (
        2,
        7,
        len(task.conformer_samplers),
    )
    assert task.per_block_conformer_sampler_allowed.device == torch_device
    assert task.per_block_chi_expansion.shape == (2, 7, 21, 4)
    assert task.per_block_chi_expansion.device == torch_device


def test_set_packer_task_ctor(ubq_pdb, torch_device):
    palette = PackerPalette()
    p1 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=5)
    p2 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=7)
    poses = PoseStackBuilder.from_poses([p1, p2], torch_device)

    task = PackerTask(poses, palette)
    task.restrict_to_repacking()
    set_task = SetPackerTask.from_packer_task(task)

    assert set_task.per_block_n_considered_block_types.shape == (2, 7)
    assert set_task.per_block_n_considered_block_types.device == torch_device
    assert set_task.per_block_considered_block_types.shape == (2, 7, 21)
    assert set_task.per_block_considered_block_types.device == torch_device
    assert set_task.per_block_is_block_type_allowed.shape == (2, 7, 21)
    assert set_task.per_block_is_block_type_allowed.device == torch_device
    assert set_task.per_block_orig_block_type.shape == (2, 7)
    assert set_task.per_block_orig_block_type.device == torch_device
    assert set_task.restrict_to_repacking_masks.shape == (2, 7, 21)
    assert set_task.restrict_to_repacking_masks.device == torch_device
    assert len(set_task.conformer_samplers) == len(palette.default_conformer_samplers())
    for sampler in set_task.conformer_samplers:
        assert id(sampler) in set_task.conformer_sampler_index
        ind = set_task.conformer_sampler_index[id(sampler)]
        assert set_task.conformer_samplers[ind] is sampler
    assert set_task.per_block_conformer_sampler_allowed.shape == (
        2,
        7,
        len(set_task.conformer_samplers),
    )
    assert set_task.per_block_conformer_sampler_allowed.device == torch_device
    assert set_task.per_block_chi_expansion.shape == (2, 7, 21, 4)
    assert set_task.per_block_chi_expansion.device == torch_device

    assert set_task.cons_bt_pose.shape == (12 * 21,)
    assert set_task.cons_bt_block.shape == (12 * 21,)
    assert set_task.cons_bt_which_block_type.shape == (12 * 21,)
    assert set_task.cons_bt_block_type.shape == (12 * 21,)
    assert set_task.global_block_ind_for_considered_block_types.shape == (12 * 21,)

    n_allowed = 12
    assert set_task.allowed_bt_pose.shape == (n_allowed,)
    assert set_task.allowed_bt_block.shape == (n_allowed,)
    assert set_task.allowed_bt_which_block_type.shape == (n_allowed,)
    assert set_task.allowed_bt_block_type.shape == (n_allowed,)
    assert set_task.is_cons_bt_allowed.shape == (12 * 21,)
    assert set_task.allowed_cons_bt.shape == (n_allowed,)
