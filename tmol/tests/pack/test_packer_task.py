import torch

from tmol.pack.packer_task import PackerPalette, PackerTask
from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.io import pose_stack_from_pdb


def test_packer_palette_smoke(default_restype_set):
    pp = PackerPalette(default_restype_set)
    assert pp


def test_packer_palette_design_to_canonical_aas(
    fresh_default_restype_set, fresh_default_packed_block_types
):
    pp = PackerPalette()
    pbt = fresh_default_packed_block_types

    arg_index, _ = next(
        (i, rt)
        for i, rt in enumerate(fresh_default_restype_set.residue_types)
        if rt.name == "ARG"
    )
    n_allowed, allowed_bts = pp.block_types_from_original(
        pbt, torch.tensor([[arg_index]], dtype=torch.int64)
    )
    assert n_allowed[0, 0] == 21


def test_packer_palette_design_to_canonical_aas_backward_compat(
    fresh_default_restype_set, fresh_default_packed_block_types
):
    pp = PackerPalette()
    pbt = fresh_default_packed_block_types

    arg_index, arg_rt = next(
        (i, rt)
        for i, rt in enumerate(fresh_default_restype_set.residue_types)
        if rt.name == "ARG"
    )
    # allowed = pp.block_types_from_original_old(arg_rt)
    n_allowed, allowed_bts = pp.block_types_from_original(
        pbt, torch.tensor([[arg_index]], dtype=torch.int64)
    )
    # assert len(allowed) == 21
    assert n_allowed[0, 0] == 21


def test_packer_palette_design_to_canonical_aas2_backward_compat(
    fresh_default_restype_set, fresh_default_packed_block_types
):
    pp = PackerPalette()
    pbt = fresh_default_packed_block_types
    gly_ind, gly_rt = next(
        (i, rt)
        for i, rt in enumerate(fresh_default_restype_set.residue_types)
        if rt.name == "GLY"
    )
    n_allowed, allowed_bts = pp.block_types_from_original(
        pbt, torch.tensor([[gly_ind]], dtype=torch.int64)
    )
    assert n_allowed[0, 0] == 21


def test_packer_task_smoke(ubq_pdb, default_restype_set, torch_device):
    p1 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=5)
    p2 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=7)
    poses = PoseStackBuilder.from_poses([p1, p2], torch_device)
    palette = PackerPalette()
    task = PackerTask(poses, palette)
    assert task


def test_residue_level_task_his_restrict_to_repacking_backward_compat(
    ubq_pdb, default_restype_set, torch_device
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
    n_allowed, allowed_bts = palette.block_types_from_original(
        pbt, torch.tensor([[i]], dtype=torch.int64)
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
    assert len(task.blts) == 2
    assert len(task.blts[0]) == 5
    assert len(task.blts[1]) == 7
