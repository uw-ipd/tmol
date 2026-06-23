import torch

from tmol.pack.packer_task import PackerPalette, BlockLevelTask, PackerTask
from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.io import pose_stack_from_pdb


def test_packer_palette_smoke(default_restype_set):
    pp = PackerPalette(default_restype_set)
    assert pp


def test_packer_palette_design_to_canonical_aas(
    fresh_default_restype_set, fresh_default_packed_block_types
):
    pp = PackerPalette(fresh_default_restype_set)
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
    pp = PackerPalette(fresh_default_restype_set)
    pbt = fresh_default_packed_block_types

    arg_index, arg_rt = next(
        (i, rt)
        for i, rt in enumerate(fresh_default_restype_set.residue_types)
        if rt.name == "ARG"
    )
    allowed = pp.block_types_from_original_old(arg_rt)
    n_allowed, allowed_bts = pp.block_types_from_original(
        pbt, torch.tensor([[arg_index]], dtype=torch.int64)
    )
    assert len(allowed) == 21
    assert n_allowed[0, 0] == 21
    allowed_set = set([id(bt) for bt in allowed])

    for i in range(n_allowed[0, 0]):
        bt_ind = allowed_bts[0, 0, i].item()
        bt = pbt.active_block_types[bt_ind]
        assert id(bt) in allowed_set


def test_packer_palette_design_to_canonical_aas2_backward_compat(
    fresh_default_restype_set, fresh_default_packed_block_types
):
    pp = PackerPalette(fresh_default_restype_set)
    pbt = fresh_default_packed_block_types
    gly_ind, gly_rt = next(
        (i, rt)
        for i, rt in enumerate(fresh_default_restype_set.residue_types)
        if rt.name == "GLY"
    )
    allowed = pp.block_types_from_original_old(gly_rt)
    n_allowed, allowed_bts = pp.block_types_from_original(
        pbt, torch.tensor([[gly_ind]], dtype=torch.int64)
    )
    assert len(allowed) == 21
    assert n_allowed[0, 0] == 21
    allowed_set = set([id(bt) for bt in allowed])
    for i in range(n_allowed[0, 0]):
        bt_ind = allowed_bts[0, 0, i].item()
        bt = pbt.active_block_types[bt_ind]
        assert id(bt) in allowed_set


def test_packer_task_smoke(ubq_pdb, default_restype_set, torch_device):
    p1 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=5)
    p2 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=7)
    poses = PoseStackBuilder.from_poses([p1, p2], torch_device)
    palette = PackerPalette(default_restype_set)
    task = PackerTask(poses, palette)
    assert task


def test_residue_level_task_his_restrict_to_repacking_backward_compat(
    ubq_pdb, default_restype_set, torch_device
):
    palette = PackerPalette(default_restype_set)

    pose_stack = pose_stack_from_pdb(ubq_pdb, torch_device)
    pbt = pose_stack.packed_block_types
    i, his_res = next(
        (i, res)
        for (i, res) in enumerate(pbt.active_block_types)
        if res.name in ["HIS", "HIS_D"]
    )
    assert his_res
    blt = BlockLevelTask(i, his_res, palette)
    assert len(blt.considered_block_types) == 21
    assert len(blt.block_type_allowed) == 21
    assert sum(blt.block_type_allowed) == 21
    blt.restrict_to_repacking()
    assert sum(blt.block_type_allowed) == 2
    n_allowed, allowed_bts = palette.block_types_from_original(
        pbt, torch.tensor([[i]], dtype=torch.int64)
    )
    restrict_to_repacking_masks = palette.create_restrict_to_repacking_mask(
        pbt, torch.tensor([[i]], dtype=torch.int64)
    )
    assert n_allowed[0, 0] == 21
    assert sum(restrict_to_repacking_masks[0, 0].to(torch.int64)) == 2


def test_packer_task_ctor(ubq_pdb, default_restype_set, torch_device):
    palette = PackerPalette(default_restype_set)
    p1 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=5)
    p2 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=7)
    poses = PoseStackBuilder.from_poses([p1, p2], torch_device)

    task = PackerTask(poses, palette)
    assert len(task.blts) == 2
    assert len(task.blts[0]) == 5
    assert len(task.blts[1]) == 7
