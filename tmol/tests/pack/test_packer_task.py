from tmol.pack.packer_task import PackerPalette, ResidueLevelTask, PackerTask
from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.io import pose_stack_from_pdb


def test_packer_palette_smoke(default_restype_set):
    pp = PackerPalette(default_restype_set)
    assert pp


def test_packer_palette_design_to_canonical_aas(default_restype_set):
    pp = PackerPalette(default_restype_set)
    arg_rt = next(rt for rt in default_restype_set.residue_types if rt.name == "ARG")
    allowed = pp.restypes_from_original(arg_rt)
    assert len(allowed) == 21


def test_packer_palette_design_to_canonical_aas2(default_restype_set):
    pp = PackerPalette(default_restype_set)
    gly_rt = next(rt for rt in default_restype_set.residue_types if rt.name == "GLY")
    allowed = pp.restypes_from_original(gly_rt)
    assert len(allowed) == 21


def test_packer_task_smoke(ubq_pdb, default_restype_set, torch_device):

    # p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
    #     default_restype_set.chem_db, ubq_res[:5], torch_device
    # )
    # p2 = PoseStackBuilder.one_structure_from_polymeric_residues(
    #     default_restype_set.chem_db, ubq_res[:7], torch_device
    # )
    p1 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=5)
    p2 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=7)
    poses = PoseStackBuilder.from_poses([p1, p2], torch_device)
    palette = PackerPalette(default_restype_set)
    task = PackerTask(poses, palette)
    assert task


def test_residue_level_task_his_restrict_to_repacking(
    ubq_pdb, default_restype_set, torch_device
):
    palette = PackerPalette(default_restype_set)

    pose_stack = pose_stack_from_pdb(ubq_pdb, torch_device)
    pbt = pose_stack.packed_block_types
    i, his_res = next(
        (i, res)
        for (i, res) in enumerate(pbt.active_block_types)
        if res.residue_type.name in ["HIS", "HIS_D"]
    )
    assert his_res
    rlt = ResidueLevelTask(i, his_res, palette)
    assert len(rlt.allowed_restypes) == 21
    rlt.restrict_to_repacking()
    assert len(rlt.allowed_restypes) == 2


def test_packer_task_ctor(ubq_pdb, default_restype_set, torch_device):
    palette = PackerPalette(default_restype_set)

    # p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
    #     default_restype_set.chem_db, ubq_res[:5], torch_device
    # )
    # p2 = PoseStackBuilder.one_structure_from_polymeric_residues(
    #     default_restype_set.chem_db, ubq_res[:7], torch_device
    # )
    p1 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=5)
    p2 = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=7)
    poses = PoseStackBuilder.from_poses([p1, p2], torch_device)

    task = PackerTask(poses, palette)
    assert len(task.rlts) == 2
    assert len(task.rlts[0]) == 5
    assert len(task.rlts[1]) == 7
