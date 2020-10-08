from tmol.system.restypes import ResidueTypeSet
from tmol.pack.packer_task import PackerPalette, ResidueLevelTask, PackerTask
from tmol.system.pose import Pose, Poses


def test_packer_palette_smoke(default_restype_set):
    pp = PackerPalette(default_restype_set)


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


def test_packer_task_smoke(ubq_res, default_restype_set, torch_device):
    p1 = Pose.from_residues_one_chain(ubq_res[:5], torch_device)
    p2 = Pose.from_residues_one_chain(ubq_res[:7], torch_device)
    poses = Poses.from_poses([p1, p2], torch_device)
    palette = PackerPalette(default_restype_set)
    task = PackerTask(poses, palette)


def test_residue_level_task_his_restrict_to_repacking(ubq_res, default_restype_set):
    palette = PackerPalette(default_restype_set)

    i, his_res = next(
        (i, res)
        for (i, res) in enumerate(ubq_res)
        if res.residue_type.name in ["HIS", "HIS_D"]
    )
    assert his_res
    rlt = ResidueLevelTask(i, his_res.residue_type, palette)
    assert len(rlt.allowed_restypes) == 21
    rlt.restrict_to_repacking()
    assert len(rlt.allowed_restypes) == 2


def test_packer_task_ctor(ubq_res, default_restype_set, torch_device):
    palette = PackerPalette(default_restype_set)

    p1 = Pose.from_residues_one_chain(ubq_res[:5], torch_device)
    p2 = Pose.from_residues_one_chain(ubq_res[:7], torch_device)
    poses = Poses.from_poses((p1, p2), torch_device)

    task = PackerTask(poses, palette)
    assert len(task.rlts) == 2
    assert len(task.rlts[0]) == 5
    assert len(task.rlts[1]) == 7
