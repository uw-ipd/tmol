from tmol.pack.packer_task import PackerPalette, ResidueLevelTask, PackerTask
from tmol.system.pose import Pose, Poses


def test_packer_palette_smoke(default_database):
    pp = PackerPalette(default_database.chemical)


def test_packer_palette_design_to_canonical_aas(default_database):
    pp = PackerPalette(default_database.chemical)
    arg_rt = next(rt for rt in default_database.chemical.residues if rt.name == "ARG")
    allowed = pp.restypes_from_original(arg_rt)
    assert len(allowed) == 21


def test_packer_palette_design_to_canonical_aas2(default_database):
    pp = PackerPalette(default_database.chemical)
    gly_rt = next(rt for rt in default_database.chemical.residues if rt.name == "GLY")
    allowed = pp.restypes_from_original(gly_rt)
    assert len(allowed) == 21


def test_packer_task_smoke(ubq_res, default_database, torch_device):
    p1 = Pose.from_residues_one_chain(
        ubq_res[:5], default_database.chemical, torch_device
    )
    p2 = Pose.from_residues_one_chain(
        ubq_res[:7], default_database.chemical, torch_device
    )
    poses = Poses.from_poses([p1, p2], default_database.chemical, torch_device)
    palette = PackerPalette(default_database.chemical)
    task = PackerTask(poses, palette)


# def test_residue_level_task_his_restrict_to_repacking(ubq_system, default_database):
#     palette = PackerPalette(default_database.chemical)
#
#     i, his_res = next(
#         (i, res)
#         for (i, res) in enumerate(ubq_system.residues)
#         if res.residue_type.name in ["HIS", "HIS_D"]
#     )
#     assert his_res
#     rlt = ResidueLevelTask(i, his_res.residue_type, palette)
#     assert len(rlt.allowed_restypes) == 21
#     rlt.restrict_to_repacking()
#     assert len(rlt.allowed_restypes) == 2
#
#
# def test_packer_task_ctor(ubq_system, default_database):
#     palette = PackerPalette(default_database.chemical)
#     task = PackerTask(ubq_system, palette)
#
#     assert len(task.rlts) == len(ubq_system.residues)
