from tmol._load_ext import load_ops

_ops = load_ops(
    __name__,
    __file__,
    [
        "compiled.ops.cpp",
        "backbone_torsion_pose_score.cpu.cpp",
        "backbone_torsion_pose_score.cuda.cu",
    ],
    "tmol_bb_torsion",
)

backbone_torsion_pose_score = _ops.backbone_torsion_pose_score
backbone_torsion_rotamer_score = _ops.backbone_torsion_rotamer_score
