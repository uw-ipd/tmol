from tmol._load_ext import _load_ops

_ops = _load_ops(
    __name__,
    __file__,
    [
        "compiled.ops.cpp",
        "../../common/whole_pose_scoring.cuda.cu",
        "backbone_torsion_pose_score.cpu.cpp",
        "backbone_torsion_pose_score.cuda.cu",
    ],
    "tmol_bb_torsion",
)

backbone_torsion_pose_score = _ops.backbone_torsion_pose_score
backbone_torsion_rotamer_score = _ops.backbone_torsion_rotamer_score


def _iter_packing_rotamer_scores(*args):
    *score_args, topology_only = args
    op = (
        _ops.backbone_torsion_rotamer_score_topology
        if topology_only
        else backbone_torsion_rotamer_score
    )
    yield op(*score_args)
