from tmol._load_ext import _load_ops

_ops = _load_ops(
    __name__,
    __file__,
    [
        "compiled.ops.cpp",
        "../../common/whole_pose_scoring.cuda.cu",
        "cartbonded_pose_score.cpu.cpp",
        "cartbonded_pose_score.cuda.cu",
    ],
    "tmol_cartbonded",
)

_cartbonded_pose_scores = _ops.cartbonded_pose_scores
_cartbonded_rotamer_scores = _ops.cartbonded_rotamer_scores


def _iter_packing_rotamer_scores(*args):
    *score_args, topology_only = args
    op = (
        _ops.cartbonded_rotamer_scores_topology
        if topology_only
        else _cartbonded_rotamer_scores
    )
    yield op(*score_args)
