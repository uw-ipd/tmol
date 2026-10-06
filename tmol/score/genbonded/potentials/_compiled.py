import torch

from tmol._load_ext import ensure_compiled_or_jit

if ensure_compiled_or_jit():
    from tmol.utility import load, relpaths, modulename, cuda_if_available

    load(
        modulename(__name__),
        cuda_if_available(
            relpaths(
                __file__,
                [
                    "compiled.ops.cpp",
                    "../../common/whole_pose_scoring.cuda.cu",
                    "genbonded_pose_score.cpu.cpp",
                    "genbonded_pose_score.cuda.cu",
                ],
            )
        ),
        is_python_module=False,
    )

    _ops = torch.ops.tmol_genbonded
else:
    _ops = torch.ops.tmol_genbonded

genbonded_pose_scores = _ops.genbonded_pose_scores
genbonded_rotamer_scores = _ops.genbonded_rotamer_scores


def iter_packing_rotamer_scores(*args):
    *score_args, topology_only = args
    op = (
        _ops.genbonded_rotamer_scores_topology
        if topology_only
        else genbonded_rotamer_scores
    )
    yield op(*score_args)
