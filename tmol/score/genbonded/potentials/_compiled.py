import torch

from tmol._load_ext import _ensure_compiled_or_jit

if _ensure_compiled_or_jit():
    from tmol.utility._cpp_extension import (
        _load,
        _relpaths,
        _modulename,
        _cuda_if_available,
    )

    _load(
        _modulename(__name__),
        _cuda_if_available(
            _relpaths(
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


def _iter_packing_rotamer_scores(*args):
    *score_args, topology_only = args
    op = (
        _ops.genbonded_rotamer_scores_topology
        if topology_only
        else genbonded_rotamer_scores
    )
    yield op(*score_args)
