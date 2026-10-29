"""Compiled state transitions for the segmented Armijo line search.

Scoring remains a separate call because it is the expensive, differentiable
part of the search and may be captured in a CUDA graph.  These operations fuse
the small comparisons, masks, and selections around each score evaluation.
They are deliberately functional: returning fresh tensors keeps the optimizer
compatible with both eager execution and a graph-captured scorer.
"""

from tmol._load_ext import _load_ops

_ops = _load_ops(
    __name__,
    __file__,
    [
        "armijo_compiled.ops.cpp",
        "armijo_compiled.cpu.cpp",
        "armijo_compiled.cuda.cu",
    ],
    "tmol_optimization",
)

_armijo_start = _ops.armijo_start
_armijo_classify = _ops.armijo_classify
_armijo_trial = _ops.armijo_trial
_armijo_update = _ops.armijo_update
_armijo_finalize = _ops.armijo_finalize
