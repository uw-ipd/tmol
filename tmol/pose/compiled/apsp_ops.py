import torch

from tmol._cpp_lib import _ensure_loaded

_ensure_loaded()

_ops = torch.ops.tmol_apsp


def stacked_apsp(weights, threshold=-1):
    """Compute all pairs shortest paths given a set of (integer) weights / distances
    separating all pairs of nodes in a "stack" of graphs. A sentinel value of -1
    indicates that the distance between the nodes is unknown / there is no direct
    edge between them. If the "threshold" argument is provided, then the algorithm
    can stop after the distance between them has been determined to exceed the
    threshold. A sentinel value of "-1" for the threshold can be used to indicate
    that the exact distance should be computed for all pairs of nodes.

    There are separate implementations for this algorithm on the CPU and the GPU
    and so as a secondary step, the distances between all pairs of nodes are
    thresholded for the GPU version as its implementation does not respect the
    threshold parameter.
    """
    _ops.apsp_op(weights, threshold)
    if threshold > 0 and weights.device != torch.device("cpu"):
        weights[weights > threshold] = threshold
