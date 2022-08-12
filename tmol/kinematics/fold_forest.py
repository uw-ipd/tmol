import numpy
import attr
import enum

from typing import Sequence
from tmol.types.array import NDArray
from tmol.chemical.restypes import Residue


class EdgeType(enum.IntEnum):
    polymer = 0
    jump = enum.auto()
    chemical = enum.auto()


@attr.s(auto_attribs=True, frozen=True)
class FoldForest:
    """The fold forest will define the fold trees for the poses in a PoseStack.
    Each tensor in the class has its first dimension over the number of poses.

    The primary definition of a FoldTree is the Edge. The Edge defines a connection
    between two parts of a Pose. The two most commonly used edges are PolymerEdges
    (analgous to the previously named PeptideEdges from Rosetta++ and Rosetta3) and
    JumpEdges. A polymer edge spans a contiguous range of polymeric block types where
    the "up" connection of residue i is connected to the "down" connection of residue
    i+1 for all i in the range between the "start" and "end" blocks.

    Not sure yet what I want this class to do or how I want to construct it;
    should it be unmodifyable or should it let you add edges?
    """

    max_n_edges: int
    n_edges: NDArray[int][:]
    edges: NDArray[int][:, :, 4]
    roots: NDArray[int][:]

    @classmethod
    # soon! @validate_args
    def polymeric_forest(cls, n_res_per_tree: NDArray[numpy.int32][:]):
        # n_trees = len(residues)
        # n_res_per_tree = [len(reslist) for reslist in residues]
        n_trees = n_res_per_tree.shape[0]

        edges = numpy.full((n_trees, 1, 4), -1, dtype=int)
        edges[:, 0, 0] = EdgeType.polymer
        edges[:, 0, 1] = 0
        edges[:, 0, 2] = numpy.array(n_res_per_tree, dtype=int) - 1
        roots = numpy.full((n_trees,), 0, dtype=int)
        return cls(
            max_n_edges=1,
            n_edges=numpy.ones(n_trees, dtype=int),
            edges=edges,
            roots=roots,
        )
