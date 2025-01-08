import numpy
import attr
import enum
from typing import Union

from tmol.types.array import NDArray


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

    @classmethod
    def from_edges(cls, edges: NDArray[int][:, :, 4]):
        roots = cls.roots_from_edges(edges)
        return cls.from_roots_and_edges(roots, edges)

    @classmethod
    def from_roots_and_edges(cls, roots: NDArray[int][:], edges: NDArray[int][:, :, 4]):
        return cls(
            max_n_edges=edges.shape[1],
            n_edges=numpy.sum(edges[:, :, 0] != -1, axis=1),
            edges=edges,
            roots=roots,
        )

    @classmethod
    def roots_from_edges(cls, edges: NDArray[int][:, :, 4]):
        # somewhat slow examination of the edges to find the roots
        # TO DO: numba-fy this
        roots = numpy.full((edges.shape[0],), -1, dtype=int)
        max_n_edges = edges.shape[1]
        for i in range(edges.shape[0]):
            verts = set([])
            non_root_verts = []
            for j in range(edges.shape[1]):
                if edges[i, j, 0] == -1:
                    break
                verts.add(edges[i, j, 1])
                verts.add(edges[i, j, 2])
                non_root_verts.append(edges[i, j, 2])
            rootish_verts = list(verts - set(non_root_verts))
            assert len(rootish_verts) == 1
            roots[i] = rootish_verts[0]
        return roots
