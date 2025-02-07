import numpy
import attr
import enum

from tmol.types.array import NDArray


class EdgeType(enum.IntEnum):
    polymer = 0
    jump = enum.auto()
    root_jump = enum.auto()


@attr.s(auto_attribs=True, frozen=True)
class FoldForest:
    """The fold forest will define the fold trees for the poses in a PoseStack.
    Each tensor in the class has its first dimension over the number of poses.

    The primary definition of a FoldTree is the Edge. The Edge defines a connection
    between two parts of a Pose. The three types of edges are 1. polymer edges
    (analgous to the previously named "peptide edges" from Rosetta++ and Rosetta3), 2.
    jump edges which connect any pair of residues in the Pose, and 3. root-jump
    edges, which originate at the explicit virtual root and connect to a particular
    residue. A polymer edge spans a contiguous range of polymeric block types where
    the "up" connection of residue i is connected to the "down" connection of residue
    i+1 for all i in the range between the "start" and "end" blocks.

    Each edge is described by a 4-tuple of integers (type, start, end, jump-index);
    where type is one of the EdgeType enum values, start is the index of the upstream
    residue of the edge, end is the index of the downstream residue of the edge, and
    jump-index is used to assign an id to any particular jump edge; jump-edge indices
    must be unique and ascending from 0 to n_jumps-1. "Root jump" edges take their
    "identity" from the downstream residue of the edge, so they do not need an index.

    The FoldForest in tmol differs from the FoldTree in Rosetta3 in that there
    is always a virtual root at the origin and any residue (block) may be
    connected to this root by a "root jump". Such root-jump residues are defined
    by listing the residue that the root is connected to as the "end" residue;
    the "start" residue field should be left as -1. An example FoldForest for a
    ten-residue protein might be:
      (polymer, 0, 4)
      (jump   , 0, 7)
      (polymer, 7, 9)
      (polymer, 7, 6)
      (root-jump, -1, 0)
      (root-jump, -1, 5)
    where both residues are 0 and 5 are connected to the root.

    Note that in the MoveMap, the root-jumps are distinct from the non-root-jumps.
    """

    max_n_edges: int
    n_edges: NDArray[int][:]
    edges: NDArray[int][:, :, 4]

    @classmethod
    # soon! @validate_args
    def polymeric_forest(cls, n_res_per_tree: NDArray[numpy.int32][:]):
        """Create an N->C fold tree for a collection of monomers in a PoseStack."""
        # n_trees = len(residues)
        # n_res_per_tree = [len(reslist) for reslist in residues]
        n_trees = n_res_per_tree.shape[0]

        edges = numpy.full((n_trees, 2, 4), -1, dtype=int)
        edges[:, 0, 0] = EdgeType.root_jump
        edges[:, 0, 1] = -1
        edges[:, 0, 2] = 0
        n_res_gt_1 = n_res_per_tree > 1
        edges[n_res_gt_1, 1, 0] = EdgeType.polymer
        edges[n_res_gt_1, 1, 1] = 0
        edges[n_res_gt_1, 1, 2] = (n_res_per_tree[n_res_gt_1] - 1).astype(int)
        n_edges = numpy.full(n_trees, 1, dtype=int)
        n_edges[n_res_gt_1] = 2
        return cls(
            max_n_edges=2,
            n_edges=n_edges,
            edges=edges,
        )

    @classmethod
    def from_edges(cls, edges: NDArray[int][:, :, 4]):
        return cls(
            max_n_edges=edges.shape[1],
            n_edges=numpy.sum(edges[:, :, 0] != -1, axis=1),
            edges=edges,
        )
