import attr

import numpy
# from numba.cuda import to_device as to_cuda_device

from tmol.types.attrs import ValidateAttrs
from tmol.types.functional import validate_args
from tmol.types.array import NDArray

from .scan_paths_jit import (
    mark_path_children_and_count_nonpath_children,
    list_nonpath_children,
    find_derivsum_path_depths,
    compute_branching_factor,
    find_refold_path_depths,
)


@attr.s(auto_attribs=True, frozen=True, slots=True)
class PathPartitioning(ValidateAttrs):
    """Partitioning of a tree into linear subpaths.

    Source tree contains a single root node, at index 0, and any number nodes
    at index (i) > 0, each with a single parent with index (p_i) "higher" in the
    tree (p_i < i). The tree structure is fully defined by parent pointers, an
    index array (parent) of len(tree_size) where parent[0] == [0] and parent[i]
    == p_i.

    The source tree implies a per-node child list of length >= 0, defined by
    all nodes for which the given node is a parent. Nodes with no children are
    leaves.

    Scan paths cuts the tree into linear paths, where each non-leaf node (i)
    has a *single* child (c_i) "lower" in the tree (i < c_i). The path
    structure is fully defined by child pointers, an index array (subpath_child)
    of subpath_child[i] == c_i (non-leaves) or subpath_child[i] = -1 (leaves).

    The path partitioning implies "subpath roots", the set of nodes which are
    *not* the subpath child of their parent (subpath_child[parent[i]] != i).
    """

    # [natoms]
    parent: NDArray(int)[:]
    subpath_child: NDArray(int)[:]

    # Per-node derived information on path structure
    n_nonpath_children: NDArray(int)[:]
    is_subpath_root: NDArray(bool)[:]
    is_subpath_leaf: NDArray(bool)[:]

    # [natoms, max_num_nonpath_children]
    nonpath_children: NDArray(int)[:, :]

    # Per-path derived information on path structure, indexed
    # by node id. TODO IS THIS DATA VALID FOR NON_ROOT/LEAF INDICES?
    subpath_length: NDArray(int)[:]
    subpath_depth_from_root: NDArray(int)[:]
    subpath_depth_from_leaf: NDArray(int)[:]

    @classmethod
    @validate_args
    def minimum_subpath_depth(cls, parent: NDArray(int)[:]):
        """Generate paths minimizing the maximum path depth."""

        # Calculate branch factor for each node
        branching_factor = numpy.full_like(parent, -1, dtype=int)
        subpath_child = numpy.full_like(parent, -1, dtype=int)

        compute_branching_factor(
            natoms=len(parent),
            parent=parent,
            out_branching_factor=branching_factor,
            out_branchiest_child=subpath_child,
        )

        return cls.from_subpath_children(parent, subpath_child)

    @classmethod
    @validate_args
    def from_subpath_children(
            cls,
            parent: NDArray(int)[:],
            subpath_child: NDArray(int)[:],
    ):
        assert parent.shape == subpath_child.shape

        # Determine the derived pathing structure for each node
        n_nonpath_children = numpy.full_like(parent, 0, dtype=int)
        is_subpath_root = numpy.full_like(parent, False, dtype="bool")
        is_subpath_leaf = numpy.full_like(parent, False, dtype="bool")

        mark_path_children_and_count_nonpath_children(
            natoms=len(parent),
            parent_ko=parent,
            subpath_child_ko=subpath_child,
            out_n_nonpath_children_ko=n_nonpath_children,
            out_is_subpath_root_ko=is_subpath_root,
            out_is_subpath_leaf_ko=is_subpath_leaf,
        )

        # Get list of non_path children for subpath root
        nonpath_children = numpy.full(
            (len(parent), n_nonpath_children.max()),
            -1,
            dtype=int,
        )

        list_nonpath_children(
            natoms=len(parent),
            is_subpath_root_ko=is_subpath_root,
            parent_ko=parent,
            out_non_path_children_ko=nonpath_children,
        )

        # Mark path depths and path lengths
        subpath_length = numpy.zeros_like(parent, dtype=int)
        subpath_depth_from_leaf = numpy.full_like(parent, -1, dtype=int)
        subpath_depth_from_root: NDArray(int)[:]

        find_derivsum_path_depths(
            natoms=len(parent),
            subpath_child_ko=subpath_child,
            non_path_children_ko=nonpath_children,
            is_subpath_root_ko=is_subpath_root,
            out_derivsum_path_depth_ko=subpath_depth_from_leaf,
            out_subpath_length_ko=subpath_length,
        )

        subpath_depth_from_root = numpy.full_like(parent, 0, dtype=int)

        find_refold_path_depths(
            natoms=len(parent),
            parent_ko=parent,
            is_subpath_root_ko=is_subpath_root,
            out_refold_atom_depth_ko=subpath_depth_from_root,
        )

        return cls(
            parent=parent,
            subpath_child=subpath_child,
            n_nonpath_children=n_nonpath_children,
            nonpath_children=nonpath_children,
            is_subpath_root=is_subpath_root,
            is_subpath_leaf=is_subpath_leaf,
            subpath_length=subpath_length,
            subpath_depth_from_root=subpath_depth_from_root,
            subpath_depth_from_leaf=subpath_depth_from_leaf,
        )
