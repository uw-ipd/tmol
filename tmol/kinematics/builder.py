from typing import Optional, Tuple, Union

import attr
from toolz import first

import numpy
import pandas
import scipy.sparse as sparse
import scipy.sparse.csgraph as csgraph

from tmol.types.array import NDArray
from tmol.types.functional import convert_args, validate_args

from .datatypes import (NodeType, KinTree, KinTreeNode)


def kintree_root_factory():
    kintree_root = KinTree.full(1, 0)
    kintree_root[0] = KinTreeNode(-1, NodeType.root, 0, 0, 0, 0)
    return kintree_root


# fd  this returns a numpy data structure and not torch
#     is that reasonable?
@validate_args
def kintree_connections(kintree: KinTree) -> NDArray(int)[:, 2]:
    """Return parent-id <-> child-id pairs for non-root dofs in kintree."""
    msk = kintree.doftype != NodeType.root
    assert not msk[0], "kintree is not rooted"

    child = kintree.id[msk]
    parent = kintree.id[kintree.parent.squeeze()][msk]

    retval = numpy.empty([len(child), 2], dtype=int)
    retval[:, 0] = child
    retval[:, 1] = parent

    return retval


@attr.s(auto_attribs=True, frozen=True)
class KinematicBuilder:
    kintree: KinTree = attr.Factory(kintree_root_factory)

    @classmethod
    @convert_args
    def bond_csgraph(
            cls,
            bonds: NDArray(int)[:, 2],
            weights: NDArray(float)[:] = numpy.ones(1),
            system_size: Optional[int] = None,
    ) -> sparse.csr_matrix:
        if not system_size:
            system_size = bonds.max() + 1

        weights = numpy.broadcast_to(weights, bonds[:, 0].shape)

        return sparse.csr_matrix(
            (weights, (bonds[:, 0], bonds[:, 1])),
            shape=(system_size, system_size),
        )

    @classmethod
    @validate_args
    def bonds_to_connected_component(
            cls,
            root: int,
            bonds: Union[NDArray(int)[:, 2], sparse.spmatrix],
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        if isinstance(bonds, numpy.ndarray):
            # Bonds are a non-prioritized set of edges, assume arbitrary
            # connectivity from the root is allowed.
            bond_graph = cls.bond_csgraph(bonds)
        else:
            # Sparse graph with per-bond weights, generate the minimum
            # spanning tree of the connections before traversing from
            # the root.
            bond_graph = csgraph.minimum_spanning_tree(bonds.tocsr())

        # Perform breadth first traversal from the root of the component
        # to generate the kinematic tree.
        ids, preds = csgraph.breadth_first_order(
            bond_graph, root, directed=False, return_predecessors=True
        )
        parents = preds[ids]
        assert parents[0] == -9999
        assert numpy.all(parents[1:] >= 0)

        return ids, parents

    @convert_args
    def append_connected_component(
            self,
            ids: NDArray(int)[:],
            parent_ids: NDArray(int)[:],
            component_parent=0,
    ):
        assert ids.shape == parent_ids.shape, "elements and parents must be of same length"
        assert len(ids) > 3, "Bonded ktree must have at least three entries"
        assert component_parent < len(self.kintree) and component_parent >= -1

        # Assert that there is single connected component?

        # Root node is self-parented for the purpose of verifying the
        # connected component tree structure, will be rewritten with jump to
        # the component root frame.
        parent_ids[0] = ids[0]

        pandas.Index
        # Create an index of all component ids in the graph and get component
        # parent and parent-parent indices
        id_index = pandas.Index(ids)
        parent_indices = id_index.get_indexer(parent_ids)
        grandparent_indices = parent_indices[parent_indices]

        # Verify that ids are unique and that all parent references are valid,
        # get_indexer returns -1 if a target value is not present in index.
        assert not id_index.has_duplicates, "Duplicated id in component"
        assert numpy.all(parent_indices >= 0
                         ), ("Parent id not present in component.")

        # Allocate entries for the new subtree and store the provided ids in
        # the kintree id column. All internal kintree references,
        # (parent/frame) will be wrt kinetree indices, not ids.
        kin_stree = KinTree.full(len(ids), 0)
        kin_stree.set_ids(ids)

        # Calculate the start index of the kinematic tree block this new
        # subtree will occupy, construct all parent & frame references wrt this
        # start index.
        kin_start = len(self.kintree)

        # Start by writing the the standard, non-root entries of the graph.
        kin_stree.set_doftypes(NodeType.bond)
        kin_stree.set_parents(parent_indices + kin_start)
        kin_stree.set_frameX(numpy.arange(len(ids)) + kin_start)  # self
        kin_stree.set_frameY(parent_indices + kin_start)
        kin_stree.set_frameZ(grandparent_indices + kin_start)

        # Go back and rewrite the entries for the root and its children
        # Define the jump DOF of the root, connecting back into existing kintree
        kin_stree.doftype[0] = NodeType.jump
        kin_stree.parent[0] = component_parent

        # Fixup the orientation frame frame of the root and its children.
        # The rootis self-parented at zero, so drop the first match.
        _, *root_children = numpy.flatnonzero(parent_indices == 0)
        assert len(
            root_children
        ) >= 2, "root of bonded tree must have two children"
        assert _ == 0, "root must be self parented, was set above"
        root_c1, *root_sibs = root_children

        print(parent_indices)

        # fd: Alex, I'm not 100% sure of the logic here
        # shouldn't the root also be set
        #   (previously the frame was (1,1,1) which is wrong)
        componentRoot = kin_stree[0]
        componentRoot.frame_x = root_c1 + kin_start
        componentRoot.frame_y = 0 + kin_start
        componentRoot.frame_z = first(root_sibs) + kin_start
        kin_stree[0] = componentRoot

        firstChild = kin_stree[root_c1]
        firstChild.frame_x = root_c1 + kin_start
        firstChild.frame_y = 0 + kin_start
        firstChild.frame_z = first(root_sibs) + kin_start
        kin_stree[root_c1] = firstChild

        for sib in root_sibs:
            subseqChild = kin_stree[sib]
            subseqChild.frame_x = sib + kin_start
            subseqChild.frame_y = 0 + kin_start
            subseqChild.frame_z = root_c1 + kin_start
            kin_stree[sib] = subseqChild

        print(kin_stree)

        # Append the subtree onto the kintree.
        return attr.evolve(self, kintree=self.kintree.concatenate(kin_stree))
