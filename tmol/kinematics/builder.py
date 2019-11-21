from typing import Optional, Tuple, Union

import attr
from toolz import first

import torch
import numpy
import pandas
import scipy.sparse as sparse
import scipy.sparse.csgraph as csgraph

from tmol.types.array import NDArray
from tmol.types.torch import Tensor
from tmol.types.functional import convert_args, validate_args
from tmol.types.tensor import cat

from .datatypes import NodeType, KinTree

ChildParentTuple = Tuple[NDArray(int)[:], NDArray(int)[:]]


@attr.s(auto_attribs=True, frozen=True)
class KinematicBuilder:
    """Supports assembly of sets of bonded atoms into a valid KinTree.

    Provides utility methods to perform incremental assembly of sets of bonded
    atoms ("connected components") into a valid KinTree. This involves
    determination of a spanning DAG for the component beginning from a
    specified root atom, initialization of reference frames for each atom, and
    concatenation of this component onto the KinTree via a jump.

    The builder supports "prioritized bonds" within a connected component,
    which must be present as parent-child relationships within the resulting
    tree. This ensures that specific dofs are explicitly represented in cases
    where a cycle in bonded connectivity results in multiple valid spanning
    trees. (Eg. The proline backbone ring.)
    """

    kintree: KinTree = attr.Factory(KinTree.root_node)

    @classmethod
    @convert_args
    def component_for_prioritized_bonds(
        cls,
        roots: Union[NDArray(int)[:], int],
        mandatory_bonds: Union[NDArray(int)[:, 3], NDArray(int)[:, 2]],
        all_bonds: Union[NDArray(int)[:, 3], NDArray(int)[:, 2]],
        system_size: Optional[int] = None
    ) -> ChildParentTuple:
        assert mandatory_bonds.shape[1] == all_bonds.shape[1]
        if not isinstance(roots, numpy.ndarray):
            # create array from the single integer root input
            roots = numpy.array([roots], dtype=int)

        # interpret an Nx2 bonds array as representing a single stack
        if mandatory_bonds.shape[1] == 2:
            mandatory_bonds = numpy.concatenate(
                (numpy.zeros((mandatory_bonds.shape[0],1),dtype=int), mandatory_bonds),
                axis=1
            )
            all_bonds = numpy.concatenate(
                (numpy.zeros((all_bonds.shape[0],1),dtype=int), all_bonds),
                axis=1
            )

        if not system_size:
            system_size = max(mandatory_bonds[:, 1:3].max(), all_bonds[:, 1:3].max()) + 1

        weighted_bonds = (
            # All entries must be non-zero or sparse graph tools will entries.
            cls.bonds_to_csgraph(all_bonds, [-1], system_size)
            + cls.bonds_to_csgraph(mandatory_bonds, [-1e-5], system_size)
            + cls.faux_bonds_between_roots(
                roots=roots,
                weights=[-1],
                natoms_total=system_size*roots.shape[0],
            )
        )

        ids, parents = cls.bonds_to_connected_component(roots, weighted_bonds)

        # Verify construction
        component_bond_graph = cls.bonds_to_csgraph(
            numpy.block([[parents[1:], ids[1:]], [ids[1:], parents[1:]]]).T
        )
        bond_present = component_bond_graph[
            system_size*mandatory_bonds[:, 0] + mandatory_bonds[:, 1],
            system_size*mandatory_bonds[:, 0] + mandatory_bonds[:, 2]
        ]
        bond_absent = numpy.array((bond_present == 0), dtype=bool)

        assert numpy.all(
            bond_present
        ), "Unable to generate component containing all mandatory bonds."

        return ids, parents

    @classmethod
    @convert_args
    def bonds_to_csgraph(
        cls,
        bonds: Union[NDArray(int)[:, 3], NDArray(int)[:,2]],
        weights: NDArray(float)[:] = numpy.ones(1),  # noqa
        system_size: Optional[int] = None,
    ) -> sparse.csr_matrix:

        # interpret an Nx2 bonds array as representing a single stack
        if bonds.shape[1] == 2:
            bonds = numpy.concatenate(
                (numpy.zeros((bonds.shape[0],1),dtype=int), bonds),
                axis=1
            )

        if not system_size:
            system_size = (bonds[:,1:3].max() + 1)

        bonds_reindexed = system_size * bonds[:,0][:,None] + bonds[:,1:3]
        weights = numpy.broadcast_to(weights, bonds_reindexed[:, 0].shape)

        nats_tot = system_size * (bonds[:,0].max() + 1)
        bonds_csr = sparse.csr_matrix(
            (weights, (bonds_reindexed[:, 0], bonds_reindexed[:, 1])),
            shape=(nats_tot, nats_tot)
        )
        return bonds_csr

    @classmethod
    @convert_args
    def faux_bonds_between_roots(
        cls,
        roots: NDArray(int)[:],
        weights: NDArray(float)[:],
        natoms_total: int
    ) -> Union[sparse.spmatrix, int]:
        """Construct a csgraph with edges from the first root to all
        other roots, if there are other roots. Otherwise, return 0.
        The size of this csgraph should be the total number of
        atoms in all stacks.
        """
        if roots.shape[0] > 1:
            root_faux_bonds = numpy.full((roots.shape[0]-1, 3), roots[0], dtype=int)
            root_faux_bonds[:,0] = 0
            root_faux_bonds[:,1] = roots[1:]
            return cls.bonds_to_csgraph(
                root_faux_bonds,
                weights=weights,
                system_size=natoms_total
            )
        else:
            return 0

    @classmethod
    @validate_args
    def bonds_to_connected_component(
        cls,
        roots: Union[NDArray(int)[:], int],
        bonds: Union[NDArray(int)[:, 3], NDArray(int)[:, 2], sparse.spmatrix],
        system_size: Optional[int] = None
    ) -> ChildParentTuple:

        if not isinstance(roots, numpy.ndarray):
            # create a numpy array from the integer input
            roots = numpy.array([roots], dtype=int)

        if isinstance(bonds, numpy.ndarray):
            # Bonds are a non-prioritized set of edges, assume arbitrary
            # connectivity from the root is allowed.
            if bonds.shape[1] == 2:
                bonds = numpy.concatenate(
                    (numpy.zeros((bonds.shape[0],1),dtype=int), bonds),
                    axis=1
                )
            if system_size is None:
                system_size = max(bonds[:,1].max(), bonds[:,2].max()) + 1

            bond_graph = (
                cls.bonds_to_csgraph(bonds, system_size=system_size) +
                cls.faux_bonds_between_roots(
                    roots=roots,
                    weights=[1],
                    natoms_total=int(system_size*(1+bonds[:,0].max()))
                )
            )
        else:
            # Sparse graph with per-bond weights, generate the minimum
            # spanning tree of the connections before traversing from
            # the root.
            bond_graph = csgraph.minimum_spanning_tree(bonds.tocsr())

        # Perform breadth first traversal from the root of the component
        # to generate the kinematic tree.
        ids, preds = csgraph.breadth_first_order(
            bond_graph, roots[0], directed=False, return_predecessors=True
        )
        parents = preds[ids]
        assert parents[roots[0]] == -9999
        assert numpy.all(parents[1:] >= 0)

        return ids.astype(int), parents.astype(int)

    @convert_args
    def append_connected_components(
        self,
        roots: Tensor(int)[:],
        ids: Tensor(int)[:],
        parent_ids: Tensor(int)[:],
        component_parent=0,
    ):
        assert(
            ids.shape == parent_ids.shape,
            "elements and parents must be of same length"
        )
        assert len(ids) > 3, "Bonded ktree must have at least three entries"
        assert component_parent < len(self.kintree) and component_parent >= -1

        # root = roots[0]
        id_index = pandas.Index(ids)
        root_indices = torch.LongTensor(id_index.get_indexer(roots))
        for root in root_indices:
            parent_ids[root] = ids[root]

        parent_indices = torch.LongTensor(id_index.get_indexer(parent_ids))
        grandparent_indices = parent_indices[parent_indices]

        kin_stree = KinTree.full(len(ids), 0)
        kin_stree.id[:] = ids

        kin_start = len(self.kintree)

        kin_stree.doftype[:] = NodeType.bond
        kin_stree.parent[:] = parent_indices + kin_start
        kin_stree.frame_x[:] = torch.arange(len(ids)) + kin_start
        kin_stree.frame_y[:] = parent_indices + kin_start
        kin_stree.frame_z[:] = grandparent_indices + kin_start

        # Go back and rewrite the entries for the roots and their children.
        # Define the jump DOF of the root, connecting back into existing
        # kintree.
        kin_stree.doftype[root_indices] = NodeType.jump
        kin_stree.parent[root_indices] = component_parent

        # Fixup the orientation frames of the root and its children.
        # The root is self-parented, so drop the first match.
        # I feel like this loop will be slow when building trees
        # for rotamers, where (nroots x natoms) is large.
        for root in root_indices:

            int_root, *root_children = [int(i) for i in torch.nonzero(parent_indices == root)]
            assert len(root_children) >= 2, "root of bonded tree must have two children"
            assert root == int_root
            root_c1, *root_sibs = root_children
            root_sibs = torch. LongTensor(root_sibs)

            kin_stree.frame_x[[int_root, root_c1]] = root_c1 + kin_start
            kin_stree.frame_y[[int_root, root_c1]] = int_root + kin_start
            kin_stree.frame_z[[int_root, root_c1]] = (
                first(root_sibs).to(dtype=torch.int) + kin_start
            )

            kin_stree.frame_x[root_sibs] = root_sibs.to(dtype=torch.int) + kin_start
            kin_stree.frame_y[root_sibs] = int_root + kin_start
            kin_stree.frame_z[root_sibs] = root_c1 + kin_start

        return attr.evolve(self, kintree=cat((self.kintree, kin_stree)))


    @convert_args
    def append_connected_component(
        self, ids: Tensor(int)[:], parent_ids: Tensor(int)[:], component_parent=0
    ):
        return self.append_connected_components(
            roots=torch.zeros((1,), dtype=torch.int32),
            ids=ids, parent_ids=parent_ids,
            component_parent=component_parent )
        #  assert (
        #      ids.shape == parent_ids.shape
        #  ), "elements and parents must be of same length"
        #  assert len(ids) > 3, "Bonded ktree must have at least three entries"
        #  assert component_parent < len(self.kintree) and component_parent >= -1
        #  
        #  # Assert that there is a single connected component?
        #  
        #  # Root node is self-parented for the purpose of verifying the
        #  # connected component tree structure, will be rewritten with jump to
        #  # the component root frame.
        #  parent_ids[0] = ids[0]
        #  
        #  # Create an index of all component ids in the graph and get component
        #  # parent and parent-parent indices
        #  id_index = pandas.Index(ids)
        #  parent_indices = torch.LongTensor(id_index.get_indexer(parent_ids))
        #  grandparent_indices = parent_indices[parent_indices]
        #  
        #  # Verify that ids are unique and that all parent references are valid,
        #  # get_indexer returns -1 if a target value is not present in index.
        #  assert not id_index.has_duplicates, "Duplicated id in component"
        #  assert numpy.all(parent_indices >= 0), "Parent id not present in component."
        #  
        #  # Allocate entries for the new subtree and store the provided ids in
        #  # the kintree id column. All internal kintree references,
        #  # (parent/frame) will be wrt kinetree indices, not ids.
        #  kin_stree = KinTree.full(len(ids), 0)
        #  kin_stree.id[:] = ids
        #  
        #  # Calculate the start index of the kinematic tree block this new
        #  # subtree will occupy, construct all parent & frame references wrt this
        #  # start index.
        #  kin_start = len(self.kintree)
        #  
        #  # Start by writing the the standard, non-root entries of the graph.
        #  kin_stree.doftype[:] = NodeType.bond
        #  kin_stree.parent[:] = parent_indices + kin_start
        #  kin_stree.frame_x[:] = torch.arange(len(ids)) + kin_start
        #  kin_stree.frame_y[:] = parent_indices + kin_start
        #  kin_stree.frame_z[:] = grandparent_indices + kin_start
        #  
        #  # Go back and rewrite the entries for the root and its children
        #  # Define the jump DOF of the root, connecting back into existing kintree
        #  kin_stree.doftype[0] = NodeType.jump
        #  kin_stree.parent[0] = component_parent
        #  
        #  # Fixup the orientation frame of the root and its children.
        #  # The root is self-parented at zero, so drop the first match.
        #  root, *root_children = [int(i) for i in torch.nonzero(parent_indices == 0)]
        #  assert len(root_children) >= 2, "root of bonded tree must have two children"
        #  assert root == 0, "root must be self parented, was set above"
        #  root_c1, *root_sibs = root_children
        #  root_sibs = torch.LongTensor(root_sibs)
        #  
        #  kin_stree.frame_x[[root, root_c1]] = root_c1 + kin_start
        #  kin_stree.frame_y[[root, root_c1]] = root + kin_start
        #  kin_stree.frame_z[[root, root_c1]] = (
        #      first(root_sibs).to(dtype=torch.int) + kin_start
        #  )
        #  
        #  kin_stree.frame_x[root_sibs] = root_sibs.to(dtype=torch.int) + kin_start
        #  kin_stree.frame_y[root_sibs] = root + kin_start
        #  kin_stree.frame_z[root_sibs] = root_c1 + kin_start
        #  
        #  # Append the subtree onto the kintree.
        #  return attr.evolve(self, kintree=cat((self.kintree, kin_stree)))
