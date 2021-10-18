from typing import Optional, Tuple, Union

import attr
from toolz import first

import torch
import numpy
import numba
import pandas
import scipy.sparse as sparse
import scipy.sparse.csgraph as csgraph

from tmol.types.array import NDArray
from tmol.types.torch import Tensor
from tmol.types.functional import convert_args, validate_args
from tmol.types.tensor import cat

from .datatypes import NodeType, KinForest
from .scan_ordering import get_children

ChildParentTuple = Tuple[NDArray[int][:], NDArray[int][:]]


@attr.s(auto_attribs=True, frozen=True)
class KinematicBuilder:
    """Supports assembly of sets of bonded atoms into a valid KinForest.

    The primary way in which KinForests are built is to provide the set of
    potential (directed) edges between atoms in individual trees as well as
    the list of root atoms, one for each tree in the forest. If certain bonds
    should be included for whatever reason (e.g., they hold the DOFs that the
    user will likely optimize), then these can be given as prioritized edges.
    However, if the prioritized edge are not present in the set of potential
    directed edges, then they will not be included in the tree, thus making
    it safe to, e.g., list the bonds for all named torsions in a PoseStack
    even if some of those bonds span cutpoints; as long as the cutpoint edge
    is not listed in the set of potential edges.

    For both the potential- and priority edges, it is safe to include both
    (a,b) and (b,a) as at most one of the pair will appear in the resulting
    forest.

    Internally, the builder is going to connect the root atoms together
    calculate a minimum spanning tree, before removing the connections
    between the root atoms.

    Deprecated:
    Provides utility methods to perform incremental assembly of sets of bonded
    atoms ("connected components") into a valid KinForest. This involves
    determination of a spanning DAG for the component beginning from a
    specified root atom, initialization of reference frames for each atom, and
    concatenation of this component onto the KinForest 
    """

    kinforest: KinForest = attr.Factory(KinForest.root_node)

    @classmethod
    @convert_args
    def define_trees_with_prioritized_bonds(
        # def component_for_prioritized_bonds(
        cls,
        roots: NDArray[int][:],
        potential_bonds: NDArray[int][:, 2],
        priotized_bonds: NDArray[int][:, 2],
        n_atoms_total: int,
    ) -> ChildParentTuple:
        assert potential_bonds.shape[1] == prioritized_bonds.shape[1]
        if not isinstance(roots, numpy.ndarray):
            # create array from the single integer root input
            roots = numpy.array([roots], dtype=int)

        # interpret an Nx2 bonds array as representing a single stack
        if prioritized_bonds.shape[1] == 2:
            prioritized_bonds = numpy.concatenate(
                (
                    numpy.zeros((prioritized_bonds.shape[0], 1), dtype=int),
                    prioritized_bonds,
                ),
                axis=1,
            )
            potential_bonds = numpy.concatenate(
                (
                    numpy.zeros((potential_bonds.shape[0], 1), dtype=int),
                    potential_bonds,
                ),
                axis=1,
            )

        weighted_bonds = (
            # All entries must be non-zero or sparse graph tools will entries.
            cls.bonds_to_csgraph(potential_bonds, [-1], n_atoms_total)
            + cls.bonds_to_csgraph(prioritized_bonds, [-.125], n_atoms_total)
            + cls.faux_bonds_between_roots(
                roots=roots, weights=[-1], n_atoms_total=n_atoms_total
            )
        )

        # ids, parents = cls.bonds_to_connected_component(roots, weighted_bonds)
        to_ids_in_kfo, to_parents_in_kfo = cls.bonds_to_forest(roots, weighted_bonds)

        return to_ids_in_kfo, to_parents_in_kfo

    @classmethod
    @convert_args
    def bonds_to_csgraph(
        cls,
        n_atoms_total: int,
        bonds: NDArray[int][:, 2],
        weights: NDArray[float][:] = numpy.ones(1),  # noqa
        # system_size: Optional[int] = None,
    ) -> sparse.csr_matrix:

        # if atoms are stored in a stack, the caller of this function
        # must first convert the indices of their atoms into a single
        # global indexing
        #
        # if not system_size:
        #     system_size = bonds[:, 1:3].max() + 1
        # bonds_reindexed = system_size * bonds[:, 0][:, None] + bonds[:, 1:3]

        weights = numpy.broadcast_to(weights, bonds[:, 0].shape)

        bonds_csr = sparse.csr_matrix(
            (weights, (bonds_reindexed[:, 0], bonds_reindexed[:, 1])),
            shape=(n_atoms_total, n_atoms_total),
        )
        return bonds_csr

    @classmethod
    @convert_args
    def faux_bonds_between_roots(
        cls, roots: NDArray[int][:], weights: NDArray[float][:], natoms_total: int
    ) -> Union[sparse.spmatrix, int]:
        """Construct a csgraph with edges from the first root to all
        other roots, if there are other roots. Otherwise, return 0.
        The size of this csgraph should be the total number of
        atoms in all stacks.
        """
        if roots.shape[0] > 1:
            root_faux_bonds = numpy.full((roots.shape[0] - 1, 3), roots[0], dtype=int)
            root_faux_bonds[:, 0] = 0
            root_faux_bonds[:, 1] = roots[1:]
            return cls.bonds_to_csgraph(
                root_faux_bonds, weights=weights, system_size=natoms_total
            )
        else:
            return 0

    @classmethod
    @validate_args
    # def bonds_to_connected_component(
    def bonds_to_forest(
        cls, roots: NDArray[int][:], bonds: Union[NDArray[int][:, 2], sparse.spmatrix]
    ) -> ChildParentTuple:

        if isinstance(bonds, numpy.ndarray):
            # Bonds are a non-prioritized set of edges, assume arbitrary
            # connectivity from the root is allowed.
            n_atoms_total = max(bonds[:, 1].max(), bonds[:, 2].max()) + 1

            bond_graph = cls.bonds_to_csgraph(
                bonds, n_atoms_total=n_atoms_total
            ) + cls.faux_bonds_between_roots(
                roots=roots, weights=[1], n_atoms_total=n_atoms_total
            )
        else:
            # Sparse graph with per-bond weights, generate the minimum
            # spanning tree of the connections before traversing from
            # the root.
            bond_graph = csgraph.minimum_spanning_tree(bonds.tocsr())

        # Perform breadth first traversal from the root of the component
        # to generate the kinematic tree.
        to_ids_in_kfo, preds = csgraph.breadth_first_order(
            bond_graph, roots[0], directed=False, return_predecessors=True
        )
        to_parents_in_kfo = preds[to_ids_in_kfo]

        # make sure that all nodes were reached in the BFS traversal of
        # the graph; only the first root node should have a -9999 parent
        # and the parents of the other roots should all be the first root.
        assert to_parents_in_kfo[roots[0]] == -9999
        assert numpy.all(to_parents_in_kfo[roots[1:]] == roots[0])
        is_non_root = numpy.full(to_parents_in_kfo.shape, True, dtype=bool)
        is_non_root[roots] = False
        assert numpy.all(to_parents_in_kfo[is_non_root] >= 0)

        to_parents_in_kfo[roots] = -9999

        return to_ids_in_kfo.astype(int), to_parents_in_kfo.astype(int)

    @convert_args
    def append_connected_components(
        self,
        to_roots: Tensor[int][:],
        to_ids: Tensor[int][:],
        to_parents: Tensor[int][:],
        to_jump_nodes: Tensor[int][:],
        component_parent=0,
    ):
        """After having created a forest, as identified by a depth-first
        travsersal of the input graph, we need to convert the target-order
        indices into the kin-forest order. To do this we could construct
        a target-order-2-kin-forest-order mapping, or we could use the
        get_indexer method of the pandas.Index object.
        """

        assert (
            ids.shape == parent_ids.shape
        ), "elements and parents must be of same length"

        assert len(ids) >= 3, "Bonded ktree must have at least three entries"

        assert component_parent < len(self.kinforest) and component_parent >= -1

        n_atoms = len(ids)

        id_index = pandas.Index(to_ids)
        # kfo_roots = torch.LongTensor(id_index.get_indexer(to_roots))

        kfo_roots = id_index.get_indexer(to_roots)
        kfo_jump_nodes = id_index.get_indexer(to_jump_nodes)
        kfo_parents = id_index.get_indexer(to_parents)

        # Root nodes are self-parented for the purpose of verifying the
        # connected component tree structure, will be rewritten with jump to
        # the component root frame.
        kfo_parents[kfo_roots] = kfo_roots

        kfo_grandparents = kfo_parents[kfo_parents]
        doftype = numpy.zeros(n_atoms, numpy.int64)
        frame_x = numpy.zeros(n_atoms, numpy.int64)
        frame_y = numpy.zeros(n_atoms, numpy.int64)
        frame_z = numpy.zeros(n_atoms, numpy.int64)

        kin_stree = KinForest.full(len(ids), 0)
        kin_stree.id[:] = torch.tensor(to_ids)

        kin_start = len(self.kinforest)

        doftype[:] = NodeType.bond
        # kfo_parents_final = kfo_parents + kin_start
        frame_x[:] = numpy.arange(len(to_ids), dtype=numpy.int64) + kin_start
        frame_y[:] = kfo_parents + kin_start
        frame_z[:] = kfo_grandparents + kin_start

        doftype[kfo_roots] = NodeType.jump
        doftype[kfo_jump_nodes] = NodeType.jump

        parent[kfo_roots] = component_parent

        fix_jump_nodes(
            kfo_parents, doftype, frame_x, frame_y, frame_z, kfo_roots, kfo_jump_nodes
        )

        kfo_roots += kin_start
        kfo_parents += kin_start
        # kfo_jump_nodes += kin_start
        frame_x += kin_start
        frame_y += kin_start
        frame_z += kin_start

        def _t(x):
            return torch.tensor(x, dtype=torch.int64)

        # what does the s stand for in sforest? Shit, I don't know
        kin_sforest = KinForest(
            id=_t(to_id),
            roots=_t(kfo_roots),
            doftype=_t(doftype),
            parent=_t(kfo_parents),
            frame_x=_t(frame_x),
            frame_y=_t(frame_y),
            frame_z=_t(frame_z),
        )

        return attr.evolve(self, kinforest=cat(self.kinforest, self.kin_sforest))

        # Go back and rewrite the entries for the root and its children
        # Define the jump DOF of the root, connecting back into existing kintree

        # Fixup the orientation frame frame of the root and its children.
        # The root is self-parented at zero, so drop the first match.
        # for root in kfo_roots:
        #     int_root, *root_children = [
        #         int(i) for i in torch.nonzero(kfo_parents == root, as_tuple=False)
        #     ]
        #     assert root == int_root, "root must be self parented, was set above"
        #     root_c1, *root_sibs = root_children
        #
        #     assert len(root_children) >= 1, "root must have at least one child"
        #     c1_children = [int(i) for i in torch.nonzero(kfo_parents == root_c1)]
        #     if len(c1_children) > 0:
        #         c1_children = torch.LongTensor(c1_children)
        #         root_sibs = torch.LongTensor(root_sibs)
        #
        #         kin_stree.frame_x[[int_root, root_c1]] = root_c1 + kin_start
        #         kin_stree.frame_y[[int_root, root_c1]] = int_root + kin_start
        #         kin_stree.frame_z[[int_root, root_c1]] = (
        #             first(c1_children).to(dtype=torch.int) + kin_start
        #         )
        #
        #         kin_stree.frame_x[root_sibs] = root_sibs.to(dtype=torch.int) + kin_start
        #         kin_stree.frame_y[root_sibs] = int_root + kin_start
        #         kin_stree.frame_z[root_sibs] = root_c1 + kin_start
        #     else:
        #         assert len(root_children) >= 2, (
        #             "root of bonded tree must have two children if the"
        #             " first child of the root has no children"
        #         )
        #
        #         root_c1, *root_sibs = root_children
        #         root_sibs = torch.LongTensor(root_sibs)
        #
        #         kin_stree.frame_x[[int_root, root_c1]] = root_c1 + kin_start
        #         kin_stree.frame_y[[int_root, root_c1]] = int_root + kin_start
        #         kin_stree.frame_z[[int_root, root_c1]] = (
        #             first(root_sibs).to(dtype=torch.int) + kin_start
        #         )
        #
        #         kin_stree.frame_x[root_sibs] = root_sibs.to(dtype=torch.int) + kin_start
        #         kin_stree.frame_y[root_sibs] = int_root + kin_start
        #         kin_stree.frame_z[root_sibs] = root_c1 + kin_start

        # return attr.evolve(self, kintree=cat((self.kintree, kin_stree)))

    @convert_args
    def append_connected_component(
        self, ids: Tensor[int][:], parent_ids: Tensor[int][:], component_parent=0
    ):
        return self.append_connected_components(
            roots=torch.zeros((1,), dtype=torch.int32),
            ids=ids,
            parent_ids=parent_ids,
            component_parent=component_parent,
        )


@numba.jit(nopython=True)
def stub_defined_for_jump_atom(jump_atom, atom_is_jump, child_list_span, child_list):
    #  have to handle a couple of cases here:
    #
    #  note -- in counting dependent atoms, exclude JumpAtom's
    #
    #
    #  1. no dependent atoms --> no way to define new coord sys
    #     on this end. ergo take parent's M and my xyz
    #
    #  2. one dependent atom --> no way to define unique coord
    #     on this end, still take parent's M and my xyz
    #
    #  3. two or more dependent atoms
    #     a) if my first atom has a dependent atom, use
    #        myself, my first atom, and his first atom
    #
    #     b) otherwise, use
    #        myself, my first atom, my second atom

    first_nonjump_child = -1
    for child_ind in range(
        child_list_span[jump_atom, 0], child_list_span[jump_atom, 1]
    ):
        child_atom = child_list[child_ind]
        if atom_is_jump[child_atom]:
            continue
        if first_nonjump_child == -1:
            first_nonjump_child = child_atom
        else:
            return True
    if first_nonjump_child != -1:
        for grandchild_ind in range(
            child_list_span[first_nonjump_child, 0],
            child_list_span[first_nonjump_child, 1],
        ):
            if not atom_is_jump[child_list[grandchild_ind]]:
                return True
    return False


@numba.jit(nopython=True)
def get_c1_and_c2_atoms(
    jump_atom: int,
    atom_is_jump: NDArray[int][:],
    child_list_span: NDArray[int][:],
    child_list: NDArray[int][:],
    parents: NDArray[int][:],
) -> tuple:
    """Preferably a jump should steal DOFs from its first (nonjump) child
    and its first (nonjump) grandchild, but if the first child does not
    have any children, then it can steal a DOF from its second (nonjump)
    child. If a jump does not have a sufficient number of descendants, then
    we must recurse to its parent.
    """

    first_nonjump_child = -1
    second_nonjump_child = -1
    for child_ind in range(
        child_list_span[jump_atom, 0], child_list_span[jump_atom, 1]
    ):
        child_atom = child_list[child_ind]
        if atom_is_jump[child_atom]:
            continue
        if first_nonjump_child == -1:
            first_nonjump_child = child_atom
        else:
            second_nonjump_child = child_atom
            break

    if first_nonjump_child == -1:
        jump_parent = parents[jump_atom]
        assert jump_parent != jump_atom
        return get_c1_and_c2_atoms(
            jump_parent, atom_is_jump, child_list_span, child_list, parents
        )

    for grandchild_ind in range(
        child_list_span[first_nonjump_child, 0], child_list_span[first_nonjump_child, 1]
    ):
        grandchild_atom = child_list[grandchild_ind]
        if not atom_is_jump[grandchild_atom]:
            return first_nonjump_child, grandchild_atom

    if second_nonjump_child == -1:
        jump_parent = parents[jump_atom]
        assert jump_parent != jump_atom
        return get_c1_and_c2_atoms(
            jump_parent, atom_is_jump, child_list_span, child_list, parents
        )

    return first_nonjump_child, second_nonjump_child


@numba.jit(nopython=True)
def fix_jump_nodes(
    parents: NDArray[int][:],
    frame_x: NDArray[int][:],
    frame_y: NDArray[int][:],
    frame_z: NDArray[int][:],
    roots: NDArray[int][:],
    jumps: NDArray[int][:],
):
    nelts = parents.shape[0]
    n_children, child_list_span, child_list = get_children(parents)

    atom_is_jump = numpy.full(parents.shape, 0, dtype=numpy.int64)
    atom_is_jump[roots] = 1
    atom_is_jump[jumps] = 1

    for root in roots:
        assert stub_defined_for_jump_atom(
            root, atom_is_jump, child_list_span, child_list
        )

        root_c1, second_descendent = get_c1_and_c2_atoms(
            root, atom_is_jump, child_list_span, child_list, parents
        )

        # set the frame_x, _y, and _z to the same values for both the root
        # and the root's first child

        frame_x[root] = root_c1
        frame_y[root] = root
        frame_z[root] = second_descendent

        frame_x[root_c1] = root_c1
        frame_y[root_c1] = root
        frame_z[root_c1] = second_descendent

        # all the other children of the root need an updated kinematic description
        for child_ind in range(child_list_span[root, 0] + 1, child_list_span[root, 1]):
            child = child_list[child_ind]
            if atom_is_jump[child]:
                continue
            if child == root_c1:
                continue
            frame_x[child] = child
            frame_y[child] = root
            frame_z[child] = root_c1

    for jump in jumps:
        if stub_defined_for_jump_atom(jump, atom_is_jump, child_list_span, child_list):

            jump_c1, jump_c2 = get_c1_and_c2_atoms(
                jump, atom_is_jump, child_list_span, child_list, parents
            )

            # set the frame_x, _y, and _z to the same values for both the jump
            # and the jump's first child

            frame_x[jump] = jump_c1
            frame_y[jump] = jump
            frame_z[jump] = jump_c2

            frame_x[jump_c1] = jump_c1
            frame_y[jump_c1] = jump
            frame_z[jump_c1] = jump_c2

            # all the other children of the jump need an updated kinematic description
            for child_ind in range(
                child_list_span[jump, 0] + 1, child_list_span[jump, 1]
            ):
                child = child_list[child_ind]
                if atom_is_jump[child]:
                    continue
                if child == jump_c1:
                    continue
                frame_x[child] = child
                frame_y[child] = jump
                frame_z[child] = jump_c1
        else:
            # ok, so... I don't understand the atom tree well enough to understand this
            # situation. If the jump has no non-jump children, then certainly none
            # of them need their frame definitions update
            c1, c2 = get_c1_and_c2_atoms(
                parents[jump], atom_is_jump, child_list_span, child_list, parents
            )

            frame_x[jump] = c1
            frame_y[jump] = jump
            frame_z[jump] = c2

            # the jump may have one child; it's not entirely clear to me
            # what frame the child should have!
            # TO DO: figure this out
            for child_ind in range(
                child_list_span[jump, 0] + 1, child_list_span[jump, 1]
            ):
                child = child_list[child_ind]
                if atom_is_jump[child]:
                    continue
                frame_x[child] = c1
                frame_y[child] = jump
                frame_z[child] = c2
