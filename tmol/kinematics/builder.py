from typing import Optional, Tuple, Union

import attr

import torch
import numpy
import numba

# import pandas
import scipy.sparse as sparse
import scipy.sparse.csgraph as csgraph

from tmol.types.array import NDArray
from tmol.types.functional import convert_args, validate_args
from tmol.types.tensor import cat

from tmol.utility.ndarray.common_operations import invert_mapping

from .datatypes import NodeType, KinForest
from .scan_ordering import get_children

ChildParentTuple = Tuple[NDArray[int][:], NDArray[int][:]]


@attr.s(auto_attribs=True, frozen=True)
class KinematicBuilder:
    """Supports assembly of sets of bonded atoms into a valid KinForest.

    Usage: invoke KinematicBuilder.append_connected_components with the outputs from
    KinematicBuilder.define_trees_with_prioritized_bonds and retrieve the kinforest
    from the resulting KinematicBuilder object.

    The primary way in which KinForests are built is to provide bonds_to_forest with
    the set of *potential* (directed) edges between atoms in individual trees as well
    as the list of root atoms, one for each tree in the forest. If certain bonds
    should be included for whatever reason (e.g., they hold the DOFs that the
    user will likely optimize), then these can be given as *prioritized* edges.
    E.g., for proline, you would list CA-CB and CB-CG as prioritized edges, and list
    CA-CB, CB-CG, CG-CD, and CD-N as potential edges.

    Prioritzed edges are not guaranteed to end up in the KinForest, however.
    If any prioritized edge is absent from the set of potential
    edges, then it will not be included in the tree. A prioritized
    edge must also be a potential edge to be included. Thus it is
    safe to list as priority edges the bonds for all named torsions in a PoseStack
    even if some of those bonds span cutpoints. As long as the cutpoint edges
    are not listed in the set of potential edges.

    (Thus the FoldForest should be consulted when figuring out which inter-
    block bonds should be included in the potential edges, but it does not
    need to be consulted when figuring out which named torsions should
    be included in the prioritized edges.)

    For both the potential- and priority edges, it is safe to include both
    (a,b) and (b,a) as at most one of the pair will appear in the resulting
    forest.

    Internally, the builder is going to connect the root atoms together to
    calculate a minimum spanning tree, before removing the connections
    between the root atoms.

    Deprecated as it is inefficient:
    Provides utility methods to perform incremental assembly of sets of bonded
    atoms ("connected components") into a valid KinForest. This involves
    determination of a spanning DAG for the component beginning from a
    specified root atom, initialization of reference frames for each atom, and
    concatenation of this component onto the KinForest.
    """

    kinforest: KinForest = attr.Factory(KinForest.root_node)

    @classmethod
    @convert_args
    def define_trees_with_prioritized_bonds(
        # def component_for_prioritized_bonds( -- old name
        cls,
        roots: Union[int, NDArray[numpy.int32][:]],
        potential_bonds: NDArray[numpy.int32][:, 2],  # old name: bonds
        prioritized_bonds: NDArray[numpy.int32][:, 2],  #
        max_to_atom_index: int = 0,
    ) -> ChildParentTuple:
        assert potential_bonds.shape[1] == prioritized_bonds.shape[1]
        if not isinstance(roots, numpy.ndarray):
            # create array from the single integer root input
            roots = numpy.array([roots], dtype=numpy.int32)

        all_atoms = numpy.concatenate(
            (
                numpy.concatenate((potential_bonds, prioritized_bonds), axis=0).reshape(
                    -1
                ),
                roots,
            ),
            axis=0,
        )
        if max_to_atom_index == 0:
            max_to_atom_index = all_atoms.max()

        weighted_bonds = (
            # All entries must be non-zero or sparse graph tools will entries (??)
            cls.bonds_to_csgraph(max_to_atom_index, potential_bonds, [-1])
            + cls.bonds_to_csgraph(max_to_atom_index, prioritized_bonds, [-0.125])
            + cls.faux_bonds_between_roots(
                max_to_atom_index=max_to_atom_index, roots=roots, weights=[-1]
            )
        )

        kfo_2_to, to_parents_in_kfo = cls.bonds_to_forest(roots, weighted_bonds)

        # validation:
        # Make sure that all of the target atoms that we know about were
        # reached during the breadth-first search. All atoms listed in the
        # potential_bonds and prioritized_bonds arrays should be listed in
        # the kfo_2_to array. Construct the to_2_kfo inverse mapping to make
        # sure.

        atom_seen = numpy.zeros((kfo_2_to.max() + 1,), dtype=numpy.int32)
        atom_seen[potential_bonds[:, 0]] = 1
        atom_seen[potential_bonds[:, 1]] = 1
        atom_seen[prioritized_bonds[:, 0]] = 1
        atom_seen[prioritized_bonds[:, 1]] = 1

        to_2_kfo = numpy.full(atom_seen.shape, -1, dtype=numpy.int32)
        to_2_kfo[kfo_2_to] = numpy.arange(kfo_2_to.shape[0], dtype=numpy.int32)

        assert numpy.all(to_2_kfo[atom_seen != 0] != -1)

        return kfo_2_to, to_parents_in_kfo

    @classmethod
    @convert_args
    def bonds_to_csgraph(
        cls,
        max_to_atom_index: int,
        bonds: NDArray[int][:, 2],
        weights: NDArray[float][:] = numpy.ones(1),  # noqa
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
            (weights, (bonds[:, 0], bonds[:, 1])),
            shape=(max_to_atom_index + 1, max_to_atom_index + 1),
        )
        return bonds_csr

    @classmethod
    @convert_args
    def faux_bonds_between_roots(
        cls, roots: NDArray[int][:], weights: NDArray[float][:], max_to_atom_index: int
    ) -> Union[sparse.spmatrix, int]:
        """Construct a csgraph with edges from the first root to all
        other roots, if there are other roots. Otherwise, return 0.
        The size of this csgraph should be the total number of
        atoms in all stacks.
        """
        if roots.shape[0] > 1:
            root_faux_bonds = numpy.full((roots.shape[0] - 1, 2), roots[0], dtype=int)
            root_faux_bonds[:, 0] = 0
            root_faux_bonds[:, 1] = roots[1:]
            return cls.bonds_to_csgraph(
                max_to_atom_index=max_to_atom_index,
                bonds=root_faux_bonds,
                weights=weights,
            )
        else:
            return 0

    @classmethod
    @validate_args
    # def bonds_to_connected_component(
    def bonds_to_forest(
        cls,
        roots: NDArray[numpy.int32][:],
        bonds: Union[NDArray[numpy.int32][:, 2], sparse.spmatrix],
    ) -> ChildParentTuple:
        """Build a forest-ordering of the atoms in the target system
        and return the target-order-to-kin-forest-order conversion array
        (aka the "ids") and the forest-defining index of the parent atom
        for each atom in the system.

        The "bonds" input can either be 1) a symmetric list of the directed edges
        (i.e. if the edge (a, b) is in the list then the edge (b, a) should also
        be in the list) in which case a deterministic but hard-to-predict
        depth-first traversal from the root nodes will create a selection of
        which edges to include in the tree, or 2) a sparse matrix representing
        weighted edges, in which case a minimum spanning tree will be used to
        select the set of edges that define a tree. In this latter case, all edges
        should have negative weights and the edges with the highest priority
        should have the largest magnitude.
        """

        if isinstance(bonds, numpy.ndarray):
            # Bonds are a non-prioritized set of edges, assume arbitrary
            # connectivity from the root is allowed.
            max_to_atom_index = max(bonds[:, 0].max(), bonds[:, 1].max())

            bond_graph = cls.bonds_to_csgraph(
                max_to_atom_index=max_to_atom_index, bonds=bonds
            ) + cls.faux_bonds_between_roots(
                roots=roots, weights=[1], max_to_atom_index=max_to_atom_index
            )
        else:
            # Sparse graph with per-bond weights, generate the minimum
            # spanning tree of the connections before traversing from
            # the root.
            bond_graph = csgraph.minimum_spanning_tree(bonds.tocsr())

        # Perform breadth first traversal from the root of the component
        # to generate the kinematic tree.
        kfo_2_to, preds = csgraph.breadth_first_order(
            bond_graph, roots[0], directed=False, return_predecessors=True
        )
        print("kfo_2_to", kfo_2_to)
        print("preds", preds)
        to_parents_in_kfo = preds[kfo_2_to]

        n_target_atoms = numpy.max(kfo_2_to) + 1
        to_2_kfo = invert_mapping(kfo_2_to, n_target_atoms)

        kfo_roots = to_2_kfo[roots]
        # make sure that all nodes were reached in the BFS traversal of
        # the graph; only the first root node should have a -9999 parent
        # and the parents of the other roots should all be the first root.
        assert to_parents_in_kfo[kfo_roots[0]] == -9999
        assert numpy.all(to_parents_in_kfo[kfo_roots[1:]] == roots[0])
        is_non_root = numpy.full(to_parents_in_kfo.shape, True, dtype=bool)
        is_non_root[kfo_roots] = False
        assert numpy.all(to_parents_in_kfo[is_non_root] >= 0)

        # to_parents_in_kfo[roots] = -9999

        return kfo_2_to.astype(int), to_parents_in_kfo.astype(int)

    @convert_args
    def append_connected_components(
        self,
        to_roots: NDArray[numpy.int32][:],
        kfo_2_to: NDArray[numpy.int32][:],
        to_parents_in_kfo: NDArray[numpy.int32][:],
        to_jump_nodes: Optional[NDArray[numpy.int32][:]] = None,
        component_parent=0,
    ):
        """After having created a forest, as identified by a depth-first
        travsersal of the input graph, we need to convert the target-order (to)
        indices into the kin-forest order (kfo). To do this we construct
        a target-order-2-kin-forest-order mapping which is simply the inverse
        mapping that "kfo_2_to" represents.
        """

        assert (
            kfo_2_to.shape == to_parents_in_kfo.shape
        ), "elements and parents must be of same length"

        assert len(kfo_2_to) >= 3, "Bonded ktree must have at least three entries"

        assert component_parent < len(self.kinforest.id) and component_parent >= -1
        if to_jump_nodes is None:
            to_jump_nodes = numpy.array([], dtype=numpy.int32)

        n_kf_atoms = len(kfo_2_to)
        n_target_atoms = numpy.max(kfo_2_to) + 1

        to_2_kfo = invert_mapping(kfo_2_to, n_target_atoms)

        kfo_roots = to_2_kfo[to_roots]
        kfo_jump_nodes = to_2_kfo[to_jump_nodes]
        kfo_parents = numpy.full((n_kf_atoms,), -1, dtype=numpy.int32)
        kfo_parents[to_parents_in_kfo >= 0] = to_2_kfo[
            to_parents_in_kfo[to_parents_in_kfo >= 0]
        ]

        # Root nodes are self-parented for the purpose of verifying the
        # connected component tree structure, will be rewritten with jump to
        # the component root frame.
        kfo_parents[kfo_roots] = kfo_roots

        kfo_grandparents = kfo_parents[kfo_parents]
        doftype = numpy.zeros(n_kf_atoms, numpy.int32)
        doftype[:] = NodeType.bond
        doftype[kfo_roots] = NodeType.jump
        doftype[kfo_jump_nodes] = NodeType.jump

        frame_x = numpy.zeros(n_kf_atoms, numpy.int32)
        frame_y = numpy.zeros(n_kf_atoms, numpy.int32)
        frame_z = numpy.zeros(n_kf_atoms, numpy.int32)

        kin_stree = KinForest.full(n_kf_atoms, -1)
        kin_stree.id[:] = torch.tensor(kfo_2_to)

        kin_start = len(self.kinforest)

        # Set the coordinate-frame-defining atoms of all the atoms in the
        # system as if they are all bonded atoms; the logic for roots and
        # jumps is more complex, so we will handle them separately.
        frame_x[:] = numpy.arange(n_kf_atoms, dtype=numpy.int32)
        frame_y[:] = kfo_parents
        frame_z[:] = kfo_grandparents

        # Now go and set the coordinate-frame-defining atoms for jumps
        fix_jump_nodes(
            kfo_parents, frame_x, frame_y, frame_z, kfo_roots, kfo_jump_nodes
        )

        # Now prep the arrays for concatenation with the existing kinforest.
        # The KinBuilder will continue to support the concatenation model, in
        # which additional trees for the same forest can be concatenated onto
        # the growing set of trees. In this model, the indices that were just
        # calculated for the parents, and the frame-defining atom indices need
        # to be offset by however many atoms there already are in the tree. In
        # the case that this is the first and only set of trees concatenated,
        # then the kin_start offset is still the non-zero value of 1, representing
        # the global root of the system located at the origin. The parents for
        # the roots will be reset with the index of this global root.
        # (The code pretends that the index of this global root can be anything
        # other than 0, but all of the other code downstream of here and upstream
        # of here relies on this root being atom 0).

        kfo_parents += kin_start
        kfo_parents[kfo_roots] = component_parent
        kfo_roots += kin_start
        frame_x += kin_start
        frame_y += kin_start
        frame_z += kin_start

        def _t(x):
            return torch.tensor(x, dtype=torch.int32)

        extended_kin_forest = KinForest(
            id=_t(kfo_2_to),
            # roots=_t(kfo_roots),
            doftype=_t(doftype),
            parent=_t(kfo_parents),
            frame_x=_t(frame_x),
            frame_y=_t(frame_y),
            frame_z=_t(frame_z),
        )

        return attr.evolve(self, kinforest=cat((self.kinforest, extended_kin_forest)))


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
    # nelts = parents.shape[0]
    n_children, child_list_span, child_list = get_children(parents)

    atom_is_jump = numpy.full(parents.shape, 0, dtype=numpy.int32)
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
            # of them need their frame definitions updated
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
