import torch
import numpy
import attrs

from collections import defaultdict
from numba import jit

import scipy.sparse as sparse
import scipy.sparse.csgraph as csgraph
from tmol.types.array import NDArray

from tmol.io.canonical_ordering import (
    default_canonical_ordering,
    default_packed_block_types,
    canonical_form_from_pdb,
)
from tmol.io.pose_stack_construction import pose_stack_from_canonical_form
from tmol.kinematics.fold_forest import EdgeType
from tmol.kinematics.scan_ordering import get_children


@jit
def get_branch_depth(parents):
    # modeled off get_children
    nelts = parents.shape[0]

    n_immediate_children = numpy.full(nelts, 0, dtype=numpy.int32)
    for i in range(nelts):
        p = parents[i]
        assert p <= i
        if p == i:
            continue
        n_immediate_children[p] += 1

    child_list = numpy.full(nelts, -1, dtype=numpy.int32)
    child_list_span = numpy.empty((nelts, 2), dtype=numpy.int32)

    child_list_span[0, 0] = 0
    child_list_span[0, 1] = n_immediate_children[0]
    for i in range(1, nelts):
        child_list_span[i, 0] = child_list_span[i - 1, 1]
        child_list_span[i, 1] = child_list_span[i, 0] + n_immediate_children[i]

    # Pass 3, fill the child list for each parent.
    # As we do this,


def jump_bt_atom(bt, spanning_tree):
    # CA! TEMP!!! Replace with code that connects up conn atom to down conn atom
    # in the spanning tree and chooses the midpoing along that path, but for now,
    # CA is atom 1.
    return 1


@attrs.define
class GenSegScanPaths:
    n_gens: NDArray[numpy.int64][:, :]  # n-input x n-output
    nodes_for_generation: NDArray[numpy.int64][
        :, :, :, :
    ]  # n-input x n-output x max-n-gen x max-n-ats-per-gen
    n_scans: NDArray[numpy.int64][:, :, :]
    scan_starts: NDArray[numpy.int64][:, :, :, :]
    scan_is_inter_block: NDArray[bool][:, :, :, :]
    scan_lengths: NDArray[numpy.int64][:, :, :, :]


def test_kin_tree_construction(ubq_pdb):
    torch_device = torch.device("cpu")

    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    canonical_form = canonical_form_from_pdb(co, ubq_pdb, torch_device)
    pose_stack = pose_stack_from_canonical_form(co, pbt, **canonical_form)

    # okay!
    # 1. let's create some annotations of the packed block types
    bt_list = [bt for bt in pbt.active_block_types if bt.name == "LEU"]

    # for bt in pbt.active_block_types:
    for bt in bt_list:
        n_conn = len(bt.connections)

        n_input_types = n_conn + 2  # n_conn + jump input + root "input"
        n_output_types = n_conn + 1  # n_conn + jump output

        n_gens = numpy.zeros((n_input_types, n_output_types), dtype=numpy.int64)
        nodes_for_generation = [
            [[] for _ in range(n_output_types)] for _2 in range(n_input_types)
        ]
        n_scans = [[[] for _ in range(n_output_types)] for _2 in range(n_input_types)]
        scan_starts = [
            [[] for _ in range(n_output_types)] for _2 in range(n_input_types)
        ]
        scan_is_inter_block = [
            [[] for _ in range(n_output_types)] for _2 in range(n_input_types)
        ]
        scan_lengths = [
            [[] for _ in range(n_output_types)] for _2 in range(n_input_types)
        ]

        def _bonds_to_csgraph(
            bonds: NDArray[int][:, 2], edge_weight: float
        ) -> sparse.csr_matrix:
            weights_array = numpy.full((1,), edge_weight, dtype=numpy.float32)
            weights = numpy.broadcast_to(weights_array, bonds[:, 0].shape)

            bonds_csr = sparse.csr_matrix(
                (weights, (bonds[:, 0], bonds[:, 1])),
                shape=(bt.n_atoms, bt.n_atoms),
            )
            return bonds_csr

        # create a bond graph and then we will create the prioritized edges
        # and all edges
        potential_bonds = _bonds_to_csgraph(bt.bond_indices, -1)
        print("potential bonds", potential_bonds)
        tor_atoms = [
            (uaids[1][0], uaids[2][0])
            for tor, uaids in bt.torsion_to_uaids.items()
            if uaids[1][0] >= 0 and uaids[2][0] >= 0
        ]
        if len(tor_atoms) == 0:
            tor_atoms = numpy.zeros((0, 2), dtype=numpy.int64)
        else:
            tor_atoms = numpy.array(tor_atoms)
        print("tor atoms:", tor_atoms)

        prioritized_bonds = _bonds_to_csgraph(tor_atoms, -0.125)
        print("prioritized bonds", prioritized_bonds)
        bond_graph = potential_bonds + prioritized_bonds
        bond_graph_spanning_tree = csgraph.minimum_spanning_tree(bond_graph.tocsr())

        mid_bt_atom = jump_bt_atom(bt, bond_graph_spanning_tree)

        is_conn_atom = numpy.zeros((bt.n_atoms,), dtype=bool)
        for i in range(n_conn):
            is_conn_atom[bt.ordered_connection_atoms[i]] = True

        for i in range(n_input_types):

            i_conn_atom = bt.ordered_connection_atoms[i] if i < n_conn else mid_bt_atom
            bfto_2_orig, preds = csgraph.breadth_first_order(
                bond_graph_spanning_tree,
                i_conn_atom,
                directed=False,
                return_predecessors=True,
            )
            print(bt.name, i, bfto_2_orig, preds)
            print([bt.atom_name(bfto_2_orig[bfs_ind]) for bfs_ind in range(bt.n_atoms)])
            for j in range(n_output_types):

                if i == j and i < n_conn:
                    # we cannot enter from one inter-residue connection point and then
                    # leave by that same inter-residue connection point unless we are
                    # building a jump
                    continue

                # now we start at the j_conn_atom and work backwards toward the root
                # which marks the first scan path for this block type: the "primary exit path"
                gen_scan_paths = defaultdict(list)

                j_conn_atom = (
                    bt.ordered_connection_atoms[j] if j < n_conn else mid_bt_atom
                )

                first_descendant = numpy.full((bt.n_atoms,), -9999, dtype=numpy.int64)
                is_on_primary_exit_path = numpy.zeros((bt.n_atoms,), dtype=bool)
                is_on_primary_exit_path[i_conn_atom] = True

                focused_atom = j_conn_atom
                primary_exit_scan_path = []
                while focused_atom != i_conn_atom:
                    print("exit path:", bt.atom_name(focused_atom))
                    is_on_primary_exit_path[focused_atom] = True
                    primary_exit_scan_path.append(focused_atom)
                    pred = preds[focused_atom]
                    first_descendant[pred] = focused_atom
                    focused_atom = pred
                primary_exit_scan_path.append(i_conn_atom)
                primary_exit_scan_path.reverse()
                # we need to prioritize exit paths of all stripes
                # in constructing the trees
                is_on_exit_path = is_on_primary_exit_path.copy()
                for k in range(n_conn):
                    if k == i or k == j:
                        continue  # truly unnecessary; nothing changes if I remove these two lines
                    is_on_exit_path[bt.ordered_connection_atoms[k]] = True

                print("primary_exit_scan_path:", primary_exit_scan_path)
                gen_scan_paths[0].append(primary_exit_scan_path)

                # Create a list of children for each atom.
                n_kids = numpy.zeros((bt.n_atoms,), dtype=numpy.int64)
                atom_kids = [[] for _ in range(bt.n_atoms)]
                for k in range(bt.n_atoms):
                    if preds[k] < 0:
                        assert (
                            k == i_conn_atom
                        ), f"bad predecesor for atom {k} in {bt.name}, {preds[k]}"
                        continue  # the root
                    n_kids[preds[k]] += 1
                    atom_kids[preds[k]].append(k)

                # now we label each node with its "generation depth" using a
                # leaf-to-root traversal perscribed by the original DFS, taking
                # into account the fact that priority must be given to
                # exit paths
                gen_depth = numpy.ones((bt.n_atoms,), dtype=numpy.int64)
                on_path_from_conn_to_i_conn_atom = numpy.zeros(
                    (bt.n_atoms,), dtype=bool
                )
                for k in range(bt.n_atoms - 1, -1, -1):
                    k_atom_ind = bfto_2_orig[k]
                    # print("recursing upwards", i, "i_conn atom", i_conn_atom, j, "j_conn_atom", j_conn_atom, k, k_atom_ind)
                    k_kids = atom_kids[k_atom_ind]
                    # print("kids:", k_kids)
                    if len(k_kids) == 0:
                        continue
                    # from here forward, we know that k_atom_ind has > 0 children

                    def gen_depth_given_first_descendant():
                        # first set the first_descendant for k_atom_ind
                        # then the logic is: we have to add one to the
                        # gen-depth of every child but the first descendant
                        # which we get "for free"
                        # print(f"atom {bt.atom_name(k_atom_ind)} with first descendant {bt.atom_name(first_descendant[k_atom_ind]) if first_descendant[k_atom_ind] >= 0 else 'None'} and depth {gen_depth[first_descendant[k_atom_ind]] if first_descendant[k_atom_ind] >= 0 else -9999}")
                        return max(
                            [
                                (
                                    gen_depth[k_kid] + 1
                                    if k_kid != first_descendant[k_atom_ind]
                                    else gen_depth[k_kid]
                                )
                                for k_kid in k_kids
                            ]
                        )

                    if is_on_primary_exit_path[k_atom_ind]:
                        # in this case, the first_descendant for this atom
                        # has already been decided
                        # print("on exit path:", bt.atom_name(k_atom_ind), first_descendant[k_atom_ind], is_conn_atom[k_atom_ind])
                        if k_atom_ind == j_conn_atom:
                            # the first descendent is the atom on the next residue to which
                            # this residue is connected
                            gen_depth[k_atom_ind] = (
                                max([gen_depth[l] for l in k_kids]) + 1
                            )
                        else:
                            # first_descendant is already determined for this atom
                            gen_depth[k_atom_ind] = gen_depth_given_first_descendant()
                    else:

                        if is_conn_atom[k_atom_ind]:
                            # in this case, "the" connection (there can possibly be more than one!)
                            # will be the first child and the other descendants will be second children
                            # we save the gen depth, but when calculating the gen depth of the
                            # fold-forest, if this residue is at the upstream end of an edge, then
                            # its depth will have to be calculated as the min gen-depth of the
                            # intra-residue bits and the gen-depth of the nodes downstream of it.
                            gen_depth[k_atom_ind] = (
                                max([gen_depth[l] for l in k_kids]) + 1
                            )
                        else:
                            # most-common case: an atom not on the primary-exit path, and that isn't
                            # itself a conn atom.
                            # First we ask: are we on one or more exit paths?
                            # NOTE: this just chooses the first exit path atom it encounters
                            # as the first descendant and so I pause and think: if we have
                            # a block type with 4 inter-residue connections where the fold
                            # forest branches at this residue, then the algorithm for constructing
                            # the most number-of-generations-efficient KinForest here is going
                            # will fail: we are treating all exit paths out of this residue
                            # as interchangable and we might say connection c vs c' should
                            # be first in a case where c' leads to more generations than c.
                            # The case I am designing for here is: there's a jump that has
                            # landed at a beta-amino acid's CA atom and there are exit paths
                            # through the N- and C-terminal ends of the residue and if the
                            # primary exit path is the C-term, then the N-term exit path should
                            # still have priority over the side-chain path.
                            #
                            #         R
                            #         |
                            # ...     CB    C
                            #     \ /   \  / \
                            #      N      CA   ...
                            #
                            # The path starting at CB should go towards N and not towards R.
                            # If we are only dealing with polymeric residues that have an
                            # up- and a down connection that that's it (e.g. nucleic acids),
                            # then this algorithm will still produce optimal KinForests.

                            for kid in k_kids:
                                if is_on_exit_path[kid]:
                                    first_descendant[k_atom_ind] = kid
                                    is_on_exit_path[k_atom_ind] = True

                            if not is_on_exit_path[k_atom_ind]:
                                # which should be the first descendant? the one with the greatest gen depth
                                first_descendant[k_atom_ind] = k_kids[
                                    numpy.argmax(
                                        numpy.array([gen_depth[kid] for kid in k_kids])
                                    )
                                ]
                            gen_depth[k_atom_ind] = gen_depth_given_first_descendant()
                            # print("gen_depth", bt.atom_name(k_atom_ind), "d:", gen_depth[k_atom_ind])
                # print("gen_depth", gen_depth)

                # OKAY!
                # now we have paths rooted at each node up to the root
                # we need to turn these paths into scan paths
                processed_node_into_scan_path = is_on_primary_exit_path.copy()
                gen_to_build_atom = numpy.full((bt.n_atoms,), -1, dtype=numpy.int64)
                gen_to_build_atom[processed_node_into_scan_path] = 0
                print("gen depth", gen_depth)
                print("starting bfs:", processed_node_into_scan_path)
                for k in range(bt.n_atoms):
                    k_atom_ind = bfto_2_orig[k]
                    if processed_node_into_scan_path[k_atom_ind]:
                        continue

                    # if we arrive here, that means k_atom_ind is the root of a
                    # new scan path
                    path = []
                    # we have already processed the first scan path
                    # from the entrace-point atom to the first exit-point atom
                    assert k_atom_ind != i_conn_atom
                    # put the parent of this new root at the beginning of
                    # the scan path
                    path.append(preds[k_atom_ind])
                    focused_atom = k_atom_ind

                    gen_to_build_atom[focused_atom] = (
                        gen_to_build_atom[preds[focused_atom]] + 1
                    )
                    print(
                        f"gen to build {bt.atom_name(focused_atom)} from {bt.atom_name(preds[focused_atom])}",
                        f"with gen {gen_to_build_atom[focused_atom]}",
                    )
                    while focused_atom >= 0:
                        path.append(focused_atom)
                        processed_node_into_scan_path[focused_atom] = True
                        focused_atom = first_descendant[focused_atom]
                        if focused_atom >= 0:
                            gen_to_build_atom[focused_atom] = gen_to_build_atom[
                                preds[focused_atom]
                            ]
                    if is_on_exit_path[k_atom_ind]:
                        gen_scan_paths[gen_to_build_atom[k_atom_ind]].insert(0, path)
                    else:
                        gen_scan_paths[gen_to_build_atom[k_atom_ind]].append(path)
                # Now we need to assemble the scan paths in a compact way:
                print("gen scan paths", gen_scan_paths)

                ij_n_gens = gen_depth[i_conn_atom]
                print("ij_n_gens", i, j, ij_n_gens)
                ij_n_scans = [len(gen_scan_paths[k]) for k in range(ij_n_gens)]
                print("ij_n_scans", i, j, ij_n_scans)
                ij_scan_starts = [[0] * ij_n_scans[k] for k in range(ij_n_gens)]
                print("ij_scan_starts", i, j, ij_scan_starts)
                ij_scan_lengths = [
                    [len(gen_scan_paths[k][l]) for l in range(len(gen_scan_paths[k]))]
                    for k in range(ij_n_gens)
                ]
                print("ij_scan_lengths", i, j, ij_scan_lengths)
                # ij_n_nodes_for_gen =
                ij_n_nodes_for_gen = [
                    sum(len(path) for path in gen_scan_paths[k])
                    for k in range(ij_n_gens)
                ]
                print("ij_n_nodes_for_gen", ij_n_nodes_for_gen)


def test_decide_scan_paths_for_foldforest(ubq_pdb):
    torch_device = torch.device("cpu")

    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    canonical_form = canonical_form_from_pdb(
        co, ubq_pdb, torch_device, residue_start=0, residue_end=10
    )
    pose_stack = pose_stack_from_canonical_form(co, pbt, **canonical_form)

    fold
