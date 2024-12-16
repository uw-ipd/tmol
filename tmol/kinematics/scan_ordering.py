import attr
import numpy
import torch

from .datatypes import (
    KinForest,
    KinForestScanData,
    KinematicModuleData,
    BTGenerationalSegScanPathSegs,
    PBTGenerationalSegScanPathSegs,
)

from numba import jit
from tmol.types.array import NDArray
from tmol.types.torch import Tensor
from tmol.types.tensor import TensorGroup
from tmol.types.attrs import ConvertAttrs, ValidateAttrs

from tmol.types.functional import validate_args

from collections import defaultdict
from numba import jit

import scipy.sparse as sparse
import scipy.sparse.csgraph as csgraph
from tmol.types.torch import Tensor

from tmol.io.canonical_ordering import (
    default_canonical_ordering,
    default_packed_block_types,
    canonical_form_from_pdb,
)
from tmol.io.pose_stack_construction import pose_stack_from_canonical_form
from tmol.kinematics.datatypes import NodeType
from tmol.kinematics.fold_forest import EdgeType
from .check_fold_forest import validate_fold_forest

# from tmol.kinematics.scan_ordering import get_children
from tmol.kinematics.compiled import inverse_kin, forward_kin_op

from tmol.utility.tensor.common_operations import exclusive_cumsum1d


@jit(nopython=True)
def get_children(parents):
    nelts = parents.shape[0]

    # Pass 1, count number of children for each parent.
    n_immediate_children = numpy.full(nelts, 0, dtype=numpy.int32)
    for i in range(nelts):
        p = parents[i]
        assert p <= i, "Invalid kinematic tree ordering, parent index >= child index."
        if p == i:  # root
            continue
        n_immediate_children[p] += 1

    # Pass 2, mark the span of each parent in the child node list.
    child_list = numpy.full(nelts, -1, dtype=numpy.int32)
    child_list_span = numpy.empty((nelts, 2), dtype=numpy.int32)

    child_list_span[0, 0] = 0
    child_list_span[0, 1] = n_immediate_children[0]
    for i in range(1, nelts):
        child_list_span[i, 0] = child_list_span[i - 1, 1]
        child_list_span[i, 1] = child_list_span[i, 0] + n_immediate_children[i]

    # Pass 3, fill the child list for each parent.
    # As we do this, sum total # of descendents at each node, used to
    #   prioritize scan ordering in DFS
    n_descendents = numpy.ones(nelts, dtype=numpy.int32)
    for i in range(nelts - 1, 0, -1):
        p = parents[i]
        if p == i:  # root
            continue
        n_descendents[p] += n_descendents[i]
        child_list[child_list_span[p, 0] + n_immediate_children[p] - 1] = i
        n_immediate_children[p] -= 1

    return n_descendents, child_list_span, child_list


# jitted scan operation
# inputs:
#    parents - the "parents" array from the kinematic tree
#              assumes proper ordering: child index > parent index
#    roots - the initial root nodes.  The code will efficiently support
#            a graph with many disconnected components
# returns:
#    nodes - a 1D array with the node indices of each scan, concatenated
#    scanStarts - a 1D index array in 'nodes' where each individual scan begins
#    genStarts - an N x 2 array giving indices where each gen begins
#         genStarts[:,0] indexes nodes
#         genStarts[:,1] indexes scanStarts
@jit(nopython=True)
def get_scans(parents, roots):
    """Partitioning of a tree into linear subpaths.

    The tree is stored as a list of pointers to parent indices, with an
    arbitrary set of roots.  Each root implies a connected subtree.
    Each index i has a parent index (p_i) "higher" in the tree (p_i < i).
    The tree structure is fully defined by parent pointers, an index array
    (parent) of len(tree_size) where parent[i] == i for all root nodes.

    The source tree implies a per-node child list of length >= 0, defined by
    all nodes for which the given node is a parent. Nodes with no children are
    leaves.

    Scan paths cuts the tree into linear paths, where each non-leaf node (i)
    has a *single* child (c_i) "lower" in the tree (i < c_i). The path
    structure is fully defined by child pointers, an index array (subpath_child)
    of subpath_child[i] == c_i (non-leaves) or subpath_child[i] = -1 (leaves).

    The path partitioning implies "subpath roots", the set of nodes which are
    *not* the subpath child of their parent (subpath_child[parent[i]] != i).

    Each path in the tree is labeled with a depth: a path with depth
    i may depend on the values computed for atoms with depths 0..i-1.
    All of the paths of the same depth can be processed in a single
    kernel execution with segmented scan.
    """

    nelts = parents.shape[0]
    n_descendents, child_list_span, child_list = get_children(parents)

    # scan storage - allocate upper bound for 4-connected graph
    # scan indices are emitted as a 1D array of nodes ('scans') with:
    #   scanStarts: indices in 'scans' where individual scans start
    #   genStarts: indices in 'scanStarts' where generations start
    nodes = numpy.full(4 * nelts, -1, dtype=numpy.int32)
    scanStarts = numpy.full(nelts, -1, dtype=numpy.int32)
    genStarts = numpy.full((nelts, 2), -1, dtype=numpy.int32)

    # curr idx in each array
    genidx, scanidx, nodeidx = 0, 0, 0

    # store the active pool we are expanding
    activeFront = numpy.full(4 * nelts, -1, dtype=numpy.int32)
    nActiveFront = roots.shape[0]
    activeFront[:nActiveFront] = roots

    # DFS traversal through forest-of-rooted-trees drawing generational scan
    # paths. Each pass through the DFS search follows a set of scan paths
    # beginning from the current set of roots, following nodes to the full
    # depth of the scan path. At each node the scan is extended through the
    # child with the most descendents and the node is added to the generation
    # n+1 roots, where it will root 0-or-more scans in the n+1 generation
    # passing through its additional children.
    #
    # In typical kinematic trees this will minimize the total number of scan
    # generations by tracing a long backbone path with many short side paths.
    marked = numpy.zeros(nelts, dtype=numpy.int32)
    marked[0] = 1
    while not numpy.all(marked):
        genStarts[genidx, :] = [nodeidx, scanidx]
        for i in range(nActiveFront):
            currRoot = activeFront[i]

            # Active front nodes are members of an existing scan passing through
            # to one child (or are gen 0) and may root 1-or-more scans through
            # any additional children.
            for j in range(child_list_span[currRoot, 0], child_list_span[currRoot, 1]):
                expandedNode = child_list[j]

                if marked[expandedNode] != 0:
                    # Child node has already been 'scanned' in a prev
                    # generation and is not the first child of another scan.
                    continue

                # add the node as the start of a new scan
                # -> numbering is w.r.t. generation (hence subtracting gen start)
                scanStarts[scanidx] = nodeidx - genStarts[genidx, 0]
                scanidx += 1

                # Set root as start of the scan.
                nodes[nodeidx] = currRoot
                nodeidx += 1

                while True:
                    # Extend scan path into selected child and mark it.
                    marked[expandedNode] = 1
                    nodes[nodeidx] = expandedNode
                    nodeidx += 1

                    if (
                        child_list_span[expandedNode, 0]
                        == child_list_span[expandedNode, 1]
                    ):
                        # At a leaf node, scan path terminates.
                        break
                    else:
                        # Extend path through the child with
                        #  greatest number of descendants.
                        nextExtension = child_list[child_list_span[expandedNode, 0]]
                        for k in range(
                            child_list_span[expandedNode, 0] + 1,
                            child_list_span[expandedNode, 1],
                        ):
                            candidate = child_list[k]
                            if n_descendents[candidate] > n_descendents[nextExtension]:
                                nextExtension = candidate

                        expandedNode = nextExtension

        # Mark any node scanned in this generation as a potential root.
        # Old roots and nodes with 1 child will be filtered during expansion.
        if genStarts[genidx, 1] < scanidx:
            lastgenScan0 = genStarts[genidx, 0]
            activeFront.fill(-1)
            nActiveFront = nodeidx - lastgenScan0
            activeFront[:nActiveFront] = nodes[lastgenScan0:nodeidx]

        # next generation
        genidx += 1

    # Pad genStarts by 1 to make downstream code cleaner
    genStarts[genidx, :] = [nodeidx, scanidx]
    genidx += 1

    return nodes[:nodeidx], scanStarts[:scanidx], genStarts[:genidx, :]


@attr.s(auto_attribs=True, frozen=True)
class KinForestScanOrdering(ValidateAttrs):
    """Scan plans for parallel kinematic operations.

    The KinForestScanOrdering class divides the tree into a set of paths. Along
    each path is a continuous chain of atoms that either (1) require their
    coordinate frames computed as a cumulative product of homogeneous
    transforms for the coordinate update algorithm, or (2) require the
    cumulative sum of their f1f2 vectors for the derivative calculation
    algorithm. In both cases, these paths can be processed efficiently
    on the GPU using an algorithm called "scan" and batches of these paths
    can be processed at once in a variant called "segmented scan."

    The same set of paths is used for both the refold algorithm and
    the derivative summation; the refold algorithm starts at path
    roots and multiplies homogeneous transforms towards the leaves.
    The derivative summation algorithm starts at the leaves and sums
    upwards towards the roots.

    To accomplish this, the GPUKinForestReordering class reorders the atoms from
    the original KinForest order ("ko") where atoms are known by their
    kinforest-index ("ki") into 1) their refold order ("ro") where atoms are
    known by their refold index ("ri") and 2) their deriv-sum order ("dso")
    where atoms are known by their deriv-sum index.

    Each scan operation is performed as a series of depth-based "generations",
    in which the scans at generation n are *only* dependent on the results of
    scans in generations [0...n-1] inclusive. In the forward-scan generations
    are ordered from the kinematic root, and scan segments begin with a value
    derived from the result of a generation [0...n-1] scan. In the
    backward-scan generations are ordered from kinematic leaves, and scan
    segments pull summation results from multiple generation [0...n-1] scans.

    For further details on parallel segmented scan operations see:

    * Mark Harris, "Parallel Prefix Sum with CUDA."
      GPU Gems 3, Nvidia Corporation
      https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html
      http://developer.download.nvidia.com/
      compute/cuda/2_2/sdk/website/projects/scan/doc/scan.pdf

    * Sengupta, Shubhabrata, et al. "Scan primitives for GPU computing."
      Graphics hardware. Vol. 2007. 2007.
      http://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Sengupta07.pdf

    * Sean Baxter, "moderngpu"
      http://moderngpu.github.io/moderngpu
    """

    kinforest_cache_key = "__KinForestScanOrdering_cache__"

    forward_scan_paths: KinForestScanData
    backward_scan_paths: KinForestScanData

    @classmethod
    @validate_args
    def for_kinforest(cls, kinforest):
        """Calculate and cache refold ordering over kinforest

        KinForest data structure is frozen; so it is safe to cache the gpu scan
        ordering for a single object. Store as a private property of the input
        kinforest, lifetime of the cache will then be managed via the target
        object.
        ."""

        if not hasattr(kinforest, cls.kinforest_cache_key):
            object.__setattr__(
                kinforest,
                cls.kinforest_cache_key,
                cls.calculate_from_kinforest(kinforest),
            )

        return getattr(kinforest, cls.kinforest_cache_key)

    @classmethod
    @validate_args
    def calculate_from_kinforest(cls, kinforest: KinForest):
        """Setup for operations over KinForest.
        ``device`` is inferred from kinforest tensor device.
        """

        nodes, scanStarts, genStarts = get_scans(
            kinforest.parent.cpu().numpy(), numpy.array([0])
        )
        forward_scan_paths = KinForestScanData(
            nodes=torch.from_numpy(nodes).to(device=kinforest.parent.device),
            scans=torch.from_numpy(scanStarts).to(device=kinforest.parent.device),
            gens=torch.from_numpy(genStarts),
        )  # keep gens on CPU!

        # reverse the scan paths for derivative scans
        nodesR = numpy.ascontiguousarray(numpy.flipud(nodes))

        genStartsR = numpy.zeros_like(genStarts)
        genStartsR[1:, 0] = nodes.shape[0] - numpy.flipud(genStarts[:-1, 0])
        genStartsR[1:, 1] = scanStarts.shape[0] - numpy.flipud(genStarts[:-1, 1])

        # perhaps this can be simplified? (reversing scan indices)
        ngens = genStarts.shape[0] - 1
        scanStartsR = numpy.zeros_like(scanStarts)
        for i in range(ngens):
            genstart = genStartsR[i, 1]
            genstop = genStartsR[i + 1, 1]
            nodes_i = genStartsR[i + 1, 0] - genStartsR[i, 0]
            scan_i = nodes_i - numpy.ascontiguousarray(
                numpy.flipud(
                    scanStarts[
                        (scanStarts.shape[0] - genstop) : (
                            scanStarts.shape[0] - genstart
                        )
                    ]
                )
            )
            scanStartsR[genstart] = 0
            scanStartsR[(genstart + 1) : genstop] = scan_i[:-1]

        backward_scan_paths = KinForestScanData(
            nodes=torch.from_numpy(nodesR).to(device=kinforest.parent.device),
            scans=torch.from_numpy(scanStartsR).to(device=kinforest.parent.device),
            gens=torch.from_numpy(genStartsR),
        )  # keep gens on CPU!

        return KinForestScanOrdering(
            forward_scan_paths=forward_scan_paths,
            backward_scan_paths=backward_scan_paths,
        )


def construct_kin_module_data_for_pose(
    pose_stack,
    fold_forest_edges,
):
    from tmol.kinematics.compiled.compiled_ops import (
        calculate_ff_edge_delays,
        get_block_parent_connectivity_from_toposort,
        get_kinforest_scans_from_stencils2,
        get_kfo_indices_for_atoms,
        get_kfo_atom_parents,
        get_children,
        get_id_and_frame_xyz,
    )

    # validate_fold_forest()

    device = pose_stack.device
    pbt = pose_stack.packed_block_types
    _annotate_packed_block_type_with_gen_scan_path_segs(pbt)
    pbt_gssps = pbt.gen_seg_scan_path_segs

    ff_edges_cpu = fold_forest_edges.cpu()
    ff_edges_device = fold_forest_edges.to(device)

    # print("1")
    result = calculate_ff_edge_delays(
        pose_stack.block_coord_offset,  # TView<Int, 2, D> pose_stack_block_coord_offset,         // P x L
        pose_stack.block_type_ind,  # TView<Int, 2, D> pose_stack_block_type,                 // x - P x L
        ff_edges_cpu,  # TView<Int, 3, CPU> ff_edges_cpu,                        // y - P x E x 4 -- 0: type, 1: start, 2: stop, 3: jump ind
        pbt_gssps.scan_path_seg_that_builds_output_conn,  # TVIew<Int, 5, D> block_type_kts_conn_info,              // y - T x I x O x C x 2 -- 2 is for gen (0) and scan (1)
        pbt_gssps.nodes_for_gen,  # TView<Int, 5, D> block_type_nodes_for_gens,             // y - T x I x O x G x N
        pbt_gssps.scan_path_seg_starts,  # TView<Int, 5, D> block_type_scan_path_starts            // y - T x I x O x G x S
    )
    # print("2")

    (
        dfs_order_of_ff_edges,
        n_ff_edges,
        ff_edge_parent,
        first_ff_edge_for_block,
        pose_stack_ff_parent,
        max_gen_depth_of_ff_edge,
        first_child_of_ff_edge,
        delay_for_edge,
        toposort_index_for_edge,
    ) = tuple(x.to(device) for x in result)

    # print("dfs_order_of_ff_edges", dfs_order_of_ff_edges)
    # print("ff_edge_parent", ff_edge_parent)
    # print("first_child_of_ff_edge", first_child_of_ff_edge)
    # print("first_ff_edge_for_block", first_ff_edge_for_block)
    # print("3")

    pose_stack_block_in_and_first_out = get_block_parent_connectivity_from_toposort(
        pose_stack.block_type_ind,
        pose_stack.inter_residue_connections,
        pose_stack_ff_parent,
        dfs_order_of_ff_edges,
        n_ff_edges,
        ff_edges_device,
        first_ff_edge_for_block,
        first_child_of_ff_edge,
        delay_for_edge,
        toposort_index_for_edge,
        pbt.n_conn,
        pbt.polymeric_conn_inds,
    )

    # print("4")
    (block_kfo_offset, kfo_2_orig_mapping, atom_kfo_index) = get_kfo_indices_for_atoms(
        pose_stack.block_coord_offset,
        pose_stack.block_type_ind,
        pbt.n_atoms,
        pbt.atom_is_real,
    )

    # print("5")
    kfo_atom_parents, kfo_atom_grandparents = get_kfo_atom_parents(
        pose_stack.block_type_ind,
        pose_stack.inter_residue_connections,
        pose_stack_ff_parent,
        # ff_conn_to_parent,
        pose_stack_block_in_and_first_out,
        pbt_gssps.parents,
        kfo_2_orig_mapping,
        atom_kfo_index,
        pbt_gssps.jump_atom,
        pbt.n_conn,
        pbt.conn_atom,
    )

    # print("6")
    n_children, child_list_span, child_list, is_atom_jump = get_children(
        pose_stack.block_type_ind,
        pose_stack_block_in_and_first_out,
        kfo_2_orig_mapping,
        kfo_atom_parents,
        pbt.n_conn,
    )

    # print("7")
    id, frame_x, frame_y, frame_z = get_id_and_frame_xyz(
        pose_stack.coords.shape[1],
        pose_stack.block_coord_offset,
        kfo_2_orig_mapping,
        kfo_atom_parents,
        child_list_span,
        child_list,
        is_atom_jump,
    )

    # print("8")
    nodes_fw, scans_fw, gens_fw, nodes_bw, scans_bw, gens_bw = (
        get_kinforest_scans_from_stencils2(
            pose_stack.max_n_atoms,
            pose_stack.block_coord_offset,
            pose_stack.block_type_ind,
            pose_stack.inter_residue_connections,
            ff_edges_device,
            torch.max(delay_for_edge).item(),
            delay_for_edge,
            toposort_index_for_edge,
            first_ff_edge_for_block,
            pose_stack_ff_parent,
            pose_stack_block_in_and_first_out,
            pbt_gssps.parents,
            kfo_2_orig_mapping,
            atom_kfo_index,
            pbt_gssps.jump_atom,
            pbt.n_conn,
            pbt.polymeric_conn_inds,
            pbt_gssps.n_gens,
            pbt_gssps.scan_path_seg_that_builds_output_conn,
            pbt_gssps.nodes_for_gen,
            pbt_gssps.n_scan_path_segs,
            pbt_gssps.scan_path_seg_starts,
            pbt_gssps.scan_path_seg_is_real,
            pbt_gssps.scan_path_seg_is_inter_block,
            pbt_gssps.scan_path_seg_lengths,
        )
    )

    # print("9")
    # This feels so clunky after all that slick C++
    is_res_real = pose_stack.block_type_ind != -1
    is_atom_real = pbt.atom_is_real[pose_stack.block_type_ind[is_res_real]]

    block_atom_dof_type = pbt_gssps.dof_type[
        pose_stack.block_type_ind[is_res_real],
        pose_stack_block_in_and_first_out[is_res_real][:, 0],
    ]
    doftype = torch.zeros((id.shape[0],), dtype=torch.int32, device=id.device)
    doftype[1:] = block_atom_dof_type[is_atom_real]

    return KinematicModuleData(
        forest=KinForest(
            id=id,
            doftype=doftype,
            parent=kfo_atom_parents,
            frame_x=frame_x,
            frame_y=frame_y,
            frame_z=frame_z,
        ),
        scan_data_fw=KinForestScanData(
            nodes=nodes_fw,
            scans=scans_fw,
            gens=gens_fw.cpu(),
        ),
        scan_data_bw=KinForestScanData(
            nodes=nodes_bw,
            scans=scans_bw,
            gens=gens_bw.cpu(),
        ),
    )


def jump_atom_for_bt(bt):
    """Return the index of the atom that will be jumped to or jumped from"""
    # TEMP: CA if CA is present; ow, atom 0
    return bt.atom_to_idx["CA"] if "CA" in bt.atom_names_set else 0


# TO DO: jit this!
def _annotate_block_type_with_gen_scan_path_segs(bt):
    if hasattr(bt, "gen_seg_scan_path_segs"):
        return
    n_conn = len(bt.connections)

    n_input_types = n_conn + 2  # n_conn + jump input + root "input"
    n_output_types = n_conn + 2  # n_conn + jump output + no output at all

    n_gens = numpy.zeros((n_input_types, n_output_types), dtype=numpy.int64)
    nodes_for_generation = [
        [[] for _ in range(n_output_types)] for _2 in range(n_input_types)
    ]
    n_scan_path_segs = [
        [[] for _ in range(n_output_types)] for _2 in range(n_input_types)
    ]
    scan_path_seg_starts = [
        [[] for _ in range(n_output_types)] for _2 in range(n_input_types)
    ]
    scan_path_seg_is_inter_block = [
        [[] for _ in range(n_output_types)] for _2 in range(n_input_types)
    ]
    scan_path_seg_lengths = [
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
    # print("potential bonds", potential_bonds)
    tor_atoms = [
        (uaids[1][0], uaids[2][0])
        for tor, uaids in bt.torsion_to_uaids.items()
        if uaids[1][0] >= 0 and uaids[2][0] >= 0
    ]
    if len(tor_atoms) == 0:
        tor_atoms = numpy.zeros((0, 2), dtype=numpy.int64)
    else:
        tor_atoms = numpy.array(tor_atoms)
    # print("tor atoms:", tor_atoms)

    prioritized_bonds = _bonds_to_csgraph(tor_atoms, -0.125)
    # print("prioritized bonds", prioritized_bonds)
    bond_graph = potential_bonds + prioritized_bonds
    bond_graph_spanning_tree = csgraph.minimum_spanning_tree(bond_graph.tocsr())

    mid_bt_atom = jump_atom_for_bt(bt)

    # As we are iterating across atoms, we need to keep track of which atoms
    # are bridges to other resiudes, so write down the reverse mapping from
    # atom index to the inter-residue connection index
    is_conn_atom = numpy.zeros((bt.n_atoms,), dtype=bool)
    conn_ind_for_atom = numpy.full((bt.n_atoms,), -1, dtype=numpy.int64)
    for i in range(n_conn):
        is_conn_atom[bt.ordered_connection_atoms[i]] = True
        conn_ind_for_atom[bt.ordered_connection_atoms[i]] = i

    scan_path_segment_data = {}
    parents = numpy.full((n_input_types, bt.n_atoms), -1, dtype=numpy.int64)
    input_conn_atom = numpy.zeros((n_input_types,), dtype=numpy.int64)
    dof_type = numpy.full(
        (
            n_input_types,
            bt.n_atoms,
        ),
        NodeType.bond,
        dtype=numpy.int64,
    )
    for i in range(n_input_types):

        i_conn_atom = bt.ordered_connection_atoms[i] if i < n_conn else mid_bt_atom
        input_conn_atom[i] = i_conn_atom
        bfto_2_orig, preds = csgraph.breadth_first_order(
            bond_graph_spanning_tree,
            i_conn_atom,
            directed=False,
            return_predecessors=True,
        )
        parents[i, :] = preds
        if i >= n_conn:
            dof_type[i, i_conn_atom] = NodeType.jump
        # Now, the parent of the i_conn_atom comes from the previous residue, so we will
        # need to fix this atom when we are hooking the blocks together. For now, leave
        # it as -9999 (which is what csgraph labels it as) so that we can tell if we have
        # not corrected this parent index later on.
        # print(bt.name, i, bfto_2_orig, preds)
        # print([bt.atom_name(bfto_2_orig[bfs_ind]) for bfs_ind in range(bt.n_atoms)])
        for j in range(n_output_types):
            target = False
            # if bt.name == "ILE" and i == 3 and j == 2:
            #     target = True
            #     print(bt.name, i, j)
            if i == j and i < n_conn:
                # we cannot enter from one inter-residue connection point and then
                # leave by that same inter-residue connection point unless we are
                # building a jump
                continue

            # we will generate a list of scan-path segments for each generation
            # and as part of this building process, we will track which scan-
            # path segments are exit paths to other blocks.
            gen_scan_path_segments = defaultdict(list)
            atom_rooting_scan_path_segment_for_interres_conn = numpy.full(
                (n_conn,), -1, dtype=numpy.int64
            )
            interres_conn_scan_path_segment_rooted_by_atom = numpy.full(
                (bt.n_atoms,), -1, dtype=numpy.int64
            )
            scan_path_segment_building_interres_conn = numpy.full(
                (n_conn,), -1, dtype=numpy.int64
            )
            gen_of_scan_path_segment_building_interres_conn = numpy.full(
                (n_conn,), -1, dtype=numpy.int64
            )

            is_on_primary_exit_sp_seg = numpy.zeros((bt.n_atoms,), dtype=bool)
            if j <= n_conn:
                # Case 1: we have a designated exit from this block type to
                # the next block in the kinematic tree.
                #
                # Start at the j_conn_atom and work backwards toward the root,
                # which marks the first scan-path segment for this block type:
                # the "primary exit scan-path segment"
                j_conn_atom = (
                    bt.ordered_connection_atoms[j] if j < n_conn else mid_bt_atom
                )

                first_descendant = numpy.full((bt.n_atoms,), -9999, dtype=numpy.int64)
                is_on_primary_exit_sp_seg[i_conn_atom] = True

                focused_atom = j_conn_atom
                primary_exit_scan_path_segment = []
                while focused_atom != i_conn_atom:
                    # print("exit path:", bt.atom_name(focused_atom))
                    is_on_primary_exit_sp_seg[focused_atom] = True
                    primary_exit_scan_path_segment.append(focused_atom)
                    pred = preds[focused_atom]
                    first_descendant[pred] = focused_atom
                    focused_atom = pred
                primary_exit_scan_path_segment.append(i_conn_atom)
                primary_exit_scan_path_segment.reverse()
                # we need to prioritize exit scan-path segments of all stripes
                # in constructing the trees
                is_on_exit_sp_segment = is_on_primary_exit_sp_seg.copy()
                for k in range(n_conn):
                    if k == i or k == j:
                        continue  # truly unnecessary; nothing changes if I remove these two lines
                    k_conn_atom = bt.ordered_connection_atoms[k]
                    is_on_exit_sp_segment[k_conn_atom] = True
                    atom_rooting_scan_path_segment_for_interres_conn[k] = k_conn_atom
                    interres_conn_scan_path_segment_rooted_by_atom[k_conn_atom] = k
                # print("primary_exit_scan_path_segment:", primary_exit_scan_path_segment)
                gen_scan_path_segments[0].append(primary_exit_scan_path_segment)
                # our first exit scan path segment: keep track of the gen/scan-path-seg indices
                # for exit scan-path segments using inter-residue connections. We don't have
                # to worry about scan paths that exit by jump or that dont exit.
                if j < n_conn:
                    gen_of_scan_path_segment_building_interres_conn[j] = 0
                    scan_path_segment_building_interres_conn[j] = 0
            else:
                # Case 2: A leaf node of the kinematic tree.
                # we will not be exiting from any connection point.
                # NOTE: this is an inter-block segment
                primary_exit_scan_path_segment = []
                is_on_exit_sp_segment = numpy.zeros((bt.n_atoms,), dtype=bool)
                pass

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
            # exit scan-path segments -- that is, we must describe exit spsegs
            # being the first children of their parents and the other children
            # as being younger siblings.
            gen_depth = numpy.ones((bt.n_atoms,), dtype=numpy.int64)
            on_sp_seg_from_conn_to_i_conn_atom = numpy.zeros((bt.n_atoms,), dtype=bool)
            for k in range(bt.n_atoms - 1, -1, -1):
                k_atom_ind = bfto_2_orig[k]
                # if target:
                #     print(
                #         "recursing upwards",
                #         i,
                #         "i_conn atom",
                #         i_conn_atom,
                #         j,
                #         "j_conn_atom",
                #         j_conn_atom,
                #         k,
                #         k_atom_ind,
                #         bt.atom_name(k_atom_ind),
                #     )
                k_kids = atom_kids[k_atom_ind]
                # print("kids:", k_kids)
                if len(k_kids) == 0:
                    continue
                # from here forward, we know that k_atom_ind has > 0 children

                def gen_depth_given_first_descendant():
                    # First, set the first_descendant for k_atom_ind.
                    # Then, the logic is: we have to add one to the
                    # gen-depth of every child except the first descendant
                    # which we get "for free" since it will be built
                    # along the same-scan path segment as k_atom_ind
                    # print(f"atom {bt.atom_name(k_atom_ind)} with first descendant
                    # {bt.atom_name(first_descendant[k_atom_ind]) if first_descendant[k_atom_ind] >= 0
                    # else 'None'} and depth
                    # {gen_depth[first_descendant[k_atom_ind]] if first_descendant[k_atom_ind] >= 0 else -9999}")
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

                if is_on_primary_exit_sp_seg[k_atom_ind]:
                    # in this case, the first_descendant for this atom
                    # has already been decided
                    # if j == n_conn + 1:
                    #     print("on exit spseg:", bt.atom_name(k_atom_ind), first_descendant[k_atom_ind], is_conn_atom[k_atom_ind])
                    if k_atom_ind == j_conn_atom:
                        # this atom's first descendent is the atom on the next residue
                        # to which this residue is connected
                        gen_depth[k_atom_ind] = max([gen_depth[l] for l in k_kids]) + 1
                    else:
                        # first_descendant is already determined for this atom
                        gen_depth[k_atom_ind] = gen_depth_given_first_descendant()
                else:

                    if is_conn_atom[k_atom_ind]:
                        # In this case, "the" connection (there can possibly be more than one!)
                        # will be the first child and the other descendants will be second children.
                        # We save the gen depth, but when calculating the gen depth of the
                        # fold-forest, if this residue is at the upstream end of an edge, then
                        # its depth will have to be calculated as the min gen-depth of the
                        # intra-residue bits and the gen-depth of the nodes downstream of it.
                        # TO DO: This case needs to be properly handled when calculating the
                        # maximum number of generations to run gen-seg-scan.
                        # if target:
                        #     print("conn atom", bt.atom_name(k_atom_ind))
                        gen_depth[k_atom_ind] = max([gen_depth[l] for l in k_kids]) + 1
                    else:
                        # most-common case: an atom not on the primary-exit sp seg, and that isn't
                        # itself a connection atom.
                        # First we ask: are we on one or more exit scan path segments?
                        # NOTE: this just chooses the first exit spseg atom it encounters
                        # as the first descendant and so I pause and think: if we have
                        # a block type with 4 inter-residue connections where the fold
                        # forest branches at this residue, then the algorithm for constructing
                        # the fewest-number-of-generations KinForest here is going
                        # to fail: we are treating all exit paths out of this residue
                        # as interchangable and we might say connection c should be
                        # ahead of connection c' in a case where c' has a greater gen_depth
                        # than c. We will still get a valid KinForest, but it will lack
                        # the "fewest number of generations possible" property.
                        #
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
                        # up- and a down connection and that's it (e.g. nucleic acids),
                        # then this algorithm will still produce optimal KinForests.
                        # (I have to use a beta-amino acid as an example here because if
                        # we consider the case of an alpha-amino acid, then the exit path
                        # at N is already the root of a new scan path and there's no decision
                        # making that has to be made.)
                        #
                        # A case that this would fail to deliver the optimally-efficient
                        # (fewest number of generations) KinForest would be if this R group
                        # also contained an inter-residue connection and there were an
                        # edge in the FoldForest (a "chemical edge") leaving from that
                        # connection to some further chain, e.g., it could be a sugar
                        # group attached to a beta-ASN. Now if the path (CA->CB->N) takes
                        # precedence over the path (CA->CB->R), then everything down-
                        # stream of the R would have a generation-delay one greater than
                        # it would otherwise. Again, a KinForest produced by this algorithm
                        # is still valid, it could just be slightly slower to fold through
                        # than it would be otherwise.
                        # if target:
                        #     print("common case", k, bt.atom_name(k_atom_ind))
                        if j != n_conn + 1:
                            for kid in k_kids:
                                if is_on_exit_sp_segment[kid]:
                                    # print("bt", bt.name, "kid", kid, bt.atom_name(kid), "is on ext")
                                    first_descendant[k_atom_ind] = kid
                                    is_on_exit_sp_segment[k_atom_ind] = True
                                    assert (
                                        interres_conn_scan_path_segment_rooted_by_atom[
                                            kid
                                        ]
                                        >= 0
                                    )
                                    kid_conn_ind = (
                                        interres_conn_scan_path_segment_rooted_by_atom[
                                            kid
                                        ]
                                    )
                                    # k_atom_ind becomes the new root of the scan path
                                    # building to the kid_conn_ind interresidue connection
                                    interres_conn_scan_path_segment_rooted_by_atom[
                                        k_atom_ind
                                    ] = kid_conn_ind
                                    interres_conn_scan_path_segment_rooted_by_atom[
                                        kid
                                    ] = -1
                                    atom_rooting_scan_path_segment_for_interres_conn[
                                        kid_conn_ind
                                    ] = k_atom_ind
                                    # stop now to ensure that we do not ovewrite the first_descendant
                                    # of k_atom_ind if it should happen to have two kids that
                                    # are on exit paths!
                                    break

                        if not is_on_exit_sp_segment[k_atom_ind]:
                            # which should be the first descendant? the one with the greatest gen depth
                            first_descendant[k_atom_ind] = k_kids[
                                numpy.argmax(
                                    numpy.array([gen_depth[kid] for kid in k_kids])
                                )
                            ]
                            # if j == n_conn + 1:
                            #     print("Selecting first descendant of", bt.atom_name(k_atom_ind), "as", bt.atom_name(first_descendant[k_atom_ind]))
                        gen_depth[k_atom_ind] = gen_depth_given_first_descendant()
                        # print("gen_depth", bt.atom_name(k_atom_ind), "d:", gen_depth[k_atom_ind])
            # print("gen_depth", gen_depth)
            # print("is on exit path", bt.name, i, j, ":", is_on_exit_path)
            # OKAY!
            # if j == n_conn + 1:
            #     print("first descendants", first_descendant)
            # now we have paths rooted at each node up to the root
            # we need to turn these paths into scan paths
            # Let's now traverse the atoms in bfs order and build the scan paths
            # along the way
            processed_node_into_scan_path_segment = is_on_primary_exit_sp_seg.copy()
            gen_to_build_atom = numpy.full((bt.n_atoms,), -1, dtype=numpy.int64)
            gen_to_build_atom[is_on_primary_exit_sp_seg] = 0
            # print("gen depth", gen_depth)
            # print("starting bfs:", processed_node_into_scan_path_segment)
            for k in range(bt.n_atoms):
                k_atom_ind = bfto_2_orig[k]
                if processed_node_into_scan_path_segment[k_atom_ind]:
                    # we have already added this atom and its first
                    # descendant (and their first descendant and so on)
                    # to a scan path segment, so we can continue
                    continue

                # if we arrive here, that means k_atom_ind is the root of a
                # new scan path segment
                path = []
                # we have already processed the first scan path segment
                # from the entrace-point atom to the first exit-point atom
                # unless we are process the "is-a-leaf-node" case
                assert k_atom_ind != i_conn_atom or j == n_conn + 1
                # put the _parent_ of this new root at the beginning of
                # the scan path segment since we build the root's coordinate
                # frame from its parent's coordinate frame
                if k_atom_ind != i_conn_atom:
                    path.append(preds[k_atom_ind])
                    focused_atom = k_atom_ind

                    gen_to_build_atom[focused_atom] = (
                        gen_to_build_atom[preds[focused_atom]] + 1
                    )
                else:
                    focused_atom = k_atom_ind
                    gen_to_build_atom[focused_atom] = 0
                # print(
                #     f"gen to build {bt.atom_name(focused_atom)} from {bt.atom_name(preds[focused_atom])}",
                #     f"with gen {gen_to_build_atom[focused_atom]}",
                # )

                # now we traverse the path along each atom's first descendant
                while focused_atom >= 0:
                    path.append(focused_atom)
                    processed_node_into_scan_path_segment[focused_atom] = True
                    focused_atom = first_descendant[focused_atom]
                    if focused_atom >= 0:
                        gen_to_build_atom[focused_atom] = gen_to_build_atom[
                            preds[focused_atom]
                        ]

                if is_on_exit_sp_segment[k_atom_ind]:
                    # we will go ahead and put exit sp segs at the beginning of the
                    # list of scan path segs for a generation, however, there is no
                    # demand that we must do so.
                    gen_scan_path_segments[gen_to_build_atom[k_atom_ind]].insert(
                        0, path
                    )
                else:
                    gen_scan_path_segments[gen_to_build_atom[k_atom_ind]].append(path)
            # Now we need to assemble the scan path segments in a compact way:
            # print("gen scan path segments", gen_scan_path_segments)

            ij_n_gens = gen_depth[i_conn_atom]
            # print("ij_n_gens", i, j, ij_n_gens)
            ij_n_scan_path_segments = numpy.array(
                [len(gen_scan_path_segments[k]) for k in range(ij_n_gens)], dtype=int
            )
            # print("ij_n_scans", i, j, ij_n_scans)
            ij_scan_path_segment_starts = [
                numpy.zeros((ij_n_scan_path_segments[k],), dtype=int)
                for k in range(ij_n_gens)
            ]
            ij_scan_path_segment_lengths = [
                numpy.array(
                    [
                        len(gen_scan_path_segments[k][l])
                        for l in range(len(gen_scan_path_segments[k]))
                    ],
                    dtype=int,
                )
                for k in range(ij_n_gens)
            ]
            # print("ij_scan_path_segment_lengths", i, j, ij_scan_lengths)
            for k in range(ij_n_gens):
                offset = 0
                for l in range(ij_n_scan_path_segments[k]):
                    ij_scan_path_segment_starts[k][l] = offset
                    offset += ij_scan_path_segment_lengths[k][l]
            # print("ij_scan_starts", i, j, ij_scan_starts)
            # print("ij_scan_lengths cumsum?", numpy.cumsum(ij_scan_lengths))
            ij_scan_path_segment_is_inter_block = [
                numpy.zeros((ij_n_scan_path_segments[k],), dtype=bool)
                for k in range(ij_n_gens)
            ]

            for k in range(ij_n_gens):
                for l in range(ij_n_scan_path_segments[k]):
                    l_first_at = gen_scan_path_segments[k][l][0 if k == 0 else 1]
                    # if target:
                    #     print(k, l, "l_first_at", l_first_at)
                    # "interblock" is really asking "does this scan path segment
                    # exit to a different block?". This is "answered" by whether the
                    # last atom in the scan path segment is a connection atom.
                    # The SPSs that are inter-block are going to be roots of SPs
                    # in the forward pass, and they are likely to not be roots
                    # of SPs in the backward pass as long as there are edges leaving
                    # from the connection atoms.
                    kl_last_atom = gen_scan_path_segments[k][l][-1]
                    # if target:
                    #     print(k, l, "kl_last_atom", kl_last_atom)
                    ij_scan_path_segment_is_inter_block[k][l] = (
                        is_conn_atom[kl_last_atom] and j != n_conn + 1
                    ) or (  # is the last atom in the path a connection atom?
                        k == 0 and l == 0
                    )  # the first scan path segment is always inter-block
                    conn_for_path = interres_conn_scan_path_segment_rooted_by_atom[
                        l_first_at
                    ]
                    # if target:
                    #     print(k, l, "conn_for_path", conn_for_path)
                    if conn_for_path != -1:
                        # print(
                        #     bt.name,
                        #     i,
                        #     j,
                        #     "setting conn for path",
                        #     conn_for_path,
                        #     "as",
                        #     k,
                        #     l,
                        # )
                        gen_of_scan_path_segment_building_interres_conn[
                            conn_for_path
                        ] = k
                        scan_path_segment_building_interres_conn[conn_for_path] = l

            # print("ij_scan_is_inter_block", ij_scan_is_inter_block)
            # ij_n_nodes_for_gen =
            ij_n_nodes_for_gen = numpy.array(
                [
                    sum(len(path) for path in gen_scan_path_segments[k])
                    for k in range(ij_n_gens)
                ],
                dtype=int,
            )
            # if j == n_conn + 1:
            #     print(bt.name, i, j, "gen_scan_path_segments", gen_scan_path_segments)
            scan_path_segment_data[(i, j)] = dict(
                n_gens=ij_n_gens,
                n_nodes_for_gen=ij_n_nodes_for_gen,
                nodes_for_gen=gen_scan_path_segments,
                n_scan_path_segs=ij_n_scan_path_segments,
                gen_building_output_conn=gen_of_scan_path_segment_building_interres_conn,
                scan_path_seg_building_output_conn=scan_path_segment_building_interres_conn,
                scan_path_seg_starts=ij_scan_path_segment_starts,
                scan_path_seg_is_inter_block=ij_scan_path_segment_is_inter_block,
                scan_path_seg_lengths=ij_scan_path_segment_lengths,
            )
        # end for j
    # end for i

    # Now let's count out the maximum number of generations, scans, and nodes-per-gen
    # so we can create the BTGenerationalSegScanPaths object
    max_n_gens = max(
        scan_path_segment_data[(i, j)]["n_gens"]
        for i in range(n_input_types)
        for j in range(n_output_types)
        if (i, j) in scan_path_segment_data
    )
    max_n_scan_path_segments = max(
        max(
            scan_path_segment_data[(i, j)]["n_scan_path_segs"][k]
            for k in range(scan_path_segment_data[(i, j)]["n_gens"])
        )
        for i in range(n_input_types)
        for j in range(n_output_types)
        if (i, j) in scan_path_segment_data
    )
    max_n_nodes_per_gen = max(
        max(
            scan_path_segment_data[(i, j)]["n_nodes_for_gen"][k]
            for k in range(scan_path_segment_data[(i, j)]["n_gens"])
        )
        for i in range(n_input_types)
        for j in range(n_output_types)
        if (i, j) in scan_path_segment_data
    )
    bt_gen_seg_scan_path_segments = BTGenerationalSegScanPathSegs.empty(
        n_input_types,
        n_output_types,
        bt.n_atoms,
        n_conn,
        max_n_gens,
        max_n_scan_path_segments,
        max_n_nodes_per_gen,
    )
    bt_gen_seg_scan_path_segments.jump_atom = jump_atom_for_bt(bt)
    bt_gen_seg_scan_path_segments.parents = parents
    bt_gen_seg_scan_path_segments.dof_type[:] = dof_type
    bt_gen_seg_scan_path_segments.input_conn_atom = input_conn_atom
    # Finally, we populate the BTGenerationalSegScanPathSegs object
    for i in range(n_input_types):
        for j in range(n_output_types):
            if (i, j) not in scan_path_segment_data:
                continue
            ij_n_gens = scan_path_segment_data[(i, j)]["n_gens"]
            bt_gen_seg_scan_path_segments.n_gens[i, j] = ij_n_gens
            bt_gen_seg_scan_path_segments.scan_path_seg_that_builds_output_conn[
                i, j, :, 0
            ] = scan_path_segment_data[(i, j)]["gen_building_output_conn"]
            bt_gen_seg_scan_path_segments.scan_path_seg_that_builds_output_conn[
                i, j, :, 1
            ] = scan_path_segment_data[(i, j)]["scan_path_seg_building_output_conn"]
            for k in range(ij_n_gens):
                bt_gen_seg_scan_path_segments.n_nodes_for_gen[i, j, k] = (
                    scan_path_segment_data[(i, j)]["n_nodes_for_gen"][k]
                )
                bt_gen_seg_scan_path_segments.n_scan_path_segs[i, j, k] = (
                    scan_path_segment_data[(i, j)]["n_scan_path_segs"][k]
                )
                bt_gen_seg_scan_path_segments.scan_path_seg_is_real[
                    i, j, k, : bt_gen_seg_scan_path_segments.n_scan_path_segs[i, j, k]
                ] = True

                ijk_n_scan_path_segs = scan_path_segment_data[(i, j)][
                    "n_scan_path_segs"
                ][k]
                bt_gen_seg_scan_path_segments.scan_path_seg_starts[
                    i, j, k, :ijk_n_scan_path_segs
                ] = scan_path_segment_data[(i, j)]["scan_path_seg_starts"][k]
                bt_gen_seg_scan_path_segments.scan_path_seg_is_inter_block[
                    i, j, k, :ijk_n_scan_path_segs
                ] = scan_path_segment_data[(i, j)]["scan_path_seg_is_inter_block"][k]
                bt_gen_seg_scan_path_segments.scan_path_seg_lengths[
                    i, j, k, :ijk_n_scan_path_segs
                ] = scan_path_segment_data[(i, j)]["scan_path_seg_lengths"][k]
                # for l in range(scan_path_data[(i, j)]["n_scans"][k]):
                # bt_gen_seg_scan_paths.scan_starts[i, j, k, l] = scan_path_data[(i, j)]["scan_starts"][k][l]
                # bt_gen_seg_scan_paths.scan_is_inter_block[i, j, k, l] = scan_path_data[(i, j)]["scan_is_inter_block"][k][l]
                # bt_gen_seg_scan_paths.scan_lengths[i, j, k, l] = scan_path_data[(i, j)]["scan_lengths"][k][l]
                for l in range(ijk_n_scan_path_segs):
                    m_offset = scan_path_segment_data[(i, j)]["scan_path_seg_starts"][
                        k
                    ][l]
                    for m in range(
                        len(scan_path_segment_data[(i, j)]["nodes_for_gen"][k][l])
                    ):
                        bt_gen_seg_scan_path_segments.nodes_for_gen[
                            i, j, k, m_offset + m
                        ] = scan_path_segment_data[(i, j)]["nodes_for_gen"][k][l][m]
                # print("nodes for gen", i, j, k, bt_gen_seg_scan_paths.nodes_for_gen[i, j, k, :])

    setattr(bt, "gen_seg_scan_path_segs", bt_gen_seg_scan_path_segments)


def _annotate_packed_block_type_with_gen_scan_path_segs(pbt):
    for bt in pbt.active_block_types:
        _annotate_block_type_with_gen_scan_path_segs(bt)
    max_n_input_types = max(
        bt.gen_seg_scan_path_segs.n_gens.shape[0] for bt in pbt.active_block_types
    )
    max_n_output_types = max(
        bt.gen_seg_scan_path_segs.n_gens.shape[1] for bt in pbt.active_block_types
    )
    # max_n_atoms : pbt already provides this!
    # max_n_conn : pbt already provides this!
    max_n_gens = max(
        bt.gen_seg_scan_path_segs.n_nodes_for_gen.shape[2]
        for bt in pbt.active_block_types
    )
    max_n_scan_path_segs = max(
        bt.gen_seg_scan_path_segs.scan_path_seg_starts.shape[3]
        for bt in pbt.active_block_types
    )
    max_n_nodes_per_gen = max(
        bt.gen_seg_scan_path_segs.nodes_for_gen.shape[3]
        for bt in pbt.active_block_types
    )

    gen_seg_scan_path_segs = PBTGenerationalSegScanPathSegs.empty(
        pbt.device,
        pbt.n_types,
        max_n_input_types,
        max_n_output_types,
        pbt.max_n_atoms,
        pbt.max_n_conn,
        max_n_gens,
        max_n_scan_path_segs,
        max_n_nodes_per_gen,
    )
    gen_seg_scan_path_segs.jump_atom[:] = torch.tensor(
        [bt.gen_seg_scan_path_segs.jump_atom for bt in pbt.active_block_types],
        dtype=torch.int32,
        device=pbt.device,
    )
    varnames = [
        "parents",
        "dof_type",
        "input_conn_atom",
        "n_gens",
        "n_nodes_for_gen",
        "nodes_for_gen",
        "n_scan_path_segs",
        "scan_path_seg_starts",
        "scan_path_seg_is_real",
        "scan_path_seg_is_inter_block",
        "scan_path_seg_lengths",
    ]
    for i, bt in enumerate(pbt.active_block_types):
        bt_gssps = bt.gen_seg_scan_path_segs
        # this data member doesn't fit the same mold as the others
        shape_sptboc = bt_gssps.scan_path_seg_that_builds_output_conn.shape
        gen_seg_scan_path_segs.scan_path_seg_that_builds_output_conn[
            i, : shape_sptboc[0], : shape_sptboc[1], : shape_sptboc[2], :
        ] = torch.tensor(
            bt_gssps.scan_path_seg_that_builds_output_conn,
            dtype=torch.int32,
            device=pbt.device,
        )
        for vname in varnames:
            dst = getattr(gen_seg_scan_path_segs, vname)
            src = getattr(bt_gssps, vname)
            src = torch.tensor(
                src,
                dtype=(torch.int32 if src.dtype == numpy.int64 else torch.bool),
                device=pbt.device,
            )
            if len(src.shape) == 1:
                dst[i, : src.shape[0]] = src
            elif len(src.shape) == 2:
                dst[i, : src.shape[0], : src.shape[1]] = src
            elif len(src.shape) == 3:
                dst[i, : src.shape[0], : src.shape[1], : src.shape[2]] = src
            elif len(src.shape) == 4:
                dst[
                    i, : src.shape[0], : src.shape[1], : src.shape[2], : src.shape[3]
                ] = src
            else:
                raise ValueError("unhandled shape")
    setattr(pbt, "gen_seg_scan_path_segs", gen_seg_scan_path_segs)
