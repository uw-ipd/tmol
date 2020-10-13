import attr
import numpy
import torch

from .datatypes import KinTree

from numba import jit
from tmol.types.torch import Tensor
from tmol.types.tensor import TensorGroup
from tmol.types.attrs import ConvertAttrs, ValidateAttrs

from tmol.types.functional import validate_args


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
    n_children = numpy.ones(nelts, dtype=numpy.int32)
    for i in range(nelts - 1, 0, -1):
        p = parents[i]
        if p == i:  # root
            continue
        n_children[p] += n_children[i]
        child_list[child_list_span[p, 0] + n_immediate_children[p] - 1] = i
        n_immediate_children[p] -= 1

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
                            if n_children[candidate] > n_children[nextExtension]:
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
class KinTreeScanData(TensorGroup, ConvertAttrs):
    nodes: Tensor(torch.int)
    scans: Tensor(torch.int)
    gens: Tensor(torch.int)


@attr.s(auto_attribs=True, frozen=True)
class KinTreeScanOrdering(ValidateAttrs):
    """Scan plans for parallel kinematic operations.

    The KinTreeScanOrdering class divides the tree into a set of paths. Along
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

    To accomplish this, the GPUKinTreeReordering class reorders the atoms from
    the original KinTree order ("ko") where atoms are known by their
    kintree-index ("ki") into 1) their refold order ("ro") where atoms are
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

    kintree_cache_key = "__KinTreeScanOrdering_cache__"

    forward_scan_paths: KinTreeScanData
    backward_scan_paths: KinTreeScanData

    @classmethod
    @validate_args
    def for_kintree(cls, kintree):
        """Calculate and cache refold ordering over kintree

        KinTree data structure is frozen; so it is safe to cache the gpu scan
        ordering for a single object. Store as a private property of the input
        kintree, lifetime of the cache will then be managed via the target
        object.
        ."""

        if not hasattr(kintree, cls.kintree_cache_key):
            object.__setattr__(
                kintree, cls.kintree_cache_key, cls.calculate_from_kintree(kintree)
            )

        return getattr(kintree, cls.kintree_cache_key)

    @classmethod
    @validate_args
    def calculate_from_kintree(cls, kintree: KinTree):
        """Setup for operations over KinTree.
        ``device`` is inferred from kintree tensor device.
        """

        nodes, scanStarts, genStarts = get_scans(
            kintree.parent.cpu().numpy(), numpy.array([0])
        )
        forward_scan_paths = KinTreeScanData(
            nodes=torch.from_numpy(nodes).to(device=kintree.parent.device),
            scans=torch.from_numpy(scanStarts).to(device=kintree.parent.device),
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

        backward_scan_paths = KinTreeScanData(
            nodes=torch.from_numpy(nodesR).to(device=kintree.parent.device),
            scans=torch.from_numpy(scanStartsR).to(device=kintree.parent.device),
            gens=torch.from_numpy(genStartsR),
        )  # keep gens on CPU!

        return KinTreeScanOrdering(
            forward_scan_paths=forward_scan_paths,
            backward_scan_paths=backward_scan_paths,
        )
