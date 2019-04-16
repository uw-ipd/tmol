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
#    scanStarts - a 1D index array in 'scans' where each individual scan begins
#    genStarts - an N x 2 array giving indices where each gen begins
#         genStarts[:,0] indexes scans
#         genStarts[:,1] indexes scanStarts
@jit(nopython=True)
def get_scans(parents, roots):
    nelts = parents.shape[0]

    # count number of children
    nchildren = numpy.ones(nelts, dtype=numpy.int32)
    for i in range(nelts - 1, 0, -1):
        nchildren[parents[i]] += nchildren[i]

    # map parent idx -> child idx
    p2c = numpy.full((nelts, 4), -1, dtype=numpy.int32)
    p2c_i = numpy.full(nelts, 0, dtype=numpy.int32)
    for i in range(1, nelts):
        par_i = parents[i]
        p2c[par_i, p2c_i[par_i]] = i
        p2c_i[par_i] += 1

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
    activeFront = numpy.full(nelts, -1, dtype=numpy.int32)
    nActiveFront = roots.shape[0]
    activeFront[:nActiveFront] = roots

    # DFS traversal using #children to choose paths
    marked = numpy.zeros(nelts, dtype=numpy.int32)
    marked[0] = 1
    while not numpy.all(marked):
        genStarts[genidx, :] = [nodeidx, scanidx]
        for i in range(nActiveFront):
            currRoot = activeFront[i]

            # active front "roots" have already been generated
            #   so can participate in >1 scan
            for j in range(p2c_i[currRoot]):
                expandedNode = p2c[currRoot, j]
                if marked[expandedNode] != 0:
                    continue

                # this is the first root expansion,
                #  add the node as the start of a new scan
                scanStarts[scanidx] = nodeidx - genStarts[genidx, 0]
                # -> numbering is w.r.t. generation
                scanidx += 1
                nodes[nodeidx] = currRoot
                nodes[nodeidx + 1] = expandedNode
                nodeidx += 2

                marked[expandedNode] = 1

                while expandedNode != -1:
                    prevExpNode = expandedNode
                    expandedNode = -1
                    for k in range(p2c_i[prevExpNode]):
                        candidate = p2c[prevExpNode, k]
                        if marked[candidate] != 0:
                            continue
                        if (
                            expandedNode == -1
                            or nchildren[candidate] > nchildren[expandedNode]
                        ):
                            expandedNode = candidate

                    if expandedNode != -1:
                        marked[expandedNode] = 1
                        nodes[nodeidx] = expandedNode
                        nodeidx += 1

        # generate active front for next generation
        #  as anything added this generation (if anything was added)
        if genStarts[genidx, 1] < scanidx:
            lastgenScan0 = genStarts[genidx, 0]
            activeFront.fill(-1)
            nActiveFront = nodeidx - lastgenScan0
            activeFront[:nActiveFront] = nodes[lastgenScan0:nodeidx]

        # next generation
        genidx += 1

    # pad scanStarts and genStarts by 1 to make downstream code cleaner
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
    """Scan ordering for parallel kinematic operations.

    Following the previous version, this is attached to the kintree object.
    Unlike previous version this is sent to device on creation.
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
        ngens = genStarts.shape[0]
        scanStartsR = numpy.zeros_like(scanStarts)
        for i in range(1, ngens):
            genstart = genStartsR[i - 1, 1]
            genstop = genStartsR[i, 1]
            nodes_i = genStartsR[i, 0] - genStartsR[i - 1, 0]
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
