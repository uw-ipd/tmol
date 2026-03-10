import numpy
import attr
import enum

from tmol.types.array import NDArray
from tmol.pose.pose_stack import PoseStack


class EdgeType(enum.IntEnum):
    polymer = 0
    jump = enum.auto()
    root_jump = enum.auto()


def _build_pose_fold_forest(bti_p, irc_p, up_c, down_c):
    """Build fold forest edges for a single pose from polymer connectivity only.

    Only backbone (up/down) connections are considered.  The resulting graph is
    a disjoint union of simple paths (linear chains) and simple cycles (C→N
    cyclisation).  Non-polymer connections (disulfides, etc.) are ignored.

    poly_succ / poly_pred are built with a vectorised numpy gather:
    for each residue r, r's *up*-conn slot points to the C-terminal neighbour s,
    and s's *down*-conn slot points back to r.  (In tmol nomenclature "up" is
    the C-terminal / higher-index direction, mirroring Rosetta's upper_connect.)

    Chain walking is still a Python loop but is O(n_real) total iterations
    across all chains (each residue is visited exactly once).

    Cyclic polymers (C→N cyclisation) are broken at the bond entering the
    lowest-index residue; that bond is simply dropped from the fold forest
    (not emitted as a jump) so the result remains a valid tree.

    Returns a list of [type, start, end, jump_idx] integer lists.
    """
    n_res = len(bti_p)
    rows = numpy.arange(n_res)

    real_mask = bti_p >= 0
    real_res = numpy.where(real_mask)[0]

    if len(real_res) == 0:
        return []

    # ------------------------------------------------------------------
    # Vectorised poly_succ / poly_pred construction.
    #
    # For each residue r:
    #   1. Look up r's up-conn slot (points toward C-terminal neighbour).
    #   2. Fetch the target residue s and its connection slot cs via irc_p.
    #   3. Accept the bond only if cs == down-conn slot of s (the N-terminal
    #      side of s), confirming a proper polymer bond in the right direction.
    # ------------------------------------------------------------------
    bt_safe = numpy.where(real_mask, bti_p, 0)  # safe block-type indices

    uc_r = up_c[bt_safe]  # up-conn slot per residue
    uc_safe = numpy.maximum(uc_r, 0)

    succ_raw = irc_p[rows, uc_safe, 0]  # candidate C-terminal neighbour
    succ_cs = irc_p[rows, uc_safe, 1]  # connection slot on that neighbour

    s_safe = numpy.maximum(succ_raw, 0)
    dc_s = down_c[numpy.maximum(bti_p[s_safe], 0)]  # down-conn slot on s

    valid = (
        real_mask
        & (uc_r >= 0)
        & (succ_raw >= 0)
        & real_mask[s_safe]
        & (succ_cs == dc_s)
    )

    poly_succ_arr = numpy.full(n_res, -1, dtype=numpy.int64)
    poly_pred_arr = numpy.full(n_res, -1, dtype=numpy.int64)

    valid_r = numpy.where(valid)[0]
    valid_s = succ_raw[valid]
    poly_succ_arr[valid_r] = valid_s
    poly_pred_arr[valid_s] = valid_r

    # ------------------------------------------------------------------
    # Walk linear chains from each N-terminus (real residue, no predecessor).
    # O(n_real) total Python iterations across all chains.
    # ------------------------------------------------------------------
    chains = []  # (start, end)
    visited = numpy.zeros(n_res, dtype=bool)

    for r in numpy.where(real_mask & (poly_pred_arr < 0))[0]:
        r = int(r)
        cur = r
        while poly_succ_arr[cur] >= 0:
            visited[cur] = True
            cur = int(poly_succ_arr[cur])
        visited[cur] = True
        chains.append((r, cur))

    # ------------------------------------------------------------------
    # Handle cyclic polymers (C→N cyclisation).
    # Every unvisited real residue belongs to a simple cycle.
    # Break each cycle at the bond entering its lowest-index residue;
    # that bond is dropped (not emitted as a jump) to keep the tree acyclic.
    # ------------------------------------------------------------------
    for r in numpy.where(real_mask & ~visited)[0]:
        r = int(r)
        if visited[r]:
            continue
        cycle = []
        cur = r
        while not visited[cur]:
            visited[cur] = True
            cycle.append(cur)
            cur = int(poly_succ_arr[cur])
        n_term = min(cycle)
        c_term = int(poly_pred_arr[n_term])
        chains.append((n_term, c_term))

    # ------------------------------------------------------------------
    # Emit edges: one root-jump per chain, one polymer edge if len > 1.
    # ------------------------------------------------------------------
    result = []
    jump_idx = 0

    for start, end in sorted(chains, key=lambda c: c[0]):
        result.append([int(EdgeType.root_jump), -1, start, jump_idx])
        jump_idx += 1
        if start != end:
            result.append([int(EdgeType.polymer), start, end, -1])

    return result


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
    def reasonable_fold_forest(cls, pose_stack: PoseStack):
        """Create a fold forest for each pose using only backbone (up/down)
        polymer connectivity.

        Each linear chain receives one root-jump and one polymer edge.
        Cyclic polymers (C→N cyclisation) are broken at the bond entering
        the lowest-index residue; that bond is dropped to keep the forest
        a valid tree.  Non-polymer connections (disulfides, etc.) are ignored.
        """
        irc = pose_stack.inter_residue_connections.cpu().numpy()
        bti = pose_stack.block_type_ind.cpu().numpy()
        pbt = pose_stack.packed_block_types
        up_c = pbt.up_conn_inds.cpu().numpy()
        down_c = pbt.down_conn_inds.cpu().numpy()

        all_pose_edges = [
            _build_pose_fold_forest(bti[p], irc[p], up_c, down_c)
            for p in range(pose_stack.n_poses)
        ]

        max_n_edges = max((len(e) for e in all_pose_edges), default=1)
        n_poses = pose_stack.n_poses

        edges = numpy.full((n_poses, max_n_edges, 4), -1, dtype=numpy.int64)
        n_edges = numpy.zeros(n_poses, dtype=int)

        for p, edges_p in enumerate(all_pose_edges):
            n = len(edges_p)
            if n > 0:
                edges[p, :n] = edges_p
            n_edges[p] = n

        return cls(max_n_edges=max_n_edges, n_edges=n_edges, edges=edges)

    @classmethod
    def from_edges(cls, edges: NDArray[int][:, :, 4]):
        return cls(
            max_n_edges=edges.shape[1],
            n_edges=numpy.sum(edges[:, :, 0] != -1, axis=1),
            edges=edges,
        )
