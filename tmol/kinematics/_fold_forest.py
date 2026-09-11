import numpy
import attr
import enum

from tmol.types import NDArray
from tmol.pose import PoseStack, annotate_packed_block_types_w_dslf_conn_inds


class EdgeType(enum.IntEnum):
    polymer = 0
    jump = enum.auto()
    root_jump = enum.auto()
    chemical = enum.auto()


def _build_pose_fold_forest(bti_p, irc_p, up_c, down_c, dslf_c, n_conn, chain_id_p):
    """Build fold forest edges for a single pose from its chemical connectivity.

    Backbone (up/down) connections give a disjoint union of simple paths
    (linear chains) and simple cycles (C→N cyclisation).  Every other
    connection except the disulfide is a chemical bond joining two such
    chains, and becomes a chemical edge; disulfides are left to close on
    their own, as they always have.

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
    cyclized_res = numpy.where(real_mask & ~visited)[0]
    for r in cyclized_res:
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

    chains = sorted(chains, key=lambda c: c[0])

    # ------------------------------------------------------------------
    # Contract each chain to a node and span the chemical bonds between
    # them.  Chemical bonds are the connections that are neither polymeric
    # nor the disulfide: a glycan on a serine, a ligand on a lysine.
    # ------------------------------------------------------------------
    chem_parent = _chemical_spanning_forest(
        chains, poly_succ_arr, bti_p, irc_p, up_c, down_c, dslf_c, n_conn, real_mask
    )

    # ------------------------------------------------------------------
    # Emit edges.  A chain is entered either through a chemical bond, or --
    # as before -- by a jump from the residue preceding its N-terminus when
    # that residue belongs to the same biological chain (a chain gap), or
    # by a root jump.  Polymer edges then run outward from the entry
    # residue in both directions.
    # ------------------------------------------------------------------
    # Every edge must start at a block some other edge ends at, so a polymer
    # edge is split wherever a chemical edge leaves from its middle -- a glycan
    # on a serine partway along a chain. Rosetta splits its peptide edges at
    # branch points for the same reason.
    branch_blocks = {parent_res for parent_res, _, _ in chem_parent.values()}

    result = []
    jump_idx = 0

    for ci, (start, end) in enumerate(chains):
        if ci in chem_parent:
            parent_res, parent_conn, entry = chem_parent[ci]
            result.append([int(EdgeType.chemical), parent_res, entry, parent_conn])
        else:
            entry = start
            prev = start - 1
            if prev >= 0 and bti_p[prev] >= 0 and chain_id_p[prev] == chain_id_p[start]:
                result.append([int(EdgeType.jump), prev, start, jump_idx])
                # only true jumps are numbered; a root jump is identified by its
                # downstream block, and numbering it would leave a gap in the
                # jump indices, which must run contiguously from 0
                jump_idx += 1
            else:
                result.append([int(EdgeType.root_jump), -1, start, -1])
        for run_end in (start, end):
            if entry == run_end:
                continue
            step = 1 if entry < run_end else -1
            a = entry
            for b in range(entry + step, run_end + step, step):
                if b in branch_blocks and b != run_end:
                    result.append([int(EdgeType.polymer), a, b, -1])
                    a = b
            result.append([int(EdgeType.polymer), a, run_end, -1])

    return result


def _chemical_spanning_forest(
    chains, poly_succ_arr, bti_p, irc_p, up_c, down_c, dslf_c, n_conn, real_mask
):
    """Which chemical bond, if any, builds each polymer chain.

    Each chain is contracted to a node and the bonds between them -- the
    connections that are neither polymeric nor the disulfide -- are spanned
    breadth-first from each component's lowest-index chain.  A bond reaching
    an already-visited chain would close a cycle and is dropped; because the
    polymer chains are contracted first, a cycle of mixed polymer and
    chemical bonds can only break at a chemical bond.

    Returns a map from chain index to (parent residue, connection on it, the
    residue it builds).
    """
    chain_of = numpy.full(len(real_mask), -1, dtype=numpy.int64)
    for ci, (start, end) in enumerate(chains):
        cur = start
        chain_of[cur] = ci
        while cur != end:
            cur = int(poly_succ_arr[cur])
            chain_of[cur] = ci

    incident = [[] for _ in chains]
    for r in numpy.where(real_mask)[0]:
        r = int(r)
        bt = int(bti_p[r])
        structural = {int(up_c[bt]), int(down_c[bt]), int(dslf_c[bt])}
        for c in range(int(n_conn[bt])):
            if c in structural:
                continue
            partner = int(irc_p[r, c, 0])
            if partner < 0 or not real_mask[partner]:
                continue
            incident[chain_of[r]].append((r, c, partner))
    for bonds in incident:
        bonds.sort()

    chem_parent = {}
    seen = numpy.zeros(len(chains), dtype=bool)
    for ci in range(len(chains)):
        if seen[ci]:
            continue
        seen[ci] = True
        queue = [ci]
        while queue:
            cur = queue.pop(0)
            for r, c, partner in incident[cur]:
                child = int(chain_of[partner])
                if seen[child]:
                    continue
                seen[child] = True
                chem_parent[child] = (r, c, partner)
                queue.append(child)
    return chem_parent


@attr.s(auto_attribs=True, frozen=True)
class FoldForest:
    """The fold forest will define the fold trees for the poses in a PoseStack.
    Each tensor in the class has its first dimension over the number of poses.

    The primary definition of a FoldTree is the Edge. The Edge defines a connection
    between two parts of a Pose. The four types of edges are 1. polymer edges
    (analgous to the previously named "peptide edges" from Rosetta++ and Rosetta3), 2.
    jump edges which connect any pair of residues in the Pose, 3. root-jump
    edges, which originate at the explicit virtual root and connect to a particular
    residue, and 4. chemical edges, which join two residues through a single
    non-polymeric bond (Rosetta's Edge::CHEMICAL). A polymer edge spans a contiguous range of polymeric block types where
    the "up" connection of residue i is connected to the "down" connection of residue
    i+1 for all i in the range between the "start" and "end" blocks.

    Each edge is described by a 4-tuple of integers (type, start, end, jump-index);
    where type is one of the EdgeType enum values, start is the index of the upstream
    residue of the edge, end is the index of the downstream residue of the edge, and
    jump-index is used to assign an id to any particular jump edge; jump-edge indices
    must be unique and ascending from 0 to n_jumps-1. A chemical edge stores in that
    fourth slot the index of the connection on its start residue that leads to its
    end residue; the connection on the end residue follows from the PoseStack's
    inter-residue connections. "Root jump" edges take their
    "identity" from the downstream residue of the edge, so they do not need an index.

    The FoldForest in tmol differs from the FoldTree in Rosetta3 in that there
    is always a virtual root at the origin and any residue (block) may be
    connected to this root by a "root jump". Such root-jump residues are defined
    by listing the residue that the root is connected to as the "end" residue;
    the "start" residue field should be left as -1. An example FoldForest for a
    ten-residue protein might be::

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

        Each biological chain (same chain_id) is rooted with a single
        root-jump to its first residue.  Polymer gaps within that chain
        (chain breaks) become ordinary jump edges connecting the last
        residue before the gap to the first residue after it.  Gaps
        between different biological chains produce separate root-jumps.
        Cyclic polymers (C→N cyclisation) are broken at the bond entering
        the lowest-index residue; that bond is dropped to keep the forest
        a valid tree.  Every remaining connection except the disulfide --
        a glycan on a serine, a ligand on a lysine -- becomes a chemical
        edge, so torsions across it propagate downstream; a chemical bond
        that would close a cycle is the one dropped.
        """
        irc = pose_stack.inter_residue_connections.cpu().numpy()
        bti = pose_stack.block_type_ind.cpu().numpy()
        chain_id = pose_stack.chain_id.cpu().numpy()
        pbt = pose_stack.packed_block_types
        up_c = pbt.up_conn_inds.cpu().numpy()
        down_c = pbt.down_conn_inds.cpu().numpy()
        n_conn = pbt.n_conn.cpu().numpy()
        annotate_packed_block_types_w_dslf_conn_inds(pbt)
        dslf_c = pbt.canonical_dslf_conn_ind.cpu().numpy()

        all_pose_edges = [
            _build_pose_fold_forest(
                bti[p], irc[p], up_c, down_c, dslf_c, n_conn, chain_id[p]
            )
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
