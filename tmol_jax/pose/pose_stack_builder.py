"""PoseStackBuilder — helpers to construct a ``PoseStack`` from higher-level inputs.

Upstream reference: ``tmol/pose/pose_stack_builder.py`` (torch, ~1.7k lines
that deal with sequences, monomer polymers, disulfides, all-pairs-shortest-path
over the connection graph, etc.). In the JAX port we only need the slice of
functionality that bridges **numpy arrays of block-type indices / coordinates**
into a :class:`tmol_jax.pose.pose_stack.PoseStack`. Anything that operates on
string sequences, disulfides, or arbitrary inter-residue graphs is intentionally
out of scope — those flows either don't exist yet in the JAX port or will be
added alongside the tests that exercise them.

Two entry points are provided:

- :meth:`PoseStackBuilder.from_block_types_and_coords` — the primary "here is
  a list of poses, each with a block-type array and a flat coord tensor, go
  pack them" builder. Chain adjacency is used to infer peptide-bond
  ``inter_residue_connections`` (``conn 0 = down``, ``conn 1 = up``), and
  ``inter_block_bondsep`` gets a simple 1/6 approximation: ``1`` for directly
  connected residue pairs (same chain, consecutive residue numbers), ``6``
  for everything else (where ``6 == MAX_SIG_BOND_SEPARATION``).
- :meth:`PoseStackBuilder.empty` — lower-level factory that just sizes out
  all-zero / all-sentinel arrays, for callers that want to fill in coords
  themselves.

Both builders are pure-numpy at construction time and only cast to
``jax.numpy`` when assembling the final ``PoseStack``; this avoids
accidentally baking array creation into a traced graph.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants matching upstream tmol
# ---------------------------------------------------------------------------

# The upstream chemical module defines MAX_SIG_BOND_SEPARATION = 6.
# Any inter-residue connection farther than this is clamped to this value
# in the count-pair machinery, so `6` effectively means "not directly bonded".
MAX_SIG_BOND_SEPARATION = 6

# Convention for which connection index is the "down" (N-terminal side) and
# which is the "up" (C-terminal side) slot on polymeric residues. These
# match upstream's standard amino-acid block types.
_DOWN_CONN = 0
_UP_CONN = 1


def _as_np_int(arr, dtype=np.int32) -> np.ndarray:
    """Coerce input to a contiguous numpy array of the given int dtype."""
    a = np.asarray(arr)
    if a.dtype != dtype:
        a = a.astype(dtype)
    return np.ascontiguousarray(a)


def _pbt_attr(pbt: Any, name: str, default=None):
    """Get an attribute off a PackedBlockTypes-like object, tolerating
    objects that haven't been fully populated yet (e.g. ``empty``).

    Arrays are coerced to numpy for the builder's bookkeeping.
    """
    value = getattr(pbt, name, default)
    if value is None:
        return default
    if hasattr(value, "__array__") or isinstance(value, (list, tuple)):
        try:
            return np.asarray(value)
        except Exception:
            return value
    return value


class PoseStackBuilder:
    """Static namespace of constructors for :class:`PoseStack`."""

    # ------------------------------------------------------------------
    # Primary builder
    # ------------------------------------------------------------------
    @classmethod
    def from_block_types_and_coords(
        cls,
        pbt: Any,
        block_type_inds_per_pose: Sequence[np.ndarray],
        coords_per_pose: Sequence[np.ndarray],
        chain_id_per_pose: Sequence[Sequence[Any]],
        res_num_per_pose: Sequence[Sequence[int]],
        res_name_per_pose: Optional[Sequence[Sequence[str]]] = None,
        disulfide_pairs_per_pose: Optional[Sequence[Sequence[Tuple[int, int]]]] = None,
    ) -> Any:
        """Build a :class:`PoseStack` from per-pose numpy arrays.

        Parameters
        ----------
        pbt :
            A ``PackedBlockTypes``-like object with at least ``n_atoms``,
            ``max_n_atoms``, and ``max_n_conn`` attributes populated.
        block_type_inds_per_pose :
            List of length ``n_poses``. Each element is a 1D ``int`` numpy
            array of length ``n_blocks_this_pose`` giving the block-type
            index within ``pbt`` for each residue. Values < 0 are treated
            as padding and ignored.
        coords_per_pose :
            List of length ``n_poses``. Each element is a
            ``(n_atoms_this_pose, 3)`` numpy array. ``n_atoms_this_pose``
            must equal the sum of ``pbt.n_atoms[block_type_inds_per_pose[p]]``
            over non-padding blocks.
        chain_id_per_pose :
            List of length ``n_poses``. Each element is a sequence of
            length ``n_blocks_this_pose`` containing chain IDs (anything
            hashable; typically a ``str`` like ``"A"`` or an ``int``).
            These are used both for the numeric ``PoseStack.chain_id_int``
            tensor (if present) and for ``PDBInfo.chain_id`` string labels.
        res_num_per_pose :
            List of length ``n_poses``. Each element is a sequence of
            length ``n_blocks_this_pose`` of integer residue numbers for
            the ``PDBInfo`` record.
        res_name_per_pose :
            Optional; per-pose 3-letter residue names. If omitted, we
            look them up from ``pbt.active_block_types[i].name`` when
            possible, falling back to ``"UNK"``.

        Returns
        -------
        PoseStack
            A ``PoseStack`` with the ``inter_residue_connections`` and
            ``inter_block_bondsep`` tensors filled in per the chain
            adjacency approximation described in the module docstring.
        """
        # Local (lazy) imports so that importing this module does not
        # require the sibling modules to exist yet — useful for the
        # parallel port where sibling modules are being written by
        # peer agents.
        import jax.numpy as jnp

        from tmol_jax.pose.pose_stack import PoseStack
        from tmol_jax.pose.pdb_info import PDBInfo

        n_poses = len(block_type_inds_per_pose)
        if n_poses == 0:
            raise ValueError("from_block_types_and_coords: got zero poses")

        # Normalise inputs to numpy early.
        block_type_inds_per_pose = [_as_np_int(bti) for bti in block_type_inds_per_pose]
        coords_per_pose = [np.asarray(c, dtype=np.float32) for c in coords_per_pose]

        n_blocks_per_pose = np.asarray(
            [len(bti) for bti in block_type_inds_per_pose], dtype=np.int32
        )
        max_n_blocks = int(n_blocks_per_pose.max()) if n_poses > 0 else 0

        # PackedBlockTypes shape info.
        n_atoms_per_bt = _pbt_attr(pbt, "n_atoms")
        if n_atoms_per_bt is None:
            raise ValueError(
                "from_block_types_and_coords: pbt.n_atoms must be populated"
            )
        n_atoms_per_bt = np.asarray(n_atoms_per_bt, dtype=np.int32)

        max_n_atoms_per_block = int(_pbt_attr(pbt, "max_n_atoms", 0) or 0)
        if max_n_atoms_per_block == 0 and n_atoms_per_bt.size > 0:
            max_n_atoms_per_block = int(n_atoms_per_bt.max())

        max_n_conn = int(_pbt_attr(pbt, "max_n_conn", 2) or 2)
        # Polymeric residues need at least 2 connection slots (down, up).
        max_n_conn = max(max_n_conn, 2)

        # ------------------------------------------------------------------
        # Allocate output arrays (numpy for now, cast to jax at the end).
        # ------------------------------------------------------------------
        block_type_ind = np.full((n_poses, max_n_blocks), -1, dtype=np.int32)

        # Sum up atom counts so we can pack coords into an
        # [n_poses, max_n_atoms_per_pose, 3] slab.
        n_atoms_per_block = np.zeros((n_poses, max_n_blocks), dtype=np.int32)
        for p, bti in enumerate(block_type_inds_per_pose):
            n_b = len(bti)
            block_type_ind[p, :n_b] = bti
            real_mask = bti >= 0
            safe = np.where(real_mask, bti, 0)
            counts = n_atoms_per_bt[safe]
            counts = np.where(real_mask, counts, 0)
            n_atoms_per_block[p, :n_b] = counts

        # exclusive cumsum along the block axis = start offsets
        cumsum = np.cumsum(n_atoms_per_block, axis=1, dtype=np.int32)
        block_coord_offset = cumsum - n_atoms_per_block
        total_atoms_per_pose = (
            cumsum[:, -1] if max_n_blocks > 0 else np.zeros((n_poses,), dtype=np.int32)
        )
        max_n_atoms_per_pose = int(total_atoms_per_pose.max()) if n_poses > 0 else 0

        # Pack coords.
        coords = np.zeros((n_poses, max_n_atoms_per_pose, 3), dtype=np.float32)
        for p, c in enumerate(coords_per_pose):
            expected = int(total_atoms_per_pose[p])
            if c.shape[0] != expected:
                raise ValueError(
                    f"coords_per_pose[{p}] has shape {c.shape} but the sum of "
                    f"pbt.n_atoms over its block_type_inds is {expected}."
                )
            if c.shape[0] > 0:
                coords[p, : c.shape[0]] = c

        # ------------------------------------------------------------------
        # chain_id (integer) — remap per-pose chain labels to 0..k-1.
        # We do this per-pose so pose A's "A" and pose B's "A" may
        # independently be chain 0 without implying they're the same
        # physical chain.
        # ------------------------------------------------------------------
        chain_id_int = np.full((n_poses, max_n_blocks), -1, dtype=np.int32)
        chain_labels_per_pose: List[List[str]] = []
        for p in range(n_poses):
            labels = list(chain_id_per_pose[p])
            if len(labels) != n_blocks_per_pose[p]:
                raise ValueError(
                    f"chain_id_per_pose[{p}] has len {len(labels)} but "
                    f"expected {n_blocks_per_pose[p]} matching the block "
                    f"count."
                )
            seen: dict = {}
            int_labels = np.full((n_blocks_per_pose[p],), -1, dtype=np.int32)
            for i, lab in enumerate(labels):
                if block_type_ind[p, i] < 0:
                    continue
                if lab not in seen:
                    seen[lab] = len(seen)
                int_labels[i] = seen[lab]
            chain_id_int[p, : n_blocks_per_pose[p]] = int_labels
            chain_labels_per_pose.append([str(x) for x in labels])

        # ------------------------------------------------------------------
        # inter_residue_connections
        #
        # Shape: [n_poses, max_n_blocks, max_n_conn, 2]
        # Last-dim: (other_block, other_conn). Sentinel -1 means "no
        # connection here".
        #
        # Rule: for every pair of consecutive residues in the same chain
        # with increasing residue numbers, wire the earlier residue's
        # "up" connection to the later residue's "down" connection (and
        # vice versa). Per-block-type connection indices are looked up
        # dynamically: a base residue has ``down`` at slot 0 and
        # ``up`` at slot 1, but an ``:nterm`` variant only has ``up``
        # (which then becomes slot 0), etc.
        # ------------------------------------------------------------------
        inter_residue_connections = np.full(
            (n_poses, max_n_blocks, max_n_conn, 2), -1, dtype=np.int32
        )

        # Cache per-block-type up/down connection indices so we only
        # hit Python attribute lookups once per unique block type.
        abt_lookup = _pbt_attr(pbt, "active_block_types", None)

        def _up_down_for_bt(bt_ind: int) -> Tuple[int, int]:
            if abt_lookup is None or bt_ind < 0 or bt_ind >= len(abt_lookup):
                # Legacy fallback: assume the default (down, up) order.
                return (_DOWN_CONN, _UP_CONN)
            bt = abt_lookup[bt_ind]
            up = getattr(bt, "up_connection_ind", -1)
            down = getattr(bt, "down_connection_ind", -1)
            # Legacy fallback: if the block type doesn't advertise these
            # fields (e.g. mock objects in tests), scan the ``connections``
            # list for name matches.
            if up == -1 and down == -1:
                for k, c in enumerate(getattr(bt, "connections", ()) or ()):
                    if getattr(c, "name", None) == "up":
                        up = k
                    elif getattr(c, "name", None) == "down":
                        down = k
            if up == -1 and down == -1:
                return (_DOWN_CONN, _UP_CONN)
            return (int(down), int(up))

        # The chain-adjacency matrix we build for bondsep also needs the
        # same list of (i, j) "directly connected" pairs, so we capture
        # them here, tagged with the specific (up, down) slot indices.
        directly_connected_pairs: List[List[tuple]] = []
        for p in range(n_poses):
            nb_p = int(n_blocks_per_pose[p])
            pairs_p: List[tuple] = []
            if nb_p == 0:
                directly_connected_pairs.append(pairs_p)
                continue

            chain_p = chain_id_int[p, :nb_p]
            bti_p = block_type_ind[p, :nb_p]

            # Scan consecutive block slots. If two real residues share a
            # chain, they are "directly connected" by a peptide bond.
            #
            # NOTE: we deliberately do NOT require residue-number
            # consecutivity here (i.e. we do NOT check ``res_num[j] -
            # res_num[i] == 1``). Upstream tmol's
            # ``determine_chain_ending_status`` only breaks the peptide
            # bond on (a) a true chain-id change or (b) an explicit
            # ``res_not_connected`` flag. A same-chain residue-number
            # *gap* (e.g. 3plc with author residues 26 -> 35 across an
            # 8-residue missing-density region) is still treated as a
            # connected pair by upstream — the resulting rama / omega /
            # cart-torsion contributions on the residues bracketing the
            # gap are mathematically "wrong" but they are also the
            # numbers we are measuring parity against. Adding a
            # res_num delta check here used to under-count those
            # contributions by ~15-30 % on disulfide-rich PDBs like
            # 3plc (rama, omega, dunbrack_rotdev, cart_torsions all
            # diverged from upstream). See docs/PARITY.md.
            for i in range(nb_p - 1):
                j = i + 1
                if bti_p[i] < 0 or bti_p[j] < 0:
                    continue
                if chain_p[i] != chain_p[j] or chain_p[i] < 0:
                    continue
                _, up_i = _up_down_for_bt(int(bti_p[i]))
                down_j, _ = _up_down_for_bt(int(bti_p[j]))
                if up_i < 0 or down_j < 0:
                    # One side is a terminal variant with no matching
                    # connection -- skip the wiring.
                    continue
                inter_residue_connections[p, i, up_i, 0] = j
                inter_residue_connections[p, i, up_i, 1] = down_j
                inter_residue_connections[p, j, down_j, 0] = i
                inter_residue_connections[p, j, down_j, 1] = up_i
                pairs_p.append((i, j, up_i, down_j))
            directly_connected_pairs.append(pairs_p)

        # ------------------------------------------------------------------
        # Disulfide-bond inter-residue connections (CYD-SG to CYD-SG).
        #
        # The peptide-bond wiring above hits ``down`` / ``up`` connection
        # slots; CYD additionally carries a ``dslf`` connection on its
        # SG atom (slot index 2 for mid-chain CYD, 1 for terminal CYD
        # variants). When the caller passes ``disulfide_pairs_per_pose``
        # we wire each (i, j) pair so cart-bonded sees the SG-SG bond
        # and bondsep counts it as one chemical bond (not "infinity").
        # ------------------------------------------------------------------
        def _dslf_conn_for_bt(bt_ind: int) -> int:
            """Return the ``dslf`` connection index for a CYD-flavoured
            block type, or -1 if the block has no such connection.
            """
            if abt_lookup is None or bt_ind < 0 or bt_ind >= len(abt_lookup):
                return -1
            bt = abt_lookup[bt_ind]
            for k, c in enumerate(getattr(bt, "connections", ()) or ()):
                if getattr(c, "name", None) == "dslf":
                    return k
            return -1

        # Capture disulfide pair wiring per pose, tagged with the
        # (dslf_i, dslf_j) connection-slot pair so bondsep wiring below
        # can mirror what we wrote into inter_residue_connections.
        disulfide_wired_per_pose: List[List[tuple]] = []
        for p in range(n_poses):
            dslf_pairs_p_out: List[tuple] = []
            if disulfide_pairs_per_pose is None:
                disulfide_wired_per_pose.append(dslf_pairs_p_out)
                continue
            pairs = (
                disulfide_pairs_per_pose[p] if p < len(disulfide_pairs_per_pose) else ()
            )
            nb_p = int(n_blocks_per_pose[p])
            for i, j in pairs:
                if i < 0 or j < 0 or i >= nb_p or j >= nb_p:
                    continue
                bt_i = int(block_type_ind[p, i])
                bt_j = int(block_type_ind[p, j])
                if bt_i < 0 or bt_j < 0:
                    continue
                dslf_i = _dslf_conn_for_bt(bt_i)
                dslf_j = _dslf_conn_for_bt(bt_j)
                if dslf_i < 0 or dslf_j < 0:
                    # One of the two CYDs doesn't carry a dslf
                    # connection slot — would happen if the variant
                    # promotion fell back to vanilla CYS. Skip the
                    # wiring; cart-bonded will still see a long SG-SG
                    # "non-bond" distance, but at least the
                    # peptide-backbone is intact.
                    continue
                inter_residue_connections[p, i, dslf_i, 0] = j
                inter_residue_connections[p, i, dslf_i, 1] = dslf_j
                inter_residue_connections[p, j, dslf_j, 0] = i
                inter_residue_connections[p, j, dslf_j, 1] = dslf_i
                dslf_pairs_p_out.append((i, j, dslf_i, dslf_j))
            disulfide_wired_per_pose.append(dslf_pairs_p_out)

        # ------------------------------------------------------------------
        # inter_block_bondsep
        #
        # Shape: [n_poses, max_n_blocks, max_n_blocks, max_n_conn, max_n_conn].
        # Encodes the bond separation between every pair of inter-residue
        # connection points across every pair of residues. Upstream
        # populates this via:
        #   1. intra-block conn-to-conn distance for the (k, k) diagonal
        #      from ``bt.intrares_path_distance[conn_atom_a, conn_atom_b]``
        #      (e.g. PRO down->up = 2, ALA down->up = 3),
        #   2. value 1 for directly-bonded conn pairs (peptide bonds, dslf),
        #   3. all-pairs-shortest-path (Floyd-Warshall) propagating sums of
        #      the above through the inter-residue connection graph.
        # Unused conn slots (above ``n_conn`` for a block type) stay at
        # MAX_SIG_BOND_SEPARATION (== 6). This mirrors upstream's
        # ``PoseStackBuilder._calculate_interblock_bondsep_from_connectivity_graph``
        # closely enough for fa_elec / fa_ljatr / etc. count_pair scaling
        # to match bit-exact for non-trivial chains (1r21's 5xHIS_POS panel
        # was the original parity blocker).
        # ------------------------------------------------------------------
        inter_block_bondsep = np.full(
            (n_poses, max_n_blocks, max_n_blocks, max_n_conn, max_n_conn),
            MAX_SIG_BOND_SEPARATION,
            dtype=np.int32,
        )

        # Per-block intra-block conn-to-conn distances, looked up from the
        # block type's ``intrares_path_distance`` (an [n_atoms, n_atoms]
        # bondsep matrix). Indexed by the *connection* slot index, which
        # we map to the underlying atom index via ``bt.connections[c].atom``.
        def _conn_atom_indices(bt: Any) -> List[int]:
            atom_to_idx = {a.name: i for i, a in enumerate(bt.atoms)}
            out: List[int] = []
            for c in getattr(bt, "connections", ()) or ():
                at_name = getattr(c, "atom", None)
                out.append(int(atom_to_idx.get(at_name, -1)))
            return out

        for p in range(n_poses):
            for k in range(int(n_blocks_per_pose[p])):
                bt_ind = int(block_type_ind[p, k])
                if bt_ind < 0 or abt_lookup is None:
                    continue
                bt = abt_lookup[bt_ind]
                ipd = getattr(bt, "intrares_path_distance", None)
                conn_atoms_for_bt = _conn_atom_indices(bt)
                n_conn_bt = len(conn_atoms_for_bt)
                if ipd is None or n_conn_bt == 0:
                    # Fallback: at least put 0 on the (ci, ci) entries we
                    # actually have.
                    for ci in range(n_conn_bt):
                        inter_block_bondsep[p, k, k, ci, ci] = 0
                    continue
                ipd_arr = np.asarray(ipd, dtype=np.int32)
                for ci in range(n_conn_bt):
                    ai = conn_atoms_for_bt[ci]
                    if ai < 0 or ai >= ipd_arr.shape[0]:
                        continue
                    for cj in range(n_conn_bt):
                        aj = conn_atoms_for_bt[cj]
                        if aj < 0 or aj >= ipd_arr.shape[1]:
                            continue
                        d = int(ipd_arr[ai, aj])
                        if d < 0 or d > MAX_SIG_BOND_SEPARATION:
                            d = MAX_SIG_BOND_SEPARATION
                        inter_block_bondsep[p, k, k, ci, cj] = d

        # Wire the direct-connection pairs using the per-block-type
        # connection indices recorded above.
        for p, pairs_p in enumerate(directly_connected_pairs):
            for i, j, up_i, down_j in pairs_p:
                inter_block_bondsep[p, i, j, up_i, down_j] = 1
                inter_block_bondsep[p, j, i, down_j, up_i] = 1

        # Disulfide bondsep: SG-SG is one chemical bond.
        for p, dslf_pairs_p in enumerate(disulfide_wired_per_pose):
            for i, j, dslf_i, dslf_j in dslf_pairs_p:
                inter_block_bondsep[p, i, j, dslf_i, dslf_j] = 1
                inter_block_bondsep[p, j, i, dslf_j, dslf_i] = 1

        # All-pairs-shortest-path over the inter-residue connection graph
        # to propagate multi-hop bondseps. Each "node" is a (block, conn)
        # pair; the edge weight from (i, ca) to (j, cb) is the value already
        # written above. We run Floyd-Warshall on the flat (B * C)-sized
        # graph per pose, clamping to MAX_SIG_BOND_SEPARATION.
        if max_n_blocks > 0 and max_n_conn > 0:
            BC = max_n_blocks * max_n_conn
            for p in range(n_poses):
                # Flatten [B, B, C, C] to [B*C, B*C] for the APSP loop.
                # Mapping: pconn_index = block_index * max_n_conn + conn_index.
                flat = (
                    inter_block_bondsep[p]
                    .transpose(0, 2, 1, 3)  # [B, C, B, C]
                    .reshape(BC, BC)
                    .copy()
                )
                # Mark "unreal" pconn slots (above n_conn for a block, or
                # for padding blocks) so APSP doesn't route through them.
                # We do this by checking the diagonal: real pconn slots
                # have flat[i, i] == 0 (set above), padding has 6.
                real_pconn = flat.diagonal() == 0
                # Floyd-Warshall.
                for k_node in range(BC):
                    if not real_pconn[k_node]:
                        continue
                    fk = flat[:, k_node : k_node + 1] + flat[k_node : k_node + 1, :]
                    np.minimum(flat, fk, out=flat)
                np.minimum(flat, MAX_SIG_BOND_SEPARATION, out=flat)
                inter_block_bondsep[p] = flat.reshape(
                    max_n_blocks, max_n_conn, max_n_blocks, max_n_conn
                ).transpose(0, 2, 1, 3)

        # ------------------------------------------------------------------
        # PDBInfo
        # ------------------------------------------------------------------
        # res_num as dense padded array.
        res_num = np.full((n_poses, max_n_blocks), -1, dtype=np.int32)
        for p in range(n_poses):
            rn = np.asarray(res_num_per_pose[p], dtype=np.int32)
            if len(rn) != n_blocks_per_pose[p]:
                raise ValueError(
                    f"res_num_per_pose[{p}] has len {len(rn)} but expected "
                    f"{n_blocks_per_pose[p]}."
                )
            res_num[p, : len(rn)] = rn

        # Per-block residue names.
        abt = _pbt_attr(pbt, "active_block_types", None)

        def _bt_name(bt_ind: int) -> str:
            if bt_ind < 0 or abt is None or bt_ind >= len(abt):
                return "UNK"
            return str(getattr(abt[bt_ind], "name", "UNK"))

        if res_name_per_pose is None:
            res_name_per_pose = []
            for p in range(n_poses):
                names_p: List[str] = []
                for b in range(int(n_blocks_per_pose[p])):
                    names_p.append(_bt_name(int(block_type_ind[p, b])))
                res_name_per_pose.append(names_p)
        res_name_per_pose = [list(r) for r in res_name_per_pose]

        # Per-atom record type: default all atoms "ATOM".
        atom_record_type: List[List[List[str]]] = []
        for p in range(n_poses):
            pose_rec: List[List[str]] = []
            for b in range(int(n_blocks_per_pose[p])):
                bt_ind = int(block_type_ind[p, b])
                if bt_ind < 0:
                    pose_rec.append([])
                    continue
                pose_rec.append(["ATOM"] * int(n_atoms_per_bt[bt_ind]))
            atom_record_type.append(pose_rec)

        pdb_info = PDBInfo(
            chain_id=chain_labels_per_pose,
            res_num=res_num,
            res_name=res_name_per_pose,
            atom_record_type=atom_record_type,
        )

        # ------------------------------------------------------------------
        # Cast to jax arrays and assemble the PoseStack.
        # ------------------------------------------------------------------
        block_coord_offset_j = jnp.asarray(block_coord_offset, dtype=jnp.int32)

        return PoseStack(
            packed_block_types=pbt,
            coords=jnp.asarray(coords, dtype=jnp.float32),
            block_coord_offset=block_coord_offset_j,
            inter_residue_connections=jnp.asarray(
                inter_residue_connections, dtype=jnp.int32
            ),
            inter_block_bondsep=jnp.asarray(inter_block_bondsep, dtype=jnp.int32),
            block_type_ind=jnp.asarray(block_type_ind, dtype=jnp.int32),
            pdb_info=pdb_info,
        )

    # ------------------------------------------------------------------
    # Low-level "empty" factory
    # ------------------------------------------------------------------
    @classmethod
    def empty(cls, pbt: Any, n_poses: int, max_n_blocks: int) -> Any:
        """Create an all-zero / all-sentinel ``PoseStack`` skeleton.

        The caller is expected to fill in the ``coords``, ``block_type_ind``,
        etc. via :meth:`PoseStack.replace` or jax functional updates.

        - ``coords`` is zero-filled with shape
          ``(n_poses, max_n_blocks * pbt.max_n_atoms, 3)``. This is a
          loose upper bound rather than the tight packing performed by
          :meth:`from_block_types_and_coords`, but it matches the common
          "one block type, fixed size" pattern used by unit tests.
        - ``block_coord_offset`` is the trivial ``i * max_n_atoms`` grid
          so that callers writing into ``coords[p, offset : offset + n]``
          produce valid data without re-deriving offsets.
        - ``block_type_ind`` is all ``-1`` (all padding).
        - ``inter_residue_connections`` is all ``-1``.
        - ``inter_block_bondsep`` is all ``MAX_SIG_BOND_SEPARATION`` with
          zero on the self-diagonal.
        - ``pdb_info`` is a matching-shape empty ``PDBInfo``.
        """
        import jax.numpy as jnp

        from tmol_jax.pose.pose_stack import PoseStack
        from tmol_jax.pose.pdb_info import PDBInfo

        max_n_atoms_per_block = int(_pbt_attr(pbt, "max_n_atoms", 0) or 0)
        if max_n_atoms_per_block == 0:
            n_atoms_per_bt = _pbt_attr(pbt, "n_atoms")
            if n_atoms_per_bt is not None:
                n_atoms_per_bt = np.asarray(n_atoms_per_bt)
                if n_atoms_per_bt.size > 0:
                    max_n_atoms_per_block = int(n_atoms_per_bt.max())
        max_n_conn = max(int(_pbt_attr(pbt, "max_n_conn", 2) or 2), 2)

        max_n_atoms_per_pose = max_n_blocks * max(max_n_atoms_per_block, 1)

        coords = np.zeros((n_poses, max_n_atoms_per_pose, 3), dtype=np.float32)

        # Block_coord_offset: i * max_n_atoms grid.
        block_coord_offset = np.arange(max_n_blocks, dtype=np.int32)[None, :] * max(
            max_n_atoms_per_block, 1
        )
        block_coord_offset = np.broadcast_to(
            block_coord_offset, (n_poses, max_n_blocks)
        ).copy()

        block_type_ind = np.full((n_poses, max_n_blocks), -1, dtype=np.int32)

        inter_residue_connections = np.full(
            (n_poses, max_n_blocks, max_n_conn, 2), -1, dtype=np.int32
        )
        inter_block_bondsep = np.full(
            (n_poses, max_n_blocks, max_n_blocks, max_n_conn, max_n_conn),
            MAX_SIG_BOND_SEPARATION,
            dtype=np.int32,
        )
        if max_n_blocks > 0:
            diag_idx = np.arange(max_n_blocks)
            inter_block_bondsep[:, diag_idx, diag_idx, :, :] = 0

        pdb_info = PDBInfo.empty(n_poses, max_n_blocks)

        block_coord_offset_j = jnp.asarray(block_coord_offset, dtype=jnp.int32)

        return PoseStack(
            packed_block_types=pbt,
            coords=jnp.asarray(coords, dtype=jnp.float32),
            block_coord_offset=block_coord_offset_j,
            inter_residue_connections=jnp.asarray(
                inter_residue_connections, dtype=jnp.int32
            ),
            inter_block_bondsep=jnp.asarray(inter_block_bondsep, dtype=jnp.int32),
            block_type_ind=jnp.asarray(block_type_ind, dtype=jnp.int32),
            pdb_info=pdb_info,
        )


__all__ = [
    "PoseStackBuilder",
    "MAX_SIG_BOND_SEPARATION",
]
