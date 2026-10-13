"""Count-pair connectivity weight function and bonded-path-length helper.

Short-range non-bonded energies are damped (or zeroed) for atom pairs
that are close along the bonded graph. Rosetta's convention:

    path_length  > 4  -> weight 1.0  (full interaction)
    path_length == 4  -> weight 0.2  (1-4 interaction, scaled)
    path_length  < 4  -> weight 0.0  (1-1, 1-2, 1-3: excluded)

:func:`connectivity_weight` operates elementwise on arrays and is safe
under ``jit``/``vmap``. :func:`pairwise_bonded_path_length` is the
per-pose-stack setup helper that builds the full ``[P, N, N]``
bond-distance matrix for a ``PoseStack`` by threading through
inter-residue connections. It is implemented as a pure-JAX jit-able
function that lives on-device, so repeated setup calls for equivalently
shaped ``PoseStack``s amortise the first-call XLA compilation cost.
"""

from __future__ import annotations

import collections
import weakref
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from tmol_jax.types import Array

# Share setup across terms without retaining dead poses or trusting recycled IDs.
_CACHE_CAP = 128


def _cache_put(cache, key, value, owners):
    def release(reference):
        entry = cache.get(key)
        if entry is not None and any(ref is reference for ref in entry[0]):
            cache.pop(key, None)

    cache[key] = (tuple(weakref.ref(owner, release) for owner in owners), value)
    cache.move_to_end(key)
    if len(cache) > _CACHE_CAP:
        cache.popitem(last=False)


def _cache_get(cache, key, owners):
    entry = cache.get(key)
    if entry is None:
        return None
    references, value = entry
    if len(references) != len(owners) or any(
        ref() is not owner for ref, owner in zip(references, owners)
    ):
        cache.pop(key)
        return None
    cache.move_to_end(key)
    return value


def _cache_owners(pose, pbt):
    return (
        pose,
        pbt,
        pose.block_type_ind,
        pose.block_coord_offset,
        pose.inter_block_bondsep,
        pbt.n_atoms,
        pbt.conn_atoms,
        pbt.intrares_path_distance,
    )


# Max chemical-bond separation that matters for count-pair. Anything
# larger is clamped to this value. The weight function returns 1.0 for
# ``path_length > 4``, so any value >= 5 has the same effect; we use 6
# for safety (matches upstream's ``MAX_SIG_BOND_SEPARATION`` slot
# padding).
MAX_COUNTPAIR_SEPARATION: int = 6


def connectivity_weight(bonded_path_length: Array) -> Array:
    """Count-pair weight for the given bonded path length(s).

    Returns a float32 array matching the shape of ``bonded_path_length``.
    """
    bpl = jnp.asarray(bonded_path_length)
    # Start from the "too close" case (weight 0), then override for
    # 1-4 and further-than-1-4 in ascending precedence. Using
    # ``jnp.where`` means this is a pure functional cascade with no
    # Python branching on traced values.
    w = jnp.where(bpl == 4, jnp.float32(0.2), jnp.float32(0.0))
    w = jnp.where(bpl > 4, jnp.float32(1.0), w)
    return w


# --------------------------------------------------------------------- #
# JIT-able pose-level pair lookups                                       #
# --------------------------------------------------------------------- #


@partial(jax.jit, static_argnames=("max_n_atoms_per_pose",))
def pose_atom_block_index(
    block_type_ind: jax.Array,
    block_coord_offset: jax.Array,
    n_atoms_per_type: jax.Array,
    max_n_atoms_per_pose: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Map every pose-level atom slot to its owning block and local index.

    Given the standard PoseStack layout arrays, return three ``[P, N]``
    arrays:

    * ``block_for_atom``: index of the block containing this atom
      (``0 <= idx < max_n_blocks``). Padding atoms (slots beyond every
      real block's span) get ``0``; callers should gate with
      ``atom_is_real``.
    * ``local_for_atom``: index of this atom *within* its block
      (``0 <= idx < n_atoms_per_type[bti]``). Zero for padding.
    * ``atom_is_real``: bool, ``True`` iff this slot falls inside the
      span of a real block.

    Pure JAX, ``jit`` / ``vmap`` safe. Uses a broadcast comparison against
    block-coord-offset and end-offset to vectorise the "which block owns
    this atom" question.

    Wrapped in ``jax.jit`` so repeated setup calls (every energy term
    computes this once) amortise the compilation. Only
    ``max_n_atoms_per_pose`` is static.
    """
    P, B = block_type_ind.shape
    real_block = block_type_ind >= 0
    safe_t = jnp.where(real_block, block_type_ind, 0)
    n_atoms_b = jnp.where(real_block, n_atoms_per_type[safe_t], 0)
    end_offsets = block_coord_offset + n_atoms_b  # [P, B]

    atom_idx = jnp.arange(max_n_atoms_per_pose)  # [N]
    in_block = (atom_idx[None, None, :] >= block_coord_offset[:, :, None]) & (
        atom_idx[None, None, :] < end_offsets[:, :, None]
    )  # [P, B, N]
    atom_is_real = in_block.any(axis=1)  # [P, N]
    block_for_atom = jnp.argmax(in_block.astype(jnp.int32), axis=1).astype(
        jnp.int32
    )  # [P, N]
    P_ar = jnp.arange(P)[:, None]
    local_for_atom = (
        atom_idx[None, :] - block_coord_offset[P_ar, block_for_atom]
    ).astype(jnp.int32)
    return block_for_atom, local_for_atom, atom_is_real


def scatter_per_atom(
    per_type_table: jax.Array,
    block_type_ind: jax.Array,
    block_coord_offset: jax.Array,
    n_atoms_per_type: jax.Array,
    max_n_atoms_per_pose: int,
    default: Any = 0,
) -> jax.Array:
    """Scatter a ``[n_types, max_n_atoms_per_block[, ...]]`` palette-level
    table into ``[P, max_n_atoms_per_pose[, ...]]`` pose-atom layout.

    This is the vectorised replacement for the typical numpy "per-pose,
    per-block slice-and-copy" loop that every energy term setup used to
    run. Works for float/int/bool per-atom tables via a single gather
    along the atom axis.

    ``default`` fills padding slots (non-real atoms). Uses the semantics
    of ``jnp.where(atom_is_real[..., None..None], gather, default)``.
    """
    block_for_atom, local_for_atom, atom_is_real = pose_atom_block_index(
        block_type_ind,
        block_coord_offset,
        n_atoms_per_type,
        max_n_atoms_per_pose,
    )  # [P, N], [P, N], [P, N]
    P = block_type_ind.shape[0]
    safe_bti = jnp.where(block_type_ind >= 0, block_type_ind, 0)
    # bti_of_atom[p, i] = block_type_ind[p, block_for_atom[p, i]]
    bti_of_atom = safe_bti[jnp.arange(P)[:, None], block_for_atom]  # [P, N]
    # gather per_type_table[bti_of_atom, local_for_atom]
    vals = per_type_table[bti_of_atom, local_for_atom]
    # Mask padding
    default_arr = jnp.asarray(default, dtype=vals.dtype)
    # Broadcast: vals may have trailing axes (e.g. [P, N, 3] for coords).
    extra_axes = vals.ndim - atom_is_real.ndim
    mask = atom_is_real.reshape(atom_is_real.shape + (1,) * extra_axes)
    return jnp.where(mask, vals, default_arr)


# --------------------------------------------------------------------- #
# Pose-level index cache + bulk scatter                                 #
# --------------------------------------------------------------------- #
#
# Per-pose-stack cache of the (block_for_atom, local_for_atom,
# atom_is_real, bti_of_atom) tuple. Every energy term computes this
# quantity once to drive palette-table scatter; memoising the result by
# ``id(pose_stack)`` (same convention as _BPL_CACHE) lets the 9-ish LJLK
# scatters, the 7-ish LKBall scatters, and the elec / hbond scatters
# share one ``pose_atom_block_index`` trace + gather.
_POSE_INDEX_CACHE: "collections.OrderedDict[int, tuple]" = collections.OrderedDict()


def get_pose_atom_index(pose_stack: Any, packed_block_types: Any = None) -> tuple:
    """Return cached (block_for_atom, local_for_atom, atom_is_real,
    bti_of_atom) for a pose stack.

    Every element is ``[P, N]`` int32 / bool. ``bti_of_atom[p, i]`` is
    the block-type index of the block that owns atom ``i`` in pose ``p``
    (the "safe" version — padding atoms read from bt 0; gate on
    ``atom_is_real`` before using the result).

    Keyed by ``id(pose_stack)`` so repeated setup calls across energy
    terms hit the cache.
    """
    if packed_block_types is None:
        packed_block_types = pose_stack.packed_block_types
    key = id(pose_stack)
    owners = _cache_owners(pose_stack, packed_block_types)
    cached = _cache_get(_POSE_INDEX_CACHE, key, owners)
    if cached is not None:
        return cached
    bti = jnp.asarray(pose_stack.block_type_ind, dtype=jnp.int32)
    offsets = jnp.asarray(pose_stack.block_coord_offset, dtype=jnp.int32)
    n_atoms_per_type = jnp.asarray(packed_block_types.n_atoms, dtype=jnp.int32)
    A = int(pose_stack.coords.shape[1])
    block_for_atom, local_for_atom, atom_is_real = pose_atom_block_index(
        bti, offsets, n_atoms_per_type, A
    )
    P = bti.shape[0]
    safe_bti = jnp.where(bti >= 0, bti, 0)
    bti_of_atom = safe_bti[jnp.arange(P)[:, None], block_for_atom]
    result = (block_for_atom, local_for_atom, atom_is_real, bti_of_atom)
    _cache_put(_POSE_INDEX_CACHE, key, result, owners)
    return result


def scatter_many_per_atom(
    tables: "tuple[jax.Array, ...]",
    pose_stack: Any,
    packed_block_types: Any = None,
    defaults: "tuple | None" = None,
) -> "tuple[jax.Array, ...]":
    """Batched ``scatter_per_atom`` for a set of per-BT palette tables.

    Computes ``pose_atom_block_index`` (+ ``bti_of_atom``) exactly once
    via :func:`get_pose_atom_index`, then performs a simple fancy-index
    gather per table. Each individual gather is a cheap op (no jit
    dispatch per call); the expensive index construction is shared.

    Arguments
    ---------
    tables
        Tuple of per-BT tables shaped ``[n_types, max_n_atoms[, ...]]``.
        All tables share the same leading axes; trailing axes may vary
        (e.g. int/bool/float, vector/scalar).
    pose_stack
        The pose stack being scattered onto.
    packed_block_types
        Matching PBT; defaults to ``pose_stack.packed_block_types``.
    defaults
        Optional per-table default values for padding slots. If
        supplied must match ``len(tables)``. Defaults to zero of the
        table's dtype.
    """
    if packed_block_types is None:
        packed_block_types = pose_stack.packed_block_types
    if defaults is None:
        defaults = (0,) * len(tables)
    elif len(defaults) != len(tables):
        raise ValueError(
            f"defaults length {len(defaults)} does not match tables length {len(tables)}"
        )

    _, local_for_atom, atom_is_real, bti_of_atom = get_pose_atom_index(
        pose_stack, packed_block_types
    )

    outputs: list[jax.Array] = []
    for table, default in zip(tables, defaults):
        vals = table[bti_of_atom, local_for_atom]
        default_arr = jnp.asarray(default, dtype=vals.dtype)
        extra_axes = vals.ndim - atom_is_real.ndim
        mask = atom_is_real.reshape(atom_is_real.shape + (1,) * extra_axes)
        outputs.append(jnp.where(mask, vals, default_arr))
    return tuple(outputs)


def pose_atom_real_mask(
    block_type_ind: jax.Array,
    block_coord_offset: jax.Array,
    n_atoms_per_type: jax.Array,
    max_n_atoms_per_pose: int,
) -> jax.Array:
    """Return ``bool[P, max_n_atoms_per_pose]`` real-atom mask.

    Pure JAX convenience wrapper around :func:`pose_atom_block_index`.
    """
    _, _, atom_is_real = pose_atom_block_index(
        block_type_ind,
        block_coord_offset,
        n_atoms_per_type,
        max_n_atoms_per_pose,
    )
    return atom_is_real


# --------------------------------------------------------------------- #
# Pairwise bonded-path-length matrix                                     #
# --------------------------------------------------------------------- #


def _extract_static(pose_stack: Any, packed_block_types: Any) -> tuple:
    pbt = packed_block_types
    bti = jnp.asarray(pose_stack.block_type_ind, dtype=jnp.int32)
    offsets = jnp.asarray(pose_stack.block_coord_offset, dtype=jnp.int32)
    ibs = jnp.asarray(pose_stack.inter_block_bondsep, dtype=jnp.int32)
    n_atoms_per_type = jnp.asarray(pbt.n_atoms, dtype=jnp.int32)
    conn_atoms = jnp.asarray(pbt.conn_atoms, dtype=jnp.int32)
    intra_pd = jnp.asarray(pbt.intrares_path_distance, dtype=jnp.int32)
    max_n_atoms_per_pose = int(pose_stack.coords.shape[1])
    return (
        bti,
        offsets,
        ibs,
        n_atoms_per_type,
        conn_atoms,
        intra_pd,
        max_n_atoms_per_pose,
    )


@partial(jax.jit, static_argnames=("max_n_atoms_per_pose", "max_sep"))
def _pairwise_bpl_jit(
    block_type_ind: jax.Array,  # [P, B] int32
    block_coord_offset: jax.Array,  # [P, B] int32
    inter_block_bondsep: jax.Array,  # [P, B, B, C, C] int32
    n_atoms_per_type: jax.Array,  # [T] int32
    conn_atoms: jax.Array,  # [T, C] int32
    per_type_pair_table: jax.Array,  # [T, M, M] int32 — intra-block pair BPL
    max_n_atoms_per_pose: int,
    max_sep: int = MAX_COUNTPAIR_SEPARATION,
) -> jax.Array:
    """JIT-friendly core. Builds ``[P, N, N]`` pair BPL using Floyd-Warshall
    across residue-connection points.

    ``per_type_pair_table`` is the per-BT *intra-block* pair path-length
    table used for both the (atom -> conn) and (conn -> conn) intra-block
    edges. For the standard LJLK / HBond / LK-ball path this is
    ``intrares_path_distance`` from the PBT; for the elec path (which uses
    representative atom remapping) the caller should supply the
    ``intra_repr_path_distance`` table.

    Floyd-Warshall runs on the ``[B * C, B * C]`` connection-point graph
    per pose. For proteins of ≲1500 atoms (``B * C`` on the order of 2-300)
    this is a ≪10 ms ``lax.fori_loop``.
    """
    P, B = block_type_ind.shape
    C = conn_atoms.shape[1]
    Nn = B * C
    BIG = max_sep + 1

    real_block = block_type_ind >= 0
    safe_t = jnp.where(real_block, block_type_ind, 0)

    conn_atoms_pose = conn_atoms[safe_t]
    conn_atoms_pose = jnp.where(real_block[:, :, None], conn_atoms_pose, -1)
    valid_conn = conn_atoms_pose >= 0
    safe_ca = jnp.where(valid_conn, conn_atoms_pose, 0)

    # ---- Intra-block conn-conn edges ---- #
    pd_blocks = per_type_pair_table[safe_t]  # [P, B, M, M]
    pd_intra_conn = pd_blocks[
        jnp.arange(P)[:, None, None, None],
        jnp.arange(B)[None, :, None, None],
        safe_ca[:, :, :, None],
        safe_ca[:, :, None, :],
    ]  # [P, B, C, C]
    valid_intra = valid_conn[:, :, :, None] & valid_conn[:, :, None, :]
    pd_intra_conn = jnp.where(valid_intra, pd_intra_conn, BIG)

    # ---- Inter-block conn-conn edges ---- #
    ibs = inter_block_bondsep  # [P, B, B, C, C]
    valid_ba = valid_conn[:, :, None, :, None]
    valid_bb = valid_conn[:, None, :, None, :]
    ibs = jnp.where(valid_ba & valid_bb, ibs, BIG)
    block_diag_sel = jnp.arange(B)[:, None] == jnp.arange(B)[None, :]
    ibs = jnp.where(block_diag_sel[None, :, :, None, None], BIG, ibs)

    # Combine into a [P, B, B, C, C] unified table with intra-block on
    # the diagonal and inter-block elsewhere.
    pd_intra_full = (
        jnp.zeros_like(ibs).at[:, jnp.arange(B), jnp.arange(B), :, :].set(pd_intra_conn)
    )
    unified = jnp.where(block_diag_sel[None, :, :, None, None], pd_intra_full, ibs)

    # Reshape to [P, Nn, Nn]: node index = b * C + c.
    D = unified.transpose(0, 1, 3, 2, 4).reshape(P, Nn, Nn)
    diag_idx = jnp.arange(Nn)
    D = D.at[:, diag_idx, diag_idx].set(0)

    # ---- Floyd-Warshall ---- #
    def fw_step(k, D):
        Dk_row = jax.lax.dynamic_slice_in_dim(D, k, 1, axis=1)
        Dk_col = jax.lax.dynamic_slice_in_dim(D, k, 1, axis=2)
        return jnp.minimum(D, Dk_col + Dk_row)

    D = jax.lax.fori_loop(0, Nn, fw_step, D)
    D = jnp.minimum(D, max_sep)  # [P, Nn, Nn]

    # ---- Map every (p, i) to (block, local) ---- #
    n_atoms_b = jnp.where(real_block, n_atoms_per_type[safe_t], 0)
    end_offsets = block_coord_offset + n_atoms_b
    atom_idx = jnp.arange(max_n_atoms_per_pose)
    in_block = (atom_idx[None, None, :] >= block_coord_offset[:, :, None]) & (
        atom_idx[None, None, :] < end_offsets[:, :, None]
    )  # [P, B, N]
    atom_real = in_block.any(axis=1)  # [P, N]
    block_for_atom = jnp.argmax(in_block.astype(jnp.int32), axis=1).astype(jnp.int32)
    local_for_atom = (
        atom_idx[None, :] - block_coord_offset[jnp.arange(P)[:, None], block_for_atom]
    ).astype(jnp.int32)
    bti_of_atom = safe_t[jnp.arange(P)[:, None], block_for_atom]  # [P, N]

    # ---- Intra-block atom pair BPL ---- #
    # intra_bpl[p, i, j] = per_type_pair_table[bti_i, local_i, local_j]
    # only valid when block_for_atom[i] == block_for_atom[j].
    intra_bpl = per_type_pair_table[
        bti_of_atom[:, :, None],
        local_for_atom[:, :, None],
        local_for_atom[:, None, :],
    ]  # [P, N, N]

    # ---- Inter-block atom pair BPL ---- #
    # For each atom i, we need the intra-block distance from its local
    # position to each connection atom on its block.
    conn_atom_i = conn_atoms_pose[jnp.arange(P)[:, None], block_for_atom, :]
    valid_conn_i = valid_conn[jnp.arange(P)[:, None], block_for_atom, :]
    safe_conn_i = jnp.where(valid_conn_i, conn_atom_i, 0)

    # da[p, i, ca] = path inside i's block from local_i to the conn atom
    da = per_type_pair_table[
        bti_of_atom[:, :, None],
        local_for_atom[:, :, None],
        safe_conn_i,
    ]  # [P, N, C]
    da = jnp.where(valid_conn_i, da, BIG)

    # db[p, j, cb] = path inside j's block from the conn atom to local_j
    db = per_type_pair_table[
        bti_of_atom[:, :, None],
        safe_conn_i,
        local_for_atom[:, :, None],
    ]  # [P, N, C]
    db = jnp.where(valid_conn_i, db, BIG)

    # Gather the connection-graph distance.
    node_i = block_for_atom[:, :, None] * C + jnp.arange(C)[None, None, :]
    node_j = block_for_atom[:, :, None] * C + jnp.arange(C)[None, None, :]

    Dg = D[
        jnp.arange(P)[:, None, None, None, None],
        node_i[:, :, None, :, None],
        node_j[:, None, :, None, :],
    ]  # [P, N, N, C, C]

    combined = da[:, :, None, :, None] + Dg + db[:, None, :, None, :]
    inter_bpl = jnp.min(combined, axis=(-1, -2))
    inter_bpl = jnp.minimum(inter_bpl, max_sep)

    same_block = block_for_atom[:, :, None] == block_for_atom[:, None, :]
    intra_bpl_clamped = jnp.minimum(intra_bpl, max_sep)
    bpl = jnp.where(same_block, intra_bpl_clamped, inter_bpl)

    real_pair = atom_real[:, :, None] & atom_real[:, None, :]
    bpl = jnp.where(real_pair, bpl, max_sep)
    # The canonical clamp is 6, so retain the persistent O(N^2) table in
    # one byte. Keep unusual public callers with a larger clamp lossless.
    dtype = jnp.uint8 if max_sep <= 255 else jnp.int32
    return bpl.astype(dtype)


# Per-pose-stack cache for the APSP bonded-path-length matrix. Keyed by
# ``id(pose_stack)`` so any energy term computing this expensive quantity
# reuses the first result. Bounded LRU (``_CACHE_CAP``): a batched sweep of
# many pose stacks evicts old entries instead of leaking [P, N, N] arrays.
_BPL_CACHE: "collections.OrderedDict[int, jax.Array]" = collections.OrderedDict()


def pairwise_bonded_path_length(
    pose_stack: Any,
    packed_block_types: Any = None,
    max_sep: int = MAX_COUNTPAIR_SEPARATION,
) -> jnp.ndarray:
    """Build the exact ``[P, N, N]`` bonded-path-length matrix.

    For intra-residue pairs the value is
    ``packed_block_types.intrares_path_distance[bt, a, b]``.

    For cross-residue pairs we first run an APSP (all-pairs shortest
    paths) over the residue-connection-point graph, so that the
    shortest path between any two connection points in the pose is
    known, threading through any number of intermediate residues. We
    then combine each atom's intra-block distance to a connection
    point with the inter-connection APSP distance:

        out[A, B] = min over (ca on A's block, cb on B's block) of:
            intra_path(A, conn_atom_ca) + conn_apsp[(ba, ca), (bb, cb)]
            + intra_path(conn_atom_cb, B)

    Values are clipped above by ``max_sep`` (the weight function
    returns 1.0 for anything ``> 4`` so the clip is only for integer
    bookkeeping).

    This mirrors the ``CountPair::inter_block_separation`` logic and
    the APSP-based ``min_block_bondsep`` tensor built by
    ``PoseStackBuilder._find_inter_block_separation_*`` in upstream.
    The direct ``inter_block_bondsep`` connection distance alone (1 for
    peptide-bonded pairs, max_sep otherwise) is correct for adjacent
    residues but misses multi-residue paths -- e.g. res_i:C -> res_{i+2}:N
    is 4 bonds, which must receive the 0.2 "1-4 interaction" weight rather
    than the full weight that ``max_sep`` implies. The APSP relaxation
    below recovers those paths.

    Implemented as a ``jax.jit``'d routine; the Floyd-Warshall relaxation
    over residue-connection points runs via ``lax.fori_loop`` and the
    atom-level combine uses broadcasted gather. For a 1ubq-class pose
    (~1200 atoms, ~70 blocks) this returns in ~10 ms once the XLA graph
    has been cached for this shape.

    Parameters
    ----------
    pose_stack
        A :class:`~tmol_jax.pose.pose_stack.PoseStack`. We use
        ``block_type_ind``, ``block_coord_offset``,
        ``inter_block_bondsep``, and ``coords.shape[1]``.
    packed_block_types
        The matching :class:`~tmol_jax.pose.packed_block_types.PackedBlockTypes`.
        Defaults to ``pose_stack.packed_block_types``.
    max_sep
        Upper clamp on the integer path length. Defaults to
        :data:`MAX_COUNTPAIR_SEPARATION`.

    Returns
    -------
    ``jnp.uint8[n_poses, max_n_atoms_per_pose, max_n_atoms_per_pose]``
        Every pair (including ``i == j`` which reads 0 from
        ``intrares_path_distance``) gets a chemical-bond path length.
        Cross-residue pairs for which no connection pair can reach
        within ``max_sep`` bonds are left at ``max_sep``. A noncanonical
        ``max_sep > 255`` returns ``int32`` instead.
    """
    if packed_block_types is None:
        packed_block_types = pose_stack.packed_block_types

    # Per-pose-stack memoisation — same (pose_stack, max_sep) call from
    # LJLK / LKBall / HBond / elec should hit the cache without redoing
    # the APSP. Keyed by object ``id`` (pose stacks are long-lived
    # containers and not hashable as pytrees), with ``max_sep`` in the
    # key too to respect the rare callers that change it.
    cache_key = (id(pose_stack), int(max_sep))
    owners = _cache_owners(pose_stack, packed_block_types)
    cached = _cache_get(_BPL_CACHE, cache_key, owners)
    if cached is not None:
        return cached

    (
        bti,
        offsets,
        ibs,
        n_atoms_per_type,
        conn_atoms,
        intra_pd,
        max_n_atoms_per_pose,
    ) = _extract_static(pose_stack, packed_block_types)

    result = _pairwise_bpl_jit(
        bti,
        offsets,
        ibs,
        n_atoms_per_type,
        conn_atoms,
        intra_pd,
        max_n_atoms_per_pose,
        max_sep,
    )
    _cache_put(_BPL_CACHE, cache_key, result, owners)
    return result


@partial(jax.jit, static_argnames=("max_n_atoms_per_pose", "max_sep"))
def _pairwise_bpl_direct_jit(
    block_type_ind: jax.Array,  # [P, B] int32
    block_coord_offset: jax.Array,  # [P, B] int32
    inter_block_bondsep: jax.Array,  # [P, B, B, C, C] int32
    n_atoms_per_type: jax.Array,  # [T] int32
    conn_atoms: jax.Array,  # [T, C] int32
    per_type_intra_rpd: jax.Array,  # [T, M, M] int32
    per_type_inter_rpd: jax.Array,  # [T, M, M] int32 (path_distance[a, rep[b]])
    max_n_atoms_per_pose: int,
    max_sep: int = MAX_COUNTPAIR_SEPARATION,
) -> jax.Array:
    """Direct (no-APSP) variant — uses ``inter_block_bondsep`` only for
    1-hop cross-residue connections. Matches upstream's elec
    ``inter_block_separation`` semantics: intra-block uses
    ``intra_repr_path_distance`` (= ``path[rep[a], rep[b]]``), cross-block
    uses ``inter_repr_path_distance`` (= ``path[conn_atom, rep[.]]``) plus
    ``inter_block_bondsep``.

    Does NOT run Floyd-Warshall because the elec path-length convention
    only follows one connection hop.
    """
    P, B = block_type_ind.shape
    C = conn_atoms.shape[1]

    real_block = block_type_ind >= 0
    safe_t = jnp.where(real_block, block_type_ind, 0)
    n_atoms_b = jnp.where(real_block, n_atoms_per_type[safe_t], 0)

    conn_atoms_pose = conn_atoms[safe_t]
    conn_atoms_pose = jnp.where(real_block[:, :, None], conn_atoms_pose, -1)
    valid_conn = conn_atoms_pose >= 0

    # Map every (p, i) to (block, local).
    end_offsets = block_coord_offset + n_atoms_b
    atom_idx = jnp.arange(max_n_atoms_per_pose)
    in_block = (atom_idx[None, None, :] >= block_coord_offset[:, :, None]) & (
        atom_idx[None, None, :] < end_offsets[:, :, None]
    )
    atom_real = in_block.any(axis=1)
    block_for_atom = jnp.argmax(in_block.astype(jnp.int32), axis=1).astype(jnp.int32)
    local_for_atom = (
        atom_idx[None, :] - block_coord_offset[jnp.arange(P)[:, None], block_for_atom]
    ).astype(jnp.int32)
    bti_of_atom = safe_t[jnp.arange(P)[:, None], block_for_atom]  # [P, N]

    # Intra-block pair BPL: per_type_intra_rpd[bti_i, local_i, local_j]
    intra_bpl = per_type_intra_rpd[
        bti_of_atom[:, :, None],
        local_for_atom[:, :, None],
        local_for_atom[:, None, :],
    ]  # [P, N, N]
    intra_bpl_clamped = jnp.minimum(intra_bpl, max_sep)

    # Inter-block pair BPL:
    #   best = min_{ca, cb} [per_type_inter_rpd[bti_i, ca_atom, local_i]
    #                        + inter_block_bondsep[p, ba, bb, ca, cb]
    #                        + per_type_inter_rpd[bti_j, cb_atom, local_j]]
    # Read the conn-atom indices (per-BT) for each atom's owning block.
    conn_atom_i = conn_atoms_pose[jnp.arange(P)[:, None], block_for_atom, :]
    valid_conn_i = valid_conn[jnp.arange(P)[:, None], block_for_atom, :]
    safe_conn_i = jnp.where(valid_conn_i, conn_atom_i, 0)

    # da[p, i, ca] = per_type_inter_rpd[bti_i, ca_atom, local_i]
    da = per_type_inter_rpd[
        bti_of_atom[:, :, None], safe_conn_i, local_for_atom[:, :, None]
    ]
    da = jnp.where(valid_conn_i, da, max_sep + 1)

    # db[p, j, cb] = per_type_inter_rpd[bti_j, cb_atom, local_j]
    db = per_type_inter_rpd[
        bti_of_atom[:, :, None], safe_conn_i, local_for_atom[:, :, None]
    ]
    db = jnp.where(valid_conn_i, db, max_sep + 1)

    # Gather inter_block_bondsep entries into [P, N, N, C, C].
    # ibs[p, ba, bb, ca, cb]. ba = block_for_atom[p, i], bb = block_for_atom[p, j].
    ibs_gather = inter_block_bondsep[
        jnp.arange(P)[:, None, None, None, None],
        block_for_atom[:, :, None, None, None],
        block_for_atom[:, None, :, None, None],
        jnp.arange(C)[None, None, None, :, None],
        jnp.arange(C)[None, None, None, None, :],
    ]  # [P, N, N, C, C]

    # Combine (outer sum over ca, cb).
    combined = da[:, :, None, :, None] + ibs_gather + db[:, None, :, None, :]
    inter_bpl = jnp.min(combined, axis=(-1, -2))
    inter_bpl = jnp.minimum(inter_bpl, max_sep)

    same_block = block_for_atom[:, :, None] == block_for_atom[:, None, :]
    bpl = jnp.where(same_block, intra_bpl_clamped, inter_bpl)

    real_pair = atom_real[:, :, None] & atom_real[:, None, :]
    bpl = jnp.where(real_pair, bpl, max_sep)
    dtype = jnp.uint8 if max_sep <= 255 else jnp.int32
    return bpl.astype(dtype)


# Cache for elec-style direct bpl, same lifecycle semantics as
# :data:`_BPL_CACHE`. Keyed by ``(id(pose_stack), id(intra_rpd),
# id(inter_rpd), max_sep)`` so callers with different rep tables don't
# collide.
_BPL_DIRECT_CACHE: "collections.OrderedDict[tuple, jax.Array]" = (
    collections.OrderedDict()
)


def pair_bonded_path_length_direct(
    pose_stack: Any,
    packed_block_types: Any,
    per_type_intra_rpd: jax.Array,
    per_type_inter_rpd: jax.Array,
    max_sep: int = MAX_COUNTPAIR_SEPARATION,
) -> jax.Array:
    """Elec-style pair BPL: uses rep-remapped tables + direct inter-bondsep
    (no APSP over connection points).

    Memoised per ``(id(pose_stack), id(intra_rpd), id(inter_rpd),
    max_sep)`` so repeated setup calls of the elec term amortise the
    jit compilation + gather work.
    """
    cache_key = (
        id(pose_stack),
        id(per_type_intra_rpd),
        id(per_type_inter_rpd),
        int(max_sep),
    )
    owners = _cache_owners(pose_stack, packed_block_types) + (
        per_type_intra_rpd,
        per_type_inter_rpd,
    )
    cached = _cache_get(_BPL_DIRECT_CACHE, cache_key, owners)
    if cached is not None:
        return cached
    bti = jnp.asarray(pose_stack.block_type_ind, dtype=jnp.int32)
    offsets = jnp.asarray(pose_stack.block_coord_offset, dtype=jnp.int32)
    ibs = jnp.asarray(pose_stack.inter_block_bondsep, dtype=jnp.int32)
    n_atoms_per_type = jnp.asarray(packed_block_types.n_atoms, dtype=jnp.int32)
    conn_atoms = jnp.asarray(packed_block_types.conn_atoms, dtype=jnp.int32)
    max_n_atoms_per_pose = int(pose_stack.coords.shape[1])
    result = _pairwise_bpl_direct_jit(
        bti,
        offsets,
        ibs,
        n_atoms_per_type,
        conn_atoms,
        jnp.asarray(per_type_intra_rpd, dtype=jnp.int32),
        jnp.asarray(per_type_inter_rpd, dtype=jnp.int32),
        max_n_atoms_per_pose,
        max_sep,
    )
    _cache_put(_BPL_DIRECT_CACHE, cache_key, result, owners)
    return result


__all__ = [
    "MAX_COUNTPAIR_SEPARATION",
    "connectivity_weight",
    "pose_atom_block_index",
    "scatter_per_atom",
    "scatter_many_per_atom",
    "get_pose_atom_index",
    "pose_atom_real_mask",
    "pairwise_bonded_path_length",
    "pair_bonded_path_length_direct",
]
