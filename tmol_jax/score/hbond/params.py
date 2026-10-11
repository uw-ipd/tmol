"""Hydrogen-bond parameter containers and resolvers.

Port of :mod:`tmol.score.hbond.params` (torch / pandas) to pure JAX.
This module owns three separable concerns:

* **Global knobs.** A handful of scalars that control the analytic
  shape of the hbond potential (sp2 chi penalty geometry, sp3 softmax
  fade temperature, the A-H cutoff distance).
* **Per-donor-type / per-acceptor-type weights.** Paired as a 2-D
  ``[n_donor_types, n_acceptor_types]`` "pair" table. The upstream code
  factors these into ``donor_weight * acceptor_weight`` and we preserve
  that factorization here for clarity, then multiply into a single pair
  weight ready for use.
* **Polynomial tables.** Four polynomial fits per (donor_type,
  acceptor_type) pair: AH distance, cos(AHD) (actually angle-not-cos,
  see ``potentials.py``), cos(BAH), and the sp2 chi penalty (often
  referred to as ``cosBAH_chi`` in the upstream code — a misnomer
  preserved for compatibility).

Polynomial layout matches the C++ ``HBondPoly`` struct: 11 coefficients
in descending-power order (``numpy.polyval`` convention), plus an
``(x_lo, x_hi)`` active range and an ``(y_lo, y_hi)`` clamp value for
samples outside that range.

The ``hbond_params_for_pose_stack`` entry point materialises everything
a downstream scoring pass needs given a PoseStack + its PackedBlockTypes
+ the raw hbond YAML-backed database:

* per-pose ``donor_H_atom`` / ``donor_heavy_atom`` index arrays
* per-pose ``acceptor_atom`` / ``acceptor_base`` / ``acceptor_base0``
  / ``acceptor_hybridization`` index arrays
* a (donor_type, acceptor_type) indexed ``HBondPolynomials`` +
  ``HBondPairParams`` structure for efficient gather during pair
  evaluation
"""

from __future__ import annotations

from typing import Any, Tuple

import jax.numpy as jnp
import numpy as np

from tmol_jax.types import pytree_dataclass

# --------------------------------------------------------------------- #
# Hybridization enum (values match tmol's C++ AcceptorHybridization)     #
# --------------------------------------------------------------------- #

HYBRIDIZATION_NONE = 0
HYBRIDIZATION_SP2 = 1
HYBRIDIZATION_SP3 = 2
HYBRIDIZATION_RING = 3

_HYB_NAME_TO_INT = {
    None: HYBRIDIZATION_NONE,
    "sp2": HYBRIDIZATION_SP2,
    "sp3": HYBRIDIZATION_SP3,
    "ring": HYBRIDIZATION_RING,
}


# --------------------------------------------------------------------- #
# Parameter pytrees                                                      #
# --------------------------------------------------------------------- #


@pytree_dataclass
class HBondGlobalParams:
    """Scalar global knobs for the hbond potential.

    All fields are 0-D ``jax.Array``s so this class is a leaf-flat
    pytree (just 6 floats). Using arrays rather than Python floats
    keeps the object jit-friendly and device-stable.
    """

    hb_sp2_range_span: jnp.ndarray  # float32 scalar
    hb_sp2_BAH180_rise: jnp.ndarray  # float32 scalar
    hb_sp2_outer_width: jnp.ndarray  # float32 scalar
    hb_sp3_softmax_fade: jnp.ndarray  # float32 scalar
    threshold_distance: jnp.ndarray  # float32 scalar
    max_ah_dist: jnp.ndarray  # float32 scalar

    @classmethod
    def from_db(cls, hbond_db: Any) -> "HBondGlobalParams":
        """Build a ``HBondGlobalParams`` from a parsed ``HBondDatabase``.

        ``max_ah_dist`` is derived — it's the largest A-H cutoff across
        all distance polynomials in the database, and gates the
        short-circuit in the pair scoring loop.
        """
        g = hbond_db.global_parameters
        polys = hbond_db.polynomial_parameters
        ahdist_polys = [p for p in polys if p.dimension == "hbgd_AHdist"]
        max_ah = max(p.xmax for p in ahdist_polys) if ahdist_polys else 6.0

        def f(v):
            return jnp.asarray(v, dtype=jnp.float32)

        return cls(
            hb_sp2_range_span=f(g.hb_sp2_range_span),
            hb_sp2_BAH180_rise=f(g.hb_sp2_BAH180_rise),
            hb_sp2_outer_width=f(g.hb_sp2_outer_width),
            hb_sp3_softmax_fade=f(g.hb_sp3_softmax_fade),
            threshold_distance=f(g.threshold_distance),
            max_ah_dist=f(max_ah),
        )


@pytree_dataclass
class HBondPairParams:
    """Per (donor_type, acceptor_type) scalar pair knobs.

    ``weight`` is the *product* of the donor-type weight and the
    acceptor-type weight, precomputed once so the inner scoring loop
    avoids an extra multiply per pair.

    ``acceptor_hybridization`` is per-acceptor-type (duplicated along
    the donor-axis for gather convenience); values use the integer
    enum in this module.
    """

    weight: jnp.ndarray  # float32[n_don_types, n_acc_types]
    acceptor_hybridization: jnp.ndarray  # int32[n_don_types, n_acc_types]


@pytree_dataclass
class HBondPolynomials:
    """Four polynomial tables, each indexed by (donor_type, acceptor_type).

    Each tensor carries its 11 coefficients + (x_lo, x_hi) + (y_lo, y_hi)
    as a trailing "param" axis of length 15 (11 + 2 + 2). This layout
    lets us gather a full polynomial spec with one ``lax.gather`` call.

    Fields:

    * ``AHdist_poly`` — A-H distance.
    * ``cosAHD_poly`` — AHD *angle* (the polynomial is fit in angle
      space despite the ``cos`` in the name; see ``potentials.py``).
    * ``cosBAH_poly`` — cos(BAH).

    The sp2 chi penalty is analytic (see :func:`chi_angle_V`), not a
    polynomial, so no chi table is stored here.
    """

    AHdist_poly: jnp.ndarray  # float32[n_don, n_acc, 15]
    cosAHD_poly: jnp.ndarray  # float32[n_don, n_acc, 15]
    cosBAH_poly: jnp.ndarray  # float32[n_don, n_acc, 15]


# --------------------------------------------------------------------- #
# Per-pose donor/acceptor identification                                 #
# --------------------------------------------------------------------- #


@pytree_dataclass
class HBondAtomParams:
    """Per-pose donor / acceptor atom indices + types.

    Shapes (``P`` poses, ``D`` max donors per pose, ``A`` max acceptors
    per pose). Padding is encoded with ``-1`` in the atom index fields.

    * ``donor_H_atom[P, D]``: atom index of the polar H.
    * ``donor_heavy_atom[P, D]``: atom index of the heavy D bonded to H.
    * ``donor_type[P, D]``: int index into the donor-type table.
    * ``acceptor_atom[P, A]``: atom index of A.
    * ``acceptor_base[P, A]``: atom index of B.
    * ``acceptor_base0[P, A]``: atom index of B0.
    * ``acceptor_type[P, A]``: int index into the acceptor-type table.
    * ``acceptor_hybridization[P, A]``: integer from the hybridization
      enum defined at the top of this module.

    B / B0 selection follows the upstream ``identification.hh`` logic
    per hybridization (see ``_pick_acceptor_bases``): sp2 walks to the
    carbonyl C and then out to a non-A neighbor; sp3 picks the H as B0
    and the heavy atom as B; ring picks two heavy ring neighbors. The
    walk uses the combined intra-residue + inter-residue ``all_bonds``
    table, so e.g. backbone O -> C -> CA works even across residue
    boundaries.
    """

    donor_H_atom: jnp.ndarray  # int32[P, D]
    donor_heavy_atom: jnp.ndarray  # int32[P, D]
    donor_type: jnp.ndarray  # int32[P, D]

    acceptor_atom: jnp.ndarray  # int32[P, A]
    acceptor_base: jnp.ndarray  # int32[P, A]
    acceptor_base0: jnp.ndarray  # int32[P, A]
    acceptor_type: jnp.ndarray  # int32[P, A]
    acceptor_hybridization: jnp.ndarray  # int32[P, A]


@pytree_dataclass
class HBondParams:
    """Everything needed to evaluate hbond energies on a PoseStack.

    Bundles five pieces the pair-eval kernel needs:

    * ``atom_params`` — per-pose donor/acceptor indices.
    * ``pair_params`` — per (donor_type, acceptor_type) weights + hyb.
    * ``polynomials`` — per (donor_type, acceptor_type) polynomials.
    * ``global_params`` — scalar knobs.
    * ``donor_acceptor_bond_sep`` — ``int32[P, D, A]`` chemical-bond
      path length between each donor-H and acceptor atom. Pairs with
      ``< 5`` bonds between them are excluded from the score (upstream
      count-pair convention; matches ``hbond_pose_score.impl.hh``).
    """

    atom_params: HBondAtomParams
    pair_params: HBondPairParams
    polynomials: HBondPolynomials
    global_params: HBondGlobalParams
    donor_acceptor_bond_sep: jnp.ndarray  # int32[P, D, A]


# --------------------------------------------------------------------- #
# Builders                                                               #
# --------------------------------------------------------------------- #


def _poly_to_array(poly_entry: Any) -> np.ndarray:
    """Pack bounded Chebyshev coefficients on [-1, 1], then x/y bounds.

    Converting the degree-10 monomial fits once in host float64 avoids their
    severe float32 cancellation without changing global JAX precision.
    """
    from numpy.polynomial import Polynomial, Chebyshev

    coefficients = [getattr(poly_entry, f"c_{i}") for i in range(11)]
    domain = [poly_entry.xmin, poly_entry.xmax]
    if not domain[1] > domain[0]:
        raise ValueError(f"invalid hydrogen-bond polynomial domain: {domain}")
    polynomial = Polynomial(coefficients).convert(kind=Chebyshev, domain=domain)
    coefficients = np.pad(polynomial.coef, (0, 11 - len(polynomial.coef)))
    return np.concatenate(
        [coefficients, domain, [poly_entry.min_val, poly_entry.max_val]]
    ).astype(np.float32)


def _build_pair_tables(
    hbond_db: Any, chemical_db: Any
) -> Tuple[HBondPairParams, HBondPolynomials, dict, dict]:
    """Build per (donor_type, acceptor_type) pair tables.

    Returns the pair-param / polynomial pytrees plus forward dicts
    mapping hbond-type name -> int index so callers can translate
    per-atom type assignments.
    """
    donor_types = list(
        getattr(hbond_db, "don_type_params", None)
        or getattr(hbond_db, "donor_type_params", [])
    )
    acceptor_types = list(
        getattr(hbond_db, "acc_type_params", None)
        or getattr(hbond_db, "acceptor_type_params", [])
    )
    donor_type_to_idx = {g.name: i for i, g in enumerate(donor_types)}
    acceptor_type_to_idx = {g.name: i for i, g in enumerate(acceptor_types)}
    n_don = len(donor_types)
    n_acc = len(acceptor_types)

    # Donor/acceptor weight outer product -> pair weight.
    d_w = np.asarray([g.weight for g in donor_types], dtype=np.float32)
    a_w = np.asarray([g.weight for g in acceptor_types], dtype=np.float32)
    weight = d_w[:, None] * a_w[None, :]

    # Acceptor hybridization, broadcast along donor axis. Upstream
    # indirection: hbond_db.acceptor_atom_types maps atom-type -> hb
    # acceptor-type; chemical_db.atom_types carries the hybridization.
    atom_hyb = {
        at.name: _HYB_NAME_TO_INT.get(at.acceptor_hybridization, 0)
        for at in chemical_db.atom_types
    }
    acc_hyb = np.zeros(n_acc, dtype=np.int32)
    for mapping in hbond_db.acceptor_atom_types:
        # Each atom-type may pair with exactly one acceptor-type in the
        # YAML, so the first one we see wins (the hybridization is a
        # property of the atom-type anyway).
        idx = acceptor_type_to_idx.get(mapping.acceptor_type)
        if idx is None:
            continue
        acc_hyb[idx] = atom_hyb.get(mapping.a, 0)
    hyb = np.broadcast_to(acc_hyb[None, :], (n_don, n_acc)).astype(np.int32)

    pair_params = HBondPairParams(
        weight=jnp.asarray(weight),
        acceptor_hybridization=jnp.asarray(hyb),
    )

    # Polynomial tables. Each pair slot gets its own full [15] vector;
    # slots with no explicit pair entry stay zero-filled, which makes
    # ``bound_poly`` return ``y_lo = y_hi = 0`` (our sentinel). We keep
    # coefficients in f64 to match upstream's ``bound_poly<11, double>``
    # precision — see ``_poly_to_array`` docstring for the rationale.
    poly_by_name = {p.name: _poly_to_array(p) for p in hbond_db.polynomial_parameters}
    zeros = np.zeros(15, dtype=np.float32)

    ahdist = np.zeros((n_don, n_acc, 15), dtype=np.float32)
    ahd = np.zeros((n_don, n_acc, 15), dtype=np.float32)
    bah = np.zeros((n_don, n_acc, 15), dtype=np.float32)

    for pp in hbond_db.pair_parameters:
        di = donor_type_to_idx.get(pp.donor_type)
        ai = acceptor_type_to_idx.get(pp.acceptor_type)
        if di is None or ai is None:
            continue
        ahdist[di, ai] = poly_by_name.get(pp.AHdist, zeros)
        ahd[di, ai] = poly_by_name.get(pp.cosAHD, zeros)
        bah[di, ai] = poly_by_name.get(pp.cosBAH, zeros)

    polynomials = HBondPolynomials(
        AHdist_poly=jnp.asarray(ahdist),
        cosAHD_poly=jnp.asarray(ahd),
        cosBAH_poly=jnp.asarray(bah),
    )

    return pair_params, polynomials, donor_type_to_idx, acceptor_type_to_idx


def _atom_type_hbond_maps(
    hbond_db: Any, donor_type_to_idx: dict, acceptor_type_to_idx: dict
):
    """Forward atom-type-name -> donor/acceptor int index (or -1)."""
    atype_to_don_type = {
        m.d: donor_type_to_idx[m.donor_type] for m in hbond_db.donor_atom_types
    }
    atype_to_acc_type = {
        m.a: acceptor_type_to_idx[m.acceptor_type] for m in hbond_db.acceptor_atom_types
    }
    return atype_to_don_type, atype_to_acc_type


def _neighbors_of(
    bt: int,
    local_atom: int,
    block_ind: int,
    bco: np.ndarray,
    bti: np.ndarray,
    all_bonds_pbt: np.ndarray,
    atom_all_bond_ranges_pbt: np.ndarray,
    conn_atoms_pbt: np.ndarray,
    n_atoms_per_type: np.ndarray,
    inter_residue_connections_pose: np.ndarray,
) -> list[int]:
    """Return global atom indices of every atom bonded to ``local_atom``
    (on block ``block_ind`` of type ``bt``), including inter-block
    chemical bonds resolved to the neighbor block's connection atom.

    Iteration order matches upstream ``BlockCentricIndexedBonds`` /
    ``bound_to``: intra-residue partners first (in sorted-by-index
    order, as stored in ``all_bonds`` by ``bonds_and_bond_ranges``),
    then inter-residue connection partners appended per-atom.
    """
    start, stop = (
        int(atom_all_bond_ranges_pbt[bt, local_atom, 0]),
        int(atom_all_bond_ranges_pbt[bt, local_atom, 1]),
    )
    if start < 0 or stop <= start:
        return []

    offset = int(bco[block_ind])
    out: list[int] = []
    for bidx in range(start, stop):
        partner_atm = int(all_bonds_pbt[bt, bidx, 1])
        if partner_atm >= 0:
            # Intra-residue bond; partner on the same block.
            out.append(offset + partner_atm)
            continue
        # Inter-residue bond: partner_atm == -1, conn_id tells us which
        # connection on this block to follow.
        conn_id = int(all_bonds_pbt[bt, bidx, 2])
        if conn_id < 0:
            continue
        # inter_residue_connections[block][conn_id] = (nbr_block, nbr_conn_id)
        nbr_block = int(inter_residue_connections_pose[block_ind, conn_id, 0])
        nbr_conn_id = int(inter_residue_connections_pose[block_ind, conn_id, 1])
        if nbr_block < 0 or nbr_conn_id < 0:
            continue  # unresolved (terminus / truncation)
        nbr_bt = int(bti[nbr_block])
        if nbr_bt < 0:
            continue
        # Atom on the neighbor block that the connection lands on.
        nbr_conn_atom = int(conn_atoms_pbt[nbr_bt, nbr_conn_id])
        if nbr_conn_atom < 0 or nbr_conn_atom >= int(n_atoms_per_type[nbr_bt]):
            continue
        nbr_offset = int(bco[nbr_block])
        out.append(nbr_offset + nbr_conn_atom)
    return out


def _resolve_icoor_ancestor(
    anc_name: str,
    bt_obj: Any,
    bt: int,
    block_ind: int,
    bco: np.ndarray,
    bti: np.ndarray,
    n_atoms_per_type: np.ndarray,
    conn_atoms_pbt: np.ndarray,
    inter_residue_connections_pose: np.ndarray,
) -> int:
    """Return the global atom index of an icoor parent/grand_parent name.

    The icoor tree can reference either an atom on this block (by atom
    name) or a connection point (by connection name, typically ``"up"``
    or ``"down"``). In the latter case, we follow the inter-residue
    connection to the neighboring block's connection atom.

    Returns ``-1`` if the name cannot be resolved (e.g. a truncated
    chain where ``down`` on residue 0 has no partner).
    """
    # Atom on this block.
    atom_idx = bt_obj.atom_to_idx.get(anc_name)
    if atom_idx is not None:
        return int(bco[block_ind]) + int(atom_idx)
    # Connection name -> walk across the connection.
    conn_to_cidx = bt_obj.connection_to_cidx
    conn_id = conn_to_cidx.get(anc_name)
    if conn_id is None or conn_id < 0:
        return -1
    nbr_block = int(inter_residue_connections_pose[block_ind, conn_id, 0])
    nbr_conn_id = int(inter_residue_connections_pose[block_ind, conn_id, 1])
    if nbr_block < 0 or nbr_conn_id < 0:
        return -1
    nbr_bt = int(bti[nbr_block])
    if nbr_bt < 0:
        return -1
    nbr_conn_atom = int(conn_atoms_pbt[nbr_bt, nbr_conn_id])
    if nbr_conn_atom < 0 or nbr_conn_atom >= int(n_atoms_per_type[nbr_bt]):
        return -1
    return int(bco[nbr_block]) + nbr_conn_atom


def _pick_acceptor_bases(
    A_global: int,
    A_local: int,
    bt: int,
    block_ind: int,
    hybridization: int,
    bt_obj: Any,
    bco: np.ndarray,
    bti: np.ndarray,
    all_bonds_pbt: np.ndarray,
    atom_all_bond_ranges_pbt: np.ndarray,
    conn_atoms_pbt: np.ndarray,
    n_atoms_per_type: np.ndarray,
    inter_residue_connections_pose: np.ndarray,
    atom_types_pbt: np.ndarray,
    atype_is_hydrogen: np.ndarray,
) -> Tuple[int, int]:
    """Return (B_global, B0_global) per upstream ``identification.hh``.

    Bond-graph walk (mirrors ``BlockCentricAcceptorBases::for_acceptor``):

    * sp2:  B = first heavy bonded to A; B0 = first neighbor of B != A
            (the B-walk skips back-to-start-residue hops so chain-ends
            see sensible frames).
    * sp3:  B0 = first H bonded to A; B = first neighbor of A != B0.
            The H anchors the softmax-faded BAH(B) + BAH(B0) blend used
            by the sp3 potential.
    * ring: B = first non-H bonded to A; B0 = next non-H neighbor of A,
            != B. Both frame atoms are ring-flanking heavy atoms, matching
            the convention the polynomial tables were fit against.

    Returns ``(-1, -1)`` when the rule can't be satisfied. Falls back to
    the icoor tree for ring acceptors only if the bond-graph walk fails
    (e.g. a truncated residue with a single ring neighbor left).
    """

    def is_h(global_idx: int) -> bool:
        # Identify hydrogen by looking up the atom-type element flag
        # through (block, local-atom). Cheap branch: try the current
        # block + neighbors first; fall back to a linear scan.
        for b_candidate in (block_ind, block_ind - 1, block_ind + 1):
            if b_candidate < 0 or b_candidate >= bti.shape[0]:
                continue
            bt_c = int(bti[b_candidate])
            if bt_c < 0:
                continue
            start_off = int(bco[b_candidate])
            n_c = int(n_atoms_per_type[bt_c])
            if start_off <= global_idx < start_off + n_c:
                local = global_idx - start_off
                atype_c = int(atom_types_pbt[bt_c, local])
                return bool(atype_is_hydrogen[atype_c]) if atype_c >= 0 else False
        for b_candidate in range(bti.shape[0]):
            bt_c = int(bti[b_candidate])
            if bt_c < 0:
                continue
            start_off = int(bco[b_candidate])
            n_c = int(n_atoms_per_type[bt_c])
            if start_off <= global_idx < start_off + n_c:
                local = global_idx - start_off
                atype_c = int(atom_types_pbt[bt_c, local])
                return bool(atype_is_hydrogen[atype_c]) if atype_c >= 0 else False
        return False

    nbrs_A = _neighbors_of(
        bt,
        A_local,
        block_ind,
        bco,
        bti,
        all_bonds_pbt,
        atom_all_bond_ranges_pbt,
        conn_atoms_pbt,
        n_atoms_per_type,
        inter_residue_connections_pose,
    )

    if hybridization == HYBRIDIZATION_SP2:
        B = -1
        for nbr in nbrs_A:
            if not is_h(nbr):
                B = nbr
                break
        if B < 0:
            return (-1, -1)

        b_block, b_bt, b_local = _locate_atom(B, bco, bti, n_atoms_per_type)
        if b_block < 0:
            return (-1, -1)
        nbrs_B = _neighbors_of(
            b_bt,
            b_local,
            b_block,
            bco,
            bti,
            all_bonds_pbt,
            atom_all_bond_ranges_pbt,
            conn_atoms_pbt,
            n_atoms_per_type,
            inter_residue_connections_pose,
        )
        B0 = -1
        for nbr in nbrs_B:
            nbr_block, _, _ = _locate_atom(nbr, bco, bti, n_atoms_per_type)
            # Matches the upstream "skip back-to-start-residue" guard:
            # if the walk has already left A's block, we don't want to
            # step back onto it for B0 (avoids degenerate frames when a
            # neighboring block is being redesigned).
            if b_block != block_ind and nbr_block == block_ind:
                continue
            if nbr != A_global:
                B0 = nbr
                break
        if B0 < 0:
            return (-1, -1)
        return (B, B0)

    if hybridization == HYBRIDIZATION_SP3:
        # First H neighbor is B0; first non-B0 neighbor is B.
        # This matches upstream ``BlockCentricAcceptorBases::sp3_acceptor_base``
        # exactly: the softmax-faded BAH blend evaluates against B (the
        # heavy-atom frame) and B0 (the proton-frame), and at canonical
        # hbond geometries one of the two gets a favourable polynomial
        # value while the other fades out.
        B0 = -1
        for nbr in nbrs_A:
            if is_h(nbr):
                B0 = nbr
                break
        if B0 < 0:
            return (-1, -1)
        B = -1
        for nbr in nbrs_A:
            if nbr != B0:
                B = nbr
                break
        if B < 0:
            return (-1, -1)
        return (B, B0)

    if hybridization == HYBRIDIZATION_RING:
        # First non-H neighbor is B; next non-H neighbor (!= B) is B0.
        # This keeps both frame atoms in the ring — the icoor grand_parent
        # for some ring nitrogens (e.g. HIS ND1) walks out of the ring to
        # the sidechain stem, which rotates the BAH axis off-canonical.
        B = -1
        for nbr in nbrs_A:
            if not is_h(nbr):
                B = nbr
                break
        if B < 0:
            return (-1, -1)
        B0 = -1
        for nbr in nbrs_A:
            if nbr != B and not is_h(nbr):
                B0 = nbr
                break
        if B0 >= 0:
            return (B, B0)
        # Fallback: icoor tree. Covers truncated rings that only have one
        # heavy neighbor of A in the bond graph (rare, but upstream
        # behaviour is "fall through to something rather than drop").
        at_to_icoor_ind = bt_obj.at_to_icoor_ind
        icoor_idx = int(at_to_icoor_ind[A_local])
        icoor = bt_obj.icoors[icoor_idx]
        B_icoor = _resolve_icoor_ancestor(
            icoor.parent,
            bt_obj,
            bt,
            block_ind,
            bco,
            bti,
            n_atoms_per_type,
            conn_atoms_pbt,
            inter_residue_connections_pose,
        )
        B0_icoor = _resolve_icoor_ancestor(
            icoor.grand_parent,
            bt_obj,
            bt,
            block_ind,
            bco,
            bti,
            n_atoms_per_type,
            conn_atoms_pbt,
            inter_residue_connections_pose,
        )
        if B_icoor < 0 or B0_icoor < 0:
            return (-1, -1)
        return (B_icoor, B0_icoor)

    return (-1, -1)


def _locate_atom(
    global_idx: int,
    bco: np.ndarray,
    bti: np.ndarray,
    n_atoms_per_type: np.ndarray,
) -> Tuple[int, int, int]:
    """Return (block_ind, block_type, local_atom_idx) for a global atom.

    Returns ``(-1, -1, -1)`` if not found (e.g. padding / invalid).
    Linear scan is fine here because this is called during setup, not
    inside a hot loop, and poses for real proteins have a few hundred
    blocks at most.
    """
    for b in range(bti.shape[0]):
        bt_c = int(bti[b])
        if bt_c < 0:
            continue
        start = int(bco[b])
        n_c = int(n_atoms_per_type[bt_c])
        if start <= global_idx < start + n_c:
            return b, bt_c, global_idx - start
    return -1, -1, -1


def _hbond_per_bt_cache(
    pbt: Any,
    chemical_db: Any,
    atype_to_don_type: dict,
    atype_to_acc_type: dict,
) -> Tuple[
    np.ndarray,
    np.ndarray,  # donor_local (H, D), donor_type
    np.ndarray,
    np.ndarray,  # n_donors_per_bt, donor_mask
    np.ndarray,
    np.ndarray,  # acc_local (A, B, B0), acc_type
    np.ndarray,
    np.ndarray,  # acc_hyb, n_accs_per_bt
    np.ndarray,  # acc_mask
]:
    """Palette-level cache of per-BT donor / acceptor lists.

    Returns padded numpy arrays keyed by block type so the per-pose
    identification loop is a pure index-offset scatter — no per-pose
    bond-graph walk, no Python per-acceptor hybridization branch.

    Layout:

    ``donor_local`` ``int32[n_types, max_n_donors, 2]``
        Per-BT list of (polar_H_local, donor_heavy_local) pairs.
    ``donor_type`` ``int32[n_types, max_n_donors]``
    ``n_donors_per_bt`` ``int32[n_types]``
    ``donor_mask`` ``bool[n_types, max_n_donors]``
    ``acc_local`` ``int32[n_types, max_n_acceptors, 3]``
        Per-BT list of (A_local, B_local, B0_local). ``-1`` sentinels
        flag "B lives via an inter-residue connection" — the per-pose
        path resolves those using
        :func:`_resolve_icoor_ancestor` semantics.
    ``acc_type`` ``int32[n_types, max_n_acceptors]``
    ``acc_hyb`` ``int32[n_types, max_n_acceptors]``
    ``n_accs_per_bt`` ``int32[n_types]``
    ``acc_mask`` ``bool[n_types, max_n_acceptors]``

    Acceptors whose B / B0 walk requires an inter-residue connection
    are stored with the appropriate ``-1`` sentinel in ``acc_local``;
    the per-pose code then runs the full
    :func:`_pick_acceptor_bases` path for those few cases only.
    """
    cache_attr = "_hbond_per_bt_cache"
    cached = getattr(pbt, cache_attr, None)
    if cached is not None:
        return cached

    atom_types_list = list(chemical_db.atom_types)
    atype_is_donor = np.asarray([a.is_donor for a in atom_types_list], dtype=bool)
    atype_is_acceptor = np.asarray([a.is_acceptor for a in atom_types_list], dtype=bool)
    atype_is_polarh = np.asarray([a.is_polarh for a in atom_types_list], dtype=bool)
    atype_is_hydrogen = np.asarray(
        [a.element == "H" for a in atom_types_list], dtype=bool
    )
    atype_hybridization = np.asarray(
        [_HYB_NAME_TO_INT.get(a.acceptor_hybridization, 0) for a in atom_types_list],
        dtype=np.int32,
    )
    atype_name_by_idx = [a.name for a in atom_types_list]

    n_atoms_per_type = np.asarray(pbt.n_atoms)
    atom_types_pbt = np.asarray(pbt.atom_types)
    bond_indices_pbt = np.asarray(pbt.bond_indices)
    n_bonds_pbt = np.asarray(pbt.n_bonds)

    n_types = int(pbt.n_types)

    # Per-BT donors: just look for polar H -> heavy donor edges via
    # intra-residue bonds.
    donors_per_bt: list[list[tuple[int, int, int]]] = [[] for _ in range(n_types)]
    # Per-BT acceptors: run the same bond-graph walk but restricted to
    # intra-block bonds. Acceptors whose B or B0 walk falls off the block
    # (returns -1) will hit the per-pose fallback.
    accs_per_bt: list[list[tuple[int, int, int, int, int]]] = [
        [] for _ in range(n_types)
    ]

    for bt in range(n_types):
        n_ats = int(n_atoms_per_type[bt])
        if n_ats <= 0:
            continue
        ats_here = atom_types_pbt[bt, :n_ats]

        n_b = int(n_bonds_pbt[bt])
        adj: list[list[int]] = [[] for _ in range(n_ats)]
        for bi in range(n_b):
            a, b = int(bond_indices_pbt[bt, bi, 0]), int(bond_indices_pbt[bt, bi, 1])
            if 0 <= a < n_ats and 0 <= b < n_ats:
                adj[a].append(b)

        for local_i in range(n_ats):
            atype_idx = int(ats_here[local_i])
            if atype_idx < 0:
                continue

            if atype_is_polarh[atype_idx]:
                for nbr in adj[local_i]:
                    nbr_atype = int(ats_here[nbr])
                    if nbr_atype < 0:
                        continue
                    if atype_is_donor[nbr_atype]:
                        dtype_name = atype_name_by_idx[nbr_atype]
                        d_idx = atype_to_don_type.get(dtype_name, -1)
                        if d_idx == -1:
                            continue
                        donors_per_bt[bt].append((local_i, nbr, d_idx))
                        break

            if atype_is_acceptor[atype_idx]:
                aname = atype_name_by_idx[atype_idx]
                a_idx = atype_to_acc_type.get(aname, -1)
                if a_idx == -1:
                    continue
                hyb = int(atype_hybridization[atype_idx])

                # Try to pick bases purely intra-block. If that fails
                # (bL or b0L comes out -1), record with -1 sentinels so
                # the per-pose path re-runs the full walk with
                # inter-residue resolution.
                bL, b0L = _pick_intra_block_acceptor_bases(
                    local_i, hyb, adj, ats_here, atype_is_hydrogen
                )
                accs_per_bt[bt].append((local_i, bL, b0L, a_idx, hyb))

    max_donors = max((len(d) for d in donors_per_bt), default=0)
    max_donors = max(max_donors, 1)
    max_accs = max((len(a) for a in accs_per_bt), default=0)
    max_accs = max(max_accs, 1)

    donor_local = np.full((n_types, max_donors, 2), -1, dtype=np.int32)
    donor_type = np.zeros((n_types, max_donors), dtype=np.int32)
    donor_mask = np.zeros((n_types, max_donors), dtype=bool)
    n_donors_per_bt = np.zeros((n_types,), dtype=np.int32)

    acc_local = np.full((n_types, max_accs, 3), -1, dtype=np.int32)
    acc_type = np.zeros((n_types, max_accs), dtype=np.int32)
    acc_hyb_arr = np.zeros((n_types, max_accs), dtype=np.int32)
    acc_mask = np.zeros((n_types, max_accs), dtype=bool)
    n_accs_per_bt = np.zeros((n_types,), dtype=np.int32)

    for bt in range(n_types):
        for j, (local_h, local_d, dt) in enumerate(donors_per_bt[bt]):
            donor_local[bt, j, 0] = local_h
            donor_local[bt, j, 1] = local_d
            donor_type[bt, j] = dt
            donor_mask[bt, j] = True
        n_donors_per_bt[bt] = len(donors_per_bt[bt])

        for j, (local_a, bL, b0L, at, hy) in enumerate(accs_per_bt[bt]):
            acc_local[bt, j, 0] = local_a
            acc_local[bt, j, 1] = bL
            acc_local[bt, j, 2] = b0L
            acc_type[bt, j] = at
            acc_hyb_arr[bt, j] = hy
            acc_mask[bt, j] = True
        n_accs_per_bt[bt] = len(accs_per_bt[bt])

    result = (
        donor_local,
        donor_type,
        n_donors_per_bt,
        donor_mask,
        acc_local,
        acc_type,
        acc_hyb_arr,
        n_accs_per_bt,
        acc_mask,
    )
    try:
        setattr(pbt, cache_attr, result)
    except (AttributeError, TypeError):
        pass  # frozen pbt; re-compute on next call is fine
    return result


def _pick_intra_block_acceptor_bases(
    local_i: int,
    hyb: int,
    adj: list[list[int]],
    ats_here: np.ndarray,
    atype_is_hydrogen: np.ndarray,
) -> Tuple[int, int]:
    """Port of :func:`_pick_acceptor_bases` restricted to intra-block bonds.

    Returns ``(bL, b0L)`` as local-atom indices (or ``-1``) within the
    block. This skips the inter-residue connection fallback; callers
    that hit a ``-1`` result must re-run the full walk for that
    acceptor in the per-pose path.
    """
    nbrs = adj[local_i]
    if hyb == HYBRIDIZATION_SP2:
        bL = -1
        for nbr in nbrs:
            at_nbr = int(ats_here[nbr])
            if at_nbr < 0:
                continue
            if not atype_is_hydrogen[at_nbr]:
                bL = nbr
                break
        if bL < 0:
            return (-1, -1)
        b0L = -1
        for nbr2 in adj[bL]:
            if nbr2 == local_i:
                continue
            at_nbr2 = int(ats_here[nbr2])
            if at_nbr2 < 0:
                continue
            b0L = nbr2
            break
        return (bL, b0L)

    if hyb == HYBRIDIZATION_SP3:
        b0L = -1
        for nbr in nbrs:
            at_nbr = int(ats_here[nbr])
            if at_nbr < 0:
                continue
            if atype_is_hydrogen[at_nbr]:
                b0L = nbr
                break
        if b0L < 0:
            return (-1, -1)
        bL = -1
        for nbr in nbrs:
            if nbr != b0L:
                at_nbr = int(ats_here[nbr])
                if at_nbr < 0:
                    continue
                bL = nbr
                break
        if bL < 0:
            return (-1, -1)
        return (bL, b0L)

    if hyb == HYBRIDIZATION_RING:
        bL = -1
        for nbr in nbrs:
            at_nbr = int(ats_here[nbr])
            if at_nbr < 0:
                continue
            if not atype_is_hydrogen[at_nbr]:
                bL = nbr
                break
        if bL < 0:
            return (-1, -1)
        b0L = -1
        for nbr in nbrs:
            if nbr != bL:
                at_nbr = int(ats_here[nbr])
                if at_nbr < 0:
                    continue
                if not atype_is_hydrogen[at_nbr]:
                    b0L = nbr
                    break
        # ring fallback (icoor tree) is handled per-pose, mark as -1
        # so the caller knows to re-run the full walk.
        return (bL, b0L)

    return (-1, -1)


def _identify_donors_acceptors_for_pose(
    pose_stack: Any,
    pose_ind: int,
    pbt: Any,
    chemical_db: Any,
    atype_to_don_type: dict,
    atype_to_acc_type: dict,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,  # donor H, heavy D, donor-type
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,  # acc atom/B/B0/type/hyb
    np.ndarray,
    np.ndarray,  # donor_block, donor_iter_in_block (for upstream-bug count-pair)
    np.ndarray,
    np.ndarray,  # acc_block, acc_iter_in_block
]:
    """Walk a single pose, return per-atom lists for donors & acceptors.

    Donor identification is local (each polar H is bonded to exactly
    one heavy donor within its own residue). Acceptor B / B0 selection
    walks the combined intra+inter-residue bond graph per
    ``_pick_acceptor_bases`` (upstream ``identification.hh``).

    Uses the per-BT cache :func:`_hbond_per_bt_cache` to avoid the
    per-pose palette walk. Only acceptors whose intra-block base walk
    fails (``-1`` sentinels in the cache) hit the full
    :func:`_pick_acceptor_bases` path with inter-residue connection
    resolution — that short-circuit saves ~2x on 1UBQ-sized proteins.

    All outputs are CPU numpy arrays; they'll be packed + padded into
    jax arrays by the caller.

    Also returns per-donor and per-acceptor ``(block_ind, iter_in_block)``
    metadata. The iter index is the donor's / acceptor's position in
    the **sorted donor/acceptor list of its source block** — this is the
    quantity upstream's tile-based count-pair query mistakenly uses as
    the tile_atom_idx when looking up path-distance for inter-block
    pairs (see :func:`hbond_params_for_pose_stack` for the bug-for-bug
    rationale). We carry it here so the cp_sep matrix can be built with
    the same effective lookup upstream uses.
    """
    (
        donor_local_table,
        donor_type_table,
        n_donors_per_bt,
        donor_mask_table,
        acc_local_table,
        acc_type_table,
        acc_hyb_table,
        n_accs_per_bt,
        acc_mask_table,
    ) = _hbond_per_bt_cache(pbt, chemical_db, atype_to_don_type, atype_to_acc_type)

    bti = np.asarray(pose_stack.block_type_ind)[pose_ind]
    bco = np.asarray(pose_stack.block_coord_offset)[pose_ind]

    # --- Pure-numpy scatter: per-pose donor / acceptor globals --- #
    # Each bti entry says which BT occupies this block; we look up its
    # per-BT table of (local_H, local_D, donor_type) and add the block's
    # coord offset. Padding blocks (bti < 0) get skipped via the mask.
    real_block = bti >= 0
    safe_bti = np.where(real_block, bti, 0)

    # Donors: [B, max_donors_per_bt, 2] (local_H, local_D)
    don_local_pb = donor_local_table[safe_bti]  # [B, D, 2]
    don_mask_pb = donor_mask_table[safe_bti] & real_block[:, None]  # [B, D]
    don_type_pb = donor_type_table[safe_bti]  # [B, D]

    # Per-block iteration index in [0, max_donors_per_bt) — broadcast to
    # the same [B, D] grid the mask is on so we can pull out
    # iter-in-block for each donor when we condense.
    max_d_per_bt = don_local_pb.shape[1]
    block_idx_arr = np.arange(don_local_pb.shape[0], dtype=np.int32)[:, None]
    don_iter_pb = np.broadcast_to(
        np.arange(max_d_per_bt, dtype=np.int32), don_local_pb.shape[:2]
    )
    don_block_pb = np.broadcast_to(block_idx_arr, don_local_pb.shape[:2])

    # Shift to global.
    bco_bc = bco[:, None]  # [B, 1]
    don_H_global = np.where(don_mask_pb, bco_bc + don_local_pb[..., 0], -1)
    don_D_global = np.where(don_mask_pb, bco_bc + don_local_pb[..., 1], -1)

    donor_H = don_H_global[don_mask_pb].astype(np.int32)
    donor_D = don_D_global[don_mask_pb].astype(np.int32)
    donor_type = don_type_pb[don_mask_pb].astype(np.int32)
    donor_block = don_block_pb[don_mask_pb].astype(np.int32)
    donor_iter_in_block = don_iter_pb[don_mask_pb].astype(np.int32)

    # Acceptors: [B, max_accs_per_bt, 3] (local_A, local_B, local_B0)
    acc_local_pb = acc_local_table[safe_bti]  # [B, A, 3]
    acc_mask_pb = acc_mask_table[safe_bti] & real_block[:, None]  # [B, A]
    acc_type_pb = acc_type_table[safe_bti]
    acc_hyb_pb = acc_hyb_table[safe_bti]

    # Per-block iteration index for acceptors (mirrors donor path above).
    max_a_per_bt = acc_local_pb.shape[1]
    acc_iter_pb = np.broadcast_to(
        np.arange(max_a_per_bt, dtype=np.int32), acc_local_pb.shape[:2]
    )
    acc_block_pb = np.broadcast_to(block_idx_arr, acc_local_pb.shape[:2])

    # Valid intra-block bases only.
    has_intra_bases = (acc_local_pb[..., 1] >= 0) & (acc_local_pb[..., 2] >= 0)
    intra_ok = acc_mask_pb & has_intra_bases

    acc_A_global = bco_bc + acc_local_pb[..., 0]
    acc_B_global = bco_bc + acc_local_pb[..., 1]
    acc_B0_global = bco_bc + acc_local_pb[..., 2]

    acc_intra = acc_A_global[intra_ok].astype(np.int32)
    acc_B_intra = acc_B_global[intra_ok].astype(np.int32)
    acc_B0_intra = acc_B0_global[intra_ok].astype(np.int32)
    acc_type_intra = acc_type_pb[intra_ok].astype(np.int32)
    acc_hyb_intra = acc_hyb_pb[intra_ok].astype(np.int32)
    acc_block_intra = acc_block_pb[intra_ok].astype(np.int32)
    acc_iter_in_block_intra = acc_iter_pb[intra_ok].astype(np.int32)

    # Acceptors whose intra-block walk failed need the per-pose
    # full-graph fallback. On 1UBQ-sized proteins there are typically
    # zero of these; iterate only if needed.
    fallback_needed = acc_mask_pb & (~has_intra_bases)
    if fallback_needed.any():
        atom_types_list = list(chemical_db.atom_types)
        atype_is_hydrogen = np.asarray(
            [a.element == "H" for a in atom_types_list], dtype=bool
        )
        n_atoms_per_type = np.asarray(pbt.n_atoms)
        atom_types_pbt = np.asarray(pbt.atom_types)
        conn_atoms_pbt = np.asarray(pbt.conn_atoms)

        # Combined intra+inter "all bonds" table (one per block type).
        n_types = int(pbt.n_types)
        max_n_atoms = int(pbt.max_n_atoms)
        active = pbt.active_block_types
        max_n_all_bonds = max((bt.all_bonds.shape[0] for bt in active), default=0)
        all_bonds_pbt = np.full((n_types, max_n_all_bonds, 3), -1, dtype=np.int32)
        atom_all_bond_ranges_pbt = np.full(
            (n_types, max_n_atoms, 2), -1, dtype=np.int32
        )
        for i, rt in enumerate(active):
            nb = rt.all_bonds.shape[0]
            if nb > 0:
                all_bonds_pbt[i, :nb, :] = rt.all_bonds.astype(np.int32)
            na = rt.n_atoms
            if na > 0:
                atom_all_bond_ranges_pbt[i, :na, :] = rt.atom_all_bond_ranges.astype(
                    np.int32
                )
        inter_residue_connections_pose = np.asarray(
            pose_stack.inter_residue_connections
        )[pose_ind]

        extra_A: list[int] = []
        extra_B: list[int] = []
        extra_B0: list[int] = []
        extra_type: list[int] = []
        extra_hyb: list[int] = []
        extra_block: list[int] = []
        extra_iter: list[int] = []
        for block_ind, a_slot in zip(*np.nonzero(fallback_needed)):
            block_ind = int(block_ind)
            a_slot = int(a_slot)
            bt = int(bti[block_ind])
            local_a = int(acc_local_table[bt, a_slot, 0])
            a_idx = int(acc_type_table[bt, a_slot])
            hyb = int(acc_hyb_table[bt, a_slot])
            global_idx = int(bco[block_ind]) + local_a
            B_global, B0_global = _pick_acceptor_bases(
                A_global=global_idx,
                A_local=local_a,
                bt=bt,
                block_ind=block_ind,
                hybridization=hyb,
                bt_obj=active[bt],
                bco=bco,
                bti=bti,
                all_bonds_pbt=all_bonds_pbt,
                atom_all_bond_ranges_pbt=atom_all_bond_ranges_pbt,
                conn_atoms_pbt=conn_atoms_pbt,
                n_atoms_per_type=n_atoms_per_type,
                inter_residue_connections_pose=inter_residue_connections_pose,
                atom_types_pbt=atom_types_pbt,
                atype_is_hydrogen=atype_is_hydrogen,
            )
            if B_global < 0 or B0_global < 0:
                continue
            extra_A.append(global_idx)
            extra_B.append(B_global)
            extra_B0.append(B0_global)
            extra_type.append(a_idx)
            extra_hyb.append(hyb)
            extra_block.append(block_ind)
            extra_iter.append(a_slot)

        if extra_A:
            acc_intra = np.concatenate([acc_intra, np.asarray(extra_A, dtype=np.int32)])
            acc_B_intra = np.concatenate(
                [acc_B_intra, np.asarray(extra_B, dtype=np.int32)]
            )
            acc_B0_intra = np.concatenate(
                [acc_B0_intra, np.asarray(extra_B0, dtype=np.int32)]
            )
            acc_type_intra = np.concatenate(
                [acc_type_intra, np.asarray(extra_type, dtype=np.int32)]
            )
            acc_hyb_intra = np.concatenate(
                [acc_hyb_intra, np.asarray(extra_hyb, dtype=np.int32)]
            )
            acc_block_intra = np.concatenate(
                [acc_block_intra, np.asarray(extra_block, dtype=np.int32)]
            )
            acc_iter_in_block_intra = np.concatenate(
                [acc_iter_in_block_intra, np.asarray(extra_iter, dtype=np.int32)]
            )

    return (
        donor_H,
        donor_D,
        donor_type,
        acc_intra,
        acc_B_intra,
        acc_B0_intra,
        acc_type_intra,
        acc_hyb_intra,
        donor_block,
        donor_iter_in_block,
        acc_block_intra,
        acc_iter_in_block_intra,
    )


def hbond_params_for_pose_stack(
    pose_stack: Any,
    pbt: Any,
    hbond_db: Any,
    chemical_db: Any | None = None,
) -> HBondParams:
    """Materialise hbond parameters for a ``PoseStack``.

    Identifies donors and acceptors per pose using the chemical-db flags
    (``is_donor`` / ``is_polarh`` / ``is_acceptor`` on ``AtomType``),
    pads per-pose lists to a common length, and returns a
    :class:`HBondParams` bundle ready for pair evaluation.

    Also computes the donor-H to acceptor chemical-bond path length
    matrix (``int32[P, D, A]``) used for the count-pair exclusion:
    upstream skips pairs with ``< 5`` bonds between them.

    Parameters
    ----------
    pose_stack:
        A :class:`tmol_jax.pose.pose_stack.PoseStack`.
    pbt:
        The :class:`tmol_jax.pose.packed_block_types.PackedBlockTypes`
        ``pose_stack`` was built with.
    hbond_db:
        Parsed :class:`tmol.database.scoring.hbond.HBondDatabase`.
    chemical_db:
        The matching chemical database (``pbt.chem_db`` by default).
    """
    if chemical_db is None:
        chemical_db = pbt.chem_db

    # The pair tables / global params are pure functions of
    # (hbond_db, chemical_db), so memoise by ``id()``.
    cache_key = (id(hbond_db), id(chemical_db))
    cached_pair = _HBOND_PAIR_TABLES_CACHE.get(cache_key)
    if cached_pair is None:
        pair_params, polynomials, d_idx, a_idx = _build_pair_tables(
            hbond_db, chemical_db
        )
        atype_to_don_type, atype_to_acc_type = _atom_type_hbond_maps(
            hbond_db, d_idx, a_idx
        )
        global_params = HBondGlobalParams.from_db(hbond_db)
        cached_pair = (
            pair_params,
            polynomials,
            atype_to_don_type,
            atype_to_acc_type,
            global_params,
        )
        _HBOND_PAIR_TABLES_CACHE[cache_key] = cached_pair
    (
        pair_params,
        polynomials,
        atype_to_don_type,
        atype_to_acc_type,
        global_params,
    ) = cached_pair

    n_poses = int(pose_stack.block_type_ind.shape[0])

    per_pose: list[tuple] = []
    for p in range(n_poses):
        per_pose.append(
            _identify_donors_acceptors_for_pose(
                pose_stack, p, pbt, chemical_db, atype_to_don_type, atype_to_acc_type
            )
        )

    # Pad to common (max_donors, max_acceptors).
    max_don = max((t[0].shape[0] for t in per_pose), default=0)
    max_acc = max((t[3].shape[0] for t in per_pose), default=0)
    max_don = max(max_don, 1)  # avoid 0-sized axes
    max_acc = max(max_acc, 1)

    def _pad(arr, width, fill):
        out = np.full(width, fill, dtype=np.int32)
        out[: arr.shape[0]] = arr
        return out

    don_H = np.stack([_pad(t[0], max_don, -1) for t in per_pose])
    don_D = np.stack([_pad(t[1], max_don, -1) for t in per_pose])
    don_T = np.stack([_pad(t[2], max_don, 0) for t in per_pose])

    acc_A = np.stack([_pad(t[3], max_acc, -1) for t in per_pose])
    acc_B = np.stack([_pad(t[4], max_acc, -1) for t in per_pose])
    acc_B0 = np.stack([_pad(t[5], max_acc, -1) for t in per_pose])
    acc_T = np.stack([_pad(t[6], max_acc, 0) for t in per_pose])
    acc_H = np.stack([_pad(t[7], max_acc, 0) for t in per_pose])

    # Per-donor / per-acceptor source-block and iter-in-block. Used to
    # build the count-pair separation matrix the same way upstream does
    # (see the cp_sep build below): upstream's tile-based kernel reads
    # ``path_dist[ii * TILE + iter_idx]`` and effectively substitutes the
    # iter_idx as the tile_atom_idx for inter-block pairs.
    don_blk = np.stack([_pad(t[8], max_don, -1) for t in per_pose])
    don_iter_blk = np.stack([_pad(t[9], max_don, -1) for t in per_pose])
    acc_blk = np.stack([_pad(t[10], max_acc, -1) for t in per_pose])
    acc_iter_blk = np.stack([_pad(t[11], max_acc, -1) for t in per_pose])

    atom_params = HBondAtomParams(
        donor_H_atom=jnp.asarray(don_H),
        donor_heavy_atom=jnp.asarray(don_D),
        donor_type=jnp.asarray(don_T),
        acceptor_atom=jnp.asarray(acc_A),
        acceptor_base=jnp.asarray(acc_B),
        acceptor_base0=jnp.asarray(acc_B0),
        acceptor_type=jnp.asarray(acc_T),
        acceptor_hybridization=jnp.asarray(acc_H),
    )

    # Donor-acceptor bond separation matrix, gathered from the full
    # [P, N, N] path-length cube. Imported lazily to avoid a circular
    # import at module load time.
    from tmol_jax.score.common.count_pair import pairwise_bonded_path_length

    bpl_pn_n = np.asarray(pairwise_bonded_path_length(pose_stack, pbt))
    # Gather per-pose [D, A] from [N, N]. Pad donors/acceptors carry -1
    # in their atom index; any ``[P, N, N]`` read at -1 is out-of-bounds,
    # so mask those to 0 (safe) and replace the result with ``max_sep``
    # (never excluded). Realness masking in ``hbond_pose_scores`` zeros
    # the contribution anyway.
    from tmol_jax.score.common.count_pair import MAX_COUNTPAIR_SEPARATION

    # Upstream-bug-compatible count-pair separation.
    #
    # The upstream tile-based hbond kernel reads its preloaded per-atom
    # path-distance table indexed by the donor/acceptor *iteration index
    # within the tile* rather than the actual ``tile_atom_idx``. See
    # :func:`_identify_donors_acceptors_for_pose` for the long-form note.
    # The net effect for an inter-block (donor, acceptor) pair is that
    # the separation is computed as if both sides were the atoms living
    # at ``block_coord_offset + iter_in_block`` — not the true donor H
    # and acceptor heavy atom. Then the value is capped at
    # ``max_important_bond_separation + 1`` (= 5), since the upstream
    # loop initialises separation there and only takes the min.
    #
    # On 1qys this matters for the (T20-V21) hydroxyl-to-backbone-O
    # pair: the *true* HG1->O separation is 6 bonds (would normally be
    # included), but upstream's stand-in atom-at-iter-position-1 in THR
    # (= CA) is only 4 bonds from VAL's N (the acceptor stand-in at
    # iter-position-0), so the buggy kernel drops the pair. To match
    # the upstream reference numbers, we reproduce the same lookup.
    #
    # For intra-block pairs upstream uses ``block_type_path_distance``
    # directly with ``don_start + don_tile_ind`` (i.e. the *correct*
    # donor / acceptor atom indices), so intra-block pairs use the true
    # path length.
    max_important_bond_sep = 4  # upstream constant

    don_acc_bond_sep = np.full(
        (n_poses, max_don, max_acc), MAX_COUNTPAIR_SEPARATION, dtype=np.int32
    )
    for p in range(n_poses):
        valid_d = don_H[p] >= 0
        valid_a = acc_A[p] >= 0
        if not valid_d.any() or not valid_a.any():
            continue

        # Compute true intra-block path distance between donor H and
        # acceptor atom. Used for the intra-block case only.
        dH = np.where(valid_d, don_H[p], 0)
        aA = np.where(valid_a, acc_A[p], 0)
        true_sep = bpl_pn_n[p, dH[:, None], aA[None, :]]

        # Compute upstream-style "fake" separation: use the
        # block-coord-offset + iter_in_block as the per-atom path-distance
        # lookup. The donor "fake atom" lives in donor's block; the
        # acceptor "fake atom" lives in acceptor's block. For inter-block
        # pairs only, cap at 5.
        bco_p = np.asarray(pose_stack.block_coord_offset)[p]
        d_blk_p = don_blk[p]
        a_blk_p = acc_blk[p]
        d_iter_p = don_iter_blk[p]
        a_iter_p = acc_iter_blk[p]
        safe_d_blk = np.where(d_blk_p >= 0, d_blk_p, 0)
        safe_a_blk = np.where(a_blk_p >= 0, a_blk_p, 0)
        safe_d_iter = np.where(d_iter_p >= 0, d_iter_p, 0)
        safe_a_iter = np.where(a_iter_p >= 0, a_iter_p, 0)
        fake_d_atom = bco_p[safe_d_blk] + safe_d_iter  # [D]
        fake_a_atom = bco_p[safe_a_blk] + safe_a_iter  # [A]
        # Clamp to valid atom range so the bpl_pn_n gather does not
        # OOB. Padding will be masked off below.
        n_atoms_total = bpl_pn_n.shape[1]
        fake_d_atom = np.clip(fake_d_atom, 0, n_atoms_total - 1)
        fake_a_atom = np.clip(fake_a_atom, 0, n_atoms_total - 1)
        fake_sep = bpl_pn_n[p, fake_d_atom[:, None], fake_a_atom[None, :]]
        # Upstream caps inter-block separation at
        # ``max_important_bond_separation + 1`` (= 5) because the loop
        # only takes the min of the (ii, jj) values against the initial
        # value of 5 — any "true" separation >= 5 stays at 5.
        fake_sep = np.minimum(fake_sep, max_important_bond_sep + 1)

        # Decide per pair: intra (use true_sep) vs inter (use fake_sep).
        same_block = d_blk_p[:, None] == a_blk_p[None, :]
        sep = np.where(same_block, true_sep, fake_sep)

        # Pairs with either side padded: max_sep so they pass the cp
        # check; the donor/acceptor real-mask in ``hbond_pose_scores``
        # zeros their contribution anyway.
        mask = valid_d[:, None] & valid_a[None, :]
        don_acc_bond_sep[p] = np.where(mask, sep, MAX_COUNTPAIR_SEPARATION)

    return HBondParams(
        atom_params=atom_params,
        pair_params=pair_params,
        polynomials=polynomials,
        global_params=global_params,
        donor_acceptor_bond_sep=jnp.asarray(don_acc_bond_sep),
    )


# Cache for the (hbond_db, chemical_db) pair tables -- pure function
# of the two databases, so memoise by ``id()`` for O(1) reuse across
# pose stacks that share the same param db.
_HBOND_PAIR_TABLES_CACHE: "dict[tuple[int, int], tuple]" = {}


__all__ = [
    "HYBRIDIZATION_NONE",
    "HYBRIDIZATION_SP2",
    "HYBRIDIZATION_SP3",
    "HYBRIDIZATION_RING",
    "HBondGlobalParams",
    "HBondPairParams",
    "HBondPolynomials",
    "HBondAtomParams",
    "HBondParams",
    "hbond_params_for_pose_stack",
]
