"""Dunbrack rotamer library energy term -- JAX port.

Upstream: :mod:`tmol.score.dunbrack.dunbrack_energy_term`. The upstream
term is a thin wrapper around a C++ CUDA kernel that (per residue):

* measures phi / psi / chi torsions,
* classifies the chi angles into a rotamer bin,
* looks up the rotamer's ``-log P`` on a (phi, psi) b-spline grid,
* adds a Gaussian (chi - mean(phi,psi))^2 / (2 sigma(phi,psi)^2)
  penalty for each chi, and
* for semirotameric residues (aromatics with a continuous last chi),
  adds a 3-D (phi, psi, chi_last) ``-log P`` contribution.

This JAX port keeps exactly that three-channel split
(``dunbrack_rot`` / ``dunbrack_rotdev`` / ``dunbrack_semirot``) but
drops the cubic b-spline in favour of bi/trilinear interpolation on
the *already-evaluated* probability grids shipped in
``dunbrack.bin``. See :mod:`.potentials` for the math.

Gotchas & design decisions
--------------------------
1. **Graceful degradation.** The upstream library tables live in
   ``dunbrack.bin`` -- a ``torch.save`` pickle that we can only
   deserialise when ``torch`` is importable
   (:mod:`tmol_jax.database.scoring.dunbrack_libraries`). If the
   tables are absent (empty ``rotameric_libraries`` /
   ``semi_rotameric_libraries``) we still expose the term, but all
   three score channels return zero for every pose. This lets the
   score function stay wired up in environments without ``torch``.

2. **Per-residue torsion atom indices.** Setup walks every block type
   in the ``PackedBlockTypes`` palette, pulls out the ``phi``,
   ``psi``, and ``chi1..chi4`` torsions (via
   ``RefinedResidueType.torsion_to_uaids``), and packs their unresolved
   atom IDs (``(atom_idx, conn_id, n_bonds_from_conn)``) into a
   ``[n_types, 6, 4, 3]`` numpy array with ``-1`` padding for missing
   torsions. The score path resolves these at runtime against
   ``inter_residue_connections`` + ``conn_atoms`` so phi on residue
   ``i`` correctly picks up the ``C`` atom of residue ``i-1``.

3. **Rotamer / table dispatch.** A per-block-type mapping
   ``aa_table_ind -> (rot_neglnprob_table, rotmean_table, rotsdev_table,
   semirot_table, is_semirotameric, n_rotameric_chi, nchi,
   bb_start, bb_step, bb_period, semirot_start, semirot_step,
   semirot_period, rotind2tableind)`` is built at setup time by
   collapsing the Dunbrack library's `rotameric_libraries` +
   `semi_rotameric_libraries` into flat ``[n_aa, ...]`` arrays. At
   score time the term simply ``jnp.take``s by AA and interpolates.

4. **Only canonical 20.** Residues that have no Dunbrack entry (ALA,
   GLY, plus any non-canonical types) get a ``-1`` ``aa_table_ind``
   and contribute zero to every channel.

5. **Non-real blocks.** Padding blocks (``block_type_ind == -1``) also
   contribute zero; they are masked out explicitly so that padding
   never produces NaNs under ``jit``.

The term is ``jit`` / ``grad`` safe: all traced computation is done with
``jnp``; Python-level branching happens only at setup time where it
produces static metadata.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.numeric.bspline import bspline_coefficients
from tmol_jax.score.common.geom import dihedral_angle
from tmol_jax.score.dunbrack.potentials import (
    _bicubic_bspline_interpolate,
    _frac_index,
    bspline3_coeffs_for_2d_table,
    dunbrack_rotdev_score,
    dunbrack_semirot_score,
)
from tmol_jax.score.energy_term import EnergyTerm
from tmol_jax.score.score_types import ScoreType
from tmol_jax.types import pytree_dataclass

# Upstream's max-chi-per-AA is 4 (ARG / LYS). Phi + psi + chi1..chi4 -> 6.
_MAX_N_CHI = 4
_N_DIHEDRALS = 2 + _MAX_N_CHI  # phi + psi + chi*


def _base3_encode_rotamers(rotamers_1based: np.ndarray) -> np.ndarray:
    """Encode ``[n, k]`` 1-based rotamer-bin tuples as base-3 ints.

    Upstream's ``rotind2tableind`` layout: most-significant digit first,
    each chi in ``{1, 2, 3}``. Returns ``[n]`` int32.
    """
    k = int(rotamers_1based.shape[1])
    exponents = np.array([3 ** (k - 1 - i) for i in range(k)], dtype=np.int32)
    return np.sum((rotamers_1based - 1) * exponents, axis=1, dtype=np.int32)


# --------------------------------------------------------------------------- #
# Pytree container for the per-pose_stack static state                        #
# --------------------------------------------------------------------------- #


@pytree_dataclass
class DunbrackTermState:
    """Static arrays precomputed once per PoseStack.

    All arrays are JAX leaves so the whole state survives ``jit`` /
    ``vmap``; Python ints / bools are aux metadata.
    """

    # --- empty-library flag -------------------------------------------- #
    # When ``True`` the score returns zeros for every pose and never
    # touches the table leaves. Aux (non-array) so we can branch on it
    # with a Python ``if`` at trace time.
    empty: bool

    # --- per-block-type palette ---------------------------------------- #
    # dihedral_uaids[i, d, a, 3] = (atom_idx, conn_id, n_bonds_from_conn)
    # for block-type ``i``, dihedral ``d`` in 0..5 (phi, psi, chi1..4),
    # atom ``a`` in 0..3. Sentinel -1 / invalid rows are filled with -1.
    dihedral_uaids: jnp.ndarray  # int32 [n_types, 6, 4, 3]
    # Per-type indices into the library arrays. aa_table_ind = -1 means
    # "no Dunbrack entry for this block type; skip it".
    aa_table_ind: jnp.ndarray  # int32 [n_types]
    n_chi_per_type: jnp.ndarray  # int32 [n_types]
    n_rotameric_chi_per_type: jnp.ndarray  # int32 [n_types]
    is_semirotameric_per_type: jnp.ndarray  # bool  [n_types]

    # --- library tables ----------------------------------------------- #
    # All libraries resampled onto a common phi/psi grid shape
    # (36 x 36 per upstream) so we can stack. If the tables are empty,
    # the shapes are (0, ...); guard with ``empty`` before use.
    rot_neglnprob_table: jnp.ndarray  # float32 [n_aa, max_nrot, N, N]
    rot_mean_table: jnp.ndarray  # float32 [n_aa, max_nrot, max_nchi, N, N]
    rot_sdev_table: jnp.ndarray  # float32 [n_aa, max_nrot, max_nchi, N, N]
    rotind2tableind: jnp.ndarray  # int32   [n_aa, 3**max_nchi]
    bb_start: jnp.ndarray  # float32 [n_aa, 2]
    bb_step: jnp.ndarray  # float32 [n_aa, 2]
    bb_period: jnp.ndarray  # float32 [n_aa, 2]

    # Semirotameric tables. ``semirot_table_ind == -1`` when the AA
    # is purely rotameric. Shape: [n_sr, max_nrot_sr, Nphi, Npsi, Nchi].
    semirot_table: jnp.ndarray  # float32 [n_sr, max_nrot_sr, N, N, Nchi]
    semirot_index_for_aa: jnp.ndarray  # int32 [n_aa]
    # Rot-bin -> semirot-table-row map (for semirotameric AAs only).
    # Shape: [n_aa, 3**max_nchi]; ``-1`` outside the populated bins.
    semirot_rotind2tableind: jnp.ndarray  # int32 [n_aa, 3**max_nchi]
    semirot_start: jnp.ndarray  # float32 [n_sr, 3]
    semirot_step: jnp.ndarray  # float32 [n_sr, 3]
    semirot_period: jnp.ndarray  # float32 [n_sr, 3]


# --------------------------------------------------------------------------- #
# Energy term                                                                 #
# --------------------------------------------------------------------------- #


class DunbrackEnergyTerm(EnergyTerm):
    """Dunbrack rotamer library energy term."""

    def __init__(self, param_db: Any):
        super().__init__(param_db=param_db)
        # ``param_db.scoring.dun`` is the legacy alias for
        # ``param_db.scoring.dunbrack``.
        self.dun_db = param_db.scoring.dunbrack
        # Library-level B-spline table cache. The rotameric /
        # semirotameric stacks depend only on ``dun_db`` (they prefilter
        # the per-AA probability grids into B-spline coefficients), so we
        # build them once and reuse across any number of pose stacks.
        # ``None`` until the first :meth:`setup_for_pose_stack` call.
        self._lib_cache: Optional[dict] = None

    @classmethod
    def class_name(cls) -> str:
        return "Dunbrack"

    def score_types_provided(self) -> List[ScoreType]:
        return [
            ScoreType.dunbrack_rot,
            ScoreType.dunbrack_rotdev,
            ScoreType.dunbrack_semirot,
        ]

    # ------------------------------------------------------------------ #
    # Setup                                                              #
    # ------------------------------------------------------------------ #

    def _get_lib_cache(self) -> dict:
        """Lazy-initialise the library-level (per-AA) B-spline cache.

        The stacked tables (rot_nlp / rot_mean / rot_sdev / ri2ti /
        bb_{start,step,period} and their semirotameric counterparts) are
        a pure function of ``self.dun_db`` — they re-prefilter the
        per-AA (phi, psi[, chi_last]) probability grids into B-spline
        coefficients via :func:`bspline_coefficients`. That numpy
        prefilter cost dominates :meth:`setup_for_pose_stack` (~20-30 s
        on the first call). Compute once; reuse across every subsequent
        pose stack that shares this term instance.
        """
        if self._lib_cache is not None:
            return self._lib_cache

        rot_libs = tuple(self.dun_db.rotameric_libraries or ())
        semi_libs = tuple(self.dun_db.semi_rotameric_libraries or ())
        all_libs = rot_libs + semi_libs
        empty = len(all_libs) == 0
        lookup = {m.residue_name: m.dun_table_name for m in self.dun_db.dun_lookup}
        tablename_to_index = {lib.table_name: i for i, lib in enumerate(all_libs)}
        n_rot = len(rot_libs)

        if empty:
            # Degenerate dummy tables so shapes type-check inside jit.
            rot_nlp = np.zeros((0, 1, 1, 1), dtype=np.float32)
            rot_mean = np.zeros((0, 1, 1, 1, 1), dtype=np.float32)
            rot_sdev = np.zeros((0, 1, 1, 1, 1), dtype=np.float32)
            ri2ti = np.zeros((0, 1), dtype=np.int32)
            bb_start = np.zeros((0, 2), dtype=np.float32)
            bb_step = np.zeros((0, 2), dtype=np.float32)
            bb_period = np.zeros((0, 2), dtype=np.float32)
            sr_table = np.zeros((0, 1, 1, 1, 1), dtype=np.float32)
            sr_index = np.full((0,), -1, dtype=np.int32)
            sr_ri2ti = np.zeros((0, 1), dtype=np.int32)
            sr_start = np.zeros((0, 3), dtype=np.float32)
            sr_step = np.zeros((0, 3), dtype=np.float32)
            sr_period = np.zeros((0, 3), dtype=np.float32)
        else:
            (
                rot_nlp,
                rot_mean,
                rot_sdev,
                ri2ti,
                bb_start,
                bb_step,
                bb_period,
            ) = self._stack_rotameric_tables(all_libs, n_rot)
            # ri2ti.shape[1] == 3 ** max_nchi; pass the dim directly so the
            # semirot helper can size its own rot-bin map without recomputing log3.
            (
                sr_table,
                sr_index,
                sr_ri2ti,
                sr_start,
                sr_step,
                sr_period,
            ) = self._stack_semirotameric_tables(
                all_libs, semi_libs, int(ri2ti.shape[1])
            )

        # Eagerly move to device so they're ready leaves for the pytree;
        # doing this once avoids the ``jnp.asarray`` host->device copy
        # on every :meth:`setup_for_pose_stack` call.
        self._lib_cache = {
            "empty": bool(empty),
            "lookup": lookup,
            "tablename_to_index": tablename_to_index,
            "n_rot": n_rot,
            "all_libs": all_libs,
            "rot_nlp": jnp.asarray(rot_nlp, dtype=jnp.float32),
            "rot_mean": jnp.asarray(rot_mean, dtype=jnp.float32),
            "rot_sdev": jnp.asarray(rot_sdev, dtype=jnp.float32),
            "ri2ti": jnp.asarray(ri2ti, dtype=jnp.int32),
            "bb_start": jnp.asarray(bb_start, dtype=jnp.float32),
            "bb_step": jnp.asarray(bb_step, dtype=jnp.float32),
            "bb_period": jnp.asarray(bb_period, dtype=jnp.float32),
            "sr_table": jnp.asarray(sr_table, dtype=jnp.float32),
            "sr_index": jnp.asarray(sr_index, dtype=jnp.int32),
            "sr_ri2ti": jnp.asarray(sr_ri2ti, dtype=jnp.int32),
            "sr_start": jnp.asarray(sr_start, dtype=jnp.float32),
            "sr_step": jnp.asarray(sr_step, dtype=jnp.float32),
            "sr_period": jnp.asarray(sr_period, dtype=jnp.float32),
        }
        return self._lib_cache

    def _annotate_pbt(self, pbt: Any) -> None:
        """Cache per-BT Dunbrack metadata on ``pbt`` (palette-level).

        After this call ``pbt`` has these four attributes cached so
        subsequent :meth:`setup_for_pose_stack` calls for the same
        palette are near-instant:

        * ``dunbrack_dihedral_uaids``    ``int32[n_types, 6, 4, 3]``
        * ``dunbrack_aa_table_ind``      ``int32[n_types]``
        * ``dunbrack_n_chi_per_type``    ``int32[n_types]``
        * ``dunbrack_n_rotameric_chi``   ``int32[n_types]``
        * ``dunbrack_is_semirotameric``  ``bool[n_types]``

        These are a pure function of ``pbt.active_block_types`` and
        ``self.dun_db`` — neither of which changes across pose stacks —
        so we compute them once and leave them on the palette.
        """
        if hasattr(pbt, "dunbrack_dihedral_uaids"):
            return

        cache = self._get_lib_cache()
        lookup = cache["lookup"]
        tablename_to_index = cache["tablename_to_index"]
        all_libs = cache["all_libs"]
        n_rot = cache["n_rot"]

        n_types = pbt.n_types
        aa_table_ind = np.full((n_types,), -1, dtype=np.int32)
        n_chi_per_type = np.zeros((n_types,), dtype=np.int32)
        n_rot_chi_per_type = np.zeros((n_types,), dtype=np.int32)
        is_sr_per_type = np.zeros((n_types,), dtype=bool)

        for bt_ind, bt in enumerate(pbt.active_block_types):
            # Upstream's dunbrack energy term keys the lookup off
            # ``block_type.base_name`` (e.g. ``HIS_POS``), not ``name3``
            # (``HIS``). For residues whose base_name is not in the
            # library -- which includes HIS_POS, HIS_D, CYD -- upstream
            # falls back to ``-1`` and skips dunbrack scoring for them.
            # We mirror that to keep the forward energies identical.
            base_name = getattr(bt, "base_name", None) or getattr(bt, "name3", None)
            if base_name is None:
                continue
            table_name = lookup.get(base_name)
            if table_name is None:
                continue
            lib_idx = tablename_to_index.get(table_name)
            if lib_idx is None:
                continue
            aa_table_ind[bt_ind] = lib_idx
            lib = all_libs[lib_idx]
            n_chi_per_type[bt_ind] = int(lib.rotameric_data.rotamers.shape[1])
            is_sr = lib_idx >= n_rot
            is_sr_per_type[bt_ind] = is_sr
            n_rot_chi_per_type[bt_ind] = n_chi_per_type[bt_ind] - (1 if is_sr else 0)

        # Per-BT dihedral uaids: [n_types, 6, 4, 3].
        dih_uaids_np = np.full((n_types, _N_DIHEDRALS, 4, 3), -1, dtype=np.int32)
        torsion_names = ("phi", "psi") + tuple(f"chi{i + 1}" for i in range(_MAX_N_CHI))
        for bt_ind, bt in enumerate(pbt.active_block_types):
            t2u = getattr(bt, "torsion_to_uaids", None)
            if t2u is None:
                continue
            for d_ind, tname in enumerate(torsion_names):
                uaids = t2u.get(tname)
                if uaids is None:
                    continue
                for a_ind in range(4):
                    dih_uaids_np[bt_ind, d_ind, a_ind, :] = uaids[a_ind][:3]

        for attr, arr, dtype in (
            ("dunbrack_dihedral_uaids", dih_uaids_np, jnp.int32),
            ("dunbrack_aa_table_ind", aa_table_ind, jnp.int32),
            ("dunbrack_n_chi_per_type", n_chi_per_type, jnp.int32),
            ("dunbrack_n_rotameric_chi", n_rot_chi_per_type, jnp.int32),
            ("dunbrack_is_semirotameric", is_sr_per_type, bool),
        ):
            setattr(pbt, attr, jnp.asarray(arr, dtype=dtype))

    def setup_for_pose_stack(
        self, pose_stack: Any, pbt: Any = None
    ) -> DunbrackTermState:
        """Build the precomputed ``DunbrackTermState``.

        Walks the ``PackedBlockTypes`` palette, pulls out phi/psi/chi
        torsion uaids for every block type, and stacks every Dunbrack
        rotamer library into flat ``[n_aa, ...]`` arrays ready for
        gather-based lookup at score time.

        All of this is per-palette / per-library — not per-pose — so we
        cache it on ``pbt`` (BT metadata) and on ``self`` (library
        tables). Only the assembly of the final pytree bag happens on a
        per-pose-stack basis, which is near-instant.
        """
        pbt = pose_stack.packed_block_types if pbt is None else pbt
        self._annotate_pbt(pbt)
        cache = self._get_lib_cache()

        return DunbrackTermState(
            empty=cache["empty"],
            dihedral_uaids=pbt.dunbrack_dihedral_uaids,
            aa_table_ind=pbt.dunbrack_aa_table_ind,
            n_chi_per_type=pbt.dunbrack_n_chi_per_type,
            n_rotameric_chi_per_type=pbt.dunbrack_n_rotameric_chi,
            is_semirotameric_per_type=pbt.dunbrack_is_semirotameric,
            rot_neglnprob_table=cache["rot_nlp"],
            rot_mean_table=cache["rot_mean"],
            rot_sdev_table=cache["rot_sdev"],
            rotind2tableind=cache["ri2ti"],
            bb_start=cache["bb_start"],
            bb_step=cache["bb_step"],
            bb_period=cache["bb_period"],
            semirot_table=cache["sr_table"],
            semirot_index_for_aa=cache["sr_index"],
            semirot_rotind2tableind=cache["sr_ri2ti"],
            semirot_start=cache["sr_start"],
            semirot_step=cache["sr_step"],
            semirot_period=cache["sr_period"],
        )

    # ---- table stacking helpers (CPU, one-shot) ---------------------- #

    @staticmethod
    def _stack_rotameric_tables(
        all_libs: Tuple[Any, ...],
        n_rot_libs: int,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Stack per-AA rotamer prob / mean / sdev tables into flat arrays.

        Each library ships::

            rotamer_probabilities : [nrot, N_phi, N_psi]
            rotamer_means          : [nrot, N_phi, N_psi, nchi]
            rotamer_stdvs          : [nrot, N_phi, N_psi, nchi]

        We pad to ``max_nrot`` / ``max_nchi`` across AAs, convert means/
        sigmas from degrees to radians, and apply the upstream wrap
        (means near ``-pi`` are shifted towards ``+pi``).

        ``n_rot_libs`` is the number of purely-rotameric libraries at the
        start of ``all_libs``; entries at index ``>= n_rot_libs`` are
        semirotameric. For semirotameric libs the ``rotamers`` table also
        encodes the (discretised) terminal-chi bin as an extra integer
        in ``[1, N_chi_last]`` (not ``[1, 3]``), so we *cannot* use the
        full ``rotamers`` array to build the ``rotind -> table-row``
        lookup. Instead we use :attr:`SemiRotamericAADunbrackLibrary.rotameric_chi_rotamers`
        which only covers the rotameric chis and maps cleanly into a
        ``3^n_rotameric_chi`` index space. Following upstream we assume
        the 18 (= 3x6) / 54 (= 9x6) / 36 (= 3x12) rotamer rows are sorted
        grouped by rotameric bin, so a representative "first sub-row"
        (``n_nonrot_sub * rotameric_bin``) is stored in ``ri2ti``.
        """
        n_aa = len(all_libs)
        # We expect the canonical 36x36 grid but derive it from the data.
        n_phi = all_libs[0].rotameric_data.rotamer_probabilities.shape[1]
        n_psi = all_libs[0].rotameric_data.rotamer_probabilities.shape[2]
        max_nrot = max(
            int(lib.rotameric_data.rotamer_probabilities.shape[0]) for lib in all_libs
        )
        max_nchi = max(int(lib.rotameric_data.rotamers.shape[1]) for lib in all_libs)

        rot_nlp = np.zeros((n_aa, max_nrot, n_phi, n_psi), dtype=np.float32)
        rot_mean = np.zeros((n_aa, max_nrot, max_nchi, n_phi, n_psi), dtype=np.float32)
        rot_sdev = np.ones((n_aa, max_nrot, max_nchi, n_phi, n_psi), dtype=np.float32)
        ri2ti = np.full((n_aa, 3**max_nchi), -1, dtype=np.int32)
        bb_start = np.zeros((n_aa, 2), dtype=np.float32)
        bb_step = np.zeros((n_aa, 2), dtype=np.float32)
        bb_period = np.full((n_aa, 2), 2.0 * np.pi, dtype=np.float32)

        for ai, lib in enumerate(all_libs):
            rd = lib.rotameric_data
            nrot = int(rd.rotamer_probabilities.shape[0])
            nchi = int(rd.rotamers.shape[1])
            is_sr = ai >= n_rot_libs

            # -log P(rotamer | phi, psi), with zeros clamped to 1e-6.
            # Upstream (``dunbrack/params.py::_compute_rotprob_coeffs``) feeds
            # this table through ``BSplineInterpolation.from_coordinates``
            # before interpolating, i.e. the tables stored on the term are
            # B-spline *coefficients*, not raw sample values. Mirror that
            # here so our interpolator (bicubic B-spline with periodic BCs)
            # matches upstream to ~float32 precision.
            probs = np.asarray(rd.rotamer_probabilities, dtype=np.float32)
            probs = np.where(probs <= 0, 1e-6, probs)
            raw_nlp = -np.log(probs)  # [nrot, N_phi, N_psi]
            rot_nlp[ai, :nrot] = bspline3_coeffs_for_2d_table(raw_nlp)

            # Mean / sdev tables; convert deg -> rad, shift means near
            # -pi towards +pi.
            means_deg = np.asarray(rd.rotamer_means, dtype=np.float32)
            means_deg = np.where(means_deg < -120.0, means_deg + 360.0, means_deg)
            means_rad = means_deg * (np.pi / 180.0)
            sdevs_rad = np.asarray(rd.rotamer_stdvs, dtype=np.float32) * (np.pi / 180.0)
            # upstream shape: [nrot, N_phi, N_psi, nchi] -> [nrot, nchi, N_phi, N_psi]
            means_rad = np.moveaxis(means_rad, -1, 1)
            sdevs_rad = np.moveaxis(sdevs_rad, -1, 1)
            sdevs_rad = np.where(sdevs_rad > 1e-6, sdevs_rad, 1.0)
            # Upstream (``dunbrack/params.py::_calculate_rot_mean_coeffs`` +
            # ``_calculate_rot_sdev_coeffs``) feeds each 2-D (N_phi, N_psi)
            # plane through ``BSplineInterpolation.from_coordinates`` at DB
            # load time, i.e. the tables on the score term are bicubic
            # B-spline *coefficients*, not raw sample values. Mirror that
            # here by applying the same separable causal / anti-causal
            # cubic prefilter (``bspline_coefficients`` with degree=3).
            rot_mean[ai, :nrot, :nchi] = bspline3_coeffs_for_2d_table(means_rad)
            rot_sdev[ai, :nrot, :nchi] = bspline3_coeffs_for_2d_table(sdevs_rad)

            # rotind2tableind: base-3 encoded rotamer bins -> row index
            # into the [nrot, ...] tables. Missing rotamers stay -1.
            if is_sr:
                # For semirotameric libs, use rotameric_chi_rotamers so
                # every entry of the source table is a pure base-3
                # rotamer-bin tuple (no discretised last chi).
                src_rotamers = np.asarray(lib.rotameric_chi_rotamers, dtype=np.int32)
                rotinds = _base3_encode_rotamers(src_rotamers)
                # Upstream picks n_nonrot_sub * bin as the representative
                # rotameric-table row (first of the discretised-last-chi
                # sub-rows, which all share the same rotameric means/sigmas).
                n_sr_bins = int(src_rotamers.shape[0])
                n_nonrot_sub = nrot // n_sr_bins if n_sr_bins > 0 else 1
                row_ind = np.arange(n_sr_bins, dtype=np.int32) * n_nonrot_sub
                ri2ti[ai, rotinds] = row_ind
            else:
                rotamers = np.asarray(rd.rotamers, dtype=np.int32)
                rotinds = _base3_encode_rotamers(rotamers)
                ri2ti[ai, rotinds] = np.arange(nrot, dtype=np.int32)
                # Apply rotamer aliases (upstream ``params.py`` ~l.469-485).
                # PRO is the only canonical AA with aliases: maps certain
                # base-3 bins to the same table row as a *defined* rotamer.
                # Shape: [n_alias, 2 * nchi] -- first nchi cols = "orig"
                # (aliased) bin, last nchi cols = "alt" (already-mapped)
                # bin. For each alias row we set ri2ti[orig] = ri2ti[alt].
                alias = getattr(rd, "rotamer_alias", None)
                if alias is not None:
                    alias_np = np.asarray(alias, dtype=np.int32)
                    if alias_np.ndim == 2 and alias_np.shape[0] > 0:
                        alias_nchi = alias_np.shape[1] // 2
                        orig_inds = _base3_encode_rotamers(alias_np[:, :alias_nchi])
                        alt_inds = _base3_encode_rotamers(alias_np[:, alias_nchi:])
                        ri2ti[ai, orig_inds] = ri2ti[ai, alt_inds]

            bb_start[ai] = np.asarray(rd.backbone_dihedral_start, dtype=np.float32) * (
                np.pi / 180.0
            )
            bb_step[ai] = np.asarray(rd.backbone_dihedral_step, dtype=np.float32) * (
                np.pi / 180.0
            )

        return rot_nlp, rot_mean, rot_sdev, ri2ti, bb_start, bb_step, bb_period

    @staticmethod
    def _stack_semirotameric_tables(
        all_libs: Tuple[Any, ...],
        semi_libs: Tuple[Any, ...],
        n_rotbins: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Stack per-AA semirotameric tables + a mapping from all-AA index.

        Returns
        -------
        sr_table
            ``[n_sr, max_nrot_sr, N_phi, N_psi, N_chi_last]`` ``-log P``
            values.
        sr_index
            ``[n_aa]`` mapping AA index -> semirot table index (or -1).
        sr_rotind2tableind
            ``[n_aa, 3 ** max_nchi]``: rot-bin -> semirot-table row.
            ``-1`` where no mapping exists. For purely rotameric AAs all
            entries are ``-1`` (unused). For semirotameric AAs only the
            rotameric-chi bins (subset of the full ``3**max_nchi`` space)
            are populated.
        sr_start, sr_step, sr_period
            ``[n_sr, 3]`` grid descriptors.
        """
        n_aa = len(all_libs)
        n_sr = len(semi_libs)
        if n_sr == 0:
            return (
                np.zeros((0, 1, 1, 1, 1), dtype=np.float32),
                np.full((n_aa,), -1, dtype=np.int32),
                np.full((n_aa, max(1, n_rotbins)), -1, dtype=np.int32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
            )

        n_phi = semi_libs[0].nonrotameric_chi_probabilities.shape[1]
        n_psi = semi_libs[0].nonrotameric_chi_probabilities.shape[2]
        n_chi = semi_libs[0].nonrotameric_chi_probabilities.shape[3]
        max_nrot_sr = max(
            int(lib.nonrotameric_chi_probabilities.shape[0]) for lib in semi_libs
        )

        sr_table = np.zeros((n_sr, max_nrot_sr, n_phi, n_psi, n_chi), dtype=np.float32)
        sr_index = np.full((n_aa,), -1, dtype=np.int32)
        sr_ri2ti = np.full((n_aa, n_rotbins), -1, dtype=np.int32)
        sr_start = np.zeros((n_sr, 3), dtype=np.float32)
        sr_step = np.zeros((n_sr, 3), dtype=np.float32)
        sr_period = np.full((n_sr, 3), 2.0 * np.pi, dtype=np.float32)

        # Map AA index in ``all_libs`` -> index in ``semi_libs``.
        # ``all_libs = rotameric_libs + semi_libs`` so semi_libs start
        # at offset len(rotameric_libs).
        offset = n_aa - n_sr
        for si in range(n_sr):
            sr_index[offset + si] = si

        for si, lib in enumerate(semi_libs):
            probs = np.asarray(lib.nonrotameric_chi_probabilities, dtype=np.float32)
            probs = np.where(probs <= 0, 1e-6, probs)
            nrot_sr = probs.shape[0]
            raw_nlp = -np.log(probs)  # [nrot_sr, N_phi, N_psi, N_chi]
            # Upstream (``dunbrack/params.py::_calc_semirot_coeffs``) feeds
            # each 3-D (N_phi, N_psi, N_chi) slice through
            # ``BSplineInterpolation.from_coordinates`` at DB load time.
            # Apply the same tricubic B-spline prefilter here so that a
            # matching tricubic evaluator at score time reproduces the
            # upstream numbers.
            for ri in range(nrot_sr):
                sr_table[si, ri] = np.asarray(
                    bspline_coefficients(raw_nlp[ri], degree=3)
                )

            # Backbone axes are in the same 10-deg spacing as rotameric.
            sr_start[si] = np.array(
                [-np.pi, -np.pi, float(lib.non_rot_chi_start) * np.pi / 180.0],
                dtype=np.float32,
            )
            sr_step[si] = np.array(
                [
                    10.0 * np.pi / 180.0,
                    10.0 * np.pi / 180.0,
                    float(lib.non_rot_chi_step) * np.pi / 180.0,
                ],
                dtype=np.float32,
            )
            sr_period[si] = np.array(
                [
                    2.0 * np.pi,
                    2.0 * np.pi,
                    float(lib.non_rot_chi_period) * np.pi / 180.0,
                ],
                dtype=np.float32,
            )

            # sr_rotind2tableind: rotameric-bin -> semirot row. The
            # semirot probability table has ``nrot_sr`` rows, one per
            # rotameric-chi bin. We map base-3 rotameric-chi indices into
            # ``[0, nrot_sr)``.
            src_rotamers = np.asarray(lib.rotameric_chi_rotamers, dtype=np.int32)
            rotinds = _base3_encode_rotamers(src_rotamers)
            sr_ri2ti[offset + si, rotinds] = np.arange(nrot_sr, dtype=np.int32)

        return sr_table, sr_index, sr_ri2ti, sr_start, sr_step, sr_period

    # ------------------------------------------------------------------ #
    # Score (hot path; runs inside jit)                                   #
    # ------------------------------------------------------------------ #

    def __call__(
        self,
        pose_stack: Any,
        term_state: DunbrackTermState,
        coords_override: Optional[jax.Array] = None,
    ) -> Dict[ScoreType, jax.Array]:
        """Compute ``{rot, rotdev, semirot}`` contributions per pose."""
        n_poses = pose_stack.n_poses

        if term_state.empty:
            zero = jnp.zeros((n_poses,), dtype=jnp.float32)
            return {
                ScoreType.dunbrack_rot: zero,
                ScoreType.dunbrack_rotdev: zero,
                ScoreType.dunbrack_semirot: zero,
            }

        coords = coords_override if coords_override is not None else pose_stack.coords
        pbt = pose_stack.packed_block_types

        # Resolve the 6 dihedrals per (pose, block) into a single
        # ``[P, B, 6]`` float array of angles (NaN for missing / padding
        # torsions, masked out below).
        dihedrals, valid_dihedral = self._compute_all_dihedrals(
            coords,
            pose_stack.block_type_ind,
            pose_stack.block_coord_offset,
            pose_stack.inter_residue_connections,
            pbt.conn_atoms,
            pbt.atom_is_real,
            term_state.dihedral_uaids,
        )

        # Vectorised per-residue score.
        rot_per_res, rotdev_per_res, semirot_per_res = jax.vmap(
            jax.vmap(_score_single_residue, in_axes=(0, 0, 0, None)),
            in_axes=(0, 0, 0, None),
        )(
            dihedrals,
            valid_dihedral,
            pose_stack.block_type_ind,
            term_state,
        )

        # Sum per pose. Invalid (block_type_ind < 0) residues are masked
        # to zero inside ``_score_single_residue``.
        return {
            ScoreType.dunbrack_rot: jnp.sum(rot_per_res, axis=1),
            ScoreType.dunbrack_rotdev: jnp.sum(rotdev_per_res, axis=1),
            ScoreType.dunbrack_semirot: jnp.sum(semirot_per_res, axis=1),
        }

    # ------------------------------------------------------------------ #
    # Dihedral resolution                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_all_dihedrals(
        coords: jax.Array,
        block_type_ind: jax.Array,
        block_coord_offset: jax.Array,
        inter_residue_connections: jax.Array,
        conn_atoms: jax.Array,
        atom_is_real: jax.Array,
        dihedral_uaids: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """Resolve and measure every torsion for every (pose, block).

        Returns
        -------
        dihedrals
            ``float32[P, B, 6]`` radians; junk in invalid slots.
        valid
            ``bool[P, B, 6]`` whether the torsion was resolvable
            (all four atoms in-bounds and real).
        """
        # Safe block-type lookup -- replace -1 padding with 0 for the
        # gather, and mask results with ``is_real_block``.
        safe_bti = jnp.where(block_type_ind >= 0, block_type_ind, 0)
        # Per-(pose, block) uaid array: [P, B, 6, 4, 3]
        uaids_pb = dihedral_uaids[safe_bti]

        # Resolve each of the 4 atoms of each torsion to a *global*
        # atom index into ``coords[P, :]``. Invalid entries get -1.
        resolved, valid_atoms = _resolve_uaids_to_atom_indices(
            uaids_pb,
            block_type_ind,
            block_coord_offset,
            inter_residue_connections,
            conn_atoms,
            atom_is_real,
        )

        # Gather 4 atom positions per torsion and compute the dihedral.
        # resolved has shape [P, B, 6, 4]. Use safe indexing (clamp to 0)
        # and mask via valid_atoms.
        safe_resolved = jnp.where(resolved >= 0, resolved, 0)

        # coords has shape [P, A, 3]; gather per-pose.
        pose_ind = jnp.arange(coords.shape[0])[:, None, None, None]  # [P,1,1,1]
        pose_ind = jnp.broadcast_to(pose_ind, safe_resolved.shape)
        gathered = coords[pose_ind, safe_resolved]  # [P, B, 6, 4, 3]

        # Dihedral is valid only if all four atoms resolved cleanly.
        valid_dihedral = jnp.all(valid_atoms, axis=-1)

        # To keep gradients well-defined on invalid slots, substitute a
        # non-degenerate fallback geometry (a canonical zigzag along x
        # / y axes) for those entries. The result is then masked to
        # zero below; using a well-defined dihedral here avoids NaN
        # gradients leaking into otherwise-independent terms through
        # JAX's non-lazy ``where`` semantics.
        safe_fallback = jnp.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
            dtype=gathered.dtype,
        )  # [4, 3]
        valid_atoms_e = valid_atoms[..., None]  # [P, B, 6, 4, 1]
        gathered = jnp.where(valid_atoms_e, gathered, safe_fallback)

        a = gathered[..., 0, :]
        b = gathered[..., 1, :]
        c = gathered[..., 2, :]
        d = gathered[..., 3, :]
        dihedrals = dihedral_angle(a, b, c, d)  # [P, B, 6]

        # Substitute upstream defaults for unresolved phi/psi:
        # terminal residues (``N`` has no preceding C, ``C`` has no
        # following N) get phi = -60 deg / psi = +60 deg, matching
        # ``dunbrack_pose_score.impl.hh`` (PHI_DEFAULT/PSI_DEFAULT).
        # Chi torsions stay at 0.0 for missing slots -- they are
        # explicitly masked out below via ``valid_dihedral``.
        phi_default = jnp.deg2rad(jnp.asarray(-60.0, dtype=dihedrals.dtype))
        psi_default = jnp.deg2rad(jnp.asarray(60.0, dtype=dihedrals.dtype))
        dihedrals = dihedrals.at[..., 0].set(
            jnp.where(valid_dihedral[..., 0], dihedrals[..., 0], phi_default)
        )
        dihedrals = dihedrals.at[..., 1].set(
            jnp.where(valid_dihedral[..., 1], dihedrals[..., 1], psi_default)
        )
        # Once phi/psi have defaults, treat them as valid for downstream
        # lookups (their interpolated mean/sigma are still meaningful).
        # Chi slots (indices 2..5) keep their original validity.
        phipsi_valid = jnp.ones_like(valid_dihedral[..., :2])
        valid_dihedral = jnp.concatenate(
            [phipsi_valid, valid_dihedral[..., 2:]], axis=-1
        )

        # Replace invalid chi dihedrals with 0 so downstream math is stable.
        dihedrals = jnp.where(valid_dihedral, dihedrals, 0.0)
        return dihedrals, valid_dihedral


def _resolve_uaids_to_atom_indices(
    uaids: jax.Array,
    block_type_ind: jax.Array,
    block_coord_offset: jax.Array,
    inter_residue_connections: jax.Array,
    conn_atoms: jax.Array,
    atom_is_real: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """Resolve unresolved-atom-ids to global atom indices inside a pose.

    A uaid is ``(atom_idx, conn_id, n_bonds_from_conn)``:

    * If ``atom_idx >= 0``: the atom is intra-residue on the current
      block. Resolve via ``block_coord_offset + atom_idx``.
    * If ``atom_idx == -1`` and ``conn_id >= 0``: the atom lives on a
      neighbouring residue. We look up the neighbour via
      ``inter_residue_connections[pose, block, conn_id] ->
      (other_block, other_conn)`` and take the connection-atom of the
      other block. ``n_bonds_from_conn`` selects *which* atom walking
      downstream from that connection; for phi/psi this is always 0
      (the partner residue's directly bonded atom).
    * Otherwise invalid; marked as ``valid=False``.

    For simplicity (and because phi/psi only ever cross a single
    connection), we ignore ``n_bonds_from_conn > 0`` requests and just
    use the connection atom of the partner residue. This matches
    upstream phi/psi behaviour exactly because the connection atom
    *is* the partner's C (for phi) or N (for psi).

    Parameters
    ----------
    uaids
        ``int32[P, B, D, 4, 3]`` -- uaids per (pose, block, dihedral,
        atom).
    block_type_ind
        ``int32[P, B]`` -- block-type index (or -1 for padding).
    block_coord_offset
        ``int32[P, B]`` -- atom-offset of each block inside
        ``coords[P, :]``.
    inter_residue_connections
        ``int32[P, B, C, 2]`` -- ``(other_block, other_conn)`` per
        outgoing connection.
    conn_atoms
        ``int32[n_types, C]`` -- atom index within each block type for
        every outgoing connection.
    atom_is_real
        ``bool[n_types, max_n_atoms]`` -- which slots are real atoms.

    Returns
    -------
    resolved
        ``int32[P, B, D, 4]`` global atom index into
        ``coords[P, :, :]``. ``-1`` for invalid entries.
    valid
        ``bool[P, B, D, 4]``.
    """
    P, B, D, _, _ = uaids.shape
    del _  # unused

    atom_ids = uaids[..., 0]  # [P, B, D, 4]
    conn_ids = uaids[..., 1]
    # n_bonds_from_conn = uaids[..., 2]  # not used here (phi/psi-only)

    is_intra = atom_ids >= 0
    is_inter = (atom_ids == -1) & (conn_ids >= 0)

    # Block coord offsets broadcast to [P, B, D, 4].
    bco = block_coord_offset[:, :, None, None]  # [P, B, 1, 1]

    # Intra-residue resolution: atom_offset + atom_idx.
    intra_global = bco + jnp.where(is_intra, atom_ids, 0)

    # Inter-residue resolution: follow the connection.
    # For each (pose, block, dihedral, atom) with a conn_id >= 0:
    #   other_block, other_conn = irc[pose, block, conn_id]
    #   other_bt = block_type_ind[pose, other_block]
    #   partner_atom_in_block = conn_atoms[other_bt, other_conn]
    #   global_atom = block_coord_offset[pose, other_block] + partner_atom_in_block

    safe_conn = jnp.where(conn_ids >= 0, conn_ids, 0)  # [P, B, D, 4]

    # Gather irc entries per (pose, block, dihedral, atom, conn) -> (P, B, D, 4, 2).
    # inter_residue_connections has shape [P, B, C, 2].
    # We need irc[pose, block, safe_conn] -> [P, B, D, 4, 2].
    # Build index helpers.
    pose_idx = jnp.arange(P)[:, None, None, None]  # [P,1,1,1]
    block_idx = jnp.arange(B)[None, :, None, None]  # [1,B,1,1]
    pose_idx = jnp.broadcast_to(pose_idx, safe_conn.shape)
    block_idx = jnp.broadcast_to(block_idx, safe_conn.shape)

    irc_gather = inter_residue_connections[pose_idx, block_idx, safe_conn]
    other_block = irc_gather[..., 0]  # [P, B, D, 4]
    other_conn = irc_gather[..., 1]

    # Mask: only treat as inter if the connection resolved.
    has_neighbour = (other_block >= 0) & is_inter

    safe_other_block = jnp.where(other_block >= 0, other_block, 0)
    safe_other_conn = jnp.where(other_conn >= 0, other_conn, 0)

    other_bt = block_type_ind[pose_idx, safe_other_block]  # [P,B,D,4]
    safe_other_bt = jnp.where(other_bt >= 0, other_bt, 0)

    # Gather the partner's connection atom within its block type.
    partner_atom_in_block = conn_atoms[safe_other_bt, safe_other_conn]
    partner_is_valid = (partner_atom_in_block >= 0) & (other_bt >= 0)

    # Global offset of the partner block inside this pose's coords.
    other_block_offset = block_coord_offset[pose_idx, safe_other_block]
    inter_global = other_block_offset + jnp.where(
        partner_atom_in_block >= 0, partner_atom_in_block, 0
    )

    # Merge.
    resolved = jnp.where(
        is_intra,
        intra_global,
        jnp.where(is_inter & has_neighbour & partner_is_valid, inter_global, -1),
    )

    # Validity: must be intra, *or* inter with a real partner. Also
    # require ``block_type_ind >= 0`` on the host block (padding).
    host_bti = block_type_ind[:, :, None, None]  # [P,B,1,1]
    host_valid = host_bti >= 0
    valid = (
        host_valid
        & (is_intra | (is_inter & has_neighbour & partner_is_valid))
        & (resolved >= 0)
    )

    return resolved, valid


# --------------------------------------------------------------------------- #
# Per-residue scoring kernel                                                  #
# --------------------------------------------------------------------------- #


def _score_single_residue(
    dihedrals: jax.Array,  # [6] radians (phi, psi, chi1..chi4)
    valid_dihedral: jax.Array,  # [6] bool
    block_type_ind: jax.Array,  # scalar int
    state: DunbrackTermState,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Compute (rot, rotdev, semirot) for one residue.

    Designed to be ``jit`` / ``vmap`` safe: all Python-level decisions
    have been lifted into static metadata on ``state``; the only traced
    branching is a few ``jnp.where`` masks to zero out padding.
    """
    # --- fetch static block-type metadata (gather). -------------------- #
    safe_bti = jnp.where(block_type_ind >= 0, block_type_ind, 0)
    aa = state.aa_table_ind[safe_bti]
    n_chi = state.n_chi_per_type[safe_bti]
    n_rot_chi = state.n_rotameric_chi_per_type[safe_bti]
    is_sr = state.is_semirotameric_per_type[safe_bti]

    valid = (block_type_ind >= 0) & (aa >= 0)

    # --- fetch per-AA library rows. ----------------------------------- #
    safe_aa = jnp.where(aa >= 0, aa, 0)

    bb_start = state.bb_start[safe_aa]
    bb_step = state.bb_step[safe_aa]
    bb_period = state.bb_period[safe_aa]

    phi = dihedrals[0]
    psi = dihedrals[1]
    chis = dihedrals[2:]
    chi_valid = valid_dihedral[2:]

    # Chi-index mask per AA: the rotdev penalty only covers
    # *rotameric* chis (upstream ``block_n_rotameric_chi`` loop).
    # For purely rotameric residues ``n_rot_chi == n_chi``; for
    # semirotameric residues ``n_rot_chi == n_chi - 1`` (the last
    # chi is continuous and handled by the semirot table).
    chi_mask = jnp.arange(_MAX_N_CHI) < n_rot_chi
    chi_mask = chi_mask & chi_valid

    # --- rotamer-bin classification ----------------------------------- #
    # Upstream's base-3 encoding. We compute it for the maximum #rot chi
    # and then mask/truncate via n_rot_chi.
    rot_bin = _classify_rot_bin_masked(chis, n_rot_chi)
    # Map bin -> table row.
    ri2ti_for_aa = state.rotind2tableind[safe_aa]
    # Clamp bin index to table length for safety.
    safe_bin = jnp.clip(rot_bin, 0, ri2ti_for_aa.shape[0] - 1)
    rot_row = ri2ti_for_aa[safe_bin]
    rot_row_valid = rot_row >= 0
    safe_rot_row = jnp.where(rot_row_valid, rot_row, 0)

    # --- dunbrack_rot: bilinear interp of -log P table ---------------- #
    # Upstream (``dunbrack_pose_score.impl.hh`` line 272) skips the rot
    # energy entirely for semirotameric residues -- their
    # ``block_rotameric_index`` is -1, gating the whole
    # ``rotameric_chi_probability_for_block`` call. The terminal chi of a
    # semirotameric residue is scored by the separate ``semirot`` table
    # which already encodes ``-log P(chi_last | rotameric-bin, phi, psi)``,
    # so adding a rotameric -log P on top would double-count the rotamer
    # prior. We replicate that gate here.
    nlp_table = state.rot_neglnprob_table[safe_aa, safe_rot_row]  # [N_phi, N_psi]
    fi = _frac_index(phi, bb_start[0], bb_step[0], bb_period[0])
    fj = _frac_index(psi, bb_start[1], bb_step[1], bb_period[1])
    # nlp_table ships as bicubic B-spline coefficients (produced at setup
    # time from the raw -log P samples); use the matching interpolator.
    rot_score = _bicubic_bspline_interpolate(nlp_table, fi, fj)
    rot_score = jnp.where(valid & rot_row_valid & ~is_sr, rot_score, 0.0)

    # --- dunbrack_rotdev: Gaussian chi penalty ------------------------ #
    # Upstream stores chi mean/sdev as prefiltered 2-D bicubic B-spline
    # *coefficients* (see ``_stack_rotameric_tables`` and
    # ``dunbrack/params.py::_calculate_rot_mean_coeffs``). Match that with
    # a bicubic B-spline evaluator at score time.
    mean_tables = state.rot_mean_table[safe_aa, safe_rot_row]  # [max_nchi, Nphi, Npsi]
    sdev_tables = state.rot_sdev_table[safe_aa, safe_rot_row]  # [max_nchi, Nphi, Npsi]

    chi_means = jax.vmap(lambda t: _bicubic_bspline_interpolate(t, fi, fj))(
        mean_tables
    )  # [max_nchi]
    chi_sigmas = jax.vmap(lambda t: _bicubic_bspline_interpolate(t, fi, fj))(
        sdev_tables
    )
    rotdev_score = dunbrack_rotdev_score(
        chis,
        chi_means,
        chi_sigmas,
        chi_mask=chi_mask,
    )
    rotdev_score = jnp.where(valid & rot_row_valid, rotdev_score, 0.0)

    # --- dunbrack_semirot: trilinear interp ---------------------------- #
    sr_ind = state.semirot_index_for_aa[safe_aa]
    sr_valid = is_sr & (sr_ind >= 0)
    safe_sr_ind = jnp.where(sr_valid, sr_ind, 0)
    sr_start = state.semirot_start[safe_sr_ind]
    sr_step = state.semirot_step[safe_sr_ind]
    sr_period = state.semirot_period[safe_sr_ind]

    # For semirotameric libs the ``semirot_table`` is indexed by the
    # *semirot-bin* index (in ``[0, nrot_sr)``), which is separate
    # from the rotameric-table row index (``rot_row``). Dispatch
    # through a dedicated ``semirot_rotind2tableind`` built from
    # ``lib.rotameric_chi_rotamers``.
    sr_ri2ti_for_aa = state.semirot_rotind2tableind[safe_aa]
    safe_sr_bin = jnp.clip(rot_bin, 0, sr_ri2ti_for_aa.shape[0] - 1)
    sr_table_row_raw = sr_ri2ti_for_aa[safe_sr_bin]
    sr_row_valid_sr = sr_table_row_raw >= 0
    sr_table_row = jnp.clip(sr_table_row_raw, 0, state.semirot_table.shape[1] - 1)
    sr_table_row = jnp.where(sr_valid & sr_row_valid_sr, sr_table_row, 0)

    sr_table = state.semirot_table[safe_sr_ind, sr_table_row]  # [Nphi, Npsi, Nchi]
    # The last chi is at index (n_chi - 1).
    last_chi_idx = jnp.clip(n_chi - 1, 0, _MAX_N_CHI - 1)
    chi_last = chis[last_chi_idx]
    chi_last_valid = chi_valid[last_chi_idx]

    sr_score = dunbrack_semirot_score(
        phi,
        psi,
        chi_last,
        sr_table,
        sr_start,
        sr_step,
        sr_period,
    )
    sr_score = jnp.where(
        valid & sr_valid & sr_row_valid_sr & chi_last_valid, sr_score, 0.0
    )

    return rot_score, rotdev_score, sr_score


def _classify_rot_bin_masked(chis: jax.Array, n_rot_chi: jax.Array) -> jax.Array:
    """Base-3 rotamer-bin index that counts only the first ``n_rot_chi`` chis.

    ``classify_rotamer_bin`` in :mod:`.potentials` takes a static int
    for the chi count. Here the count is data-dependent (different
    block types in the same pose), so we unroll over ``_MAX_N_CHI``
    and mask with ``i < n_rot_chi``.
    """
    deg120 = jnp.deg2rad(120.0)

    def bin_of(chi):
        return jnp.where(
            chi < -deg120,
            1,
            jnp.where(chi < 0.0, 2, jnp.where(chi < deg120, 0, 1)),
        )

    rot_ind = jnp.int32(0)
    for i in range(_MAX_N_CHI):
        use = i < n_rot_chi
        # Only multiply-and-add on "used" chis; otherwise pass through.
        new_val = rot_ind * 3 + bin_of(chis[i])
        rot_ind = jnp.where(use, new_val, rot_ind)
    return rot_ind


__all__ = [
    "DunbrackEnergyTerm",
    "DunbrackTermState",
]
