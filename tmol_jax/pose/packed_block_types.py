"""JAX port of ``tmol.pose.packed_block_types``.

``PackedBlockTypes`` is the "palette" of residue types used in a
``PoseStack``. It flattens the per-residue-type arrays (atom types,
ideal coords, bonds, connections, intra-residue path distances) into
packed, block-wise arrays indexed by block-type-index.

The upstream class is an ``attr.s`` class that stores ``torch.Tensor``s
plus some auxiliary Python objects (``chem_db``, ``restype_set``,
``pandas.Index`` of names). This port swaps:

  * ``torch`` → ``jax.numpy``;
  * ``attr.s`` → the tmol_jax ``@pytree_dataclass`` wrapper so the whole
    object flows through ``jit``/``vmap``/``grad`` naturally as a
    pytree, with array fields as leaves and Python containers /
    ``RefinedResidueType``s as aux data.

Only the subset of fields that do *not* depend on torsion / polymeric
connection bookkeeping (``ordered_torsions`` etc.) is populated here;
those are handled by other specialised pytrees elsewhere in the port.
The fields we expose match the "palette" as used by scoring-term
setup and by ``PoseStack``'s block-wise lookups.

Imports of the chemical-side classes (``RefinedResidueType``,
``PatchedChemicalDatabase``) are done *lazily* inside methods so this
module can be imported before the chemical subpackage is fully ported
without triggering circular imports.
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import attrs
import jax.numpy as jnp
import numpy as np

from tmol_jax.types import pytree_dataclass

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _element_index_map(chem_db: Any) -> dict:
    """Build ``{element_symbol: int_index}`` from a PatchedChemicalDatabase.

    Element indices are assigned in the order they appear in
    ``chem_db.element_types``. An "UNKNOWN"-style fallback is emitted as
    index ``-1`` by callers that consult this map and get ``None``.
    """
    return {el.name: i for i, el in enumerate(chem_db.element_types)}


def _atom_type_index_map(chem_db: Any) -> dict:
    """Build ``{atom_type_name: int_index}`` from a PatchedChemicalDatabase."""
    return {at.name: i for i, at in enumerate(chem_db.atom_types)}


def _atom_element_map(chem_db: Any) -> dict:
    """Build ``{atom_type_name: element_symbol}`` from the atom types table."""
    return {at.name: at.element for at in chem_db.atom_types}


def _path_distance_within_residue(
    n_atoms: int, bond_indices: np.ndarray, max_sep: int = 6
) -> np.ndarray:
    """Compute shortest-path distance (in #bonds) between every pair of atoms
    in a single residue type, clipped at ``max_sep``.

    BFS from each source atom -- avoids the scipy.sparse dependency of
    the upstream implementation while producing an identical result for
    the connectivity we care about (unweighted undirected bond graph).
    """
    if n_atoms == 0:
        return np.zeros((0, 0), dtype=np.int32)

    # Adjacency list from the (redundant, symmetric) bond edge list.
    adj: List[List[int]] = [[] for _ in range(n_atoms)]
    for i in range(bond_indices.shape[0]):
        a, b = int(bond_indices[i, 0]), int(bond_indices[i, 1])
        if 0 <= a < n_atoms and 0 <= b < n_atoms:
            adj[a].append(b)

    dist = np.full((n_atoms, n_atoms), max_sep, dtype=np.int32)
    for src in range(n_atoms):
        # BFS with a simple queue; max depth is ``max_sep`` since
        # further-than-that distances are clamped.
        dist[src, src] = 0
        frontier = [src]
        d = 0
        while frontier and d < max_sep:
            d += 1
            nxt: List[int] = []
            for u in frontier:
                for v in adj[u]:
                    if dist[src, v] > d:
                        dist[src, v] = d
                        nxt.append(v)
            frontier = nxt
    return dist


# --------------------------------------------------------------------------- #
# PackedBlockTypes                                                             #
# --------------------------------------------------------------------------- #


@pytree_dataclass
@attrs.define(slots=False)
class PackedBlockTypes:
    """A flat, array-backed "palette" of residue types.

    Given a list of ``RefinedResidueType``s (the *active* block types
    the caller cares about), this class produces packed 2-D / 3-D
    arrays with the leading dimension being block-type index. Unused
    slots are zero-filled (for int/float arrays) or ``False`` (for
    bool masks); the ``n_atoms`` / ``n_bonds`` / ``n_conn`` arrays
    record the true per-type counts.

    Array fields are JAX arrays and therefore act as pytree leaves;
    Python-typed fields (``chem_db``, ``active_block_types``,
    ``atom_names``, the ``max_*`` / ``n_types`` ints) are pytree aux.
    """

    # --- aux / metadata ------------------------------------------------ #
    chem_db: Any
    active_block_types: Tuple[Any, ...]
    atom_names: Tuple[Tuple[str, ...], ...]

    n_types: int
    max_n_atoms: int
    max_n_bonds: int
    max_n_conn: int

    # --- packed per-type arrays (pytree leaves) ------------------------ #
    n_atoms: jnp.ndarray  # [n_types] int32
    atom_is_real: jnp.ndarray  # [n_types, max_n_atoms] bool
    atom_elements: jnp.ndarray  # [n_types, max_n_atoms] int32
    atom_types: jnp.ndarray  # [n_types, max_n_atoms] int32
    atom_is_hydrogen: jnp.ndarray  # [n_types, max_n_atoms] bool
    atom_is_backbone: jnp.ndarray  # [n_types, max_n_atoms] bool

    bond_indices: jnp.ndarray  # [n_types, max_n_bonds, 2] int32
    n_bonds: jnp.ndarray  # [n_types] int32

    ideal_coords: jnp.ndarray  # [n_types, max_n_atoms, 3] float32

    conn_atoms: jnp.ndarray  # [n_types, max_n_conn] int32
    n_conn: jnp.ndarray  # [n_types] int32

    intrares_path_distance: jnp.ndarray
    # [n_types, max_n_atoms, max_n_atoms] int32

    # ------------------------------------------------------------------ #
    # Core builder                                                        #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_restypes(
        cls,
        chem_db: Any,
        active_types: Sequence[Any],
    ) -> "PackedBlockTypes":
        """Build a ``PackedBlockTypes`` from a list of
        ``RefinedResidueType`` instances.

        The build proceeds on the CPU via ``numpy`` (residue-type
        metadata is Python-object-heavy and iterating over it from JAX
        tracing would be nonsensical) and the final arrays are moved
        onto whatever device ``jax.numpy`` defaults to via
        ``jnp.asarray``.
        """
        from tmol_jax.chemical.ideal_coords import build_ideal_coords

        active_types = tuple(active_types)
        n_types = len(active_types)

        # Cached lookups on the chemical database.
        element_index = _element_index_map(chem_db)
        atom_type_index = _atom_type_index_map(chem_db)
        atom_element = _atom_element_map(chem_db)

        # -------------------------------------------------------------- #
        # Dimensions                                                      #
        # -------------------------------------------------------------- #
        if n_types == 0:
            max_n_atoms = 0
            max_n_bonds = 0
            max_n_conn = 0
        else:
            max_n_atoms = max(len(rt.atoms) for rt in active_types)
            max_n_bonds = max(
                (rt.bond_indices.shape[0] if rt.bond_indices is not None else 0)
                for rt in active_types
            )
            max_n_conn = max(len(rt.connections) for rt in active_types)

        # -------------------------------------------------------------- #
        # Per-type scalar counts                                          #
        # -------------------------------------------------------------- #
        n_atoms_np = np.asarray([len(rt.atoms) for rt in active_types], dtype=np.int32)
        n_bonds_np = np.asarray(
            [
                (rt.bond_indices.shape[0] if rt.bond_indices is not None else 0)
                for rt in active_types
            ],
            dtype=np.int32,
        )
        n_conn_np = np.asarray(
            [len(rt.connections) for rt in active_types], dtype=np.int32
        )

        # -------------------------------------------------------------- #
        # Per-atom packed arrays                                          #
        # -------------------------------------------------------------- #
        atom_is_real_np = np.zeros((n_types, max_n_atoms), dtype=bool)
        atom_elements_np = np.zeros((n_types, max_n_atoms), dtype=np.int32)
        atom_types_np = np.zeros((n_types, max_n_atoms), dtype=np.int32)
        atom_is_hydrogen_np = np.zeros((n_types, max_n_atoms), dtype=bool)
        atom_is_backbone_np = np.zeros((n_types, max_n_atoms), dtype=bool)
        atom_names_list: List[Tuple[str, ...]] = []

        for i, rt in enumerate(active_types):
            atoms = list(rt.atoms)
            atom_is_real_np[i, : len(atoms)] = True

            # Main-chain atom set (may be ``None`` / empty for non-
            # polymeric residues). We use *set* membership rather than
            # atom index because some variants re-order atoms.
            mc_atoms: set[str]
            try:
                polymer = rt.properties.polymer
                mc_atoms = set(polymer.mainchain_atoms or ())
            except AttributeError:
                mc_atoms = set()

            names: List[str] = []
            for j, atom in enumerate(atoms):
                names.append(atom.name)
                atype = atom.atom_type
                atom_types_np[i, j] = atom_type_index.get(atype, -1)
                elem = atom_element.get(atype, None)
                atom_elements_np[i, j] = (
                    element_index.get(elem, -1) if elem is not None else -1
                )
                atom_is_hydrogen_np[i, j] = elem == "H"
                atom_is_backbone_np[i, j] = atom.name in mc_atoms
            # pad names out to ``max_n_atoms`` with the empty string
            while len(names) < max_n_atoms:
                names.append("")
            atom_names_list.append(tuple(names))

        # -------------------------------------------------------------- #
        # Bond edges                                                      #
        # -------------------------------------------------------------- #
        bond_indices_np = np.zeros((n_types, max_n_bonds, 2), dtype=np.int32)
        for i, rt in enumerate(active_types):
            bi = rt.bond_indices
            if bi is not None and bi.shape[0] > 0:
                bond_indices_np[i, : bi.shape[0]] = bi.astype(np.int32)

        # -------------------------------------------------------------- #
        # Connections: atom index each connection exits through           #
        # -------------------------------------------------------------- #
        conn_atoms_np = np.full((n_types, max_n_conn), -1, dtype=np.int32)
        for i, rt in enumerate(active_types):
            oc = getattr(rt, "ordered_connection_atoms", None)
            if oc is not None and oc.shape[0] > 0:
                conn_atoms_np[i, : oc.shape[0]] = oc.astype(np.int32)
            else:
                # Fall back to iterating the raw connection list.
                for k, conn in enumerate(rt.connections):
                    conn_atoms_np[i, k] = rt.atom_to_idx[conn.atom]

        # -------------------------------------------------------------- #
        # Ideal coords                                                    #
        # -------------------------------------------------------------- #
        ideal_coords_np = np.zeros((n_types, max_n_atoms, 3), dtype=np.float32)
        for i, rt in enumerate(active_types):
            coords = build_ideal_coords(rt)
            ideal_coords_np[i, : len(rt.atoms)] = coords[rt.at_to_icoor_ind]

        # -------------------------------------------------------------- #
        # Intra-residue path distances                                    #
        # -------------------------------------------------------------- #
        intrares_pd_np = np.zeros((n_types, max_n_atoms, max_n_atoms), dtype=np.int32)
        for i, rt in enumerate(active_types):
            n = len(rt.atoms)
            # Prefer the cached ``path_distance`` attribute when present;
            # otherwise compute via BFS.
            pd = getattr(rt, "path_distance", None)
            if pd is None:
                bi = (
                    rt.bond_indices
                    if rt.bond_indices is not None
                    else np.zeros((0, 2), dtype=np.int32)
                )
                pd = _path_distance_within_residue(n, bi)
            intrares_pd_np[i, :n, :n] = pd[:n, :n].astype(np.int32)

        # -------------------------------------------------------------- #
        # Build the instance                                              #
        # -------------------------------------------------------------- #
        return cls(
            chem_db=chem_db,
            active_block_types=active_types,
            atom_names=tuple(atom_names_list),
            n_types=int(n_types),
            max_n_atoms=int(max_n_atoms),
            max_n_bonds=int(max_n_bonds),
            max_n_conn=int(max_n_conn),
            n_atoms=jnp.asarray(n_atoms_np),
            atom_is_real=jnp.asarray(atom_is_real_np),
            atom_elements=jnp.asarray(atom_elements_np),
            atom_types=jnp.asarray(atom_types_np),
            atom_is_hydrogen=jnp.asarray(atom_is_hydrogen_np),
            atom_is_backbone=jnp.asarray(atom_is_backbone_np),
            bond_indices=jnp.asarray(bond_indices_np),
            n_bonds=jnp.asarray(n_bonds_np),
            ideal_coords=jnp.asarray(ideal_coords_np),
            conn_atoms=jnp.asarray(conn_atoms_np),
            n_conn=jnp.asarray(n_conn_np),
            intrares_path_distance=jnp.asarray(intrares_pd_np),
        )

    # ------------------------------------------------------------------ #
    # Convenience constructors                                            #
    # ------------------------------------------------------------------ #
    @classmethod
    def default(cls, chem_db: Any | None = None) -> "PackedBlockTypes":
        """Build a ``PackedBlockTypes`` from the default patched chem-db
        including the 20 canonical amino acids **plus their termini
        variants** (``:nterm``, ``:cterm``, ``:cterm:nterm``).

        Upstream tmol's ``PoseStack`` construction detects chain termini
        and assigns the appropriate variant (e.g. first residue of a
        chain becomes ``MET:nterm``, last becomes ``GLY:cterm``). That
        changes atom counts (N-term gains H1/H2/H3 / loses H; C-term
        gains OXT) and charges (N-term NH3+, C-term COO-), which shift
        fa_elec, fa_lj*, fa_lk, hbond scores by ~2-5%. To match that
        behaviour, the default palette includes every reachable
        variant combination for the 20 canonical AAs -- 4 variants per
        AA (base / :nterm / :cterm / :cterm:nterm) for a total of ~80
        block types.

        If ``chem_db`` is not supplied, the default
        ``PatchedChemicalDatabase`` is loaded via
        ``tmol_jax.chemical.patched_chemdb.default_patched_chemdb()``.

        The palette ordering is: ALA, ALA:nterm, ALA:cterm,
        ALA:cterm:nterm, CYS, ... -- base-first then variants, so that
        callers which only rely on ``name3`` lookups of the 20 bases
        (legacy path) still find them as the first representative.
        """
        # Lazy imports (see module docstring for why).
        if chem_db is None:
            from tmol_jax.chemical.patched_chemdb import (
                default_patched_chemdb,
            )

            chem_db = default_patched_chemdb()

        from tmol_jax.chemical.restypes import RefinedResidueType  # noqa: F401

        canonical_aas = (
            "ALA",
            "CYS",
            "ASP",
            "GLU",
            "PHE",
            "GLY",
            "HIS",
            "ILE",
            "LYS",
            "LEU",
            "MET",
            "ASN",
            "PRO",
            "GLN",
            "ARG",
            "SER",
            "THR",
            "VAL",
            "TRP",
            "TYR",
        )

        # Group residues by base ``name3``; we collect every variant
        # whose ``name3`` matches one of the canonical AAs.
        by_name3: dict = {}
        for res in chem_db.residues:
            n3 = getattr(res, "name3", None)
            name = getattr(res, "name", None)
            if n3 is None or name is None:
                continue
            by_name3.setdefault(n3, []).append(res)

        def _variant_sort_key(res: Any) -> tuple:
            """Sort key: base residue first, then by variant-tag tuple.

            We rely on the residue's ``name`` being ``base`` (no colon)
            for the base variant, and ``base:<display_name>[:...]`` for
            patched ones. The key orders:

              - bases first (empty variant tuple),
              - then single variants sorted alphabetically,
              - then double+ variants sorted alphabetically.
            """
            name = getattr(res, "name", "")
            parts = name.split(":")
            # Everything after the base name is a sorted tuple of
            # display names (the chemdb already sorts them).
            variants = tuple(parts[1:])
            return (len(variants), variants, name)

        selected: List[Any] = []
        for aa in canonical_aas:
            variants = sorted(by_name3.get(aa, []), key=_variant_sort_key)
            for res in variants:
                if not isinstance(res, RefinedResidueType):
                    from tmol_jax.chemical.restypes import _refined_from_raw

                    res = _refined_from_raw(res)
                selected.append(res)

        return cls.from_restypes(chem_db, selected)

    # ------------------------------------------------------------------ #
    # Lookup helpers                                                      #
    # ------------------------------------------------------------------ #
    def inds_for_restypes(self, res_types: Sequence[Any]) -> np.ndarray:
        """Return the index of each ``res_type`` within
        ``active_block_types``, or ``-1`` when absent.

        Returned as a plain ``numpy`` array -- these are discrete
        metadata lookups, never something we want to trace through
        JAX.
        """
        name_to_idx = {rt.name: i for i, rt in enumerate(self.active_block_types)}
        return np.asarray(
            [name_to_idx.get(rt.name, -1) for rt in res_types], dtype=np.int32
        )


__all__ = ["PackedBlockTypes"]
