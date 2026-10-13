"""Canonical-form atom ordering for tmol_jax.

Ported from ``tmol.io.canonical_ordering``.

A :class:`CanonicalOrdering` holds a "palette" of per-residue-type
canonical atom orderings, built from a
:class:`~tmol_jax.chemical.patched_chemdb.PatchedChemicalDatabase`. For
each residue type it records:

- The ordered tuple of atom names (backbone ``N / CA / C / O`` followed
  by the side-chain atoms, in the order they appear in the chemical
  database).
- A mapping from atom name (and any ``atom_aliases`` alternate name) to
  the integer index of that atom within the canonical ordering. This
  lets callers fill in a per-position coord tensor given only atom
  names.

The module also exports two default-factory helpers used by the rest of
the io pipeline:

- :func:`default_canonical_ordering` returns the canonical ordering for
  the standard 20 amino acids, built from the default chemical database.
- :func:`default_packed_block_types` returns the matching
  :class:`PackedBlockTypes` "palette" for the same 20 residues and
  their N-/C-terminal variants (``:nterm``, ``:cterm``,
  ``:cterm:nterm``). Including the termini variants in the default
  palette lets ``pose_stack_from_canonical_form`` assign the
  appropriate variant to each chain's first / last residue (matching
  upstream tmol behaviour), which is necessary for correct fa_elec /
  hbond / LJ-LK scoring.

We only support the 20 canonical amino acids (plus their termini
variants) here; unknown residue types are returned with sentinel
values when queried so callers can filter them out.
"""

from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional, Tuple

import attr
import numpy as np

# --------------------------------------------------------------------- #
# Helpers                                                                #
# --------------------------------------------------------------------- #


class _OrderedSet:
    """Minimal ordered-set helper used to preserve chem-db iteration order."""

    __slots__ = ("_seen", "_ordered")

    def __init__(self) -> None:
        self._seen: set = set()
        self._ordered: List[Any] = []

    def add(self, value: Any) -> None:
        if value in self._seen:
            return
        self._seen.add(value)
        self._ordered.append(value)

    @property
    def values(self) -> List[Any]:
        return list(self._ordered)


# --------------------------------------------------------------------- #
# CanonicalOrdering                                                      #
# --------------------------------------------------------------------- #


@attr.s(auto_attribs=True, frozen=True, slots=True)
class CanonicalOrdering:
    """Per-residue-type canonical atom ordering.

    Attributes
    ----------
    max_n_canonical_atoms:
        The largest number of distinct canonical atom names across all
        residue-type equivalence classes in the palette.
    restype_io_equiv_classes:
        Tuple of 3-letter residue-type codes, in the order they were
        added. Use ``tuple.index("ALA")`` etc. to get the integer index
        canonical-form arrays use for each residue.
    restypes_ordered_atom_names:
        Mapping ``{name3: tuple(atom_name, ...)}`` of the canonical
        atom-name ordering for each residue type. Does NOT include
        alternate names / aliases.
    restypes_atom_index_mapping:
        Mapping ``{name3: {atom_name: int_index}}``. Alternate names
        (from ``atom_aliases``) are merged in and point at the same
        index as the canonical name.
    restypes_mainchain_atoms:
        Mapping ``{name3: tuple(atom_name, ...) or None}`` with the
        polymeric main-chain atoms (N / CA / C / O for standard AAs).
    """

    max_n_canonical_atoms: int
    restype_io_equiv_classes: Tuple[str, ...]
    restypes_ordered_atom_names: Mapping[str, Tuple[str, ...]]
    restypes_atom_index_mapping: Mapping[str, Mapping[str, int]]
    restypes_mainchain_atoms: Mapping[str, Optional[Tuple[str, ...]]]

    # ---------------------------------------------------------------- #
    # Properties / lookups                                              #
    # ---------------------------------------------------------------- #

    @property
    def n_restype_io_equiv_classes(self) -> int:
        return len(self.restype_io_equiv_classes)

    def restype_index(self, name3: str) -> int:
        """Return the integer index of a 3-letter code, or ``-1``."""
        try:
            return self.restype_io_equiv_classes.index(name3)
        except ValueError:
            return -1

    def atom_index(self, name3: str, atom_name: str) -> int:
        """Return the canonical-form atom index for (``name3``, ``atom_name``),
        or ``-1`` when unknown."""
        mapping = self.restypes_atom_index_mapping.get(name3)
        if mapping is None:
            return -1
        return mapping.get(atom_name, -1)

    # ---------------------------------------------------------------- #
    # Builder                                                           #
    # ---------------------------------------------------------------- #

    @classmethod
    def from_chemdb(cls, chemdb: Any) -> "CanonicalOrdering":
        """Build a :class:`CanonicalOrdering` from a ``PatchedChemicalDatabase``.

        Walks ``chemdb.residues`` collecting, for each distinct 3-letter
        code, the union of atom names across every residue type (so that
        variant-specific atoms like termini hydrogens are representable
        in the canonical form). Atom alias mappings are folded into the
        returned atom-index lookup so callers can feed in either PDBv2
        or PDBv3 atom names without needing an extra translation step.
        """
        # 1. Iterate chem-db residues, collecting a per-name3 ordered
        #    set of atom names and a per-name3 alias-to-canonical map.
        name3_order = _OrderedSet()
        all_atoms: "OrderedDict[str, _OrderedSet]" = OrderedDict()
        aliases: "Dict[str, Dict[str, str]]" = {}
        mainchain: "Dict[str, Optional[Tuple[str, ...]]]" = {}

        for restype in chemdb.residues:
            name3 = restype.name3
            io_equiv = getattr(restype, "io_equiv_class", name3)

            name3_order.add(io_equiv)
            if io_equiv not in all_atoms:
                all_atoms[io_equiv] = _OrderedSet()
            for atom in restype.atoms:
                all_atoms[io_equiv].add(atom.name)

            if io_equiv not in aliases:
                aliases[io_equiv] = {}
            for alias in getattr(restype, "atom_aliases", ()) or ():
                # Alias records have ``.name`` (canonical) and ``.alt_name``.
                prior = aliases[io_equiv].get(alias.alt_name)
                if prior is not None and prior != alias.name:
                    # Conflicting alias -- keep the first and move on.
                    continue
                aliases[io_equiv][alias.alt_name] = alias.name

            if io_equiv not in mainchain:
                mc = None
                polymer = getattr(getattr(restype, "properties", None), "polymer", None)
                if polymer is not None:
                    mc_atoms = getattr(polymer, "mainchain_atoms", None)
                    if mc_atoms:
                        mc = tuple(mc_atoms)
                mainchain[io_equiv] = mc

        # 2. Freeze atom-name orderings and build atom-index maps.
        ordered_atom_names: Dict[str, Tuple[str, ...]] = {
            name3: tuple(all_atoms[name3].values) for name3 in all_atoms
        }
        atom_index_mapping: Dict[str, Dict[str, int]] = {
            name3: {name: i for i, name in enumerate(names)}
            for name3, names in ordered_atom_names.items()
        }
        # Merge aliases: alias_name -> same index as canonical.
        for name3, alias_map in aliases.items():
            idx_map = atom_index_mapping.setdefault(name3, {})
            for alt, canon in alias_map.items():
                canon_idx = idx_map.get(canon)
                if canon_idx is not None:
                    idx_map[alt] = canon_idx

        max_n = max((len(v) for v in ordered_atom_names.values()), default=0)

        return cls(
            max_n_canonical_atoms=int(max_n),
            restype_io_equiv_classes=tuple(name3_order.values),
            restypes_ordered_atom_names=ordered_atom_names,
            restypes_atom_index_mapping=atom_index_mapping,
            restypes_mainchain_atoms=mainchain,
        )


# --------------------------------------------------------------------- #
# Default-factory helpers                                                #
# --------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def default_canonical_ordering() -> CanonicalOrdering:
    """Return the :class:`CanonicalOrdering` for the default 20-AA palette.

    Uses :func:`tmol_jax.chemical.patched_chemdb.default_patched_chemdb`.
    """
    # Imported lazily to avoid a cycle at module import time.
    from tmol_jax.chemical.patched_chemdb import default_patched_chemdb
    from tmol_jax.chemical.constants import canonical_aa_3letter

    chemdb = default_patched_chemdb()

    # Filter to just the 20 canonical AAs (the chem-db may contain
    # extras like DNA bases, termini variants, etc.). We keep the
    # palette order ALA, CYS, ASP, ... matching ``canonical_aa_3letter``.
    canonical_set = set(canonical_aa_3letter)
    filtered_residues = tuple(
        r for r in chemdb.residues if getattr(r, "name3", None) in canonical_set
    )

    # Reconstruct a minimal stand-in chem-db with just the residues we
    # care about; ``from_chemdb`` only reads ``.residues``.
    class _MiniDb:
        __slots__ = ("residues",)

    mini = _MiniDb()
    mini.residues = filtered_residues  # type: ignore[attr-defined]

    return CanonicalOrdering.from_chemdb(mini)


@lru_cache(maxsize=1)
def default_packed_block_types() -> Any:
    """Return a :class:`PackedBlockTypes` matching :func:`default_canonical_ordering`.

    This is exactly :func:`PackedBlockTypes.default` -- the 20 canonical
    amino acids **plus their N-/C-terminal variants**
    (``:nterm``, ``:cterm``, ``:cterm:nterm``). The termini variants
    differ from the base block types in atom count and charge state:
    the N-terminal NH3+ adds H1/H2/H3 and drops H, and the C-terminal
    COO- adds OXT. Including them in the default palette lets
    :func:`pose_stack_from_canonical_form` assign the correct variant
    to each chain's first / last residue, matching upstream tmol
    behaviour.
    """
    from tmol_jax.pose.packed_block_types import PackedBlockTypes

    return PackedBlockTypes.default()


# --------------------------------------------------------------------- #
# canonical_form_from_pdb                                                #
# --------------------------------------------------------------------- #


def canonical_form_from_pdb(pdb_text_or_path, canonical_ordering=None):
    """Convert a PDB to canonical protein coordinates, retaining missing atoms.

    Noncanonical residues fail explicitly; water is ignored. Chain labels
    determine polymer connectivity. Coordinate gaps do not create chain breaks.
    """
    from .pdb_parsing import parse_pdb

    return _canonical_form_from_records(
        parse_pdb(pdb_text_or_path),
        canonical_ordering or default_canonical_ordering(),
    )


def _canonical_form_from_records(records, canonical_ordering):
    atom_names = records["atom_name"]
    res_names = records["res_name"]
    chain_ids_raw = records["chain_id"]
    res_nums_raw = records["res_num"]
    insert_raw = records["insert"]
    coords_raw = records["coords"]
    occupancy_raw = records["occupancy"]
    b_factor_raw = records["b_factor"]
    record_name = records.get(
        "record_name", np.asarray(["ATOM"] * len(atom_names), dtype=object)
    )
    alt_loc_raw = records.get(
        "alt_loc", np.asarray([""] * len(atom_names), dtype=object)
    )

    n_records = len(atom_names)

    # -------- Index distinct residues and chains -------------------- #
    unique_residues: "OrderedDict[Tuple[str, int, str], int]" = OrderedDict()
    unique_chains: "OrderedDict[str, int]" = OrderedDict()

    # Per-atom: which residue row / chain slot it belongs to.
    res_ind_per_atom = np.full(n_records, -1, dtype=np.int32)
    keep_atom = np.zeros(n_records, dtype=bool)

    skipped_residue_names: Dict[str, int] = {}

    for i in range(n_records):
        rec = str(record_name[i]).strip() if record_name is not None else "ATOM"
        if rec not in ("ATOM", "HETATM", ""):
            continue
        if str(res_names[i]).strip() in {"HOH", "WAT", "DOD"}:
            continue
        alt = str(alt_loc_raw[i]).strip() if alt_loc_raw is not None else ""
        # Keep only the primary alt-loc: blank, "A", or "1".
        if alt not in ("", "A", "1"):
            continue
        resn = str(res_names[i]).strip()
        if canonical_ordering.restype_index(resn) < 0:
            skipped_residue_names[resn] = skipped_residue_names.get(resn, 0) + 1
            continue

        chain_key = str(chain_ids_raw[i])
        insert_key = str(insert_raw[i]) if insert_raw is not None else ""
        res_key = (chain_key, int(res_nums_raw[i]), insert_key)

        if chain_key not in unique_chains:
            unique_chains[chain_key] = len(unique_chains)
        if res_key not in unique_residues:
            unique_residues[res_key] = len(unique_residues)
        res_ind_per_atom[i] = unique_residues[res_key]
        keep_atom[i] = True

    if skipped_residue_names:
        raise NotImplementedError(
            "JAX protein loading does not yet support residues: "
            + ", ".join(sorted(skipped_residue_names))
        )

    n_res = len(unique_residues)
    if n_res == 0:
        raise ValueError("PDB contains no supported protein residues")
    max_n_atoms = max(1, canonical_ordering.max_n_canonical_atoms)

    # -------- Allocate canonical-form arrays ------------------------ #
    coords = np.full((1, n_res, max_n_atoms, 3), np.nan, dtype=np.float32)
    atom_present = np.zeros((1, n_res, max_n_atoms), dtype=bool)
    chain_id = np.full((1, n_res), -1, dtype=np.int32)
    res_types = np.full((1, n_res), -1, dtype=np.int32)
    res_num_arr = np.zeros((1, n_res), dtype=np.int32)
    res_name_arr = np.empty((1, n_res), dtype=object)
    chain_labels = np.empty((1, n_res), dtype=object)
    res_ins_codes = np.empty((1, n_res), dtype=object)
    occupancy_arr = np.ones((1, n_res, max_n_atoms), dtype=np.float32)
    b_factor_arr = np.zeros((1, n_res, max_n_atoms), dtype=np.float32)

    # Initialise object arrays so NumPy doesn't leave them as None.
    res_name_arr.fill("")
    chain_labels.fill("")
    res_ins_codes.fill("")

    # Fill in per-residue metadata first, using the first atom seen for
    # each residue as the source of truth.
    residue_initialised = np.zeros(n_res, dtype=bool)
    for i in range(n_records):
        if not keep_atom[i]:
            continue
        r = int(res_ind_per_atom[i])
        if residue_initialised[r]:
            continue
        resn = str(res_names[i]).strip()
        chain_key = str(chain_ids_raw[i])
        res_types[0, r] = canonical_ordering.restype_index(resn)
        chain_id[0, r] = unique_chains[chain_key]
        res_num_arr[0, r] = int(res_nums_raw[i])
        res_name_arr[0, r] = resn
        chain_labels[0, r] = chain_key
        res_ins_codes[0, r] = str(insert_raw[i]) if insert_raw is not None else ""
        residue_initialised[r] = True

    # Now splat atoms into their canonical slots.
    for i in range(n_records):
        if not keep_atom[i]:
            continue
        r = int(res_ind_per_atom[i])
        resn = str(res_names[i]).strip()
        atom_name = str(atom_names[i]).strip()
        aind = canonical_ordering.atom_index(resn, atom_name)
        if aind < 0:
            # Unknown atom for this residue type -- silently skip.
            # (Matches upstream behaviour.)
            continue
        coords[0, r, aind, 0] = coords_raw[i, 0]
        coords[0, r, aind, 1] = coords_raw[i, 1]
        coords[0, r, aind, 2] = coords_raw[i, 2]
        atom_present[0, r, aind] = True
        occupancy_arr[0, r, aind] = float(occupancy_raw[i])
        b_factor_arr[0, r, aind] = float(b_factor_raw[i])

    return {
        "coords": coords,
        "atom_is_present": atom_present,
        "chain_id": chain_id,
        "res_types": res_types,
        "res_num": res_num_arr,
        "res_name": res_name_arr,
        "chain_labels": chain_labels,
        "res_ins_codes": res_ins_codes,
        "atom_occupancy": occupancy_arr,
        "atom_b_factor": b_factor_arr,
    }


__all__ = [
    "CanonicalOrdering",
    "default_canonical_ordering",
    "default_packed_block_types",
    "canonical_form_from_pdb",
]
