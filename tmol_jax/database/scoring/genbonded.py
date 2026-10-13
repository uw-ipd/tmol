"""Generic-bonded torsion parameter database.

Port of upstream :mod:`tmol.database.scoring.genbonded`. Same attrs
classes and YAML schema as upstream; uses plain ``yaml.safe_load``
(no cattrs) — the upstream module already did this so the port is
essentially verbatim.

Energy form (proper torsion, per `potentials.hh`):

    E = sum_{n=1..4} k_n * (1 + cos(n * theta))
      + sum_{n: k_n < 0} (-2 * k_n)       # lift to non-negative
      + offset

Improper torsion: ``E = k * (theta - delta)^2``.

Parameter lookup is keyed by atom *chemical type* (e.g. ``CS``, ``C*``,
``X``) plus a bond-type bin for the central bond; see
:meth:`GenBondedDatabase.find_torsion_params` for the multiplicity scoring
rule (bond specificity dominates atom-type specificity).
"""

from __future__ import annotations

from itertools import permutations
from typing import Dict, FrozenSet, List, Optional, Tuple

import attr
import yaml

# Which Rosetta bond bins each database bond-char covers (for btidx 1-8).
# btidx = ((BondName - 1) << 1) + (2 if in_ring else 1)
# BondName: 1=single, 2=double, 3=triple, 4=aromatic.
_BOND_CHAR_COVERED_BINS: Dict[str, FrozenSet[int]] = {
    "~": frozenset({1, 2, 3, 4, 5, 6, 7, 8}),  # any bond
    "@": frozenset({2, 4, 6, 8}),  # any bond, in ring
    "-": frozenset({1, 2}),  # single, ring or not
    "=": frozenset({3, 4}),  # double
    "#": frozenset({5, 6}),  # triple
    ":": frozenset({7, 8}),  # aromatic
}

_BOND_CHAR_BIN_COUNT: Dict[str, int] = {
    c: len(bins) for c, bins in _BOND_CHAR_COVERED_BINS.items()
}


def _bond_btidx(bond_type_int: int, is_ring: bool) -> int:
    br = 2 if is_ring else 1
    return ((bond_type_int - 1) << 1) + br


@attr.s(auto_attribs=True, slots=True, frozen=True)
class GenBondedTorsionEntry:
    """One proper-torsion entry; 4 atom types + bond char + 4 k's + offset."""

    atoms: Tuple[str, str, str, str]
    bond: str
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    offset: float = 0.0


@attr.s(auto_attribs=True, slots=True, frozen=True)
class GenBondedImproperEntry:
    """One improper entry; atoms[0]=center, atoms[1..3]=bonded (unordered)."""

    atoms: Tuple[str, str, str, str]
    k: float = 0.0
    delta: float = 0.0


@attr.s(auto_attribs=True, frozen=True, slots=True)
class GenBondedDatabase:
    """Genbonded torsion + improper parameter database."""

    atom_hierarchy: Dict[str, List[str]]
    torsions: Tuple[GenBondedTorsionEntry, ...]
    impropers: Tuple[GenBondedImproperEntry, ...]
    coverage: Dict[str, int]
    multi_max: int

    # Lazy index built post-init; not part of equality / repr.
    _torsion_index: Dict[
        Tuple[str, str, str, str],
        List[Tuple[float, FrozenSet[int], GenBondedTorsionEntry]],
    ] = attr.ib(init=False, factory=dict, eq=False, repr=False)

    def __attrs_post_init__(self) -> None:
        mm4 = self.multi_max**4
        index = self._torsion_index
        for entry in self.torsions:
            covered = _BOND_CHAR_COVERED_BINS.get(entry.bond)
            if covered is None:
                continue
            bin_count = _BOND_CHAR_BIN_COUNT.get(entry.bond, 1)
            atom_cov = 1
            for at in entry.atoms:
                atom_cov *= self.coverage.get(at, 1)
            mult = bin_count * mm4 + atom_cov
            record = (mult, covered, entry)
            for key in {entry.atoms, entry.atoms[::-1]}:
                index.setdefault(key, []).append(record)
        for bucket in index.values():
            bucket.sort(key=lambda r: r[0])

    # ------------------------------------------------------------------
    @classmethod
    def from_file(cls, path: str) -> "GenBondedDatabase":
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh)

        atom_hierarchy: Dict[str, List[str]] = {}
        for atom_type, fallbacks in raw.get("atoms", {}).items():
            if fallbacks:
                atom_hierarchy[atom_type] = list(fallbacks)

        coverage: Dict[str, int] = {}
        for hier in atom_hierarchy.values():
            for h_type in hier:
                coverage[h_type] = coverage.get(h_type, 0) + 1

        multi_max = coverage.get("X", 1) // 2

        torsion_list: List[GenBondedTorsionEntry] = []
        for entry in raw.get("torsions", []):
            k_list = entry.get("K", [0.0, 0.0, 0.0, 0.0])
            torsion_list.append(
                GenBondedTorsionEntry(
                    atoms=tuple(entry["atoms"]),
                    bond=entry.get("bond"),
                    k1=float(k_list[0]),
                    k2=float(k_list[1]),
                    k3=float(k_list[2]),
                    k4=float(k_list[3]),
                    offset=float(entry.get("offset", 0.0)),
                )
            )

        impropers_list: List[GenBondedImproperEntry] = []
        for entry in raw.get("impropers", []):
            atoms_raw = (entry["center"], *entry["bonded"])
            impropers_list.append(
                GenBondedImproperEntry(
                    atoms=tuple(atoms_raw),
                    k=float(entry.get("k", 0.0)),
                    delta=float(entry.get("delta", 0.0)),
                )
            )

        return cls(
            atom_hierarchy=atom_hierarchy,
            torsions=tuple(torsion_list),
            impropers=tuple(impropers_list),
            coverage=coverage,
            multi_max=multi_max,
        )

    # ------------------------------------------------------------------
    def hierarchy_for(self, atom_type: str) -> List[str]:
        return self.atom_hierarchy.get(atom_type, [atom_type])

    def find_torsion_params(
        self,
        type1: str,
        type2: str,
        type3: str,
        type4: str,
        bond_type_int: int = 1,
        is_ring: bool = False,
    ) -> Optional[GenBondedTorsionEntry]:
        """Best-matching proper-torsion entry via Rosetta multiplicity scoring.

        Bond specificity dominates atom-type specificity (bond_bin_count *
        multi_max**4 + product of atom-type coverage). Both forward and
        reversed orderings are pre-inserted in the index so a single
        iteration direction suffices.
        """
        btidx = _bond_btidx(bond_type_int, is_ring)
        index = self._torsion_index
        h1 = self.hierarchy_for(type1)
        h2 = self.hierarchy_for(type2)
        h3 = self.hierarchy_for(type3)
        h4 = self.hierarchy_for(type4)

        best_entry: Optional[GenBondedTorsionEntry] = None
        best_mult = float("inf")
        for e1 in h1:
            for e2 in h2:
                for e3 in h3:
                    for e4 in h4:
                        bucket = index.get((e1, e2, e3, e4))
                        if bucket is None:
                            continue
                        for mult, covered, entry in bucket:
                            if mult >= best_mult:
                                break
                            if btidx in covered:
                                best_mult = mult
                                best_entry = entry
                                break
        return best_entry

    def find_improper_params(
        self, center: str, n1: str, n2: str, n3: str
    ) -> Optional[GenBondedImproperEntry]:
        """Best-matching improper entry; tries all 6 permutations of bonded."""
        hc = self.hierarchy_for(center)
        h1 = self.hierarchy_for(n1)
        h2 = self.hierarchy_for(n2)
        h3 = self.hierarchy_for(n3)

        best_entry: Optional[GenBondedImproperEntry] = None
        best_score = float("inf")
        for entry in self.impropers:
            ec, eb1, eb2, eb3 = entry.atoms
            try:
                sc = hc.index(ec)
            except ValueError:
                continue
            for pe1, pe2, pe3 in permutations((eb1, eb2, eb3)):
                try:
                    s1 = h1.index(pe1)
                    s2 = h2.index(pe2)
                    s3 = h3.index(pe3)
                except ValueError:
                    continue
                score = sc + s1 + s2 + s3
                if score < best_score:
                    best_score = score
                    best_entry = entry
        return best_entry


__all__ = [
    "GenBondedDatabase",
    "GenBondedTorsionEntry",
    "GenBondedImproperEntry",
]
