import attr
import yaml

from itertools import permutations
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Bond-type bin constants (mirror Rosetta's bin_from_bond)
#
#   btidx = ((BondName - 1) << 1) + BondRingness
#     BondName:    SingleBond=1, DoubleBond=2, TripleBond=3, AromaticBond=4
#     BondRingness: NotInRing=1, InRing=2
#
#   Results:  Single+NotInRing=1, Single+InRing=2,
#             Double+NotInRing=3, Double+InRing=4,
#             Triple+NotInRing=5, Triple+InRing=6,
#             Aromatic+NotInRing=7, Aromatic+InRing=8
#   Wildcard (~) bin: 0 (not a valid query bin, but used for bin_count)
# ---------------------------------------------------------------------------

# Which Rosetta bond bins each database bond-char covers (for btidx 1-8).
# An entry matches a query bond if query btidx is in the entry's covered set.
_BOND_CHAR_COVERED_BINS: Dict[str, frozenset] = {
    "~": frozenset({1, 2, 3, 4, 5, 6, 7, 8}),  # any of 8 bond types
    "@": frozenset({2, 4, 6, 8}),  # any bond, in ring
    "-": frozenset({1, 2}),  # single,   ring or not
    "=": frozenset({3, 4}),  # double,   ring or not
    "#": frozenset({5, 6}),  # triple,   ring or not
    ":": frozenset({7, 8}),  # aromatic, ring or not
}

# Number of bins each bond-char covers.  Mirrors Rosetta's indicesBT.size():
# more bins = less specific = higher multiplicity.
_BOND_CHAR_BIN_COUNT: Dict[str, int] = {
    c: len(bins) for c, bins in _BOND_CHAR_COVERED_BINS.items()
}


def _bond_btidx(bond_type_int: int, is_ring: bool) -> int:
    """Compute Rosetta bond-bin index from a concrete bond.

    BondType values (1=SINGLE, 2=DOUBLE, 3=TRIPLE, 4=AROMATIC) are used
    directly as the Rosetta BondName index.
    """
    br = 2 if is_ring else 1
    return ((bond_type_int - 1) << 1) + br


@attr.s(auto_attribs=True, slots=True, frozen=True)
class GenBondedTorsionEntry:
    """One torsion entry from the genbonded parameter database.

    atoms  -- four atom-type strings (may be generic, e.g. "C*", or wildcard "X")
    bond   -- bond character:
                ~ any bond
                @ ring bond
                - single bond
                = double bond
                # triple bond
                : aromatic bond
    k1..k4 -- Fourier coefficients for periods 1-4
    offset -- single phase offset applied to all Fourier terms
    """

    atoms: Tuple[str, str, str, str]
    bond: str
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    k4: float = 0.0
    offset: float = 0.0


@attr.s(auto_attribs=True, slots=True, frozen=True)
class GenBondedImproperEntry:
    """One improper torsion entry from the genbonded parameter database.

    atoms  -- four atom-type strings: atoms[0] = center, atoms[1..3] = bonded
           -- (may be generic, e.g. "C*", or wildcard "X")
    k      -- harmonic spring constant  (E = k*(theta - delta)^2)
    delta  -- ideal improper torsion angle (radians)
    """

    atoms: Tuple[str, str, str, str]
    k: float = 0.0
    delta: float = 0.0


@attr.s(auto_attribs=True, frozen=True, slots=True)
class GenBondedDatabase:
    """Database for the genbonded (generic-bonded torsional) scoring term.

    atom_hierarchy -- maps each concrete atom-type name to an ordered list of
                      types to try when looking up a parameter entry, from most
                      specific to most generic.  e.g. {"CS": ["CS", "C*", "X"]}

    torsions       -- ordered list of torsion entries (most-specific first so
                      that a linear scan finds the best match quickly).

    impropers      -- ordered list of improper torsion entries (center + 3
                      bonded atoms; order of bonded atoms is unordered).

    coverage       -- maps each atom-type string to the count of concrete types
                      whose hierarchy includes that string.  Used for Rosetta-
                      style multiplicity scoring.

    multi_max      -- half the total number of concrete atom types; scales the
                      bond-type contribution to multiplicity so that bond
                      specificity dominates atom-type specificity (mirrors
                      Rosetta's multi_max = indicesX.size() / 2).
    """

    atom_hierarchy: Dict[str, List[str]]
    torsions: Tuple[GenBondedTorsionEntry, ...]
    impropers: Tuple[GenBondedImproperEntry, ...]
    coverage: Dict[str, int]
    multi_max: int

    # Lookup index built lazily in __attrs_post_init__: maps an atom-string
    # 4-tuple to a list of (mult, covered_bins, entry) sorted by ascending
    # multiplicity.  Both forward and reversed orderings of each entry's atom
    # tuple are inserted so torsion direction is handled by a single dict
    # lookup rather than a second pass.
    _torsion_index: Dict[
        Tuple[str, str, str, str],
        List[Tuple[float, frozenset, GenBondedTorsionEntry]],
    ] = attr.ib(init=False, factory=dict, eq=False, repr=False)

    def __attrs_post_init__(self) -> None:
        mm4 = self.multi_max**4
        index = self._torsion_index  # mutate in place (frozen-safe)
        for entry in self.torsions:
            covered = _BOND_CHAR_COVERED_BINS.get(entry.bond)
            if covered is None:
                continue
            bin_count = _BOND_CHAR_BIN_COUNT.get(entry.bond, 1)
            et1, et2, et3, et4 = entry.atoms
            atom_cov = (
                self.coverage.get(et1, 1)
                * self.coverage.get(et2, 1)
                * self.coverage.get(et3, 1)
                * self.coverage.get(et4, 1)
            )
            mult = bin_count * mm4 + atom_cov
            record = (mult, covered, entry)
            # Insert under both orderings; a palindromic tuple goes in once.
            keys = {entry.atoms, entry.atoms[::-1]}
            for key in keys:
                index.setdefault(key, []).append(record)
        for bucket in index.values():
            bucket.sort(key=lambda r: r[0])

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str) -> "GenBondedDatabase":
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh)

        # --- atom hierarchy -------------------------------------------
        atom_hierarchy: Dict[str, List[str]] = {}
        for atom_type, fallbacks in raw.get("atoms", {}).items():
            # BOND / ANGLE sentinel entries have empty lists; skip them
            if fallbacks:
                atom_hierarchy[atom_type] = list(fallbacks)

        # --- coverage map (Rosetta: indices_i.size() per atom type) ---
        # For each group/wildcard type string, count how many concrete types
        # have it in their hierarchy.  Mirrors name_index_map[atm].size().
        coverage: Dict[str, int] = {}
        for hier in atom_hierarchy.values():
            for h_type in hier:
                coverage[h_type] = coverage.get(h_type, 0) + 1

        # multi_max = indicesX.size() / 2  (X covers all concrete types).
        # Use integer division to match Rosetta's int arithmetic and keep
        # the resulting multiplicities integer-valued.
        multi_max = coverage.get("X", 1) // 2

        # --- torsion entries ------------------------------------------
        torsion_list: List[GenBondedTorsionEntry] = []
        for entry in raw.get("torsions", []):
            atoms_raw = entry["atoms"]
            assert len(atoms_raw) == 4, f"Expected 4 atoms, got {atoms_raw}"

            k_list = entry.get("K")
            assert len(k_list) == 4, f"Expected 4 components, got {atoms_raw}"

            torsion_list.append(
                GenBondedTorsionEntry(
                    atoms=tuple(atoms_raw),
                    bond=entry.get("bond"),
                    k1=float(k_list[0]),
                    k2=float(k_list[1]),
                    k3=float(k_list[2]),
                    k4=float(k_list[3]),
                    offset=float(entry.get("offset", 0.0)),
                )
            )

        # --- improper entries -----------------------------------------
        impropers_list: List[GenBondedImproperEntry] = []
        for entry in raw.get("impropers", []):
            atoms_raw = [entry["center"], *entry["bonded"]]
            assert len(atoms_raw) == 4, f"Expected 4 atoms, got {atoms_raw}"

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
    # Type-index helpers
    # ------------------------------------------------------------------

    def all_type_names(self) -> List[str]:
        """Sorted list of all unique chemical type strings in the database."""
        all_types: set = set()
        for ct, hierarchy in self.atom_hierarchy.items():
            all_types.add(ct)
            for ht in hierarchy:
                all_types.add(ht)
        for entry in self.torsions:
            for at in entry.atoms:
                all_types.add(at)
        for entry in self.impropers:
            for at in entry.atoms:
                all_types.add(at)
        return sorted(all_types)

    def make_type_to_idx(self) -> Dict[str, int]:
        """Return a dict mapping every chemical type string to a unique int index."""
        return {ct: i for i, ct in enumerate(self.all_type_names())}

    # ------------------------------------------------------------------
    # Parameter lookup helpers
    # ------------------------------------------------------------------

    def hierarchy_for(self, atom_type: str) -> List[str]:
        """Return the fallback list for *atom_type*, defaulting to [atom_type]."""
        return self.atom_hierarchy.get(atom_type, [atom_type])

    def find_torsion_params(
        self,
        type1: str,
        type2: str,
        type3: str,
        type4: str,
        bond_type_int: int,
        is_ring: bool,
    ) -> Optional[GenBondedTorsionEntry]:
        """Return the best-matching torsion entry using Rosetta's multiplicity scoring.

        Multiplicity = bond_bin_count * multi_max^4 + coverage(a1)*...*coverage(a4)

        Bond specificity dominates: an entry with a more-specific bond char
        (fewer covered bins) beats one with a less-specific bond char regardless
        of atom-type generality, as long as both match.  Within the same bond
        specificity, atom-type coverage breaks ties (lower = more specific).

        This mirrors Rosetta's GenericBondedPotential multiplicity formula exactly:
          multBT   = indicesBT.size() * multi_max^4
          mult_atm = indices1.size() * indices2.size() * indices3.size() * indices4.size()
          multiplicity = multBT + mult_atm   (lower = more specific = preferred)
        """
        btidx = _bond_btidx(bond_type_int, is_ring)
        index = self._torsion_index

        h1 = self.hierarchy_for(type1)
        h2 = self.hierarchy_for(type2)
        h3 = self.hierarchy_for(type3)
        h4 = self.hierarchy_for(type4)

        best_entry = None
        best_mult = float("inf")

        # Enumerate the (small) Cartesian product of the four hierarchies and
        # probe the precomputed index.  Reversed-direction matches are already
        # represented in the index, so a single direction of iteration suffices.
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
        """Return the best-matching improper entry for (center, n1, n2, n3).

        The three bonded atoms (n1, n2, n3) are considered unordered: all six
        permutations are tried.  'Best' is the lowest total hierarchy-position
        score.  Returns None if no match is found.
        """
        hc = self.hierarchy_for(center)
        h1 = self.hierarchy_for(n1)
        h2 = self.hierarchy_for(n2)
        h3 = self.hierarchy_for(n3)

        best_entry = None
        best_score = float("inf")

        for entry in self.impropers:
            ec, eb1, eb2, eb3 = entry.atoms
            try:
                sc = hc.index(ec)
            except ValueError:
                continue

            for pe1, pe2, pe3 in permutations([eb1, eb2, eb3]):
                try:
                    s1 = h1.index(pe1)
                except ValueError:
                    continue
                try:
                    s2 = h2.index(pe2)
                except ValueError:
                    continue
                try:
                    s3 = h3.index(pe3)
                except ValueError:
                    continue

                score = sc + s1 + s2 + s3
                if score < best_score:
                    best_score = score
                    best_entry = entry

        return best_entry
