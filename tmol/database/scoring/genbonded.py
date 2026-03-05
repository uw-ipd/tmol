import attr
import yaml

from itertools import permutations
from typing import Dict, List, Optional, Tuple


@attr.s(auto_attribs=True, slots=True, frozen=True)
class GenBondedTorsionEntry:
    """One torsion entry from the genbonded parameter database.

    atoms  -- four atom-type strings (may be generic, e.g. "C*", or wildcard "X")
    bond   -- bond character:
                ~ any bond
                @ ring bond
                − single bond
                = double bond
                # triple bond
                ∶ aromatic bond
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
    """

    atom_hierarchy: Dict[str, List[str]]
    torsions: Tuple[GenBondedTorsionEntry, ...]
    impropers: Tuple[GenBondedImproperEntry, ...]

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
        )

    # ------------------------------------------------------------------
    # Type-index helpers
    # ------------------------------------------------------------------

    def all_type_names(self) -> List[str]:
        """Sorted list of all unique chemical type strings in the database.

        Includes both concrete types (hierarchy keys), generic types
        (hierarchy values), and any type strings appearing in torsion or
        improper entries.
        """
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
        self, type1: str, type2: str, type3: str, type4: str, bond_char: str = "~"
    ) -> Optional[GenBondedTorsionEntry]:
        """Return the best-matching torsion entry for the given atom types.

        'Best' is defined as the lowest total hierarchy-position score (i.e.
        most specific match wins).  Returns None if no match is found.

        bond_char: the bond character for the central bond (e.g. '-' for single,
                   '=' for double, '~' for wildcard).  Entries whose bond field
                   is neither '~' nor bond_char are skipped.
        """
        h1 = self.hierarchy_for(type1)
        h2 = self.hierarchy_for(type2)
        h3 = self.hierarchy_for(type3)
        h4 = self.hierarchy_for(type4)

        best_entry = None
        best_score = float("inf")

        for entry in self.torsions:
            if entry.bond != "~" and entry.bond != bond_char:
                continue  # bond type mismatch — skip entirely
            et1, et2, et3, et4 = entry.atoms
            # Each position matches if the entry type appears anywhere in the
            # concrete atom's hierarchy list.
            try:
                s1 = h1.index(et1)
            except ValueError:
                continue
            try:
                s2 = h2.index(et2)
            except ValueError:
                continue
            try:
                s3 = h3.index(et3)
            except ValueError:
                continue
            try:
                s4 = h4.index(et4)
            except ValueError:
                continue

            score = s1 + s2 + s3 + s4
            if score < best_score:
                best_score = score
                best_entry = entry

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
            # Try all permutations of the 3 bonded entry types against the
            # 3 bonded atom hierarchies.
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
