"""Shared parsing of Rosetta ``.params`` reference files.

The regression suite and the parity harness both need to read a Rosetta
``.params`` file into structured fields and, in particular, recover the
per-atom partial charges. ``read_params_file`` in :mod:`tmol.ligand.params_io`
builds a ``RawResidueType`` but drops the charge column, so this module
provides a light-weight, charge-bearing parser plus a ``{atom_name: charge}``
sidecar accessor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ReferenceParams:
    """Structured view of a Rosetta ``.params`` file.

    Attributes:
        name: The ``NAME`` record value (empty string when absent).
        atoms: ``(atom_name, atom_type, charge)`` tuples in file order.
        bond_types: ``(frozenset{a, b}, order, ring_flag)`` keys.
        cut_bonds: ``frozenset{a, b}`` keys for ``CUT_BOND`` records.
        chis: ``(chi_number, (a, b, c, d), biaryl_flag)`` tuples.
        proton_chis: Raw ``PROTON_CHI`` line strings.
        nbr_atom: The ``NBR_ATOM`` value (empty string when absent).
        icoor_topology: ``atom_name -> (parent, grandparent, great_grandparent)``.
    """

    name: str
    atoms: tuple[tuple[str, str, float], ...]
    bond_types: frozenset[tuple[frozenset[str], str, str]]
    cut_bonds: frozenset[frozenset[str]]
    chis: tuple[tuple[int, tuple[str, str, str, str], bool], ...]
    proton_chis: tuple[str, ...]
    nbr_atom: str
    icoor_topology: dict[str, tuple[str, str, str]] = field(default_factory=dict)

    @property
    def charges(self) -> dict[str, float]:
        """Return the ``{atom_name: charge}`` sidecar map.

        ``read_params_file`` discards charges, so this is the canonical way to
        recover the per-atom charge column from a ``.params`` reference.
        """
        return {name: charge for name, _atype, charge in self.atoms}

    @property
    def atom_types(self) -> dict[str, str]:
        """Return the ``{atom_name: atom_type}`` map."""
        return {name: atype for name, atype, _charge in self.atoms}

    @property
    def has_hydrogen(self) -> bool:
        """Return whether any atom is a hydrogen."""
        return any(_is_hydrogen(name) for name, _t, _q in self.atoms)

    def heavy_atom_names(self) -> set[str]:
        """Return the set of non-hydrogen atom names."""
        return {name for name, _t, _q in self.atoms if not _is_hydrogen(name)}

    def all_bond_pairs(self) -> set[frozenset[str]]:
        """Return every bonded atom-name pair (hydrogen-inclusive)."""
        return {pair for pair, _order, _ring in self.bond_types}


def _is_hydrogen(name: str) -> bool:
    """Return whether an atom name denotes a hydrogen."""
    return str(name).startswith("H")


def parse_reference_params(path: str | Path) -> ReferenceParams:
    """Parse a Rosetta ``.params`` file into a :class:`ReferenceParams`.

    Args:
        path: Path to the ``.params`` file.

    Returns:
        A frozen :class:`ReferenceParams` with atoms (including charges),
        bond types, cut bonds, CHI/PROTON_CHI records, neighbour atom, and
        ICOOR topology.
    """
    name = ""
    atoms: list[tuple[str, str, float]] = []
    bond_types: set[tuple[frozenset[str], str, str]] = set()
    cut_bonds: set[frozenset[str]] = set()
    chis: list[tuple[int, tuple[str, str, str, str], bool]] = []
    proton_chis: list[str] = []
    nbr_atom = ""
    icoor_topo: dict[str, tuple[str, str, str]] = {}

    with open(path) as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue

            record = parts[0]
            if record == "NAME" and len(parts) >= 2:
                name = parts[1]
            elif record == "ATOM" and len(parts) >= 5:
                atoms.append((parts[1], parts[2], float(parts[4])))
            elif record == "BOND_TYPE" and len(parts) >= 4:
                a1, a2 = parts[1].strip(), parts[2].strip()
                ring = "RING" if len(parts) >= 5 and parts[4] == "RING" else ""
                bond_types.add((frozenset([a1, a2]), parts[3], ring))
            elif record == "CUT_BOND" and len(parts) >= 3:
                cut_bonds.add(frozenset([parts[1].strip(), parts[2].strip()]))
            elif record == "CHI" and len(parts) >= 6:
                quad = (parts[2], parts[3], parts[4], parts[5])
                chis.append((int(parts[1]), quad, "#biaryl" in line))
            elif record == "PROTON_CHI":
                proton_chis.append(line.strip())
            elif record == "NBR_ATOM" and len(parts) >= 2:
                nbr_atom = parts[1]
            elif record == "ICOOR_INTERNAL" and len(parts) >= 8:
                icoor_topo[parts[1]] = (parts[5], parts[6], parts[7])

    return ReferenceParams(
        name=name,
        atoms=tuple(atoms),
        bond_types=frozenset(bond_types),
        cut_bonds=frozenset(cut_bonds),
        chis=tuple(chis),
        proton_chis=tuple(proton_chis),
        nbr_atom=nbr_atom,
        icoor_topology=icoor_topo,
    )


def as_legacy_dict(ref: ReferenceParams) -> dict:
    """Return the historical dict shape used by the existing regression suite.

    The legacy ``_parse_reference_params`` test helper returns a dict; this
    adapter lets that helper delegate here without changing its callers.
    """
    return {
        "atoms": list(ref.atoms),
        "bond_types": set(ref.bond_types),
        "cut_bonds": set(ref.cut_bonds),
        "chis": list(ref.chis),
        "proton_chis": list(ref.proton_chis),
        "nbr_atom": ref.nbr_atom,
        "icoor_topology": dict(ref.icoor_topology),
    }


def reference_charges(
    path_or_ref: "str | Path | ReferenceParams",
) -> dict[str, float]:
    """Return the ``{atom_name: charge}`` sidecar for a ``.params`` reference.

    Accepts either a path (parsed on the fly) or an already-parsed
    :class:`ReferenceParams`.
    """
    if isinstance(path_or_ref, ReferenceParams):
        return path_or_ref.charges
    return parse_reference_params(path_or_ref).charges


def compare_charges(
    generated: dict[str, float],
    reference: dict[str, float],
    *,
    tolerance: float,
) -> tuple[bool, list[tuple[str, float, float, float]]]:
    """Compare two ``{atom_name: charge}`` maps within a tolerance.

    Only atoms shared by both maps are compared. Returns ``(ok, mismatches)``
    where each mismatch is ``(name, generated, reference, delta)`` and ``ok``
    requires at least one shared atom and no out-of-tolerance delta.

    Args:
        generated: Charges produced by the pipeline, keyed by atom name.
        reference: Reference charges, keyed by atom name.
        tolerance: Maximum permitted absolute charge difference.

    Returns:
        ``(ok, mismatches)``.
    """
    shared = sorted(generated.keys() & reference.keys())
    mismatches = [
        (name, generated[name], reference[name], generated[name] - reference[name])
        for name in shared
        if abs(generated[name] - reference[name]) >= tolerance
    ]
    ok = len(shared) > 0 and len(mismatches) == 0
    return ok, mismatches
