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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tmol.ligand.equivalence import EquivalenceResult


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


@dataclass(frozen=True)
class ChargeComparison:
    """Outcome of comparing two ``{atom_name: charge}`` maps.

    Attributes:
        ok: Whether the maps agree under the requested policy.
        mismatches: ``(name, generated, reference, delta)`` for shared atoms
            whose absolute difference reaches the tolerance.
        missing_in_generated: Reference atom names absent from the generated map.
        extra_in_generated: Generated atom names absent from the reference map.
    """

    ok: bool
    mismatches: list[tuple[str, float, float, float]]
    missing_in_generated: list[str]
    extra_in_generated: list[str]


def compare_charges(
    generated: dict[str, float],
    reference: dict[str, float],
    *,
    tolerance: float,
    require_same_keys: bool = True,
) -> ChargeComparison:
    """Compare two ``{atom_name: charge}`` maps within a tolerance.

    By default this is a strict by-name comparison: every reference atom must
    be present in the generated map and vice versa, and every shared atom must
    agree within ``tolerance``. Pass ``require_same_keys=False`` to accept a
    subset comparison (shared atoms only) when that is genuinely intended.

    Args:
        generated: Charges produced by the pipeline, keyed by atom name.
        reference: Reference charges, keyed by atom name.
        tolerance: Maximum permitted absolute charge difference.
        require_same_keys: When ``True`` (default), missing or extra atoms make
            the result non-equivalent.

    Returns:
        A :class:`ChargeComparison`.
    """
    gen_keys = set(generated)
    ref_keys = set(reference)
    missing = sorted(ref_keys - gen_keys)
    extra = sorted(gen_keys - ref_keys)
    shared = sorted(gen_keys & ref_keys)
    mismatches = [
        (name, generated[name], reference[name], generated[name] - reference[name])
        for name in shared
        if abs(generated[name] - reference[name]) >= tolerance
    ]
    ok = len(shared) > 0 and len(mismatches) == 0
    if require_same_keys:
        ok = ok and not missing and not extra
    return ChargeComparison(
        ok=ok,
        mismatches=mismatches,
        missing_in_generated=missing,
        extra_in_generated=extra,
    )


def reference_bond_keys(
    ref: ReferenceParams,
) -> set[tuple[frozenset[str], str, bool]]:
    """Return normalized all-atom bond keys for a reference.

    Each key is ``(frozenset{a, b}, normalized_order, ring)`` where
    ``normalized_order`` maps the Rosetta order token (``1/2/3/4``) onto the
    tmol bond-type label (``SINGLE/DOUBLE/TRIPLE/AROMATIC``) via the shared
    table in :mod:`tmol.ligand.params_io`, so a reference and a generated
    preparation can be compared on the same vocabulary.
    """
    from tmol.ligand.params_io import _BOND_TOK_TO_TYPE

    keys: set[tuple[frozenset[str], str, bool]] = set()
    for pair, order, ring in ref.bond_types:
        label = _BOND_TOK_TO_TYPE.get(str(order).upper(), "SINGLE")
        keys.add((pair, label, ring == "RING"))
    return keys


@dataclass(frozen=True)
class GeneratedFields:
    """Per-field view of a prepared ligand for exact-by-name comparison.

    Attributes:
        atom_types: ``{atom_name: atom_type}`` over all atoms (hydrogens too).
        bond_keys: ``(frozenset{a, b}, order_label, ring)`` over all bonds.
        icoor_topology: ``atom_name -> (parent, grandparent, great_grandparent)``.
        nbr_atom: The neighbour (default jump connection) atom name.
        charges: ``{atom_name: charge}`` over all atoms.
    """

    atom_types: dict[str, str]
    bond_keys: frozenset[tuple[frozenset[str], str, bool]]
    icoor_topology: dict[str, tuple[str, str, str]]
    nbr_atom: str
    charges: dict[str, float]


def generated_fields_from_preparation(prep: object) -> GeneratedFields:
    """Extract :class:`GeneratedFields` from a ``LigandPreparation``.

    Reads atom types, all-atom bonds (with order/ring labels), ICOOR topology,
    the neighbour atom, and partial charges from ``prep.residue_type`` and
    ``prep.partial_charges``.
    """
    rt = prep.residue_type  # type: ignore[attr-defined]
    atom_types = {str(a.name): a.atom_type for a in rt.atoms}
    bond_keys = frozenset(
        (frozenset([str(a), str(b)]), str(order), bool(ring))
        for a, b, order, ring in rt.bonds
    )
    icoor_topology = {
        ic.name: (ic.parent, ic.grand_parent, ic.great_grand_parent) for ic in rt.icoors
    }
    return GeneratedFields(
        atom_types=atom_types,
        bond_keys=bond_keys,
        icoor_topology=icoor_topology,
        nbr_atom=rt.default_jump_connection_atom,
        charges=dict(prep.partial_charges),  # type: ignore[attr-defined]
    )


@dataclass(frozen=True)
class StrictComparison:
    """Outcome of an exact-by-name comparison against a ``.params`` reference."""

    ok: bool
    checks: dict[str, bool]
    details: dict[str, object]


def compare_params_strict(
    generated: GeneratedFields,
    reference: ReferenceParams,
    *,
    charge_tolerance: float = 0.01,
) -> StrictComparison:
    """Compare a prepared ligand to a ``.params`` reference exactly, by name.

    Intended for the mol2 path, where atom names, all-atom bonds (including
    hydrogens), ICOOR topology, and the neighbour atom are preserved and must
    match exactly; charges must agree within ``charge_tolerance``. Excluded
    fields (raw coordinates, numeric ``NBR_RADIUS``, cartbonded params) are not
    consulted here.

    Args:
        generated: Fields extracted from the prepared ligand.
        reference: Parsed reference ``.params``.
        charge_tolerance: Maximum permitted absolute charge difference.

    Returns:
        A :class:`StrictComparison`.
    """
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    ref_types = reference.atom_types
    type_mismatches = [
        (n, generated.atom_types.get(n), ref_types.get(n))
        for n in sorted(set(ref_types) | set(generated.atom_types))
        if generated.atom_types.get(n) != ref_types.get(n)
    ]
    checks["atom_types"] = not type_mismatches
    if type_mismatches:
        details["atom_types"] = type_mismatches

    ref_bonds = reference_bond_keys(reference)
    checks["bonds"] = generated.bond_keys == ref_bonds
    if not checks["bonds"]:
        details["bonds"] = {
            "only_in_generated": sorted(
                map(_bond_repr, generated.bond_keys - ref_bonds)
            ),
            "only_in_reference": sorted(
                map(_bond_repr, ref_bonds - generated.bond_keys)
            ),
        }

    checks["icoor_topology"] = generated.icoor_topology == reference.icoor_topology
    if not checks["icoor_topology"]:
        ref_ic = reference.icoor_topology
        gen_ic = generated.icoor_topology
        details["icoor_topology"] = [
            (n, gen_ic.get(n), ref_ic.get(n))
            for n in sorted(set(ref_ic) | set(gen_ic))
            if gen_ic.get(n) != ref_ic.get(n)
        ]

    checks["nbr_atom"] = generated.nbr_atom == reference.nbr_atom
    if not checks["nbr_atom"]:
        details["nbr_atom"] = (generated.nbr_atom, reference.nbr_atom)

    charge_cmp = compare_charges(
        generated.charges, reference.charges, tolerance=charge_tolerance
    )
    checks["charges"] = charge_cmp.ok
    if not charge_cmp.ok:
        details["charges"] = {
            "mismatches": charge_cmp.mismatches,
            "missing_in_generated": charge_cmp.missing_in_generated,
            "extra_in_generated": charge_cmp.extra_in_generated,
        }

    return StrictComparison(ok=all(checks.values()), checks=checks, details=details)


def _bond_repr(key: tuple[frozenset[str], str, bool]) -> tuple[str, str, str, bool]:
    """Flatten a bond key into a sortable ``(a, b, order, ring)`` tuple."""
    pair, order, ring = key
    a, b = sorted(pair)
    return (a, b, order, ring)


def compare_semantic(
    generated: object, reference: object, **kwargs: object
) -> "EquivalenceResult":
    """Heavy-atom isomorphism comparison of two ``LigandPreparation`` objects.

    Thin wrapper over :func:`tmol.ligand.equivalence.compare_ligand_preparations`
    so the parity harness has a single import surface for all comparison modes;
    no new isomorphism logic is introduced here.
    """
    from tmol.ligand.equivalence import compare_ligand_preparations

    return compare_ligand_preparations(generated, reference, **kwargs)  # type: ignore[arg-type]
