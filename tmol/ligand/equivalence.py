"""Shared ligand-preparation equivalence helpers.

These utilities are extracted from regression-test logic so large-scale
batch scripts can compare two ``LigandPreparation`` objects using the same
normalization rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


def _is_heavy(name: str) -> bool:
    """Return whether an atom name is non-hydrogen.

    Args:
        name: Atom name.

    Returns:
        ``True`` when the atom is not a hydrogen.
    """
    return not str(name).startswith("H")


def _cartres_heavy_key_set(params: Iterable[Any], kind: str) -> set[Any]:
    """Build normalized heavy-atom keys for a cartbonded parameter group.

    Args:
        params: Iterable of cartbonded parameter entries.
        kind: Parameter group kind (``length``, ``angle``, or ``improper``).

    Returns:
        Set of normalized keys suitable for equivalence comparison.
    """
    keys = set()
    if kind == "length":
        for p in params:
            a, b = str(p.atm1), str(p.atm2)
            if _is_heavy(a) and _is_heavy(b):
                keys.add(frozenset([a, b]))
    elif kind == "angle":
        for p in params:
            a1, c, a3 = str(p.atm1), str(p.atm2), str(p.atm3)
            if all(_is_heavy(n) for n in (a1, c, a3)):
                lo, hi = sorted([a1, a3])
                keys.add((lo, c, hi))
    elif kind == "improper":
        for p in params:
            names = [str(p.atm1), str(p.atm2), str(p.atm3), str(p.atm4)]
            if all(_is_heavy(n) for n in names):
                keys.add(tuple(sorted(names)))
    else:
        raise ValueError(f"Unknown cartbonded group kind: {kind}")
    return keys


@dataclass
class EquivalenceResult:
    is_equivalent: bool
    checks: dict[str, bool]
    details: dict[str, Any]


def compare_ligand_preparations(  # noqa: C901
    generated: Any,
    reference: Any,
    *,
    charge_tolerance: float = 0.05,
) -> EquivalenceResult:
    """Compare two ``LigandPreparation`` objects with DUD test semantics."""
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    # Atom set
    gen_atoms = {
        (str(a.name), a.atom_type)
        for a in generated.residue_type.atoms
        if _is_heavy(a.name)
    }
    ref_atoms = {
        (str(a.name), a.atom_type)
        for a in reference.residue_type.atoms
        if _is_heavy(a.name)
    }
    checks["atom_set"] = gen_atoms == ref_atoms
    if not checks["atom_set"]:
        details["atom_set"] = {
            "only_in_generated": sorted(gen_atoms - ref_atoms),
            "only_in_reference": sorted(ref_atoms - gen_atoms),
        }

    # Atom types by name
    gen_types = {
        str(a.name): a.atom_type
        for a in generated.residue_type.atoms
        if _is_heavy(a.name)
    }
    ref_types = {
        str(a.name): a.atom_type
        for a in reference.residue_type.atoms
        if _is_heavy(a.name)
    }
    mismatches = [
        (n, gen_types[n], ref_types[n])
        for n in sorted(gen_types.keys() & ref_types.keys())
        if gen_types[n] != ref_types[n]
    ]
    checks["atom_types"] = len(mismatches) == 0
    if not checks["atom_types"]:
        details["atom_types"] = mismatches

    # Bonds (delocalized normalization)
    aromatic_equiv = frozenset({"AROMATIC", "SINGLE", "DOUBLE"})

    def bond_keyset(
        bonds: Iterable[tuple[Any, Any, Any, Any]],
    ) -> set[tuple[frozenset[str], str, bool]]:
        """Normalize heavy-atom bond records into comparison keys.

        Args:
            bonds: Bond tuple records from a residue type.

        Returns:
            Set of normalized bond keys.
        """
        out = set()
        for a, b, bond_type, *rest in bonds:
            a, b = str(a), str(b)
            if not (_is_heavy(a) and _is_heavy(b)):
                continue
            ring = bool(rest[0]) if rest else False
            out.add((frozenset([a, b]), str(bond_type), ring))
        return out

    gen_bonds = bond_keyset(generated.residue_type.bonds)
    ref_bonds = bond_keyset(reference.residue_type.bonds)
    all_bonds = gen_bonds | ref_bonds

    aromatic_atoms: set[str] = set()
    for pair, btype, _ring in all_bonds:
        if btype == "AROMATIC":
            aromatic_atoms.update(pair)

    def is_delocalized(pair: frozenset[str], btype: str, ring: bool) -> bool:
        """Return whether a bond key should be treated as delocalized.

        Args:
            pair: Atom-name pair for the bond.
            btype: Bond-type label.
            ring: Whether the bond is ring-annotated.

        Returns:
            ``True`` when this bond should be normalized as delocalized.
        """
        if btype == "AROMATIC":
            return True
        if ring and btype in aromatic_equiv:
            return True
        if pair & aromatic_atoms and btype in aromatic_equiv:
            return True
        return False

    delocalized_pairs: set[frozenset] = set()
    for pair, btype, ring in all_bonds:
        if is_delocalized(pair, btype, ring):
            delocalized_pairs.add(pair)

    def normalize(
        bond_set: set[tuple[frozenset[str], str, bool]],
    ) -> set[tuple[frozenset[str], str, bool]]:
        """Normalize resonance-equivalent bonds to a common label.

        Args:
            bond_set: Set of normalized bond keys.

        Returns:
            Bond-key set with delocalized bonds collapsed to one label.
        """
        out = set()
        for pair, btype, ring in bond_set:
            if pair in delocalized_pairs and btype in aromatic_equiv:
                out.add((pair, "DELOCALIZED", ring))
            else:
                out.add((pair, btype, ring))
        return out

    gen_norm = normalize(gen_bonds)
    ref_norm = normalize(ref_bonds)
    checks["bonds"] = gen_norm == ref_norm
    if not checks["bonds"]:
        details["bonds"] = {
            "only_in_generated": sorted(gen_norm - ref_norm),
            "only_in_reference": sorted(ref_norm - gen_norm),
        }

    # Partial charges
    gen_q = generated.partial_charges
    ref_q = reference.partial_charges
    shared = sorted(gen_q.keys() & ref_q.keys())
    charge_bad = [
        (n, gen_q[n], ref_q[n], gen_q[n] - ref_q[n])
        for n in shared
        if abs(gen_q[n] - ref_q[n]) >= charge_tolerance
    ]
    checks["partial_charges"] = len(shared) > 0 and len(charge_bad) == 0
    if not checks["partial_charges"]:
        details["partial_charges"] = {
            "shared_atom_count": len(shared),
            "tolerance": charge_tolerance,
            "mismatches": charge_bad,
        }

    # Cartbonded params
    groups = [
        ("length_parameters", "length"),
        ("angle_parameters", "angle"),
        ("improper_parameters", "improper"),
    ]
    cb_diffs: dict[str, Any] = {}
    for attr_name, kind in groups:
        gen_keys = _cartres_heavy_key_set(
            getattr(generated.cartbonded_params, attr_name), kind
        )
        ref_keys = _cartres_heavy_key_set(
            getattr(reference.cartbonded_params, attr_name), kind
        )
        if gen_keys != ref_keys:
            cb_diffs[attr_name] = {
                "only_in_generated": sorted(gen_keys - ref_keys),
                "only_in_reference": sorted(ref_keys - gen_keys),
            }
    checks["cartbonded_params"] = len(cb_diffs) == 0
    if not checks["cartbonded_params"]:
        details["cartbonded_params"] = cb_diffs

    return EquivalenceResult(
        is_equivalent=all(checks.values()),
        checks=checks,
        details=details,
    )
