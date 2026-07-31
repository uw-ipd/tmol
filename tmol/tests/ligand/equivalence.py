"""Shared ligand-preparation equivalence helpers.

These utilities are extracted from regression-test logic so large-scale
batch scripts can compare two ``LigandPreparation`` objects using the same
normalization rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from typing import Any, Iterable


def _identity_name(name: str) -> str:
    """Return ``name`` unchanged (default atom-name mapping)."""
    return name


def _is_heavy(name: str) -> bool:
    """Return whether an atom name is non-hydrogen.

    Args:
        name: Atom name.

    Returns:
        ``True`` when the atom is not a hydrogen.
    """
    return not str(name).startswith("H")


_AROMATIC_EQUIV = frozenset({"AROMATIC", "SINGLE", "DOUBLE"})


def _infer_element_from_name(name: str) -> str:
    """Infer element symbol from a Rosetta-style atom name.

    Fallback only: prefer :func:`_element_from_atom_type` when an atom type is
    available. This name-based guess cannot disambiguate a PDB carbon name like
    ``CA`` (alpha carbon) from the element calcium.
    """
    s = str(name).strip()
    if not s:
        return "?"
    letters = []
    for ch in s:
        if ch.isalpha():
            letters.append(ch)
        else:
            break
    token = "".join(letters)
    if not token:
        return "?"
    if len(token) == 1:
        return token.upper()
    upper = token.upper()
    if upper in {"CL", "BR", "NA", "MG", "ZN", "FE", "CA"}:
        return upper[0] + upper[1:].lower()
    return token[0].upper()


def _element_from_atom_type(atom_type: Any) -> str | None:
    """Element symbol from a Rosetta gen-potential atom type, or ``None``.

    The element is encoded unambiguously in the type prefix (``CS1`` -> C,
    ``Oat`` -> O, ``Nad`` -> N, ``Sth`` -> S, ``Cl``/``ClR`` -> Cl,
    ``Br``/``BrR`` -> Br, ``F``/``FR`` -> F, ``I``/``IR`` -> I). Preferring this
    over name-based inference avoids the PDB-atom-name collision where a carbon
    named ``CA`` (alpha carbon) would otherwise be read as calcium.
    """
    s = str(atom_type).strip()
    if not s:
        return None
    if s[:2].capitalize() in {"Cl", "Br"}:
        return s[:2].capitalize()
    first = s[0].upper()
    return first if first.isalpha() else None


def _heavy_graph(
    residue_type: Any,
) -> tuple[list[str], dict[str, set[str]], dict[str, tuple[str, int]]]:
    """Build heavy-atom adjacency and per-node labels.

    Node element labels are taken from each atom's Rosetta ``atom_type`` (which
    encodes the element unambiguously), falling back to name-based inference
    only when no type is present.
    """
    type_by_name = {
        str(a.name): getattr(a, "atom_type", None) for a in residue_type.atoms
    }

    def _element(name: str) -> str:
        from_type = _element_from_atom_type(type_by_name.get(name))
        return from_type if from_type is not None else _infer_element_from_name(name)

    nodes = [str(a.name) for a in residue_type.atoms if _is_heavy(a.name)]
    adj: dict[str, set[str]] = {n: set() for n in nodes}
    for a, b, *_ in residue_type.bonds:
        a = str(a)
        b = str(b)
        if a not in adj or b not in adj:
            continue
        adj[a].add(b)
        adj[b].add(a)
    labels = {n: (_element(n), len(adj[n])) for n in nodes}
    return nodes, adj, labels


def _heavy_atom_name_mapping(
    generated_rt: Any,
    reference_rt: Any,
) -> dict[str, str] | None:
    """Find a heavy-atom graph isomorphism from generated -> reference names."""
    gen_nodes, gen_adj, gen_labels = _heavy_graph(generated_rt)
    ref_nodes, ref_adj, ref_labels = _heavy_graph(reference_rt)
    if len(gen_nodes) != len(ref_nodes):
        return None

    # Fast path when heavy-atom name sets already match exactly.
    if set(gen_nodes) == set(ref_nodes):
        return {n: n for n in gen_nodes}

    by_ref_label: dict[tuple[str, int], list[str]] = {}
    for r in ref_nodes:
        by_ref_label.setdefault(ref_labels[r], []).append(r)
    for lbl in by_ref_label:
        by_ref_label[lbl].sort()

    ordering = sorted(
        gen_nodes,
        key=lambda n: (
            -gen_labels[n][1],  # degree first
            gen_labels[n][0],  # element
            n,
        ),
    )

    mapping: dict[str, str] = {}
    reverse: dict[str, str] = {}

    def feasible(g_name: str, r_name: str) -> bool:
        if gen_labels[g_name] != ref_labels[r_name]:
            return False
        g_neighbors = gen_adj[g_name]
        r_neighbors = ref_adj[r_name]
        if len(g_neighbors) != len(r_neighbors):
            return False
        for g_nb in g_neighbors:
            if g_nb in mapping and mapping[g_nb] not in r_neighbors:
                return False
        for r_nb in r_neighbors:
            if r_nb in reverse and reverse[r_nb] not in g_neighbors:
                return False
        return True

    def backtrack(depth: int) -> bool:
        if depth == len(ordering):
            return True
        g_name = ordering[depth]
        label = gen_labels[g_name]
        for r_name in by_ref_label.get(label, []):
            if r_name in reverse:
                continue
            if not feasible(g_name, r_name):
                continue
            mapping[g_name] = r_name
            reverse[r_name] = g_name
            if backtrack(depth + 1):
                return True
            del mapping[g_name]
            del reverse[r_name]
        return False

    return mapping if backtrack(0) else None


def _bond_keyset(
    bonds: Iterable[tuple[Any, Any, Any, Any]],
    map_name: Callable[[str], str],
) -> set[tuple[frozenset[str], str, bool]]:
    """Normalize heavy-atom bond records into comparison keys."""
    out = set()
    for a, b, bond_type, *rest in bonds:
        a_name = map_name(str(a))
        b_name = map_name(str(b))
        if not (_is_heavy(a_name) and _is_heavy(b_name)):
            continue
        ring = bool(rest[0]) if rest else False
        out.add((frozenset([a_name, b_name]), str(bond_type), ring))
    return out


def _normalized_delocalized_bond_keys(
    bond_keys: set[tuple[frozenset[str], str, bool]],
    delocalized_pairs: set[frozenset[str]],
) -> set[tuple[frozenset[str], str, bool]]:
    """Normalize resonance-equivalent bonds to a common label."""
    out = set()
    for pair, btype, ring in bond_keys:
        if pair in delocalized_pairs and btype in _AROMATIC_EQUIV:
            out.add((pair, "DELOCALIZED", ring))
        else:
            out.add((pair, btype, ring))
    return out


def _cartres_heavy_key_set(
    params: Iterable[Any], kind: str, map_name: Callable[[str], str] | None = None
) -> set[Any]:
    """Build normalized heavy-atom keys for a cartbonded parameter group.

    Args:
        params: Iterable of cartbonded parameter entries.
        kind: Parameter group kind (``length``, ``angle``, or ``improper``).

    Returns:
        Set of normalized keys suitable for equivalence comparison.
    """
    name_fn = map_name if map_name is not None else _identity_name
    keys = set()
    if kind == "length":
        for p in params:
            a, b = name_fn(str(p.atm1)), name_fn(str(p.atm2))
            if _is_heavy(a) and _is_heavy(b):
                keys.add(frozenset([a, b]))
    elif kind == "angle":
        for p in params:
            a1 = name_fn(str(p.atm1))
            c = name_fn(str(p.atm2))
            a3 = name_fn(str(p.atm3))
            if all(_is_heavy(n) for n in (a1, c, a3)):
                lo, hi = sorted([a1, a3])
                keys.add((lo, c, hi))
    elif kind == "improper":
        for p in params:
            names = [
                name_fn(str(p.atm1)),
                name_fn(str(p.atm2)),
                name_fn(str(p.atm3)),
                name_fn(str(p.atm4)),
            ]
            if all(_is_heavy(n) for n in names):
                keys.add(tuple(sorted(names)))
    else:
        raise ValueError(f"Unknown cartbonded group kind: {kind}")
    return keys


@dataclass
class EquivalenceResult:
    """Outcome of comparing two ``LigandPreparation`` objects.

    Attributes:
        is_equivalent: Whether all active (non-skipped) checks passed.
        checks: Per-check pass/fail flags keyed by check name.
        details: Diagnostic detail for failed (or skipped) checks.
    """

    is_equivalent: bool
    checks: dict[str, bool]
    details: dict[str, Any]


def compare_ligand_preparations(  # noqa: C901
    generated: Any,
    reference: Any,
    *,
    charge_tolerance: float = 0.05,
    skip_checks: frozenset[str] | None = None,
) -> EquivalenceResult:
    """Compare two ``LigandPreparation`` objects with DUD test semantics.

    Args:
        skip_checks: Check names to omit from pass/fail (e.g. ``partial_charges``
            when comparing MMFF-derived prep to an AM1-BCC reference ``.tmol``).
            Skipped checks are recorded as passed and noted in ``details``.
    """
    skipped = skip_checks or frozenset()
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    heavy_name_map = _heavy_atom_name_mapping(
        generated.residue_type, reference.residue_type
    )
    if heavy_name_map is None:
        heavy_name_map = {}

    def _map_generated_name(name: str) -> str:
        return heavy_name_map.get(name, name)

    # Atom set
    gen_atoms = {
        (_map_generated_name(str(a.name)), a.atom_type)
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
        _map_generated_name(str(a.name)): a.atom_type
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
    gen_bonds = _bond_keyset(generated.residue_type.bonds, _map_generated_name)
    ref_bonds = _bond_keyset(reference.residue_type.bonds, lambda n: n)
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
        if ring and btype in _AROMATIC_EQUIV:
            return True
        if pair & aromatic_atoms and btype in _AROMATIC_EQUIV:
            return True
        return False

    delocalized_pairs: set[frozenset] = set()
    for pair, btype, ring in all_bonds:
        if is_delocalized(pair, btype, ring):
            delocalized_pairs.add(pair)

    gen_norm = _normalized_delocalized_bond_keys(gen_bonds, delocalized_pairs)
    ref_norm = _normalized_delocalized_bond_keys(ref_bonds, delocalized_pairs)
    checks["bonds"] = gen_norm == ref_norm
    if not checks["bonds"]:
        details["bonds"] = {
            "only_in_generated": sorted(gen_norm - ref_norm),
            "only_in_reference": sorted(ref_norm - gen_norm),
        }

    # Partial charges
    gen_q = generated.partial_charges
    ref_q = reference.partial_charges
    gen_q_heavy = {
        _map_generated_name(str(name)): q
        for name, q in gen_q.items()
        if _is_heavy(str(name))
    }
    ref_q_heavy = {str(name): q for name, q in ref_q.items() if _is_heavy(str(name))}
    shared = sorted(gen_q_heavy.keys() & ref_q_heavy.keys())
    charge_bad = [
        (n, gen_q_heavy[n], ref_q_heavy[n], gen_q_heavy[n] - ref_q_heavy[n])
        for n in shared
        if abs(gen_q_heavy[n] - ref_q_heavy[n]) >= charge_tolerance
    ]
    checks["partial_charges"] = len(shared) > 0 and len(charge_bad) == 0
    if not checks["partial_charges"]:
        details["partial_charges"] = {
            "shared_atom_count": len(shared),
            "tolerance": charge_tolerance,
            "mismatches": charge_bad,
            "mapped_heavy_atom_count_generated": len(gen_q_heavy),
            "heavy_atom_count_reference": len(ref_q_heavy),
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
            getattr(generated.cartbonded_params, attr_name),
            kind,
            map_name=_map_generated_name,
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

    for name in skipped:
        checks[name] = True
        details[name] = "skipped"

    active = {k: v for k, v in checks.items() if k not in skipped}
    return EquivalenceResult(
        is_equivalent=all(active.values()),
        checks=checks,
        details=details,
    )
