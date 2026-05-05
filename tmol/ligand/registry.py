"""Registration of dynamically created ligand residue types.

Extends tmol's ParameterDatabase with new residue types and their
scoring parameters built by the ligand preparation pipeline.
"""

import logging
import math
import copy
from dataclasses import dataclass, field
from typing import Optional

from tmol.chemical.patched_chemdb import PatchedChemicalDatabase
from tmol.database import ParameterDatabase
from tmol.database.chemical import (
    AtomType,
    RawResidueType,
)
from tmol.database.scoring.cartbonded import (
    AngleGroup,
    CartRes,
    LengthGroup,
)
from tmol.io.canonical_ordering import CanonicalOrdering
from tmol.ligand.chemistry_tables import get_hbond_properties

logger = logging.getLogger(__name__)

CacheKey = tuple[str, float, tuple[str, ...], tuple[str, ...]]


@dataclass
class LigandPreparationCache:
    """Mutable cache keyed by (res_name, ph, atom_names, elements)."""
    ligands_by_key: dict[CacheKey, RawResidueType] = field(default_factory=dict)
    charges_by_key: dict[CacheKey, dict[str, float]] = field(default_factory=dict)


_default_cache = LigandPreparationCache()


def get_default_cache() -> LigandPreparationCache:
    """Return the process-global ligand preparation cache."""
    return _default_cache


def get_cached_ligand_for_key(
    cache_key: CacheKey, cache: Optional[LigandPreparationCache] = None
) -> Optional[RawResidueType]:
    """Retrieve a cached ligand by full preparation key."""
    cache = cache or _default_cache
    cached = cache.ligands_by_key.get(cache_key)
    return copy.deepcopy(cached) if cached is not None else None


def get_cached_charges_for_key(
    cache_key: CacheKey, cache: Optional[LigandPreparationCache] = None
) -> Optional[dict[str, float]]:
    """Retrieve cached partial charges by full preparation key."""
    cache = cache or _default_cache
    cached = cache.charges_by_key.get(cache_key)
    return dict(cached) if cached is not None else None


def cache_ligand(
    res_name: str,
    restype: RawResidueType,
    charges: Optional[dict[str, float]] = None,
    *,
    cache_key: Optional[CacheKey] = None,
    cache: Optional[LigandPreparationCache] = None,
) -> None:
    """Store a prepared ligand and its charges under ``cache_key``.

    The ``res_name`` argument is kept for call-site clarity but has no
    effect on cache lookup — only ``cache_key`` is.
    """
    cache = cache or _default_cache
    if cache_key is None:
        return
    cache.ligands_by_key[cache_key] = copy.deepcopy(restype)
    if charges is not None:
        cache.charges_by_key[cache_key] = dict(charges)


def clear_cache(cache: Optional[LigandPreparationCache] = None) -> None:
    """Clear ligand cache contents."""
    cache = cache or _default_cache
    cache.ligands_by_key.clear()
    cache.charges_by_key.clear()


def _build_cartbonded_params(
    residue_type: RawResidueType,
    coords: Optional[dict[str, "tuple[float, float, float]"]] = None,
) -> CartRes:
    """Build a CartRes from a ligand's bond topology and Cartesian geometry.

    Extracts:
    - Bond lengths from every bond (K=300 kcal/mol/A^2)
    - Bond angles from every bonded triplet (K=80 kcal/mol/rad^2)

    When ``coords`` is supplied (mapping atom_name -> (x, y, z)), distances
    and angles are computed from real Cartesian positions, capturing
    ring-closure bonds that the icoor tree omits. Otherwise we fall back to
    icoor-derived geometry, which only covers tree edges.

    Does NOT generate proper torsions (Rosetta CartBonded sets K=0 for
    non-protein proper torsions). Does NOT generate improper torsions
    either — the gen_bonded scoring term covers sp2 planarity, and Frank's
    reference .tmol files leave ``improper_parameters`` empty.
    """
    import numpy as np

    atom_names = {a.name for a in residue_type.atoms}
    atom_type_by_name = {a.name: a.atom_type for a in residue_type.atoms}
    icoor_by_name = {ic.name: ic for ic in residue_type.icoors}

    np_coords = (
        {k: np.asarray(v, dtype=float) for k, v in coords.items()}
        if coords is not None
        else None
    )

    def _dist_from_coords(a: str, b: str) -> Optional[float]:
        if np_coords is None or a not in np_coords or b not in np_coords:
            return None
        return float(np.linalg.norm(np_coords[a] - np_coords[b]))

    def _angle_from_coords(a: str, b: str, c: str) -> Optional[float]:
        if np_coords is None or any(x not in np_coords for x in (a, b, c)):
            return None
        v1 = np_coords[a] - np_coords[b]
        v2 = np_coords[c] - np_coords[b]
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-9 or n2 < 1e-9:
            return None
        cos = float(np.dot(v1, v2) / (n1 * n2))
        cos = max(-1.0, min(1.0, cos))
        return math.acos(cos)

    atom_neighbors: dict[str, list[str]] = {}
    for a, b, _order, *_ in residue_type.bonds:
        atom_neighbors.setdefault(a, []).append(b)
        atom_neighbors.setdefault(b, []).append(a)

    lengths = []
    seen_lengths: set[tuple[str, str]] = set()
    if np_coords is not None:
        # Iterate every bond so ring-closure bonds are included.
        for a, b, _order, *_ in residue_type.bonds:
            if a not in atom_names or b not in atom_names:
                continue
            key = (min(a, b), max(a, b))
            if key in seen_lengths:
                continue
            seen_lengths.add(key)
            d = _dist_from_coords(a, b)
            if d is None or d <= 0:
                continue
            lengths.append(LengthGroup(atm1=a, atm2=b, x0=d, K=300.0))
    else:
        # Fallback: icoor tree edges only (legacy behavior).
        for ic in residue_type.icoors:
            if ic.name == ic.parent:
                continue
            if ic.d > 0 and ic.name in atom_names and ic.parent in atom_names:
                lengths.append(
                    LengthGroup(atm1=ic.name, atm2=ic.parent, x0=ic.d, K=300.0)
                )

    angles = []
    seen_angles: set[tuple[str, str, str]] = set()
    for center, neighbors in atom_neighbors.items():
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                a1, a3 = neighbors[i], neighbors[j]
                key = (min(a1, a3), center, max(a1, a3))
                if key in seen_angles:
                    continue
                seen_angles.add(key)

                angle_rad: Optional[float] = None
                if np_coords is not None:
                    angle_rad = _angle_from_coords(a1, center, a3)
                if angle_rad is None:
                    ic1 = icoor_by_name.get(a1)
                    ic3 = icoor_by_name.get(a3)
                    if (
                        ic1 is not None
                        and ic3 is not None
                        and ic1.parent == center
                        and ic3.parent == center
                        and ic1.theta > 0
                        and ic3.theta > 0
                    ):
                        delta = abs(ic1.phi - ic3.phi)
                        if delta > math.pi:
                            delta = (2.0 * math.pi) - delta
                        angle_rad = delta
                    else:
                        ic_center = icoor_by_name.get(center)
                        if ic_center and ic_center.theta > 0:
                            angle_rad = math.pi - ic_center.theta

                if angle_rad is not None and angle_rad > 0:
                    angles.append(
                        AngleGroup(atm1=a1, atm2=center, atm3=a3, x0=angle_rad, K=80.0)
                    )

    return CartRes(
        length_parameters=tuple(lengths),
        angle_parameters=tuple(angles),
        torsion_parameters=(),
        improper_parameters=(),
        hxltorsion_parameters=(),
    )


def _collect_new_atom_types(
    chem_db: PatchedChemicalDatabase,
    residue_type: RawResidueType,
    atom_type_elements: Optional[dict[str, str]] = None,
    *,
    strict_atom_types: bool = False,
) -> list[AtomType]:
    """Identify atom types used by the residue that aren't in the database.

    Sets hbond properties (is_donor, is_acceptor, acceptor_hybridization)
    from the HBOND_PROPERTIES lookup in atom_typing.py.
    """
    existing = {at.name for at in chem_db.atom_types}
    needed: dict[str, str] = {}
    hbond_properties = get_hbond_properties()

    for atom in residue_type.atoms:
        if atom.atom_type not in existing and atom.atom_type not in needed:
            needed[atom.atom_type] = atom.atom_type

    result = []
    atom_type_elements = atom_type_elements or {}
    for name in needed:
        props = hbond_properties.get(name, {})
        element = atom_type_elements.get(name)
        if element is None:
            if strict_atom_types:
                raise ValueError(
                    f"Unknown element mapping for atom type '{name}' while "
                    f"registering residue {residue_type.name}"
                )
            # Heuristic: treat polar-H atom types and any name starting
            # with 'H' as hydrogen, everything else as carbon. The
            # params-file path always lands here because the file format
            # encodes atom types but not their elements.
            element = (
                "H"
                if props.get("is_polarh") or name.startswith("H")
                else "C"
            )
        result.append(
            AtomType(
                name=name,
                element=element,
                is_donor=props.get("is_donor", False),
                is_acceptor=props.get("is_acceptor", False),
                is_hydroxyl=props.get("is_hydroxyl", False),
                is_polarh=props.get("is_polarh", False),
                acceptor_hybridization=props.get("acceptor_hybridization"),
            )
        )
    return result


@dataclass(frozen=True)
class LigandPreparation:
    """The unified abstraction both ligand-pipeline paths converge on.

    A ``LigandPreparation`` is everything tmol needs to inject one ligand
    into a ``ParameterDatabase``: the residue type definition, partial
    charges, cartbonded parameters, and (optionally) the element mapping
    for any new atom-type names introduced.

    Both pipeline entry points produce this same struct:

    * **AtomArray path** — :func:`tmol.ligand.prepare_single_ligand`
      runs the RDKit pipeline (protonation, atom typing, residue
      building, cartbonded extraction) and returns one
      ``LigandPreparation`` per detected ligand.
    * **Params-file path** — :func:`tmol.ligand.params_file.load_params_file`
      parses a ``.tmol`` YAML and returns ``list[LigandPreparation]``
      describing the residues defined in that file.

    Either list is then handed to :func:`inject_ligand_preparations`,
    the single chokepoint that extends the ``ParameterDatabase``. Tests
    can equally roundtrip ``AtomArray → LigandPreparation → .tmol →
    LigandPreparation`` and expect bit-equivalent injection.
    """

    residue_type: RawResidueType
    partial_charges: dict[str, float]
    cartbonded_params: CartRes
    # Optional element mapping for new atom types this ligand introduces.
    # Populated by the AtomArray path (where atom_type element is known
    # from the RDKit Mol). The params-file path leaves it None and the
    # injector falls back to an element heuristic.
    atom_type_elements: Optional[dict[str, str]] = None


def inject_ligand_preparations(
    param_db: ParameterDatabase,
    preparations: list[LigandPreparation],
    *,
    strict_atom_types: bool = False,
) -> ParameterDatabase:
    """Inject a batch of ``LigandPreparation`` records into a database.

    The single chokepoint both pipeline paths use — given a list of
    prepared ligands (regardless of whether they came from a
    ``.tmol`` file or an AtomArray), this function aggregates their
    residue types, atom types, charges, and cartbonded params and
    evolves the input ``ParameterDatabase`` exactly once via
    :func:`tmol.database.inject_residue_params`.

    Residues whose name already exists in ``param_db`` are silently
    skipped so repeat injection is idempotent.

    Args:
        param_db: Base database (not modified).
        preparations: One ``LigandPreparation`` per ligand to register.
        strict_atom_types: If True, raise when an atom type's element
            cannot be resolved from any preparation's
            ``atom_type_elements`` — otherwise fall back to a name-based
            heuristic and emit a warning.

    Returns:
        A new frozen ``ParameterDatabase`` extended with all provided
        preparations.
    """
    from tmol.database import inject_residue_params

    if not preparations:
        return param_db

    existing_names = {r.name for r in param_db.chemical.residues}
    new_preps = [p for p in preparations if p.residue_type.name not in existing_names]
    if not new_preps:
        return param_db

    new_atom_types: list[AtomType] = []
    seen_at: set[str] = set()
    for prep in new_preps:
        for at in _collect_new_atom_types(
            param_db.chemical,
            prep.residue_type,
            atom_type_elements=prep.atom_type_elements,
            strict_atom_types=strict_atom_types,
        ):
            if at.name in seen_at:
                continue
            seen_at.add(at.name)
            new_atom_types.append(at)

    for prep in new_preps:
        logger.info(
            "Registering ligand %s (%d atoms, %d bonds)",
            prep.residue_type.name,
            len(prep.residue_type.atoms),
            len(prep.residue_type.bonds),
        )

    return inject_residue_params(
        param_db,
        residue_types=[p.residue_type for p in new_preps],
        atom_types=new_atom_types or None,
        partial_charges={p.residue_type.name: p.partial_charges for p in new_preps},
        cartbonded_params={
            p.residue_type.name: p.cartbonded_params for p in new_preps
        },
    )


def rebuild_canonical_ordering(
    param_db: ParameterDatabase,
) -> CanonicalOrdering:
    """Build a new CanonicalOrdering from a (possibly extended) ParameterDatabase."""
    return CanonicalOrdering.from_chemdb(param_db.chemical)
