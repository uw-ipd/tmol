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
    ImproperGroup,
    LengthGroup,
)
from tmol.io.canonical_ordering import CanonicalOrdering

logger = logging.getLogger(__name__)

CacheKey = tuple[str, float, tuple[str, ...], tuple[str, ...]]


@dataclass
class LigandPreparationCache:
    """Mutable cache object for ligand preparation products."""

    ligands_by_key: dict[CacheKey, RawResidueType] = field(default_factory=dict)
    charges_by_key: dict[CacheKey, dict[str, float]] = field(default_factory=dict)
    ligands_by_name: dict[str, RawResidueType] = field(default_factory=dict)
    charges_by_name: dict[str, dict[str, float]] = field(default_factory=dict)


_default_cache = LigandPreparationCache()

SP2_ATOM_TYPES = frozenset(
    [
        "CD",
        "CD1",
        "CD2",
        "CDp",
        "CR",
        "CRp",
        "Oad",
        "Oal",
        "Oat",
        "Ont",
        "OG2",
        "Nad",
        "Nim",
        "Nin",
    ]
)


def get_default_cache() -> LigandPreparationCache:
    """Return the process-global ligand preparation cache."""
    return _default_cache


def get_cached_ligand(
    res_name: str, cache: Optional[LigandPreparationCache] = None
) -> Optional[RawResidueType]:
    """Retrieve a previously prepared ligand by residue name."""
    cache = cache or _default_cache
    cached = cache.ligands_by_name.get(res_name)
    return copy.deepcopy(cached) if cached is not None else None


def get_cached_ligand_for_key(
    cache_key: CacheKey, cache: Optional[LigandPreparationCache] = None
) -> Optional[RawResidueType]:
    """Retrieve a cached ligand by full preparation key."""
    cache = cache or _default_cache
    cached = cache.ligands_by_key.get(cache_key)
    return copy.deepcopy(cached) if cached is not None else None


def get_cached_charges(
    res_name: str, cache: Optional[LigandPreparationCache] = None
) -> Optional[dict[str, float]]:
    """Retrieve cached partial charges by residue name."""
    cache = cache or _default_cache
    cached = cache.charges_by_name.get(res_name)
    return dict(cached) if cached is not None else None


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
    """Store a prepared ligand and its charges in the cache."""
    cache = cache or _default_cache
    restype_snapshot = copy.deepcopy(restype)
    charges_snapshot = dict(charges) if charges is not None else None
    cache.ligands_by_name[res_name] = restype_snapshot
    if charges is not None:
        cache.charges_by_name[res_name] = charges_snapshot
    if cache_key is not None:
        cache.ligands_by_key[cache_key] = copy.deepcopy(restype_snapshot)
        if charges is not None:
            cache.charges_by_key[cache_key] = dict(charges_snapshot)


def clear_cache(cache: Optional[LigandPreparationCache] = None) -> None:
    """Clear ligand cache contents."""
    cache = cache or _default_cache
    cache.ligands_by_key.clear()
    cache.charges_by_key.clear()
    cache.ligands_by_name.clear()
    cache.charges_by_name.clear()


def _build_cartbonded_params(residue_type: RawResidueType) -> CartRes:
    """Build a CartRes from a ligand's icoors and bond topology.

    Extracts:
    - Bond lengths from icoors (K=300 kcal/mol/A^2)
    - Bond angles from icoors (K=80 kcal/mol/rad^2)
    - Improper torsions for sp2 centers (K=80 kcal/mol, x0=0)

    Does NOT generate proper torsions (Rosetta CartBonded sets K=0
    for non-protein proper torsions).
    """
    atom_names = {a.name for a in residue_type.atoms}
    atom_type_by_name = {a.name: a.atom_type for a in residue_type.atoms}
    icoor_by_name = {ic.name: ic for ic in residue_type.icoors}

    lengths = []
    for ic in residue_type.icoors:
        if ic.name == ic.parent:
            continue
        if ic.d > 0 and ic.name in atom_names and ic.parent in atom_names:
            lengths.append(LengthGroup(atm1=ic.name, atm2=ic.parent, x0=ic.d, K=300.0))

    atom_neighbors: dict[str, list[str]] = {}
    for a, b, _ in residue_type.bonds:
        atom_neighbors.setdefault(a, []).append(b)
        atom_neighbors.setdefault(b, []).append(a)

    angles = []
    seen_angles: set[tuple[str, str, str]] = set()
    for center, neighbors in atom_neighbors.items():
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                a1, a3 = neighbors[i], neighbors[j]
                angle_key = (min(a1, a3), center, max(a1, a3))
                if angle_key in seen_angles:
                    continue
                seen_angles.add(angle_key)

                ic1 = icoor_by_name.get(a1)
                ic3 = icoor_by_name.get(a3)
                angle_rad = None
                # Prefer child-local icoors when both neighbors are children of center.
                # This avoids assigning the same center angle to all neighbor pairs.
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

    impropers = []
    seen_impropers: set[tuple[str, str, str, str]] = set()
    for center, neighbors in atom_neighbors.items():
        atype = atom_type_by_name.get(center, "")
        if atype in SP2_ATOM_TYPES and len(neighbors) == 3:
            n = sorted(neighbors)
            improper_key = (n[0], n[1], center, n[2])
            if improper_key in seen_impropers:
                continue
            seen_impropers.add(improper_key)
            impropers.append(
                ImproperGroup(
                    atm1=n[0],
                    atm2=n[1],
                    atm3=center,
                    atm4=n[2],
                    k1=80.0,
                    phi1=0.0,
                )
            )

    return CartRes(
        length_parameters=tuple(lengths),
        angle_parameters=tuple(angles),
        torsion_parameters=(),
        improper_parameters=tuple(impropers),
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
    from tmol.ligand.atom_typing import HBOND_PROPERTIES

    existing = {at.name for at in chem_db.atom_types}
    needed: dict[str, str] = {}

    for atom in residue_type.atoms:
        if atom.atom_type not in existing and atom.atom_type not in needed:
            needed[atom.atom_type] = atom.atom_type

    result = []
    atom_type_elements = atom_type_elements or {}
    for name in needed:
        props = HBOND_PROPERTIES.get(name, {})
        element = atom_type_elements.get(name)
        if element is None:
            msg = (
                f"Unknown element mapping for atom type '{name}' while registering "
                f"residue {residue_type.name}"
            )
            if strict_atom_types:
                raise ValueError(msg)
            logger.warning("%s; using heuristic fallback", msg)
            element = (
                "H"
                if props.get("is_polarh")
                else ("H" if name.startswith("H") else "C")
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


def register_ligand(
    param_db: ParameterDatabase,
    residue_type: RawResidueType,
    partial_charges: Optional[dict[str, float]] = None,
    atom_type_elements: Optional[dict[str, str]] = None,
    *,
    strict_atom_types: bool = False,
) -> bool:
    """Register a new ligand residue type in the ParameterDatabase.

    Extends the chemical database with the new residue and atom types,
    and injects scoring parameters (charges, cartbonded) via the
    ParameterDatabase.add_residue_scoring_params API.

    Args:
        param_db: The ParameterDatabase to extend (mutated in place).
        residue_type: The ligand RawResidueType to register.
        partial_charges: Optional per-atom partial charges {atom_name: charge}.
        atom_type_elements: Optional mapping {atom_type: element_symbol} for
            newly introduced atom types.
        strict_atom_types: If True, fail when an unknown atom type element
            cannot be resolved from atom_type_elements.
    Returns:
        True when a new residue type is inserted into the database, otherwise
        False if the residue was already present.
    """
    chem_db = param_db.chemical

    existing_names = {r.name for r in chem_db.residues}
    if residue_type.name in existing_names:
        logger.info("Residue %s already registered, skipping", residue_type.name)
        return False

    new_atom_types = _collect_new_atom_types(
        chem_db,
        residue_type,
        atom_type_elements=atom_type_elements,
        strict_atom_types=strict_atom_types,
    )
    if new_atom_types:
        logger.info(
            "Adding %d new atom types: %s",
            len(new_atom_types),
            [at.name for at in new_atom_types],
        )

    logger.info(
        "Registering ligand %s (%d atoms, %d bonds)",
        residue_type.name,
        len(residue_type.atoms),
        len(residue_type.bonds),
    )

    param_db.add_residue_type(residue_type, new_atom_types=new_atom_types or None)

    cart_res = _build_cartbonded_params(residue_type)
    param_db.add_residue_scoring_params(
        residue_type.name,
        partial_charges=partial_charges,
        cartbonded_params=cart_res,
    )
    return True


def rebuild_canonical_ordering(
    param_db: ParameterDatabase,
) -> CanonicalOrdering:
    """Build a new CanonicalOrdering from a (possibly extended) ParameterDatabase."""
    return CanonicalOrdering.from_chemdb(param_db.chemical)
