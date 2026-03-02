"""Registration of dynamically created ligand residue types.

Extends tmol's ParameterDatabase with new residue types and their
scoring parameters built by the ligand preparation pipeline.
"""

import logging
import math
from typing import Optional

from tmol.database import ParameterDatabase
from tmol.database.chemical import (
    AtomType,
    ChemicalDatabase,
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

_ligand_cache: dict[str, RawResidueType] = {}
_ligand_charges_cache: dict[str, dict[str, float]] = {}

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


def get_cached_ligand(res_name: str) -> Optional[RawResidueType]:
    """Retrieve a previously prepared ligand from the cache."""
    return _ligand_cache.get(res_name)


def get_cached_charges(res_name: str) -> Optional[dict[str, float]]:
    """Retrieve cached partial charges for a previously prepared ligand."""
    return _ligand_charges_cache.get(res_name)


def cache_ligand(
    res_name: str,
    restype: RawResidueType,
    charges: Optional[dict[str, float]] = None,
) -> None:
    """Store a prepared ligand and its charges in the cache."""
    _ligand_cache[res_name] = restype
    if charges is not None:
        _ligand_charges_cache[res_name] = charges


def clear_cache() -> None:
    """Clear the ligand cache."""
    _ligand_cache.clear()
    _ligand_charges_cache.clear()


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
    for a, b in residue_type.bonds:
        atom_neighbors.setdefault(a, []).append(b)
        atom_neighbors.setdefault(b, []).append(a)

    angles = []
    for center, neighbors in atom_neighbors.items():
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                a1, a3 = neighbors[i], neighbors[j]
                ic_center = icoor_by_name.get(center)
                if ic_center and ic_center.theta > 0:
                    angle_rad = math.pi - ic_center.theta
                    if angle_rad > 0:
                        angles.append(
                            AngleGroup(
                                atm1=a1, atm2=center, atm3=a3, x0=angle_rad, K=80.0
                            )
                        )

    impropers = []
    for center, neighbors in atom_neighbors.items():
        atype = atom_type_by_name.get(center, "")
        if atype in SP2_ATOM_TYPES and len(neighbors) == 3:
            n = neighbors
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
    chem_db: ChemicalDatabase,
    residue_type: RawResidueType,
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
    for name in needed:
        props = HBOND_PROPERTIES.get(name, {})
        element = "H" if props.get("is_polarh") else "C"
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
) -> None:
    """Register a new ligand residue type in the ParameterDatabase.

    Extends the chemical database with the new residue and atom types,
    and injects scoring parameters (charges, cartbonded) via the
    ParameterDatabase.add_residue_scoring_params API.

    Args:
        param_db: The ParameterDatabase to extend (mutated in place).
        residue_type: The ligand RawResidueType to register.
        partial_charges: Optional per-atom partial charges {atom_name: charge}.
    """
    from tmol.chemical.patched_chemdb import PatchedChemicalDatabase

    chem_db = param_db.chemical

    existing_names = {r.name for r in chem_db.residues}
    if residue_type.name in existing_names:
        logger.info("Residue %s already registered, skipping", residue_type.name)
        return

    new_atom_types = _collect_new_atom_types(chem_db, residue_type)
    if new_atom_types:
        logger.info(
            "Adding %d new atom types: %s",
            len(new_atom_types),
            [at.name for at in new_atom_types],
        )

    cache_ligand(residue_type.name, residue_type, charges=partial_charges)

    logger.info(
        "Registering ligand %s (%d atoms, %d bonds)",
        residue_type.name,
        len(residue_type.atoms),
        len(residue_type.bonds),
    )

    new_chem_db = ChemicalDatabase(
        element_types=chem_db.element_types,
        atom_types=chem_db.atom_types + tuple(new_atom_types),
        residues=chem_db.residues + (residue_type,),
        variants=chem_db.variants,
    )
    param_db.chemical = PatchedChemicalDatabase.from_chem_db(new_chem_db)

    cart_res = _build_cartbonded_params(residue_type)
    param_db.add_residue_scoring_params(
        residue_type.name,
        partial_charges=partial_charges,
        cartbonded_params=cart_res,
    )


def rebuild_canonical_ordering(
    param_db: ParameterDatabase,
) -> CanonicalOrdering:
    """Build a new CanonicalOrdering from a (possibly extended) ParameterDatabase."""
    return CanonicalOrdering.from_chemdb(param_db.chemical)
