"""Registration of dynamically created ligand residue types.

Extends tmol's ParameterDatabase with new residue types and their
scoring parameters built by the ligand preparation pipeline.
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional

from tmol.database import (
    PatchedChemicalDatabase,
    ParameterDatabase,
)
from tmol.database.chemical import (
    AtomType,
    RawResidueType,
)
from tmol.database.scoring import (
    AngleGroup,
    CartRes,
    LengthGroup,
)
from tmol.io import CanonicalOrdering
from tmol.ligand._chemistry_tables import get_hbond_properties

logger = logging.getLogger(__name__)


def _build_cartbonded_params(  # noqa: C901
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
    icoor_by_name = {ic.name: ic for ic in residue_type.icoors}

    np_coords = (
        {k: np.asarray(v, dtype=float) for k, v in coords.items()}
        if coords is not None
        else None
    )

    def _dist_from_coords(a: str, b: str) -> Optional[float]:
        """Measure distance between two atoms from coordinate annotations.

        Args:
            a: First atom name.
            b: Second atom name.

        Returns:
            Distance in angstroms, or ``None`` if either atom is unavailable.
        """
        if np_coords is None or a not in np_coords or b not in np_coords:
            return None
        return float(np.linalg.norm(np_coords[a] - np_coords[b]))

    def _angle_from_coords(a: str, b: str, c: str) -> Optional[float]:
        """Measure the angle ``a-b-c`` from coordinate annotations.

        Args:
            a: First atom name.
            b: Vertex atom name.
            c: Third atom name.

        Returns:
            Angle in degrees, or ``None`` when coordinates are unavailable.
        """
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


def collect_new_atom_types(
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
            element = "H" if props.get("is_polarh") or name.startswith("H") else "C"
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

    * **AtomArray / SMILES path** — :func:`tmol.ligand.prepare_single_ligand`
      types the (already protonated, already charged) SMILES-derived molecule,
      builds the residue, and extracts cartbonded params, returning one
      ``LigandPreparation`` per ligand.
    * **Params-file path** — :func:`tmol.ligand._params_file.load_params_file`
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
    # Patches for this residue's chain ends, scoped to it alone. A backbone the
    # database's own termini patches were not written for carries its own.
    adds_patches: tuple = ()
    # {variant residue name: {atom: charge}} for those patches
    variant_partial_charges: Optional[dict[str, dict[str, float]]] = None


def _applied_patch(chemdb, base, variant):
    """The variant type that produced ``variant`` from ``base``.

    Several patches can share a display name -- an amine's and a substituted
    amine's -- so the suffix does not say which one ran. Each is identified by
    the atoms it declares, and the one that ran is the most specific whose
    atoms are all present.
    """
    present = {a.name for a in variant.atoms}
    absent = {a.name for a in base.atoms} - present
    best = None
    for patch in chemdb.variants:
        names = {a.name for a in patch.add_atoms}
        if not names <= present:
            continue
        if any(a.name in present for a in patch.modify_atoms if a.name in absent):
            continue
        if best is None or len(names) > len({a.name for a in best.add_atoms}):
            best = patch
    return best


def _patched_connection(base, variant, patch):
    """The connection the patch acts on, as (base atom, variant atom).

    A terminus patch replaces one polymer connection with the atoms that cap
    it, so the connection it removed is the one the base has and the variant
    does not. Its atom is where the charge redistribution is centred, and it
    is named by the residue rather than by the patch, so a reference's has to
    be found the same way rather than assumed to share the name.
    """
    kept = {c.name for c in variant.connections}
    for connection in base.connections:
        if connection.name not in kept:
            return connection.name, connection.atom
    return None, None


def terminus_charge_entries(param_db, patched_chemdb, residue_type) -> dict:
    """``{variant name: {atom: charge}}`` for every variant a residue takes.

    A patch introduces atoms the base residue has no charge for -- the acid's
    OXT, the ammonium's protons -- so a residue injected without them cannot be
    scored in any terminal position. Terminal charges are backbone chemistry
    rather than this residue's, so they are taken from a residue already in the
    database that the same patch was applied to: the substituted-amine patch
    finds a proline, the plain one an alanine, with no residue named here.

    The patch's own atoms carry the same names everywhere, since the patch
    gives them. The connection atom is the exception -- it is the residue's,
    and a gamma-linked acid calls it CD where an alpha one calls it C -- so it
    is matched to the reference's by the connection it belongs to.
    """
    by_name = {r.name: r for r in patched_chemdb.residues}
    charges: dict = {}
    for entry in param_db.scoring.elec.atom_charge_parameters:
        charges.setdefault(str(entry.res), {})[str(entry.atom)] = entry.charge

    references: dict = {}
    for name, restype in by_name.items():
        base_name, _, suffix = name.partition(":")
        if not suffix or base_name not in by_name or name not in charges:
            continue
        patch = _applied_patch(patched_chemdb, by_name[base_name], restype)
        if patch is not None:
            references.setdefault((suffix, patch.name), name)

    entries: dict = {}
    for name, restype in by_name.items():
        base_name, _, suffix = name.partition(":")
        if base_name != residue_type.name or not suffix:
            continue
        patch = _applied_patch(patched_chemdb, residue_type, restype)
        if patch is None:
            continue
        reference = references.get((suffix, patch.name))
        if reference is None:
            continue
        reference_charges = charges[reference]

        delta = {
            atom.name: reference_charges[atom.name]
            for atom in (*patch.add_atoms, *patch.modify_atoms)
            if atom.name in reference_charges
        }
        _connection, atom_name = _patched_connection(residue_type, restype, patch)
        _, reference_atom = _patched_connection(
            by_name[reference.partition(":")[0]], by_name[reference], patch
        )
        if atom_name is not None and reference_atom in reference_charges:
            delta[atom_name] = reference_charges[reference_atom]
        if delta:
            entries[name] = delta
    return entries


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
        for at in collect_new_atom_types(
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
        variants=[v for p in new_preps for v in p.adds_patches] or None,
        partial_charges=_charges_with_termini(
            param_db, new_preps, (*param_db.chemical.atom_types, *new_atom_types)
        ),
        cartbonded_params={p.residue_type.name: p.cartbonded_params for p in new_preps},
    )


def _charges_with_termini(param_db, preps, atom_types) -> dict:
    """Each preparation's charges, plus the deltas its variants need.

    The variants are read off a patched copy rather than listed, so a patch
    added to the database is covered without a change here.
    """
    patched = param_db.chemical.with_added_residues(
        [p.residue_type for p in preps],
        atom_types=atom_types,
        variants=[v for p in preps for v in p.adds_patches] or None,
    )
    charges = {}
    for prep in preps:
        charges[prep.residue_type.name] = prep.partial_charges
        charges.update(terminus_charge_entries(param_db, patched, prep.residue_type))
        charges.update(prep.variant_partial_charges or {})
    return charges


def rebuild_canonical_ordering(
    param_db: ParameterDatabase,
) -> CanonicalOrdering:
    """Build a new CanonicalOrdering from a (possibly extended) ParameterDatabase."""
    return CanonicalOrdering.from_chemdb(param_db.chemical)
