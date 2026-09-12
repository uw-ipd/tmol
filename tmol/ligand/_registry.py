"""Registration of dynamically created ligand residue types.

Extends tmol's ParameterDatabase with new residue types and their
scoring parameters built by the ligand preparation pipeline.
"""

import logging
import math
from dataclasses import dataclass, replace
from itertools import chain
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
    ConnectionCartRes,
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
    return _collect_new_atom_types(
        chem_db,
        ((residue_type.atoms, f"residue {residue_type.name}"),),
        atom_type_elements,
        strict_atom_types=strict_atom_types,
    )


def _collect_new_atom_types(
    chem_db, atom_sources, atom_type_elements=None, *, strict_atom_types=False
):
    """Collect types once across sources, including atoms introduced by patches."""
    existing = {at.name for at in chem_db.atom_types}
    needed: dict[str, str] = {}
    hbond_properties = get_hbond_properties()

    for atoms, source in atom_sources:
        for atom in atoms:
            if atom.atom_type not in existing:
                needed.setdefault(atom.atom_type, source)

    result = []
    atom_type_elements = atom_type_elements or {}
    for name in needed:
        props = hbond_properties.get(name, {})
        element = atom_type_elements.get(name)
        if element is None:
            if strict_atom_types:
                raise ValueError(
                    f"Unknown element mapping for atom type '{name}' while "
                    f"registering {needed[name]}"
                )
            # Heuristic: treat polar-H atom types and any name starting
            # with 'H' as hydrogen, everything else as carbon. The
            # Legacy params files may omit element declarations. Version-5
            # .tmol files preserve explicit maps and avoid this fallback.
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
    # from the RDKit Mol) and preserved by .tmol version 5. Legacy files may
    # omit it, in which case strict injection rejects unknown types.
    atom_type_elements: Optional[dict[str, str]] = None
    # Patches for this residue and any existing residues it attaches to.
    # Shared bundle metadata is carried once, on its first preparation.
    adds_patches: tuple = ()
    # {variant residue name: {atom: charge}} for those patches
    variant_partial_charges: Optional[dict[str, dict[str, float]]] = None
    connection_params: tuple[ConnectionCartRes, ...] = ()
    # Complete CartRes replacements for other exact residue/variant names.
    additional_cartbonded_params: Optional[dict[str, CartRes]] = None
    # Explicit exact-name replacement, guarded by the original residue,
    # effective charges, and bonded-record digest. Ordinary additions omit it.
    baseline_sha256: Optional[str] = None


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


def _unique_preparations(preparations):
    """One complete definition per name; shared metadata stays on its sources."""
    unique = {}
    fields = ("residue_type", "partial_charges", "cartbonded_params", "baseline_sha256")
    for prep in preparations:
        name = prep.residue_type.name
        previous = unique.get(name)
        if previous is None:
            unique[name] = prep
        elif any(getattr(previous, f) != getattr(prep, f) for f in fields):
            raise ValueError(f"Conflicting residue preparations: {name}")
    return list(unique.values())


def _merge_named_parameters(pairs, kind):
    merged = {}
    for name, value in pairs:
        if name in merged and merged[name] != value:
            raise ValueError(f"Conflicting {kind}: {name}")
        merged[name] = value
    return merged


def _merge_partial_charges(mappings):
    return _charges_from_rows(
        (res, atom, charge)
        for mapping in mappings
        for res, charges in mapping.items()
        for atom, charge in charges.items()
    )


def _charges_from_rows(rows):
    """Stream named charges into one map, rejecting conflicting duplicates."""
    merged = {}
    for res, atom, charge in rows:
        charges = merged.setdefault(res, {})
        if atom in charges and charges[atom] != charge:
            raise ValueError(f"Conflicting partial charges: {(res, atom)}")
        charges[atom] = charge
    return merged


def _validate_atom_type_elements(mapping):
    if not isinstance(mapping, dict) or any(
        not isinstance(k, str) or not k or not isinstance(v, str) or not v
        for k, v in mapping.items()
    ):
        raise ValueError(
            "atom_type_elements must map nonempty type names to element strings"
        )
    return mapping


def _batch_atom_type_elements(preparations, chemical=None):
    elements = _merge_named_parameters(
        (
            item
            for p in preparations
            for item in _validate_atom_type_elements(
                {} if p.atom_type_elements is None else p.atom_type_elements
            ).items()
        ),
        "atom type elements",
    )
    if chemical is not None:
        for atom_type in chemical.atom_types:
            if (
                atom_type.name in elements
                and elements[atom_type.name] != atom_type.element
            ):
                raise ValueError(
                    f"Atom type element disagrees with database: {atom_type.name}"
                )
    return elements


def _additional_cartbonded_params(preparations):
    """Collect shared records, rejecting contradictory definitions in one bundle."""
    # For guarded replacements, shared records describe the original baseline;
    # their complete target records are installed only after the digest check.
    own = {
        p.residue_type.name: p.cartbonded_params
        for p in preparations
        if p.baseline_sha256 is None
    }
    extra = {}
    for prep in preparations:
        for name, params in (prep.additional_cartbonded_params or {}).items():
            previous = extra.get(name, own.get(name))
            if previous is not None and previous != params:
                raise ValueError(f"Conflicting bonded parameters: {name}")
            extra[name] = params
    return extra


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
    evolves the input ``ParameterDatabase`` via
    :func:`tmol.database.inject_residue_params`.

    Base definitions whose name already exists in ``param_db`` are skipped.
    Their additional patches, variant charges and connection records are still
    installed; repeating an identical bundle returns the original database.
    A preparation with ``baseline_sha256`` instead replaces an exact residue
    after checking its original parameters (or an already installed result).

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
    replacements = [p for p in preparations if p.baseline_sha256 is not None]
    if not replacements:
        return _inject_additions(param_db, preparations, strict_atom_types)

    from tmol.ligand._parameter_replacements import (
        install_replacements,
        validate_replacements,
    )

    # Check existing targets before any bundle metadata can reset their charges.
    validate_replacements(param_db, replacements, allow_missing=True)
    # Conflicting source metadata is invalid even when an installed target
    # makes that metadata unnecessary for this particular database.
    _merge_partial_charges(p.variant_partial_charges or {} for p in preparations)
    _additional_cartbonded_params(preparations)
    existing_names = {r.name for r in param_db.chemical.residues}
    protected = {p.residue_type.name for p in replacements} & existing_names
    additions = [
        replace(
            p,
            variant_partial_charges={
                n: q
                for n, q in (p.variant_partial_charges or {}).items()
                if n not in protected
            }
            or None,
            additional_cartbonded_params={
                n: c
                for n, c in (p.additional_cartbonded_params or {}).items()
                if n not in protected
            }
            or None,
            connection_params=(),
        )
        for p in preparations
    ]
    extended = _inject_additions(param_db, additions, strict_atom_types)
    elements = _batch_atom_type_elements(preparations, extended.chemical)
    atom_types = _collect_new_atom_types(
        extended.chemical,
        (
            (p.residue_type.atoms, f"residue {p.residue_type.name}")
            for p in replacements
        ),
        elements,
        strict_atom_types=strict_atom_types,
    )
    return install_replacements(
        extended,
        replacements,
        tuple(dict.fromkeys(r for p in preparations for r in p.connection_params)),
        atom_types=atom_types or None,
    )


def _inject_additions(param_db, preparations, strict_atom_types):
    from tmol.database import inject_residue_params

    if not preparations:
        return param_db

    unique = _unique_preparations(p for p in preparations if p.baseline_sha256 is None)
    elements = _batch_atom_type_elements(preparations, param_db.chemical)
    existing_names = {r.name for r in param_db.chemical.residues}
    new_preps = [p for p in unique if p.residue_type.name not in existing_names]
    known_variants = {v.name: v for v in param_db.chemical.variants}
    variants = []
    for prep in preparations:
        for variant in prep.adds_patches:
            if variant.name in known_variants:
                if variant != known_variants[variant.name]:
                    raise ValueError(f"Conflicting patch definition: {variant.name}")
            else:
                known_variants[variant.name] = variant
                variants.append(variant)
    old_charges = {
        (p.res, p.atom): p.charge for p in param_db.scoring.elec.atom_charge_parameters
    }
    shared_charges = _merge_partial_charges(
        p.variant_partial_charges or {} for p in preparations
    )
    extra_charges = {
        name: changed
        for name, charges in shared_charges.items()
        if (
            changed := {
                a: q for a, q in charges.items() if old_charges.get((name, a)) != q
            }
        )
    }
    known_connections = set(param_db.scoring.cartbonded.connection_params)
    connections = tuple(
        dict.fromkeys(
            record
            for prep in preparations
            for record in prep.connection_params
            if record not in known_connections
        )
    )
    additional_cart = {
        name: params
        for name, params in _additional_cartbonded_params(preparations).items()
        if param_db.scoring.cartbonded.residue_params.get(name) != params
    }
    if (
        not new_preps
        and not variants
        and not extra_charges
        and not connections
        and not additional_cart
    ):
        return param_db

    new_atom_types = _collect_new_atom_types(
        param_db.chemical,
        chain(
            (
                (p.residue_type.atoms, f"residue {p.residue_type.name}")
                for p in new_preps
            ),
            ((chain(v.add_atoms, v.modify_atoms), f"patch {v.name}") for v in variants),
        ),
        elements,
        strict_atom_types=strict_atom_types,
    )

    for prep in new_preps:
        logger.info(
            "Registering ligand %s (%d atoms, %d bonds)",
            prep.residue_type.name,
            len(prep.residue_type.atoms),
            len(prep.residue_type.bonds),
        )

    charges = (
        _charges_with_termini(
            param_db, new_preps, (*param_db.chemical.atom_types, *new_atom_types)
        )
        if new_preps
        else {}
    )
    for name, delta in extra_charges.items():
        # The initial maps may be owned by reusable frozen preparations.
        charges[name] = {**charges.get(name, {}), **delta}
    return inject_residue_params(
        param_db,
        residue_types=[p.residue_type for p in new_preps],
        atom_types=new_atom_types or None,
        variants=variants or None,
        partial_charges=charges,
        cartbonded_params={
            **{p.residue_type.name: p.cartbonded_params for p in new_preps},
            **additional_cart,
        },
        connection_params=connections,
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
