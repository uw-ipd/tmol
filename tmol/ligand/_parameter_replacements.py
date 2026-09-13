"""Explicit residue replacements checked against their parameter baseline."""

import hashlib
import json
import math
import re

import attr
import cattr

from tmol.score.elec._params import ElecParamResolver


def _baseline_charges(index, restype):
    base, variants = ElecParamResolver._lookup_order(restype.name)
    names = [base + ":" + v if v else base for v in variants]
    charges = {}
    for atom in restype.atoms:
        for name in names:
            if (name, atom.name) in index:
                charges[atom.name] = index[name, atom.name]
                break
        else:
            raise ValueError(f"Missing baseline charge for {restype.name},{atom.name}")
    return charges


def _local_identity(restype, charges, bonded):
    # Apply the same declared scalar types as YAML loading: NumPy atom names
    # and integer-valued float fields otherwise hash differently after export.
    restype = cattr.structure(cattr.unstructure(restype), type(restype))
    if bonded is not None:
        bonded = cattr.structure(cattr.unstructure(bonded), type(bonded))
    # Bond tuples are intentionally untyped in RawResidueType, so structuring
    # alone does not cast their NumPy strings. JSON encodes them as strings.
    payload = (
        "tmol-residue-replacement-v1",
        cattr.unstructure(restype),
        {str(a): float(q) for a, q in charges.items()},
        cattr.unstructure(bonded),
    )
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _connection_key(record):
    return tuple(
        sorted(
            (
                (record.block_type1, record.connection1),
                (record.block_type2, record.connection2),
            )
        )
    )


def validate_replacements(parameter_database, rows, *, allow_missing=False):
    """Validate complete records; return whether all targets are installed.

    The digest covers the exact residue definition, effective atom charges,
    and local bonded record. It is not a fingerprint of the whole force field.
    """
    residues = {r.name: r for r in parameter_database.chemical.residues}
    charge_index = {
        (p.res, p.atom): p.charge
        for p in parameter_database.scoring.elec.atom_charge_parameters
    }
    cart = parameter_database.scoring.cartbonded
    installed = True
    seen = {}
    for row in rows:
        name = row.residue_type.name
        if not isinstance(row.baseline_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", row.baseline_sha256
        ):
            raise ValueError(f"Invalid replacement baseline_sha256 for {name}")
        if set(row.partial_charges) != {a.name for a in row.residue_type.atoms} or any(
            not math.isfinite(q) for q in row.partial_charges.values()
        ):
            raise ValueError(
                f"Replacement requires complete finite atom charges: {name}"
            )
        expected = _local_identity(
            row.residue_type, row.partial_charges, row.cartbonded_params
        )
        signature = (row.baseline_sha256, expected)
        if name in seen and seen[name] != signature:
            raise ValueError(f"Conflicting residue replacements: {name}")
        seen[name] = signature
        if name not in residues:
            if allow_missing:
                installed = False
                continue
            raise ValueError(f"Missing replacement baseline residue {name}")
        rt = residues[name]
        actual = _local_identity(
            rt,
            _baseline_charges(charge_index, rt),
            cart.residue_params.get(name, cart.residue_params.get(rt.base_name)),
        )
        if actual not in (row.baseline_sha256, expected):
            raise ValueError(f"Residue baseline changed for {name}")
        installed &= actual == expected
    return installed


def install_replacements(parameter_database, rows, connections=(), *, atom_types=None):
    """Install explicit replacements atomically, checking their baseline.

    Reinstalling the same result is a no-op. Applying it to changed local
    chemistry/charges/bonded records raises instead of silently mixing fits.
    The input database is never modified.
    """
    from tmol.database import inject_residue_params

    installed = validate_replacements(parameter_database, rows)
    cart = parameter_database.scoring.cartbonded
    replacements = {}
    for record in connections:
        key = _connection_key(record)
        old = replacements.get(key)
        if old is not None and old != record:
            raise ValueError("Conflicting replacement connection parameters")
        replacements[key] = record
    for old in cart.connection_params:
        new = replacements.get(_connection_key(old))
        if new is not None and attr.evolve(old, provenance="") != attr.evolve(
            new, provenance=""
        ):
            raise ValueError("Existing connection parameters differ")
    if (
        installed
        and not atom_types
        and all(r in cart.connection_params for r in connections)
    ):
        return parameter_database
    retained = tuple(
        r for r in cart.connection_params if _connection_key(r) not in replacements
    )
    extended = inject_residue_params(
        parameter_database,
        [],
        atom_types=atom_types,
        partial_charges={r.residue_type.name: r.partial_charges for r in rows},
    )
    updates = {r.residue_type.name: r.residue_type for r in rows}
    return attr.evolve(
        extended,
        scoring=attr.evolve(
            extended.scoring,
            cartbonded=type(cart).from_cartres_dict(
                {
                    **cart.residue_params,
                    **{r.residue_type.name: r.cartbonded_params for r in rows},
                },
                (*retained, *replacements.values()),
            ),
        ),
        chemical=attr.evolve(
            extended.chemical,
            residues=tuple(
                updates.get(r.name, r) for r in parameter_database.chemical.residues
            ),
        ),
    )
