"""tmol YAML params file format for ligand residue types.

Provides load/write/inject functions for a unified YAML format that
bundles residue type definitions, cartbonded parameters, and electrostatic
charges in a single file.  The format matches the existing tmol database
YAML schemas so entries can be copy-pasted between params files and the
main database YAMLs.

Example YAML structure::

    residues:
      - name: LIG
        base_name: LIG
        ...  # same schema as chemical.yaml residue entries

    residue_params:
      LIG:
        length_parameters: [...]
        angle_parameters: [...]
        ...  # same schema as cartbonded.yaml

    atom_charge_parameters:
      - {res: LIG, atom: C1, charge: 0.123}
      ...  # same schema as elec.yaml
"""

import logging
import math
from pathlib import Path
from typing import Any, Mapping

import attr
import cattr
import yaml

from tmol.database import ParameterDatabase, inject_residue_params
from tmol.database.chemical import RawResidueType, normalize_bond_tuples
from tmol.database.scoring.cartbonded import CartRes
from tmol.database.scoring.elec import PartialCharges
from tmol.ligand.registry import _collect_new_atom_types

logger = logging.getLogger(__name__)


def _radians_to_deg_str(val: float) -> str:
    """Convert a radian float to a degree string for YAML output."""
    return f"{math.degrees(val):.6f} deg"


def _unstructure_residue(rt: RawResidueType) -> dict[str, Any]:
    """Unstructure a RawResidueType to a YAML-friendly dict.

    Converts radian angles in icoors back to degree strings for
    human-readable output matching chemical.yaml format.
    """
    d = cattr.unstructure(rt)
    for ic in d.get("icoors", []):
        if isinstance(ic.get("phi"), (int, float)):
            ic["phi"] = _radians_to_deg_str(ic["phi"])
        if isinstance(ic.get("theta"), (int, float)):
            ic["theta"] = _radians_to_deg_str(ic["theta"])
    return d


def load_params_file(path: str | Path) -> dict[str, Any]:
    """Load a tmol params YAML file.

    Returns:
        Dict with keys ``residues`` (tuple of RawResidueType),
        ``residue_params`` (dict of str -> CartRes),
        ``atom_charge_parameters`` (tuple of PartialCharges).
    """
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping at YAML root, got {type(raw).__name__}")

    res_list = raw.get("residues") or []
    normalize_bond_tuples({"residues": res_list})
    residues = tuple(cattr.structure(item, RawResidueType) for item in res_list)

    cb_raw = raw.get("residue_params") or {}
    residue_params = {
        str(name): cattr.structure(payload, CartRes) for name, payload in cb_raw.items()
    }

    ec_raw = raw.get("atom_charge_parameters") or []
    atom_charge_parameters = tuple(
        cattr.structure(item, PartialCharges) for item in ec_raw
    )

    return {
        "residues": residues,
        "residue_params": residue_params,
        "atom_charge_parameters": atom_charge_parameters,
    }


def write_params_file(
    path: str | Path,
    residue_types: list[RawResidueType],
    charges: Mapping[str, dict[str, float]],
    cartbonded: Mapping[str, CartRes],
) -> None:
    """Write prepared ligand data to a tmol params YAML file.

    Args:
        path: Output file path.
        residue_types: Residue types to include.
        charges: Per-residue charge dicts ``{res_name: {atom: charge}}``.
        cartbonded: Per-residue CartRes ``{res_name: CartRes}``.
    """
    charge_list = [
        cattr.unstructure(PartialCharges(res=res, atom=atom, charge=charge))
        for res, cmap in charges.items()
        for atom, charge in cmap.items()
    ]

    payload: dict[str, Any] = {
        "residues": [_unstructure_residue(r) for r in residue_types],
        "residue_params": {k: cattr.unstructure(v) for k, v in cartbonded.items()},
        "atom_charge_parameters": charge_list,
    }

    with Path(path).open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)


def inject_params_file(
    param_db: ParameterDatabase,
    path: str | Path,
    *,
    strict_atom_types: bool = False,
) -> ParameterDatabase:
    """Load a params file and inject into a ParameterDatabase.

    Args:
        param_db: Base database (not modified).
        path: Path to tmol params YAML file.
        strict_atom_types: If True, raise on unknown atom type elements.

    Returns:
        New ParameterDatabase with params file data injected.
    """
    data = load_params_file(path)
    return _inject_params_data(param_db, data, strict_atom_types=strict_atom_types)


def inject_params_files(
    param_db: ParameterDatabase,
    paths: list[str | Path],
    *,
    strict_atom_types: bool = False,
) -> ParameterDatabase:
    """Load multiple params files and inject all at once.

    Merges data from all files before injecting, so only one new
    ParameterDatabase is created regardless of file count.
    """
    all_residues = []
    all_residue_params: dict[str, CartRes] = {}
    all_charges: list[PartialCharges] = []

    for path in paths:
        data = load_params_file(path)
        all_residues.extend(data["residues"])
        all_residue_params.update(data["residue_params"])
        all_charges.extend(data["atom_charge_parameters"])

    merged = {
        "residues": tuple(all_residues),
        "residue_params": all_residue_params,
        "atom_charge_parameters": tuple(all_charges),
    }
    return _inject_params_data(param_db, merged, strict_atom_types=strict_atom_types)


def _inject_params_data(
    param_db: ParameterDatabase,
    data: dict[str, Any],
    *,
    strict_atom_types: bool = False,
) -> ParameterDatabase:
    """Inject parsed params data into a ParameterDatabase."""
    residues = data["residues"]
    residue_params = data["residue_params"]
    charges = data["atom_charge_parameters"]

    existing_names = {r.name for r in param_db.chemical.residues}
    new_residues = [r for r in residues if r.name not in existing_names]

    if not new_residues:
        return param_db

    all_atom_types = []
    for rt in new_residues:
        new_ats = _collect_new_atom_types(
            param_db.chemical,
            rt,
            atom_type_elements=None,
            strict_atom_types=strict_atom_types,
        )
        all_atom_types.extend(new_ats)

    charges_by_res: dict[str, dict[str, float]] = {}
    for pc in charges:
        charges_by_res.setdefault(pc.res, {})[pc.atom] = pc.charge

    return inject_residue_params(
        param_db,
        residue_types=new_residues,
        atom_types=all_atom_types or None,
        partial_charges=charges_by_res or None,
        cartbonded_params=residue_params or None,
    )
