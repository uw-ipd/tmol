"""tmol YAML params file format for ligand residue types.

Provides load/write/inject functions for a unified YAML format that
bundles residue type definitions, cartbonded parameters, and electrostatic
charges in a single file.  The top-level shape mirrors ``ParameterDatabase``:

    chemical:
      residues:
        - name: LIG
          base_name: LIG
          atoms: [...]
          bonds: [...]
          icoors: [...]
          properties: {...}
          # atom_aliases / chi_samples / default_jump_connection_atom optional
    elec:
      atom_charge_parameters:
        - {res: LIG, atom: C1, charge: 0.123}
    cartbonded:
      residue_params:
        LIG:
          length_parameters: [...]
          angle_parameters: [...]
          torsion_parameters: [...]
          improper_parameters: [...]
          hxltorsion_parameters: []

Each subsection's schema matches the corresponding canonical database
YAML so entries can be copy-pasted between params files and
``chemical.yaml`` / ``cartbonded.yaml`` / ``elec.yaml``.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cattr
import yaml

from tmol.database import ParameterDatabase
from tmol.database.chemical import RawResidueType, normalize_bond_tuples
from tmol.database.scoring.cartbonded import CartRes
from tmol.database.scoring.elec import PartialCharges

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from tmol.ligand.registry import LigandPreparation


_RAW_RESIDUE_DEFAULTS: dict[str, Any] = {
    "atom_aliases": [],
    "chi_samples": [],
    "default_jump_connection_atom": "",
    # The ligand pipeline writes empty `torsions` (gen_bonded handles
    # ligand torsions); supply a default so files written without the
    # key still load via cattr.
    "torsions": [],
}

_POLYMER_PROPERTIES_DEFAULTS: dict[str, Any] = {
    "is_polymer": False,
    "polymer_type": "",
    "backbone_type": "",
    "mainchain_atoms": None,
    "sidechain_chirality": "",
    "termini_variants": [],
}

_PROTONATION_PROPERTIES_DEFAULTS: dict[str, Any] = {
    "protonated_atoms": [],
    "protonation_state": "neutral",
    "pH": 7,
}

_CHEMICAL_PROPERTIES_DEFAULTS: dict[str, Any] = {
    "is_canonical": False,
    "chemical_modifications": [],
    "connectivity": [],
    "virtual": [],
}


def _fill_properties_defaults(props: dict[str, Any]) -> dict[str, Any]:
    """Fill missing fields in a residue ``properties`` dict with defaults."""
    polymer = {**_POLYMER_PROPERTIES_DEFAULTS, **(props.get("polymer") or {})}
    protonation = {
        **_PROTONATION_PROPERTIES_DEFAULTS,
        **(props.get("protonation") or {}),
    }
    return {
        **_CHEMICAL_PROPERTIES_DEFAULTS,
        **props,
        "polymer": polymer,
        "protonation": protonation,
    }


def _structure_residue(item: dict[str, Any]) -> RawResidueType:
    """Apply defaults for optional fields, then structure into RawResidueType."""
    populated = {**_RAW_RESIDUE_DEFAULTS, **item}
    if "properties" in populated:
        populated["properties"] = _fill_properties_defaults(populated["properties"])
    return cattr.structure(populated, RawResidueType)


def load_params_file(path: str | Path) -> list["LigandPreparation"]:
    """Load a tmol params YAML file as a list of ``LigandPreparation``.

    The returned list is the same abstraction the AtomArray pipeline
    produces (see :func:`tmol.ligand.prepare_single_ligand`), so the
    caller can pass it directly to
    :func:`tmol.ligand.registry.inject_ligand_preparations` regardless
    of which input form (file or AtomArray) it came from.

    The ``.tmol`` schema is the nested
    ``chemical:`` / ``elec:`` / ``cartbonded:`` shape — files using the
    legacy flat schema (top-level ``residues:`` etc.) raise a
    ``ValueError`` pointing at the migration.
    """
    from tmol.ligand.registry import LigandPreparation

    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping at YAML root, got {type(raw).__name__}")

    if "chemical" not in raw and (
        "residues" in raw or "residue_params" in raw or "atom_charge_parameters" in raw
    ):
        raise ValueError(
            f"{path}: top-level keys 'residues'/'residue_params'/"
            "'atom_charge_parameters' indicate the deprecated flat schema. "
            "Migrate to the nested schema with 'chemical:', 'elec:', and "
            "'cartbonded:' top-level keys."
        )

    chem = raw.get("chemical") or {}
    elec = raw.get("elec") or {}
    cart = raw.get("cartbonded") or {}

    res_list = chem.get("residues") or []
    normalize_bond_tuples({"residues": res_list})
    residues = [_structure_residue(item) for item in res_list]

    cb_raw = cart.get("residue_params") or {}
    cart_by_res = {
        str(name): cattr.structure(payload, CartRes) for name, payload in cb_raw.items()
    }

    charges_by_res: dict[str, dict[str, float]] = {}
    for item in elec.get("atom_charge_parameters") or []:
        pc = cattr.structure(item, PartialCharges)
        charges_by_res.setdefault(pc.res, {})[pc.atom] = pc.charge

    preps = []
    for rt in residues:
        charges = charges_by_res.get(rt.name, {})
        if not charges:
            logger.warning(
                "%s: no elec charges in %s -- all partial charges will be 0.0",
                rt.name,
                path,
            )
        preps.append(
            LigandPreparation(
                residue_type=rt,
                partial_charges=charges,
                cartbonded_params=cart_by_res.get(rt.name, _empty_cartres()),
                atom_type_elements=None,
            )
        )
    return preps


def _empty_cartres() -> CartRes:
    """Construct an empty ``CartRes`` container.

    Returns:
        ``CartRes`` with all parameter collections initialized empty.
    """
    return CartRes(
        length_parameters=(),
        angle_parameters=(),
        torsion_parameters=(),
        improper_parameters=(),
        hxltorsion_parameters=(),
    )


def inject_params_file(
    param_db: ParameterDatabase,
    path: str | Path,
    *,
    strict_atom_types: bool = False,
) -> ParameterDatabase:
    """Load a single ``.tmol`` file and inject it into a ParameterDatabase."""
    from tmol.ligand.registry import inject_ligand_preparations

    return inject_ligand_preparations(
        param_db, load_params_file(path), strict_atom_types=strict_atom_types
    )


def inject_params_files(
    param_db: ParameterDatabase,
    paths: list[str | Path],
    *,
    strict_atom_types: bool = False,
) -> ParameterDatabase:
    """Load multiple ``.tmol`` files and inject them in one shot."""
    from tmol.ligand.registry import inject_ligand_preparations

    preps: list = []
    for path in paths:
        preps.extend(load_params_file(path))
    return inject_ligand_preparations(
        param_db, preps, strict_atom_types=strict_atom_types
    )
