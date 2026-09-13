"""tmol YAML params file format for ligand residue types.

Provides load/write/inject functions for a unified YAML format that
bundles residue type definitions, cartbonded parameters, and electrostatic
charges in a single file.  The top-level shape mirrors ``ParameterDatabase``:

    version: "2.0"
    chemical:
      residues:
        - name: LIG
          base_name: LIG
          atoms: [...]
          bonds: [...]
          icoors: [...]
          properties: {...}
          # atom_aliases / chi_samples / default_jump_connection_atom optional
      # Patches for the residue and any canonical attachment partners.
      adds_patches:
        - name: LIG_CarboxyTerminus
          display_name: cterm
          applies_to: { base_names: [LIG] }
          ...
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
      connection_params: []  # optional complete ConnectionCartRes records

Each subsection's schema matches the corresponding canonical database
YAML so entries can be copy-pasted between params files and
``chemical.yaml`` / ``cartbonded.yaml`` / ``elec.yaml``.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cattr

from tmol.database import ParameterDatabase
from tmol.database._yaml import safe_load
from tmol.database.chemical import (
    RawResidueType,
    VariantType,
    normalize_bond_tuples,
)
from tmol.database.scoring import (
    CartRes,
    ConnectionCartRes,
    PartialCharges,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from tmol.ligand._registry import LigandPreparation


# Current .tmol format version.  Bump the major version on breaking
# schema changes; bump the minor version on backward-compatible additions.
# Writers choose the oldest supported major that preserves the bundle;
# this is the newest supported version. Every file is checked on load.
TMOL_FORMAT_VERSION: str = "5.0"

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
    "polymer_type": None,
    "backbone_type": None,
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


def _replacement_metadata(chem, elec, cart, residues, file_major):
    """Read guarded metadata separately from ordinary additions."""
    residue_names = {r.name for r in residues}
    replacement_baselines = chem.get("replacement_baselines") or {}
    if replacement_baselines:
        if file_major not in {"4", "5"}:
            raise ValueError(
                "Replacement baselines require .tmol format version 4 or later"
            )
        if (
            not isinstance(replacement_baselines, dict)
            or not set(replacement_baselines) <= residue_names
        ):
            raise ValueError("Replacement baselines must name residues in the bundle")
        if len(residue_names) != len(residues):
            raise ValueError("Replacement bundles require unique residue definitions")
        import re

        if any(
            not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v)
            for v in replacement_baselines.values()
        ):
            raise ValueError("Invalid replacement baseline_sha256")
    baseline_charges = elec.get("replacement_baseline_charges") or {}
    baseline_cart = cart.get("replacement_baseline_params") or {}
    for records in (baseline_charges, baseline_cart):
        if not isinstance(records, dict) or not set(records) <= set(
            replacement_baselines
        ):
            raise ValueError(
                "Replacement baseline parameters must name guarded residues"
            )
    if replacement_baselines and not set(replacement_baselines) <= set(
        cart.get("residue_params") or {}
    ):
        raise ValueError("Replacement requires an explicit complete bonded record")
    return replacement_baselines, baseline_charges, baseline_cart


def _element_metadata(chem, residues, file_major):
    from tmol.ligand._registry import _validate_atom_type_elements

    elements = chem.get("atom_type_elements")
    if elements is not None:
        if file_major != "5":
            raise ValueError("atom_type_elements require .tmol format version 5")
        elements = _validate_atom_type_elements(elements)

    if elements and not residues:
        raise ValueError(
            "A params bundle with atom_type_elements must define a residue"
        )
    return elements


def load_params_file(path: str | Path) -> list["LigandPreparation"]:
    """Load a tmol params YAML file as a list of ``LigandPreparation``.

    The returned list is the same abstraction the AtomArray pipeline
    produces (see :func:`tmol.ligand.prepare_single_ligand`), so the
    caller can pass it directly to
    :func:`tmol.ligand._registry.inject_ligand_preparations` regardless
    of which input form (file or AtomArray) it came from.

    The ``.tmol`` schema is the nested
    ``chemical:`` / ``elec:`` / ``cartbonded:`` shape — files using the
    legacy flat schema (top-level ``residues:`` etc.) raise a
    ``ValueError`` pointing at the migration.
    """
    from tmol.ligand._registry import (
        LigandPreparation,
        _charges_from_rows,
    )

    path = Path(path)
    with path.open() as f:
        raw = safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping at YAML root, got {type(raw).__name__}")

    # --- .tmol format version check ---
    file_version = raw.get("version")
    if file_version is None:
        logger.warning(
            "%s: no 'version' field found; assuming format %s "
            "(consider regenerating this file with the current writer)",
            path,
            TMOL_FORMAT_VERSION,
        )
        raise ValueError(
            f"{path}: no 'version' field found for .tmol file. "
            f"Current format version is {TMOL_FORMAT_VERSION}. "
            f"Regenerate the file with the current version."
        )
    else:
        file_version = str(file_version)
        # Read legacy single-residue bundles as well as complete conjugates.
        # v2 adds shared partner patches and connection parameters; v3 adds
        # per-atom generic bonded references; v4 adds guarded exact-residue
        # replacements; v5 preserves atom-type element declarations.
        # Old readers must reject fields
        # they would otherwise silently drop.
        file_major = file_version.split(".")[0]
        if file_major not in {"1", "2", "3", "4", "5"}:
            raise ValueError(
                f"{path}: .tmol format version {file_version} is incompatible "
                f"with the current format version {TMOL_FORMAT_VERSION}. "
                f"Regenerate the file with the current writer."
            )
        if file_version != TMOL_FORMAT_VERSION:
            logger.info(
                "%s: .tmol format version %s differs from current %s "
                "(supported compatibility version)",
                path,
                file_version,
                TMOL_FORMAT_VERSION,
            )

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
    elements = _element_metadata(chem, residues, file_major)

    # Bundle-wide additions need not target a residue defined in this file:
    # a glycan, for example, brings a patch for its canonical ASN partner.
    # Carry patches once, in file order. Assigning them to individual residue
    # owners would reorder patches when a bundle's residue order changes.
    residue_names = {r.name for r in residues}
    replacement_baselines, baseline_charges, baseline_cart = _replacement_metadata(
        chem, elec, cart, residues, file_major
    )
    patches = []
    for item in chem.get("adds_patches") or []:
        patch = cattr.structure(_fill_patch_defaults(item), VariantType)
        if not residues:
            raise ValueError("A params bundle with patches must define a residue")
        patches.append(patch)

    connections = tuple(
        cattr.structure(item, ConnectionCartRes)
        for item in cart.get("connection_params") or ()
    )
    if connections and not residues:
        raise ValueError("A params bundle with connections must define a residue")

    cb_raw = cart.get("residue_params") or {}
    cart_by_res = {
        str(name): cattr.structure(payload, CartRes) for name, payload in cb_raw.items()
    }
    additional_cart = {
        name: params
        for name, params in cart_by_res.items()
        if name not in residue_names
    }
    additional_cart.update(
        {
            str(name): cattr.structure(payload, CartRes)
            for name, payload in baseline_cart.items()
        }
    )
    if additional_cart and not residues:
        raise ValueError("A params bundle with bonded parameters must define a residue")

    charge_rows = (
        cattr.structure(item, PartialCharges)
        for item in elec.get("atom_charge_parameters") or ()
    )
    charges_by_res = _charges_from_rows(
        (pc.res, pc.atom, pc.charge) for pc in charge_rows
    )

    variant_charges_by_res: dict[str, dict[str, dict[str, float]]] = {}
    if baseline_charges:
        variant_charges_by_res[residues[0].name] = baseline_charges
    for name, atom_charges in charges_by_res.items():
        base_name = name.partition(":")[0]
        if name not in residue_names:
            if not residues:
                raise ValueError("A params bundle with charges must define a residue")
            owner = base_name if base_name in residue_names else residues[0].name
            variant_charges_by_res.setdefault(owner, {})[name] = atom_charges

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
                atom_type_elements=(elements or None) if not preps else None,
                adds_patches=tuple(patches) if not preps else (),
                variant_partial_charges=variant_charges_by_res.get(rt.name) or None,
                connection_params=connections if not preps else (),
                additional_cartbonded_params=(
                    (additional_cart or None) if not preps else None
                ),
                baseline_sha256=replacement_baselines.get(rt.name),
            )
        )
    return preps


_VARIANT_DEFAULTS: dict[str, Any] = {
    "remove_atoms": [],
    "add_atoms": [],
    "add_atom_aliases": [],
    "modify_atoms": [],
    "add_connections": [],
    "add_bonds": [],
    "icoors": [],
    "add_torsions": [],
    "add_chi_samples": [],
    "applies_to": {},
}


def _fill_patch_defaults(item: dict[str, Any]) -> dict[str, Any]:
    """Fill a patch record's optional collections so a terse file still loads."""
    filled = {**_VARIANT_DEFAULTS, **item}
    filled["add_bonds"] = [list(b) for b in filled["add_bonds"]]
    return filled


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
    from tmol.ligand._registry import inject_ligand_preparations

    return inject_ligand_preparations(
        param_db, load_params_file(path), strict_atom_types=strict_atom_types
    )


def _load_params_files(paths):
    """Read each supplied path once within a batch, preserving source order."""
    return [
        prep
        for path in dict.fromkeys(Path(p) for p in paths)
        for prep in load_params_file(path)
    ]


def inject_params_files(
    param_db: ParameterDatabase,
    paths: list[str | Path],
    *,
    strict_atom_types: bool = False,
) -> ParameterDatabase:
    """Load multiple ``.tmol`` files and inject them in one shot."""
    from tmol.ligand._registry import inject_ligand_preparations

    return inject_ligand_preparations(
        param_db, _load_params_files(paths), strict_atom_types=strict_atom_types
    )
