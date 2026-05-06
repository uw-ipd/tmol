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
import math
from pathlib import Path
from typing import Any, Mapping

import cattr
import yaml

from tmol.database import ParameterDatabase
from tmol.database.chemical import RawResidueType, normalize_bond_tuples
from tmol.database.scoring.cartbonded import CartRes
from tmol.database.scoring.elec import PartialCharges

logger = logging.getLogger(__name__)


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


def _radians_to_deg_str(val: float) -> str:
    """Convert a radian float to a degree string for YAML output."""
    return f"{math.degrees(val):.6f} deg"


_OMIT_IF_EMPTY_FIELDS = ("torsions",)

_PROPERTIES_OMIT_IF_DEFAULT: dict[str, Any] = {}
_POLYMER_OMIT_IF_DEFAULT: dict[str, Any] = {}


def _unstructure_residue(rt: RawResidueType) -> dict[str, Any]:
    """Unstructure a RawResidueType to a YAML-friendly dict.

    Output matches the compact style Frank uses for ligand ``.tmol``
    files: flow-style atoms/bonds (one entry per line), icoor angles
    re-formatted as degree strings, and optional fields with default
    values omitted entirely so the output is byte-close to a
    hand-curated reference.
    """
    d = cattr.unstructure(rt)

    # Drop empty optional collections that Frank's references omit.
    for f in _OMIT_IF_EMPTY_FIELDS:
        if f in d and not d[f]:
            del d[f]

    # Trim default polymer/protonation fields when they match the
    # neutral non-polymer ligand defaults (the only setting the
    # pipeline currently emits).
    props = d.get("properties")
    if isinstance(props, dict):
        for k, default in _PROPERTIES_OMIT_IF_DEFAULT.items():
            if k in props and props[k] == default:
                del props[k]
        polymer = props.get("polymer")
        if isinstance(polymer, dict):
            for k, default in _POLYMER_OMIT_IF_DEFAULT.items():
                if k in polymer and polymer[k] == default:
                    del polymer[k]

    for ic in d.get("icoors", []):
        if isinstance(ic.get("phi"), (int, float)):
            ic["phi"] = _radians_to_deg_str(ic["phi"])
        if isinstance(ic.get("theta"), (int, float)):
            ic["theta"] = _radians_to_deg_str(ic["theta"])
        # Round bond distances to match Frank's 6-decimal convention.
        if isinstance(ic.get("d"), float):
            rounded = round(ic["d"], 6)
            ic["d"] = float(f"{rounded:g}")

    # Trim ``UnresolvedAtom`` defaults inside torsion entries and emit
    # them flow-style (``{atom: C3}``) — Frank's references keep only
    # the ``atom: <name>`` field.
    for tor in d.get("torsions", []):
        for k in ("a", "b", "c", "d"):
            ua = tor.get(k)
            if isinstance(ua, dict):
                if ua.get("connection") is None:
                    ua.pop("connection", None)
                if ua.get("bond_sep_from_conn") is None:
                    ua.pop("bond_sep_from_conn", None)
                tor[k] = _flow_atom(ua)
    return d


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
    return CartRes(
        length_parameters=(),
        angle_parameters=(),
        torsion_parameters=(),
        improper_parameters=(),
        hxltorsion_parameters=(),
    )


# --- YAML formatting helpers -------------------------------------------------
# The default `yaml.safe_dump` emits everything in block style, but Frank's
# hand-curated reference `.tmol` files use a hybrid layout: a block-style
# outer list (one entry per line) of flow-style entries (`{name: C1, ...}`).
# These helpers + `_CompactDumper` reproduce that style so writer output is
# byte-close to the reference, which keeps regression diffs readable and
# the injection-equivalence tests strict.


class _FlowList(list):
    """Marker subtype: yaml dumper emits this list in flow style ([...])."""


def _flow_list_representer(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


def _flow_dict_representer(dumper, data):
    return dumper.represent_mapping("tag:yaml.org,2002:map", data, flow_style=True)


class _CompactDumper(yaml.SafeDumper):
    pass


_CompactDumper.add_representer(_FlowList, _flow_list_representer)


def _flow_atom(d: dict[str, Any]) -> dict[str, Any]:
    """Mark an atom dict for flow-style emission."""
    out = dict(d)

    class _FlowDict(dict):
        pass

    _CompactDumper.add_representer(_FlowDict, _flow_dict_representer)
    return _FlowDict(out)


def _compactify_residue(d: dict[str, Any]) -> dict[str, Any]:
    """Block-style outer list with flow-style entries (matches Frank's tmol).

    Each atom / bond / icoor lives on its own line via the block-style
    outer ``-`` marker, but its fields are emitted on a single line via
    the flow-style dict / list. Mirrors Frank's hand-curated layout.
    """
    if "atoms" in d:
        d["atoms"] = [_flow_atom(a) for a in d["atoms"]]
    if "bonds" in d:
        # Sort bonds: SINGLE first (in input order), then non-SINGLE in
        # input order — matches Frank's reference layout.
        bonds = list(d["bonds"])
        single = [b for b in bonds if (b[2] if len(b) > 2 else "SINGLE") == "SINGLE"]
        other = [b for b in bonds if (b[2] if len(b) > 2 else "SINGLE") != "SINGLE"]
        d["bonds"] = [_FlowList(b) for b in single + other]
    if "icoors" in d:
        d["icoors"] = [_flow_atom(ic) for ic in d["icoors"]]
    return d


def write_params_file(
    path: str | Path,
    residue_types: list[RawResidueType],
    charges: Mapping[str, dict[str, float]],
    cartbonded: Mapping[str, CartRes],
) -> None:
    """Write prepared ligand data to a tmol params YAML file (nested shape).

    Output style matches Frank's reference ``.tmol`` files: flow-style
    atom/bond entries (one record per line) and omitted defaults so the
    file is byte-close to a hand-curated example.
    """
    charge_list = [
        _flow_atom(cattr.unstructure(PartialCharges(res=res, atom=atom, charge=charge)))
        for res, cmap in charges.items()
        for atom, charge in cmap.items()
    ]

    cartbonded_payload: dict[str, Any] = {}
    for k, v in cartbonded.items():
        cb = cattr.unstructure(v)
        for group_key in (
            "length_parameters",
            "angle_parameters",
            "torsion_parameters",
            "improper_parameters",
            "hxltorsion_parameters",
        ):
            if group_key in cb:
                cb[group_key] = _FlowList(_flow_atom(g) for g in cb[group_key])
        cartbonded_payload[k] = cb

    payload: dict[str, Any] = {
        "chemical": {
            "residues": [
                _compactify_residue(_unstructure_residue(r)) for r in residue_types
            ],
        },
        "elec": {
            "atom_charge_parameters": _FlowList(charge_list),
        },
        "cartbonded": {
            "residue_params": cartbonded_payload,
        },
    }

    with Path(path).open("w") as f:
        yaml.dump(
            payload, f, Dumper=_CompactDumper, sort_keys=False, default_flow_style=False
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
