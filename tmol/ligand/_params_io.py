"""Write ligand params files in tmol's ``.tmol`` format.

Single home for ligand params I/O. :func:`write_params_file` serializes a
:class:`~tmol.ligand._registry.LigandPreparation`; the tmol reader is
re-exported from :mod:`tmol.ligand._params_file`.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import cattr
import numpy as np
import yaml

from tmol.database.chemical import RawResidueType
from tmol.ligand._params_file import TMOL_FORMAT_VERSION
from tmol.database.scoring import (
    CartRes,
    PartialCharges,
)

if TYPE_CHECKING:
    from tmol.ligand._registry import LigandPreparation

logger = logging.getLogger(__name__)


# --- tmol .tmol YAML writer --------------------------------------------------
# Frank's hand-curated reference `.tmol` files use a hybrid layout: a
# block-style outer list (one entry per line) of flow-style entries
# (`{name: C1, ...}`). The helpers + `_CompactDumper` below reproduce that
# style so writer output is byte-close to the reference, which keeps regression
# diffs readable and the injection-equivalence tests strict.

_OMIT_IF_EMPTY_FIELDS = ("torsions",)


def _unstructure_residue(rt: RawResidueType) -> dict[str, Any]:
    """Unstructure a RawResidueType to a YAML-friendly dict.

    Keep numeric precision and order intact; formatting must not change the
    reconstructed residue or its derived kinematics.
    """
    d = cattr.unstructure(rt)

    # Drop empty optional collections that Frank's references omit.
    for f in _OMIT_IF_EMPTY_FIELDS:
        if f in d and not d[f]:
            del d[f]

    # Trim ``UnresolvedAtom`` defaults inside torsion entries and emit them
    # flow-style (``{atom: C3}``) — Frank's references keep only ``atom``.
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


class _FlowList(list):
    """Marker subtype: yaml dumper emits this list in flow style ([...])."""


def _flow_list_representer(dumper: Any, data: _FlowList) -> Any:
    """Represent a list in compact flow-style YAML."""
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


def _flow_dict_representer(dumper: Any, data: dict[str, Any]) -> Any:
    """Represent a dict in compact flow-style YAML."""
    return dumper.represent_mapping("tag:yaml.org,2002:map", data, flow_style=True)


class _CompactDumper(yaml.SafeDumper):
    """SafeDumper variant that emits ``_FlowList`` marker lists in flow style."""


_CompactDumper.add_representer(_FlowList, _flow_list_representer)


class _FlowDict(dict):
    """Marker subtype for compact, flow-style records."""


_CompactDumper.add_representer(_FlowDict, _flow_dict_representer)


def _np_scalar_representer(dumper: Any, data: Any) -> Any:
    """Represent numpy scalar types (np.str_, np.float64, ...) as native Python.

    Residue data coming from biotite/numpy arrays carries numpy scalar types
    (e.g. ``np.str_`` atom names) that PyYAML's SafeDumper cannot serialize.
    Coerce each to its native Python equivalent before emission.
    """
    return dumper.represent_data(data.item())


_CompactDumper.add_multi_representer(np.generic, _np_scalar_representer)


def _flow_atom(d: dict[str, Any]) -> dict[str, Any]:
    """Mark an atom dict for flow-style emission."""
    return _FlowDict(
        (key, value)
        for key, value in d.items()
        if key not in ("genbonded_type", "cartbonded_reference") or value is not None
    )


def _compactify_patch(d: dict[str, Any]) -> dict[str, Any]:
    """A patch record with its per-atom entries one to a line."""
    out = dict(d)
    for key in ("add_atoms", "add_atom_aliases", "modify_atoms", "icoors"):
        if out.get(key):
            out[key] = _FlowList(_flow_atom(entry) for entry in out[key])
        else:
            out.pop(key, None)
    if out.get("add_bonds"):
        out["add_bonds"] = _FlowList(_FlowList(b) for b in out["add_bonds"])
    for key in ("remove_atoms", "add_connections", "add_torsions", "add_chi_samples"):
        if not out.get(key):
            out.pop(key, None)
        elif key == "remove_atoms":
            out[key] = _FlowList(out[key])
    return out


def _compactify_residue(d: dict[str, Any]) -> dict[str, Any]:
    """Block-style outer list with flow-style entries (matches Frank's tmol).

    Each atom / bond / icoor lives on its own line via the block-style outer
    ``-`` marker, but its fields are emitted on a single line via the flow-style
    dict / list. Mirrors Frank's hand-curated layout.
    """
    if "atoms" in d:
        d["atoms"] = [_flow_atom(a) for a in d["atoms"]]
    if "bonds" in d:
        d["bonds"] = [_FlowList(b) for b in d["bonds"]]
    if "icoors" in d:
        d["icoors"] = [_flow_atom(ic) for ic in d["icoors"]]
    return d


def _write_tmol_params_file(
    path: str | Path,
    residue_types: list[RawResidueType],
    charges: Mapping[str, dict[str, float]],
    cartbonded: Mapping[str, CartRes],
    patches: "list | None" = None,
    connection_params: tuple = (),
    replacement_baselines: Mapping[str, str] | None = None,
    replacement_baseline_charges: Mapping[str, dict[str, float]] | None = None,
    replacement_baseline_cartbonded: Mapping[str, CartRes] | None = None,
    atom_type_elements: Mapping[str, str] | None = None,
) -> None:
    """Write prepared ligand data to a tmol params YAML (``.tmol``) file.

    Output style matches Frank's reference ``.tmol`` files: flow-style atom/bond
    entries (one record per line) and omitted defaults so the file is byte-close
    to a hand-curated example. Supports one or more residues per file.
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

    chemical: dict[str, Any] = {
        "residues": [
            _compactify_residue(_unstructure_residue(r)) for r in residue_types
        ],
    }
    if patches:
        chemical["adds_patches"] = [
            _compactify_patch(cattr.unstructure(patch)) for patch in patches
        ]
    if replacement_baselines:
        chemical["replacement_baselines"] = dict(replacement_baselines)
    if atom_type_elements:
        chemical["atom_type_elements"] = dict(atom_type_elements)

    payload: dict[str, Any] = {
        # One current version until the format settles; per-feature minimum
        # versions are not worth their complexity before 1.0.
        "version": TMOL_FORMAT_VERSION,
        "chemical": chemical,
        "elec": {
            "atom_charge_parameters": _FlowList(charge_list),
        },
        "cartbonded": {
            "residue_params": cartbonded_payload,
        },
    }
    if connection_params:
        payload["cartbonded"]["connection_params"] = [
            cattr.unstructure(record) for record in connection_params
        ]
    if replacement_baseline_charges:
        payload["elec"]["replacement_baseline_charges"] = dict(
            replacement_baseline_charges
        )
    if replacement_baseline_cartbonded:
        payload["cartbonded"]["replacement_baseline_params"] = {
            name: cattr.unstructure(record)
            for name, record in replacement_baseline_cartbonded.items()
        }

    with Path(path).open("w") as f:
        yaml.dump(
            payload, f, Dumper=_CompactDumper, sort_keys=False, default_flow_style=False
        )


def write_params_file(
    preparation: "LigandPreparation | list[LigandPreparation]",
    path: str | Path,
) -> None:
    """Write a ligand ``LigandPreparation`` as a tmol ``.tmol`` file.

    Args:
        preparation: A :class:`~tmol.ligand._registry.LigandPreparation` (its
            ``residue_type`` / ``partial_charges`` / ``cartbonded_params`` are
            used), or a list of them.
        path: Output file, holding every supplied residue.
    """
    preps = (
        list(preparation) if isinstance(preparation, (list, tuple)) else [preparation]
    )
    from tmol.ligand._registry import (
        _additional_cartbonded_params,
        _merge_partial_charges,
        _merge_named_parameters,
        _unique_preparations,
        _batch_atom_type_elements,
    )

    definitions = _unique_preparations(preps)
    charges = {p.residue_type.name: p.partial_charges for p in preps}
    cartbonded = {p.residue_type.name: p.cartbonded_params for p in preps}
    extra_cart = _additional_cartbonded_params(preps)
    cartbonded.update(extra_cart)
    shared_charges = _merge_partial_charges(
        p.variant_partial_charges or {} for p in preps
    )
    charges.update(shared_charges)
    replacements = [p for p in definitions if p.baseline_sha256 is not None]
    replacement_names = {p.residue_type.name for p in replacements}
    baseline_charges = {
        n: q for n, q in shared_charges.items() if n in replacement_names
    }
    baseline_cart = {n: c for n, c in extra_cart.items() if n in replacement_names}
    # Complete explicit replacements supersede old patch metadata in a
    # combined bundle, independently of preparation order.
    charges.update({p.residue_type.name: p.partial_charges for p in replacements})
    cartbonded.update({p.residue_type.name: p.cartbonded_params for p in replacements})
    _write_tmol_params_file(
        path,
        [p.residue_type for p in definitions],
        charges,
        cartbonded,
        patches=list(
            _merge_named_parameters(
                ((v.name, v) for p in preps for v in p.adds_patches),
                "patch definitions",
            ).values()
        ),
        connection_params=tuple(
            dict.fromkeys(record for p in preps for record in p.connection_params)
        ),
        replacement_baselines={
            p.residue_type.name: p.baseline_sha256 for p in replacements
        },
        replacement_baseline_charges=baseline_charges,
        replacement_baseline_cartbonded=baseline_cart,
        atom_type_elements=_batch_atom_type_elements(preps),
    )


def write_params_from_mol2(
    mol2_path: str | Path,
    out_path: str | Path,
    *,
    res_name: str | None = None,
    sample_proton_chi: bool = True,
) -> None:
    """Build params from a mol2 file and write a tmol ``.tmol`` file.

    Args:
        mol2_path: Input Tripos mol2 (names, coords, charges preserved verbatim).
        out_path: Output file path (see :func:`write_params_file`).
        res_name: Optional residue name override.
        sample_proton_chi: Whether to emit PROTON_CHI samples.
    """
    from tmol.ligand._detect import nonstandard_residue_info_from_mol2
    from tmol.ligand._preparation import prepare_single_ligand

    info = nonstandard_residue_info_from_mol2(mol2_path, res_name=res_name)
    prep = prepare_single_ligand(info, sample_proton_chi=sample_proton_chi)
    write_params_file(prep, out_path)
