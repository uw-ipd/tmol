"""Read and write ligand params files (Rosetta ``.params`` and tmol ``.tmol``).

Single home for ligand params I/O. :func:`write_params_file` serializes a
:class:`~tmol.ligand.registry.LigandPreparation` to either format; the Rosetta
reader :func:`read_params_file` and (re-exported) tmol reader cover the inputs.
"""

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import cattr
import numpy as np
import yaml

from tmol.database.chemical import (
    Atom,
    ChemicalProperties,
    ChiSamples,
    Icoor,
    PolymerProperties,
    ProtonationProperties,
    RawResidueType,
    Torsion,
    UnresolvedAtom,
)
from tmol.database.scoring.cartbonded import CartRes
from tmol.database.scoring.elec import PartialCharges
from tmol.ligand.params_file import TMOL_FORMAT_VERSION

if TYPE_CHECKING:
    from tmol.ligand.registry import LigandPreparation

logger = logging.getLogger(__name__)

_BOND_TOK_TO_TYPE = {
    "1": "SINGLE",
    "2": "DOUBLE",
    "3": "TRIPLE",
    "4": "AROMATIC",
    "SINGLE": "SINGLE",
    "DOUBLE": "DOUBLE",
    "TRIPLE": "TRIPLE",
    "AROMATIC": "AROMATIC",
    "ARO": "AROMATIC",
}


def _chi_number(name: str) -> int:
    """Extract the integer N from a chi torsion name like ``chi3`` (else 0)."""
    if name.startswith("chi") and name[3:].isdigit():
        return int(name[3:])
    return 0


def _write_rosetta_params_file(
    restype: RawResidueType,
    path: str | Path,
    partial_charges: dict[str, float] | None = None,
) -> None:
    """Write a RawResidueType as a Rosetta .params file.

    Args:
        restype: The residue type to write.
        path: Output file path.
        partial_charges: Optional per-atom partial charges. Keys are atom
            names matching restype.atoms[i].name.
    """
    lines: list[str] = []

    lines.append(f"NAME {restype.name}")
    lines.append(f"IO_STRING {restype.name3} Z")
    lines.append("TYPE LIGAND")
    lines.append("AA UNK")

    for atom in restype.atoms:
        charge = 0.0
        if partial_charges and atom.name in partial_charges:
            charge = partial_charges[atom.name]
        lines.append(f"ATOM {atom.name:4s} {atom.atom_type:4s} X {charge:8.4f}")

    for a, b, c, *rest in restype.bonds:
        is_ring = bool(rest[0]) if rest else False
        ring_tok = " RING" if is_ring else ""
        lines.append(f"BOND_TYPE {a:4s} {b:4s} {c:12s}{ring_tok}")

    if restype.default_jump_connection_atom:
        lines.append(f"NBR_ATOM {restype.default_jump_connection_atom}")

    lines.append("NBR_RADIUS 999.0")

    # ICOOR_INTERNAL comes BEFORE CHI / PROTON_CHI, matching Rosetta's
    # mol2genparams layout (NBR_RADIUS -> ICOOR_INTERNAL -> CHI -> PROTON_CHI).
    for ic in restype.icoors:
        phi_deg = math.degrees(ic.phi)
        theta_deg = math.degrees(ic.theta)
        lines.append(
            f"ICOOR_INTERNAL {ic.name:4s} {phi_deg:11.6f} {theta_deg:11.6f} "
            f"{ic.d:11.6f} {ic.parent:4s} {ic.grand_parent:4s} "
            f"{ic.great_grand_parent:4s}"
        )

    # CHI / PROTON_CHI rotatable-bond DOFs. One CHI line per named torsion;
    # PROTON_CHI lines carry the sample/expansion data for polar-hydrogen chis.
    # Rosetta-only annotations (e.g. a trailing "#biaryl" comment) are NOT
    # emitted — tmol keeps only the semantic content.
    proton_by_chi = {cs.chi_dihedral: cs for cs in restype.chi_samples}
    for tor in sorted(restype.torsions, key=lambda t: _chi_number(t.name)):
        n = _chi_number(tor.name)
        quad = " ".join(f"{(ua.atom or ''):>4s}" for ua in (tor.a, tor.b, tor.c, tor.d))
        lines.append(f"CHI {n:>2d} {quad}")
        cs = proton_by_chi.get(tor.name)
        if cs is not None:
            samples = " ".join(f"{s:g}" for s in cs.samples)
            if cs.expansions:
                extra = f"EXTRA {len(cs.expansions)} " + " ".join(
                    f"{e:g}" for e in cs.expansions
                )
            else:
                extra = "EXTRA 0"
            lines.append(f"PROTON_CHI {n} SAMPLES {len(cs.samples)} {samples} {extra}")

    with open(path, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")

    logger.info("Wrote params file for %s to %s", restype.name, path)


def read_params_file(path: str | Path) -> RawResidueType:
    """Read a Rosetta .params file into a RawResidueType.

    Parses ATOM, BOND, ICOOR_INTERNAL, and NBR_ATOM records. Other
    records are silently ignored.

    Args:
        path: Path to the .params file.

    Returns:
        A RawResidueType populated from the params file.
    """
    name = "UNK"
    name3 = "UNK"
    atoms: list[Atom] = []
    bonds: list[tuple[str, str, str, bool]] = []
    icoors: list[Icoor] = []
    nbr_atom = ""
    chi_torsions: dict[int, Torsion] = {}
    chi_proton: dict[int, ChiSamples] = {}

    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue

            if parts[0] == "NAME" and len(parts) >= 2:
                name = parts[1]
                name3 = name

            elif parts[0] == "ATOM" and len(parts) >= 4:
                atoms.append(Atom(name=parts[1], atom_type=parts[2]))

            elif parts[0] == "BOND" and len(parts) >= 3:
                bond_type = "SINGLE"
                if len(parts) >= 4:
                    bond_type = _BOND_TOK_TO_TYPE.get(parts[3].upper(), "SINGLE")
                bonds.append((parts[1], parts[2], bond_type, False))
            elif parts[0] == "BOND_TYPE" and len(parts) >= 4:
                order = _BOND_TOK_TO_TYPE.get(parts[3].upper(), "SINGLE")
                ring = len(parts) >= 5 and parts[4].upper() == "RING"
                bonds.append((parts[1], parts[2], order, ring))

            elif parts[0] == "CHI" and len(parts) >= 6:
                # "CHI n a b c d [#biaryl ...]" — trailing comments ignored.
                n = int(parts[1])
                chi_torsions[n] = Torsion(
                    name=f"chi{n}",
                    a=UnresolvedAtom(atom=parts[2]),
                    b=UnresolvedAtom(atom=parts[3]),
                    c=UnresolvedAtom(atom=parts[4]),
                    d=UnresolvedAtom(atom=parts[5]),
                )

            elif parts[0] == "PROTON_CHI" and "SAMPLES" in parts:
                # "PROTON_CHI n SAMPLES k v1..vk [EXTRA m e1..em]"
                n = int(parts[1])
                si = parts.index("SAMPLES")
                k = int(parts[si + 1])
                samples = tuple(float(x) for x in parts[si + 2 : si + 2 + k])
                expansions: tuple[float, ...] = ()
                if "EXTRA" in parts:
                    ei = parts.index("EXTRA")
                    n_extra = int(parts[ei + 1])
                    expansions = tuple(
                        float(x) for x in parts[ei + 2 : ei + 2 + n_extra]
                    )
                chi_proton[n] = ChiSamples(
                    chi_dihedral=f"chi{n}", samples=samples, expansions=expansions
                )

            elif parts[0] == "NBR_ATOM" and len(parts) >= 2:
                nbr_atom = parts[1]

            elif parts[0] == "ICOOR_INTERNAL" and len(parts) >= 8:
                icoors.append(
                    Icoor(
                        name=parts[1],
                        phi=math.radians(float(parts[2])),
                        theta=math.radians(float(parts[3])),
                        d=float(parts[4]),
                        parent=parts[5],
                        grand_parent=parts[6],
                        great_grand_parent=parts[7],
                    )
                )

    properties = ChemicalProperties(
        is_canonical=False,
        polymer=PolymerProperties(
            is_polymer=False,
            polymer_type="NA",
            backbone_type="NA",
            mainchain_atoms=None,
            sidechain_chirality="NA",
            termini_variants=(),
        ),
        chemical_modifications=(),
        connectivity=(),
        protonation=ProtonationProperties(
            protonated_atoms=(),
            protonation_state="neutral",
            pH=7,
        ),
        virtual=(),
    )

    return RawResidueType(
        name=name,
        base_name=name,
        name3=name3,
        io_equiv_class=name3,
        atoms=tuple(atoms),
        atom_aliases=(),
        bonds=tuple(bonds),
        connections=(),
        torsions=tuple(chi_torsions[n] for n in sorted(chi_torsions)),
        icoors=tuple(icoors),
        properties=properties,
        chi_samples=tuple(chi_proton[n] for n in sorted(chi_proton)),
        default_jump_connection_atom=nbr_atom or (atoms[0].name if atoms else ""),
    )


# --- tmol .tmol YAML writer --------------------------------------------------
# Frank's hand-curated reference `.tmol` files use a hybrid layout: a
# block-style outer list (one entry per line) of flow-style entries
# (`{name: C1, ...}`). The helpers + `_CompactDumper` below reproduce that
# style so writer output is byte-close to the reference, which keeps regression
# diffs readable and the injection-equivalence tests strict.

_OMIT_IF_EMPTY_FIELDS = ("torsions",)
_PROPERTIES_OMIT_IF_DEFAULT: dict[str, Any] = {}
_POLYMER_OMIT_IF_DEFAULT: dict[str, Any] = {}


def _radians_to_deg_str(val: float) -> str:
    """Format a radian angle as a ``"<deg> deg"`` string (Frank's convention)."""
    return f"{math.degrees(val):.6f} deg"


def _unstructure_residue(rt: RawResidueType) -> dict[str, Any]:
    """Unstructure a RawResidueType to a YAML-friendly dict.

    Output matches the compact style Frank uses for ligand ``.tmol`` files:
    flow-style atoms/bonds (one entry per line), icoor angles re-formatted as
    degree strings, and optional fields with default values omitted entirely so
    the output is byte-close to a hand-curated reference.
    """
    d = cattr.unstructure(rt)

    # Drop empty optional collections that Frank's references omit.
    for f in _OMIT_IF_EMPTY_FIELDS:
        if f in d and not d[f]:
            del d[f]

    # Trim default polymer/protonation fields when they match the neutral
    # non-polymer ligand defaults (the only setting the pipeline emits).
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
    out = dict(d)

    class _FlowDict(dict):
        pass

    _CompactDumper.add_representer(_FlowDict, _flow_dict_representer)
    return _FlowDict(out)


def _compactify_residue(d: dict[str, Any]) -> dict[str, Any]:
    """Block-style outer list with flow-style entries (matches Frank's tmol).

    Each atom / bond / icoor lives on its own line via the block-style outer
    ``-`` marker, but its fields are emitted on a single line via the flow-style
    dict / list. Mirrors Frank's hand-curated layout.
    """
    if "atoms" in d:
        d["atoms"] = [_flow_atom(a) for a in d["atoms"]]
    if "bonds" in d:
        # Sort bonds: SINGLE first (in input order), then non-SINGLE in input
        # order — matches Frank's reference layout.
        bonds = list(d["bonds"])
        single = [b for b in bonds if (b[2] if len(b) > 2 else "SINGLE") == "SINGLE"]
        other = [b for b in bonds if (b[2] if len(b) > 2 else "SINGLE") != "SINGLE"]
        d["bonds"] = [_FlowList(b) for b in single + other]
    if "icoors" in d:
        d["icoors"] = [_flow_atom(ic) for ic in d["icoors"]]
    return d


def _write_tmol_params_file(
    path: str | Path,
    residue_types: list[RawResidueType],
    charges: Mapping[str, dict[str, float]],
    cartbonded: Mapping[str, CartRes],
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

    payload: dict[str, Any] = {
        "version": TMOL_FORMAT_VERSION,
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


def write_params_file(
    preparation: "LigandPreparation | list[LigandPreparation]",
    path: str | Path,
    format: str = "rosetta",
) -> None:
    """Write a ligand ``LigandPreparation`` as a Rosetta ``.params`` or tmol ``.tmol``.

    Args:
        preparation: A :class:`~tmol.ligand.registry.LigandPreparation` (its
            ``residue_type`` / ``partial_charges`` / ``cartbonded_params`` are
            used), or a list of them.
        path: Output path. Its meaning depends on the format and whether a list
            was passed:

            * single preparation -> ``path`` is the output file (either format);
            * ``"rosetta"`` + list -> ``path`` is a **directory**; each
              preparation is written to ``<path>/<residue_type.name>.params``
              (a ``.params`` holds a single residue);
            * ``"tmol"`` (single or list) -> ``path`` is a single file holding
              all residues.
        format: ``"rosetta"`` (classic Rosetta ``.params``) or ``"tmol"``
            (tmol YAML ``.tmol``).
    """
    is_list = isinstance(preparation, (list, tuple))
    preps = list(preparation) if is_list else [preparation]
    fmt = str(format).lower()
    if fmt == "rosetta":
        if is_list:
            out_dir = Path(path)
            for prep in preps:
                _write_rosetta_params_file(
                    prep.residue_type,
                    out_dir / f"{prep.residue_type.name}.params",
                    prep.partial_charges,
                )
        else:
            prep = preps[0]
            _write_rosetta_params_file(prep.residue_type, path, prep.partial_charges)
    elif fmt == "tmol":
        _write_tmol_params_file(
            path,
            [p.residue_type for p in preps],
            {p.residue_type.name: p.partial_charges for p in preps},
            {p.residue_type.name: p.cartbonded_params for p in preps},
        )
    else:
        raise ValueError(f"unknown params format {format!r} (use 'rosetta' or 'tmol')")


def write_params_from_mol2(
    mol2_path: str | Path,
    out_path: str | Path,
    *,
    res_name: str | None = None,
    sample_proton_chi: bool = True,
    format: str = "rosetta",
) -> None:
    """Build params from a mol2 file and write Rosetta ``.params`` or tmol ``.tmol``.

    Args:
        mol2_path: Input Tripos mol2 (names, coords, charges preserved verbatim).
        out_path: Output file path (see :func:`write_params_file`).
        res_name: Optional residue name override.
        sample_proton_chi: Whether to emit PROTON_CHI samples.
        format: ``"rosetta"`` or ``"tmol"``.
    """
    from tmol.ligand.detect import nonstandard_residue_info_from_mol2
    from tmol.ligand.preparation import prepare_single_ligand

    info = nonstandard_residue_info_from_mol2(mol2_path, res_name=res_name)
    prep = prepare_single_ligand(info, sample_proton_chi=sample_proton_chi)
    write_params_file(prep, out_path, format=format)
