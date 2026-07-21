"""Dataset-driven manifest for the ligand-prep parity harness.

The parity regression is parametrized over a manifest rather than a hard-coded
molecule list. Each entry pairs a SMILES (and its protonated form) with a
Rosetta ``.params`` reference and, optionally, a prepared mol2. With no manifest
supplied the loader returns seed entries built from the committed
``designs.smi`` / ``ref{1,2}.params`` fixtures (SMILES path only, ``mol2``
unset); a supplied manifest is validated and its relative paths are resolved
against the manifest directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_DEFAULT_ROOT = Path(__file__).parent.parent / "tests" / "data" / "ligand_ground_truth"
# The dataset-driven parity set (DUD-80) lives under tests/data/ligand_test:
# ``ligand_ground_truth`` holds the Rosetta ground truth, ``ligand_tmol_generated``
# the tmol outputs saved for manual comparison. Kept local (git-ignored).
_DATASET_ROOT = (
    Path(__file__).parent.parent
    / "tests"
    / "data"
    / "ligand_test"
    / "ligand_ground_truth"
)


@dataclass(frozen=True)
class LigandParityEntry:
    """One molecule in the parity dataset.

    Attributes:
        name: Molecule identifier (matches the reference ``NAME`` record).
        input_smiles: The unprotonated input SMILES.
        expected_prot_smiles: The protonated SMILES (pins the dimorphite
            variant so the SMILES path is reproducible).
        params: Path to the Rosetta ``.params`` reference.
        mol2: Path to the prepared mol2, or ``None`` for SMILES-path-only seeds.
        charge_mode: Charge policy for preparation (default ``auto``).
        sample_proton_chi: Whether proton-chi sampling is enabled (default on).
        expected_unsupported_fields: Rosetta fields known to be unsupported by
            tmol for this molecule (e.g. ``CUT_BOND``), asserted absent-by-design.
    """

    name: str
    input_smiles: str
    expected_prot_smiles: str
    params: Path
    mol2: Optional[Path] = None
    charge_mode: str = "auto"
    sample_proton_chi: bool = True
    expected_unsupported_fields: tuple[str, ...] = ()

    @property
    def has_mol2(self) -> bool:
        """Whether this entry has a prepared mol2 (mol2-path contracts apply)."""
        return self.mol2 is not None


def _load_named_smiles(path: Path) -> dict[str, str]:
    """Load ``{name: smiles}`` from a ``<smiles><ws><name>`` file."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with open(path) as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 2:
                out[parts[1]] = parts[0]
    return out


def _seed_entries(root: Path) -> list[LigandParityEntry]:
    """Build SMILES-path-only seed entries from the committed fixtures.

    The ``ref{1,2}.mol2`` files are deliberately not used: they are orphans
    (a different molecule than ``ref{1,2}.params``), so seed entries leave
    ``mol2`` unset and exercise only the SMILES-path and serialization checks.

    The seed references carry ``PROTON_CHI`` records, so seed entries request
    proton-chi sampling (``sample_proton_chi=True``) to emit the matching
    samples — now also the production default.
    """
    inputs = _load_named_smiles(root / "designs.smi")
    protonated = _load_named_smiles(root / "designs.prot.smi")
    entries: list[LigandParityEntry] = []
    for name in sorted(inputs):
        params = root / f"{name}.params"
        if not params.exists():
            continue
        entries.append(
            LigandParityEntry(
                name=name,
                input_smiles=inputs[name],
                expected_prot_smiles=protonated.get(name, inputs[name]),
                params=params,
                mol2=None,
                sample_proton_chi=True,
            )
        )
    return entries


def _entry_from_record(record: dict, manifest_dir: Path) -> LigandParityEntry:
    """Build and validate one entry from a manifest JSON record."""
    name = record.get("name")
    input_smiles = record.get("input_smiles")
    expected_prot = record.get("expected_prot_smiles")
    if not name:
        raise ValueError(f"manifest entry missing 'name': {record!r}")
    if not input_smiles:
        raise ValueError(f"manifest entry '{name}' missing 'input_smiles'")
    if not expected_prot:
        raise ValueError(
            f"manifest entry '{name}' missing 'expected_prot_smiles' "
            "(required to pin the protonation state for the SMILES path)"
        )

    def _resolve(value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else manifest_dir / p

    params_raw = record.get("params")
    if not params_raw:
        raise ValueError(f"manifest entry '{name}' missing 'params'")
    params = _resolve(params_raw)
    if not params.exists():
        raise FileNotFoundError(
            f"manifest entry '{name}' params file not found: {params}"
        )

    mol2: Optional[Path] = None
    mol2_raw = record.get("mol2")
    if mol2_raw:
        mol2 = _resolve(mol2_raw)
        if not mol2.exists():
            raise FileNotFoundError(
                f"manifest entry '{name}' mol2 file not found: {mol2}"
            )

    unsupported = tuple(record.get("expected_unsupported_fields", ()) or ())
    return LigandParityEntry(
        name=name,
        input_smiles=input_smiles,
        expected_prot_smiles=expected_prot,
        params=params,
        mol2=mol2,
        charge_mode=record.get("charge_mode", "auto"),
        sample_proton_chi=bool(record.get("sample_proton_chi", True)),
        expected_unsupported_fields=unsupported,
    )


def load_parity_manifest(
    path: str | Path | None = None,
    *,
    root: str | Path | None = None,
) -> list[LigandParityEntry]:
    """Load parity entries from a manifest, or the committed seed fixtures.

    Args:
        path: Path to a JSON manifest (``{"molecules": [...]}`` or a bare list).
            When ``None``, seed entries are built from the committed
            ``designs.smi`` / ``ref{1,2}.params`` fixtures.
        root: Root directory for the seed fixtures (defaults to the committed
            ``ligand_ground_truth`` directory).

    Returns:
        A list of :class:`LigandParityEntry`.

    Raises:
        FileNotFoundError: If ``path`` (or a referenced params/mol2) is missing.
        ValueError: If a record is missing a required field.
    """
    if path is None:
        return _seed_entries(Path(root) if root is not None else _DEFAULT_ROOT)

    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"parity manifest not found: {manifest_path}")
    with open(manifest_path) as handle:
        data = json.load(handle)
    records = data["molecules"] if isinstance(data, dict) else data
    manifest_dir = manifest_path.parent
    return [_entry_from_record(rec, manifest_dir) for rec in records]


def default_dataset_manifest(root: str | Path | None = None) -> Path:
    """Return the conventional dataset manifest path.

    Defaults to ``tests/data/ligand_test/ligand_ground_truth/manifest.json``.
    """
    base = Path(root) if root is not None else _DATASET_ROOT
    return base / "manifest.json"
