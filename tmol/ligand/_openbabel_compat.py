"""OpenBabel fallback helpers for RDKit-fragile input formats.

RDKit's mol2 / SMILES / PDB parsers reject many real-world ligand files
that OpenBabel handles fine. The helpers in this module call OpenBabel
to read the input, then re-emit it as an SDF mol-block string that RDKit
can ingest. The result is an ``rdkit.Chem.Mol`` indistinguishable from
one produced by a successful RDKit parse, so downstream code (atom
typing, residue building, scoring) needs no further changes.

OpenBabel is a *soft* dependency. Import happens inside each helper, so
the rest of ``tmol.ligand`` loads cleanly on systems without
``openbabel-wheel``. The helpers raise a single descriptive error if a
caller invokes them without OB installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from rdkit import Chem

logger = logging.getLogger(__name__)


class OpenBabelUnavailableError(RuntimeError):
    """Raised when an OB-fallback helper is called but ``openbabel`` is missing."""


# Non-tetrahedral stereo classes. RDKit emits these for hypervalent centers
# (e.g. "[P@TB17]" on a lambda-5 phosphorane), but OpenBabel's SMILES parser
# does not support them and rejects the whole string.
_NONTETRAHEDRAL_CHIRAL_TAGS = tuple(
    tag
    for tag in (
        getattr(Chem.ChiralType, "CHI_TRIGONALBIPYRAMIDAL", None),
        getattr(Chem.ChiralType, "CHI_SQUAREPLANAR", None),
        getattr(Chem.ChiralType, "CHI_OCTAHEDRAL", None),
    )
    if tag is not None
)


def strip_nontetrahedral_stereo(smiles: str) -> str:
    """Drop stereo descriptors OpenBabel cannot parse, keeping the rest.

    Returns ``smiles`` unchanged if it has no such markers or cannot be parsed.
    """
    if not _NONTETRAHEDRAL_CHIRAL_TAGS:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    stripped = [
        atom
        for atom in mol.GetAtoms()
        if atom.GetChiralTag() in _NONTETRAHEDRAL_CHIRAL_TAGS
    ]
    if not stripped:
        return smiles
    for atom in stripped:
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    out = Chem.MolToSmiles(mol)
    logger.info(
        "stripped non-tetrahedral stereo from %d center(s) for OpenBabel: %s -> %s",
        len(stripped),
        smiles,
        out,
    )
    return out


def normalize_azide(smiles: str) -> str:
    """Rewrite the neutral cumulated azide ``N=[N]=N`` (CACTVS notation) to the
    charged form ``N=[N+]=[N-]`` that RDKit can sanitize."""
    return smiles.replace("N=[N]=N", "N=[N+]=[N-]")


def source_atom_order_from_mapped_smiles(smiles: str) -> Optional[tuple[int, ...]]:
    """Heavy-atom source indices in SMILES/mol2 order, or None if one is unmapped.

    Explicit input hydrogens are intentionally ignored: OpenBabel regenerates
    hydrogens after protonation, while only heavy atoms retain source names.
    """
    smiles = normalize_azide(strip_nontetrahedral_stereo(smiles))
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    order: list[int] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        mapnum = atom.GetAtomMapNum()
        if mapnum <= 0:
            return None
        order.append(mapnum - 1)
    return tuple(order)


def _import_openbabel() -> tuple:
    """Return ``(openbabel, pybel)`` modules, or raise a clear error.

    Imported lazily so that ``tmol.ligand`` can be loaded without the
    optional ``openbabel-wheel`` dependency.
    """
    try:
        from openbabel import openbabel, pybel  # type: ignore[import-not-found]
    except ImportError as exc:
        raise OpenBabelUnavailableError(
            "OpenBabel fallback was requested but the 'openbabel' Python "
            "package is not installed. Install with `pip install openbabel-wheel`."
        ) from exc
    return openbabel, pybel


def _obmol_to_rdkit_mol(obmol, *, sanitize: bool = False) -> Optional[Chem.Mol]:
    """Convert an OpenBabel ``OBMol`` into an RDKit ``Chem.Mol``.

    Goes through a SDF / V2000 mol-block string. This preserves bond
    orders, formal charges, and 3D coordinates; aromaticity is recomputed
    by RDKit on sanitize.

    Args:
        obmol: An OpenBabel ``OBMol`` populated with atoms, bonds, and
            (usually) 3D coordinates.
        sanitize: If True, sanitize the resulting RDKit Mol.

    Returns:
        An RDKit ``Chem.Mol``, or ``None`` if the round-trip failed.
    """
    openbabel, _ = _import_openbabel()

    conv = openbabel.OBConversion()
    conv.SetOutFormat("sdf")
    sdf = conv.WriteString(obmol)
    if not sdf:
        return None

    mol = Chem.MolFromMolBlock(sdf, sanitize=sanitize, removeHs=False)
    return mol


def obabel_read_mol2(path: str | Path) -> Optional[Chem.Mol]:
    """Read a TRIPOS mol2 file via OpenBabel and return an RDKit ``Chem.Mol``.

    Use as a fallback when ``Chem.MolFromMol2File`` returns ``None`` on
    valid mol2 files that RDKit's parser rejects.

    Returns ``None`` if OB could not parse the file. Raises
    :class:`OpenBabelUnavailableError` if OpenBabel is not installed.
    """
    openbabel, pybel = _import_openbabel()
    path = Path(path)
    try:
        pymol = next(pybel.readfile("mol2", str(path)))
    except StopIteration:
        logger.warning("OpenBabel could not read any molecule from %s", path)
        return None
    except Exception:
        logger.warning("OpenBabel failed to parse mol2 file %s", path, exc_info=True)
        return None
    return _obmol_to_rdkit_mol(pymol.OBMol)


def obabel_read_mol2_block(mol2_block: str) -> Optional[Chem.Mol]:
    """Read a TRIPOS mol2 *string* via OpenBabel and return an RDKit ``Chem.Mol``.

    In-memory analogue of :func:`obabel_read_mol2` — use as a fallback when
    ``Chem.MolFromMol2Block`` returns ``None``. Returns ``None`` if OB could not
    parse the block. Raises :class:`OpenBabelUnavailableError` if OB is missing.
    """
    _, pybel = _import_openbabel()
    try:
        pymol = pybel.readstring("mol2", mol2_block)
    except Exception:
        logger.warning("OpenBabel failed to parse in-memory mol2 block", exc_info=True)
        return None
    return _obmol_to_rdkit_mol(pymol.OBMol)


# mmff94 fails on pentavalent phosphorous.
# fallback to eem in these cases
_CHARGE_MODEL_FALLBACKS = ("eem",)


def _compute_charges_with_fallback(openbabel, pymol, forcefield: str, smiles: str):
    """Compute charges for molecule, falling back on failure.

    Returns the name of the model actually used.

    Raises:
        ValueError: If neither the primary model nor any fallback succeeds.
    """
    primary = openbabel.OBChargeModel.FindType(forcefield)
    if primary is not None and primary.ComputeCharges(pymol.OBMol):
        return forcefield

    for name in _CHARGE_MODEL_FALLBACKS:
        model = openbabel.OBChargeModel.FindType(name)
        if model is None or not model.ComputeCharges(pymol.OBMol):
            continue
        logger.warning(
            "%s partial charges are unavailable for SMILES %r; falling back to "
            "%r. These charges come from a different model and are not directly "
            "comparable to %s charges.",
            forcefield,
            smiles,
            name,
            forcefield,
        )
        return name

    raise ValueError(
        f"OpenBabel could not compute {forcefield} partial charges for "
        f"SMILES {smiles!r}, and no fallback model "
        f"({', '.join(_CHARGE_MODEL_FALLBACKS)}) could parameterize it either"
    )


def _build_charged_3d_mol2_mol(
    smiles: str,
    *,
    forcefield: str = "mmff94",
    minimize_steps: int = 50,
    seed: Optional[int] = None,
):
    """Parse a SMILES, 3D-embed, charge, and rename atoms; return the pybel mol.

    Shared core of :func:`obabel_smiles_to_mol2` (file) and
    :func:`obabel_smiles_to_mol2_block` (string). Coordinates come from the
    distance-geometry builder (:func:`tmol.ligand._conformer_generation.
    generate_conformer`); this then computes MMFF94 partial charges (so the mol2
    ``charge_type`` is normally ``MMFF94_CHARGES``) and assigns generic atom
    names. MMFF94 charges are topological, so the conformer does not affect them.

    If the named model cannot parameterize the molecule (e.g. MMFF94 has no types
    for a lambda-5 phosphorane), charges come from a fallback model instead — see
    :func:`_compute_charges_with_fallback`, which warns loudly.

    Args:
        minimize_steps: OpenBabel conjugate-gradient cleanup steps.
        seed: Fixed RNG seed for reproducible coordinates; ``None`` is random.

    Raises:
        OpenBabelUnavailableError: If the ``openbabel`` package is not installed.
        ValueError: If RDKit rejects the chemistry, 3D generation fails, or no
            partial charges can be computed.
    """
    from tmol.ligand._conformer_generation import generate_conformer

    openbabel, _ = _import_openbabel()
    smiles = normalize_azide(strip_nontetrahedral_stereo(smiles))
    try:
        pymol = generate_conformer(smiles, minimize_steps=minimize_steps, seed=seed)
    except Exception as exc:
        raise ValueError(
            f"failed to generate 3D coordinates for SMILES {smiles!r}"
        ) from exc
    _compute_charges_with_fallback(openbabel, pymol, forcefield, smiles)
    _assign_generic_atom_names(openbabel, pymol.OBMol)
    return pymol


def obabel_smiles_to_mol2_block(
    smiles: str,
    *,
    forcefield: str = "mmff94",
    minimize_steps: int = 50,
    seed: Optional[int] = None,
) -> str:
    """Return a 3D MMFF94 mol2 as an in-memory TRIPOS string (no disk I/O).

    Preferred over :func:`obabel_smiles_to_mol2` for high-throughput batches
    (e.g. millions of SMILES): the mol2 is handed downstream as a string rather
    than written to and re-read from a temp file. See
    :func:`_build_charged_3d_mol2_mol` for the protocol and raised errors.
    """
    pymol = _build_charged_3d_mol2_mol(
        smiles, forcefield=forcefield, minimize_steps=minimize_steps, seed=seed
    )
    return pymol.write("mol2")


def obabel_smiles_to_mol2(
    smiles: str,
    out_path: str | Path,
    *,
    forcefield: str = "mmff94",
    minimize_steps: int = 50,
    seed: Optional[int] = None,
) -> Path:
    """Generate a 3D MMFF94 mol2 from a SMILES and write it to ``out_path``.

    File-writing wrapper around :func:`obabel_smiles_to_mol2_block`; prefer the
    block form when no on-disk mol2 is required. See
    :func:`_build_charged_3d_mol2_mol` for the protocol and raised errors.

    Returns:
        The ``out_path`` written.
    """
    pymol = _build_charged_3d_mol2_mol(
        smiles, forcefield=forcefield, minimize_steps=minimize_steps, seed=seed
    )
    out_path = Path(out_path)
    pymol.write("mol2", str(out_path), overwrite=True)
    return out_path


def _assign_generic_atom_names(openbabel, obmol) -> None:
    """Rename atoms to generic ``<element><1-based-per-element-index>``.

    For peptide-like ligands ``make3D`` perceives amino-acid residues and writes
    PDB atom names (``CA``, ``CB``, ``OXT``) into the mol2. Those collide with
    element symbols (``CA`` = carbon-alpha vs calcium), which can confuse
    name-based element inference downstream. Generic names (``C1``, ``N1``, …)
    are element-unambiguous and match the conventional SMILES->PDB->mol2
    ligand-prep output. Chemistry (elements, coordinates, bonds, charges) is
    untouched — only the labels change.
    """
    periodic_table = Chem.GetPeriodicTable()
    counts: dict[str, int] = {}
    for atom in openbabel.OBMolAtomIter(obmol):
        symbol = periodic_table.GetElementSymbol(atom.GetAtomicNum())
        counts[symbol] = counts.get(symbol, 0) + 1
        residue = atom.GetResidue()
        if residue is not None:
            residue.SetAtomID(atom, f"{symbol}{counts[symbol]}")
