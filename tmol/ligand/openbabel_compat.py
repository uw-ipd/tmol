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


# Conformer-search defaults for the SMILES -> mol2 builder. The reference
# ligand-prep pipeline runs OpenBabel's ``--conformer -nconf 500 --score energy``
# (a genetic-algorithm search) followed by a steepest-descent minimize. That
# search (``OBConformerSearch``) crashes (double free) in the openbabel-wheel
# Python binding, so we use the stable ``OBForceField.WeightedRotorSearch``
# instead: it rotor-samples ``_CONFORMER_NCONF`` candidates, minimizing each for
# ``_CONFORMER_GEOM_STEPS`` steps, and keeps the lowest-energy geometry.
_CONFORMER_NCONF = 50
_CONFORMER_GEOM_STEPS = 25


class OpenBabelUnavailableError(RuntimeError):
    """Raised when an OB-fallback helper is called but ``openbabel`` is missing."""


def _import_openbabel():
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


def obabel_read_pdb(
    path: str | Path,
    *,
    perceive_bond_orders: bool = True,
) -> Optional[Chem.Mol]:
    """Read a PDB file via OpenBabel and return an RDKit ``Chem.Mol``.

    PDB does not encode bond orders. By default this calls
    ``ConnectTheDots`` + ``PerceiveBondOrders`` so the resulting Mol has
    chemically sensible bonds suitable for downstream RDKit usage.

    Returns ``None`` if OB could not parse the file. Raises
    :class:`OpenBabelUnavailableError` if OpenBabel is not installed.
    """
    openbabel, pybel = _import_openbabel()
    path = Path(path)
    try:
        pymol = next(pybel.readfile("pdb", str(path)))
    except StopIteration:
        logger.warning("OpenBabel could not read any molecule from %s", path)
        return None
    except Exception:
        logger.warning("OpenBabel failed to parse PDB file %s", path, exc_info=True)
        return None

    obmol = pymol.OBMol
    if perceive_bond_orders:
        # ConnectTheDots is normally called by the reader, but is idempotent
        # and cheap; call it explicitly so we know connectivity is set
        # before PerceiveBondOrders runs.
        obmol.ConnectTheDots()
        obmol.PerceiveBondOrders()
    return _obmol_to_rdkit_mol(obmol)


def obabel_read_smiles(
    smiles: str,
    *,
    generate_3d: bool = False,
    minimize: bool = False,
    forcefield: str = "mmff94",
) -> Optional[Chem.Mol]:
    """Parse a SMILES string via OpenBabel and return an RDKit ``Chem.Mol``.

    Args:
        smiles: A SMILES string.
        generate_3d: If True, embed the molecule in 3D and add explicit
            hydrogens (calls ``make3D``).
        minimize: If True (and ``generate_3d``), run a force-field
            geometry optimization after embedding.
        forcefield: Name of the force field used for 3D embedding and
            optional minimization (typically ``"mmff94"`` or ``"uff"``).

    Returns:
        An RDKit ``Chem.Mol``, or ``None`` if OB rejected the SMILES.
        Raises :class:`OpenBabelUnavailableError` if OB is not installed.
    """
    openbabel, pybel = _import_openbabel()
    try:
        pymol = pybel.readstring("smi", smiles)
    except Exception:
        logger.warning("OpenBabel failed to parse SMILES %r", smiles, exc_info=True)
        return None

    if generate_3d:
        try:
            pymol.addh()
            pymol.make3D(forcefield=forcefield, steps=50)
            if minimize:
                pymol.localopt(forcefield=forcefield, steps=500)
        except Exception:
            logger.warning(
                "OpenBabel failed to generate 3D for SMILES %r",
                smiles,
                exc_info=True,
            )
            return None

    return _obmol_to_rdkit_mol(pymol.OBMol)


def _conformer_search(
    openbabel, obmol, *, forcefield: str, minimize_steps: int
) -> bool:
    """Run a force-field rotor conformer search in place on ``obmol``.

    Replaces the reference pipeline's ``--conformer --score energy`` step with
    :meth:`OBForceField.WeightedRotorSearch` (the genetic-algorithm
    ``OBConformerSearch`` crashes in this build). Returns ``True`` if the search
    ran and the lowest-energy geometry was written back to ``obmol``; ``False``
    if the force field could not be set up (caller falls back to a plain
    minimize).
    """
    ff = openbabel.OBForceField.FindForceField(forcefield)
    if ff is None or not ff.Setup(obmol):
        return False
    ff.WeightedRotorSearch(_CONFORMER_NCONF, _CONFORMER_GEOM_STEPS)
    ff.ConjugateGradients(minimize_steps or _CONFORMER_GEOM_STEPS)
    ff.GetCoordinates(obmol)
    return True


def _build_charged_3d_mol2_mol(
    smiles: str,
    *,
    forcefield: str = "mmff94",
    make3d_steps: int = 50,
    minimize_steps: int = 500,
    conformer_search: bool = True,
):
    """Parse a SMILES, 3D-embed, charge, and rename atoms; return the pybel mol.

    Shared core of :func:`obabel_smiles_to_mol2` (file) and
    :func:`obabel_smiles_to_mol2_block` (string). Mirrors the reference
    ligand-prep protocol: add explicit hydrogens, embed 3D coordinates with the
    named force field, optionally run a rotor conformer search to find a
    low-energy geometry (``conformer_search``, default on), minimize, compute
    its partial charges (so the mol2 ``charge_type`` is ``MMFF94_CHARGES``), and
    assign generic atom names. MMFF94 charges are topological (graph-determined),
    so the exact conformer does not affect them — only the atom coordinates.

    Args:
        conformer_search: When ``True`` (default), run a rotor conformer search
            (:func:`_conformer_search`) after the initial embed, matching the
            reference pipeline. Set ``False`` for faster single-conformer
            generation (embed + minimize only).

    Raises:
        OpenBabelUnavailableError: If the ``openbabel`` package is not installed.
        ValueError: If OB rejects the SMILES, fails 3D generation, or cannot
            compute partial charges.
    """
    openbabel, pybel = _import_openbabel()
    try:
        pymol = pybel.readstring("smi", smiles)
    except Exception as exc:
        raise ValueError(f"OpenBabel could not parse SMILES {smiles!r}") from exc
    try:
        pymol.addh()
        pymol.make3D(forcefield=forcefield, steps=make3d_steps)
        searched = False
        if conformer_search:
            searched = _conformer_search(
                openbabel,
                pymol.OBMol,
                forcefield=forcefield,
                minimize_steps=minimize_steps,
            )
        if not searched and minimize_steps:
            pymol.localopt(forcefield=forcefield, steps=minimize_steps)
    except Exception as exc:
        raise ValueError(
            f"OpenBabel failed to generate 3D coordinates for SMILES {smiles!r}"
        ) from exc
    charge_model = openbabel.OBChargeModel.FindType(forcefield)
    if charge_model is None or not charge_model.ComputeCharges(pymol.OBMol):
        raise ValueError(
            f"OpenBabel could not compute {forcefield} partial charges for "
            f"SMILES {smiles!r}"
        )
    _assign_generic_atom_names(openbabel, pymol.OBMol)
    return pymol


def obabel_smiles_to_mol2_block(
    smiles: str,
    *,
    forcefield: str = "mmff94",
    make3d_steps: int = 50,
    minimize_steps: int = 500,
    conformer_search: bool = True,
) -> str:
    """Return a 3D MMFF94 mol2 as an in-memory TRIPOS string (no disk I/O).

    Preferred over :func:`obabel_smiles_to_mol2` for high-throughput batches
    (e.g. millions of SMILES): the mol2 is handed downstream as a string rather
    than written to and re-read from a temp file. ``conformer_search`` (default
    on) runs a rotor conformer search; set ``False`` for faster generation. See
    :func:`_build_charged_3d_mol2_mol` for the protocol and raised errors.
    """
    pymol = _build_charged_3d_mol2_mol(
        smiles,
        forcefield=forcefield,
        make3d_steps=make3d_steps,
        minimize_steps=minimize_steps,
        conformer_search=conformer_search,
    )
    return pymol.write("mol2")


def obabel_smiles_to_mol2(
    smiles: str,
    out_path: str | Path,
    *,
    forcefield: str = "mmff94",
    make3d_steps: int = 50,
    minimize_steps: int = 500,
    conformer_search: bool = True,
) -> Path:
    """Generate a 3D MMFF94 mol2 from a SMILES and write it to ``out_path``.

    File-writing wrapper around :func:`obabel_smiles_to_mol2_block`; prefer the
    block form when no on-disk mol2 is required. ``conformer_search`` (default
    on) runs a rotor conformer search; set ``False`` for faster generation. See
    :func:`_build_charged_3d_mol2_mol` for the protocol and raised errors.

    Returns:
        The ``out_path`` written.
    """
    pymol = _build_charged_3d_mol2_mol(
        smiles,
        forcefield=forcefield,
        make3d_steps=make3d_steps,
        minimize_steps=minimize_steps,
        conformer_search=conformer_search,
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


def is_available() -> bool:
    """Return True iff the optional ``openbabel`` Python package is importable."""
    try:
        _import_openbabel()
    except OpenBabelUnavailableError:
        return False
    return True
