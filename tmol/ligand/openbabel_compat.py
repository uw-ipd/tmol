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


def obabel_smiles_to_mol2(
    smiles: str,
    out_path: str | Path,
    *,
    forcefield: str = "mmff94",
    make3d_steps: int = 50,
    minimize_steps: int = 500,
) -> Path:
    """Generate a 3D mol2 with MMFF94 partial charges from a SMILES.

    Mirrors the reference ligand-prep protocol: parse the SMILES, add explicit
    hydrogens, embed and minimize 3D coordinates with the named force field,
    compute its partial charges, and write a TRIPOS mol2 whose ``charge_type``
    is ``MMFF94_CHARGES``. MMFF94 charges are topological (graph-determined), so
    the exact conformer does not affect them — only atom names / coordinates.

    Args:
        smiles: A (typically already pKa-protonated) SMILES string.
        out_path: Destination mol2 path.
        forcefield: Force field used for 3D embedding, minimization, and the
            partial-charge model (default ``"mmff94"``).
        make3d_steps: Steps for the initial ``make3D`` embedding.
        minimize_steps: Steps for the follow-up ``localopt`` (0 to skip).

    Returns:
        The ``out_path`` written.

    Raises:
        OpenBabelUnavailableError: If the ``openbabel`` package is not installed.
        ValueError: If OB rejects the SMILES, fails 3D generation, or cannot
            compute partial charges.
    """
    openbabel, pybel = _import_openbabel()
    out_path = Path(out_path)
    try:
        pymol = pybel.readstring("smi", smiles)
    except Exception as exc:
        raise ValueError(f"OpenBabel could not parse SMILES {smiles!r}") from exc
    try:
        pymol.addh()
        pymol.make3D(forcefield=forcefield, steps=make3d_steps)
        if minimize_steps:
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
