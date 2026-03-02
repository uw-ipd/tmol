"""3D structure generation and MMFF94 charge assignment via OpenBabel.

Converts a protonated SMILES string into a 3D molecular structure with
MMFF94 partial charges, entirely in memory (no file I/O).
"""

import logging

from openbabel import openbabel, pybel

logger = logging.getLogger(__name__)


def smiles_to_obmol(
    smiles: str,
    minimize_steps: int = 500,
    forcefield: str = "mmff94",
) -> pybel.Molecule:
    """Convert a SMILES string to a 3D molecule with partial charges.

    Generates 3D coordinates, adds explicit hydrogens, performs energy
    minimization, and assigns MMFF94 partial charges. All operations
    are performed in memory.

    Args:
        smiles: A (protonated) SMILES string.
        minimize_steps: Number of force-field minimization steps.
        forcefield: Force field for 3D generation and minimization.

    Returns:
        A pybel Molecule with 3D coordinates and partial charges set
        on each atom.

    Raises:
        RuntimeError: If 3D generation or charge computation fails.
    """
    mol = pybel.readstring("smi", smiles)
    mol.addh()
    mol.make3D(forcefield=forcefield, steps=50)
    mol.localopt(forcefield=forcefield, steps=minimize_steps)

    charge_model = openbabel.OBChargeModel.FindType(forcefield)
    if charge_model is None or not charge_model.ComputeCharges(mol.OBMol):
        logger.warning(
            "MMFF94 charge computation failed for %s, falling back to "
            "Gasteiger charges",
            smiles,
        )
        gasteiger = openbabel.OBChargeModel.FindType("gasteiger")
        if gasteiger is not None:
            gasteiger.ComputeCharges(mol.OBMol)

    return mol


def get_partial_charges_by_index(mol: pybel.Molecule) -> dict[int, float]:
    """Extract per-atom partial charges keyed by OpenBabel atom index.

    Index keys are OBMol 0-based atom indices (``OBAtom.GetIndex()``), which
    are stable across later renaming and therefore safer than name-based maps.
    """
    charges: dict[int, float] = {}
    for obatom in openbabel.OBMolAtomIter(mol.OBMol):
        charges[obatom.GetIndex()] = float(obatom.GetPartialCharge())
    return charges


def get_partial_charges(mol: pybel.Molecule) -> dict[str, float]:
    """Extract per-atom partial charges with legacy generated atom names.

    This API is kept for backward compatibility with existing tests/utilities.
    For robust pipeline usage, prefer :func:`get_partial_charges_by_index`.
    """
    charges_by_index = get_partial_charges_by_index(mol)
    charges: dict[str, float] = {}
    elem_counts: dict[str, int] = {}

    for obatom in openbabel.OBMolAtomIter(mol.OBMol):
        idx = obatom.GetIndex()
        if hasattr(openbabel, "OBElements"):
            elem = openbabel.OBElements.GetSymbol(obatom.GetAtomicNum())
        else:
            elem = openbabel.GetSymbol(obatom.GetAtomicNum())
        elem_counts[elem] = elem_counts.get(elem, 0) + 1
        name = f"{elem}{elem_counts[elem]}"
        charges[name] = charges_by_index[idx]

    return charges
