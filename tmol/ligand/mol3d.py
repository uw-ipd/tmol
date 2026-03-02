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


def get_partial_charges(mol: pybel.Molecule) -> dict[str, float]:
    """Extract per-atom partial charges from a pybel Molecule.

    Atom names are generated as element symbol + 1-based index to
    ensure uniqueness (e.g. C1, C2, O3, H4).

    Args:
        mol: A pybel Molecule with charges already computed.

    Returns:
        A dict mapping atom name to partial charge.
    """
    charges: dict[str, float] = {}
    elem_counts: dict[str, int] = {}

    for atom in mol.atoms:
        if hasattr(openbabel, "OBElements"):
            elem = openbabel.OBElements.GetSymbol(atom.atomicnum)
        else:
            elem = openbabel.GetSymbol(atom.atomicnum)
        elem_counts[elem] = elem_counts.get(elem, 0) + 1
        name = f"{elem}{elem_counts[elem]}"
        charges[name] = atom.partialcharge

    return charges
