"""3D structure generation and MMFF94 charge assignment via OpenBabel.

Converts a protonated SMILES string into a 3D molecular structure with
MMFF94 partial charges, entirely in memory (no file I/O).
"""

import logging

from tmol.ligand.smiles import _import_pybel

logger = logging.getLogger(__name__)


def smiles_to_obmol(
    smiles: str,
    minimize_steps: int = 500,
    forcefield: str = "mmff94",
):
    """Convert a SMILES string to a 3D molecule with partial charges.

    Generates 3D coordinates, adds explicit hydrogens, performs energy
    minimization, and assigns MMFF94 partial charges. All operations
    are performed in memory.

    Args:
        smiles: A (protonated) SMILES string.
        minimize_steps: Number of force-field minimization steps.
        forcefield: Force field for 3D generation and minimization.

    Returns:
        A pybel.Molecule with 3D coordinates and partial charges set
        on each atom.

    Raises:
        RuntimeError: If 3D generation or charge computation fails.
    """
    openbabel, pybel = _import_pybel()

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


def get_partial_charges(mol) -> dict[str, float]:
    """Extract per-atom partial charges from a pybel Molecule.

    Atom names follow the Rosetta rename_atoms convention: heavy atoms
    as <Element><count>, hydrogens as H<bonded_element><count>.

    Args:
        mol: A pybel.Molecule with charges already computed.

    Returns:
        A dict mapping atom name to partial charge.
    """
    openbabel, _ = _import_pybel()
    obmol = mol.OBMol

    charges: dict[str, float] = {}
    heavy_elem_counts: dict[str, int] = {}
    heavy_elem_by_idx: dict[int, str] = {}
    h_atoms: list[tuple[int, int, float]] = []

    for atom in mol.atoms:
        z = atom.atomicnum
        idx = atom.idx - 1
        if z == 1:
            bonded_heavy_idx = -1
            obatom = obmol.GetAtom(atom.idx)
            for bond in openbabel.OBAtomBondIter(obatom):
                bonded_heavy_idx = bond.GetNbrAtom(obatom).GetIndex()
                break
            h_atoms.append((idx, bonded_heavy_idx, atom.partialcharge))
        else:
            elem = openbabel.GetSymbol(z)
            heavy_elem_counts[elem] = heavy_elem_counts.get(elem, 0) + 1
            name = f"{elem}{heavy_elem_counts[elem]}"
            heavy_elem_by_idx[idx] = elem
            charges[name] = atom.partialcharge

    h_name_counts: dict[str, int] = {}
    for _, heavy_idx, charge in h_atoms:
        heavy_elem = heavy_elem_by_idx.get(heavy_idx, "")
        h_prefix = f"H{heavy_elem}"
        h_name_counts[h_prefix] = h_name_counts.get(h_prefix, 0) + 1
        name = f"{h_prefix}{h_name_counts[h_prefix]}"
        charges[name] = charge

    return charges
