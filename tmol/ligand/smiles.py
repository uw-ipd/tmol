"""SMILES perception from coordinates and protonation via dimorphite_dl.

Follows the approach from Guangfeng's ligand_prep pipeline: SMILES are
perceived from atom coordinates using OpenBabel, then protonated at a
target pH using dimorphite_dl.
"""

import logging

from tmol.ligand.detect import LigandInfo

logger = logging.getLogger(__name__)

_ELEMENT_TO_ATOMIC_NUM = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Na": 11,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "K": 19,
    "Br": 35,
    "I": 53,
}


def perceive_smiles(ligand_info: LigandInfo) -> str:
    """Perceive a SMILES string from ligand atom coordinates via OpenBabel.

    Builds an OBMol from the atoms, elements, and coordinates in the
    LigandInfo, lets OpenBabel perceive bonds, and writes a canonical
    SMILES string.

    Args:
        ligand_info: A LigandInfo with atom_names, elements, and coords.

    Returns:
        A canonical SMILES string for the molecule.

    Raises:
        ValueError: If the resulting SMILES is empty or perception fails.
    """
    from openbabel import openbabel, pybel

    obmol = openbabel.OBMol()
    obmol.BeginModify()

    for i, (elem, coord) in enumerate(zip(ligand_info.elements, ligand_info.coords)):
        obatom = obmol.NewAtom()
        atomic_num = _ELEMENT_TO_ATOMIC_NUM.get(elem.strip(), 0)
        if atomic_num == 0:
            atomic_num = openbabel.OBElements.GetAtomicNum(elem.strip())
        obatom.SetAtomicNum(atomic_num)
        obatom.SetVector(float(coord[0]), float(coord[1]), float(coord[2]))

    obmol.EndModify()
    obmol.ConnectTheDots()
    obmol.PerceiveBondOrders()

    mol = pybel.Molecule(obmol)
    mol.removeh()
    smi = mol.write("can").strip()

    if not smi:
        raise ValueError(
            f"Failed to perceive SMILES for residue {ligand_info.res_name}"
        )

    logger.debug("Perceived SMILES for %s: %s", ligand_info.res_name, smi)
    return smi


def protonate_ligand_smiles(
    smiles: str,
    ph: float = 7.4,
    precision: float = 0.1,
) -> str:
    """Protonate a SMILES string at a target pH using dimorphite_dl.

    Args:
        smiles: Input SMILES string (may be un-protonated).
        ph: Target pH for protonation (both min and max are set to this).
        precision: pKa precision in standard deviations from mean.

    Returns:
        The protonated SMILES string. If dimorphite_dl returns multiple
        variants, the first one is used. If protonation fails, the
        original SMILES is returned unchanged.
    """
    try:
        from dimorphite_dl import protonate_smiles

        variants = protonate_smiles(
            smiles,
            ph_min=ph,
            ph_max=ph,
            precision=precision,
            max_variants=1,
        )
        if variants:
            result = variants[0]
            logger.debug("Protonated SMILES at pH %.1f: %s -> %s", ph, smiles, result)
            return result
    except Exception:
        logger.warning(
            "dimorphite_dl protonation failed for %s, using original",
            smiles,
            exc_info=True,
        )
    return smiles
