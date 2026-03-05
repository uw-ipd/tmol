"""SMILES perception and protonation via dimorphite_dl.

SMILES are obtained via a tiered strategy:
  1. CCD lookup -- the LigandInfo carries a pre-resolved canonical SMILES
     from the wwPDB Chemical Component Dictionary when available.
  2. CIF sub-array -- the ligand AtomArray (with bonds from the CIF) is
     written to MOL via Biotite, loaded into RDKit, and converted to SMILES.
     If the MOL block has no bonds (novel ligand), RDKit infers them from
     coordinates via rdDetermineBonds.

Protonation at a target pH uses the vendored dimorphite_dl module in-process.
"""

import logging
import sys

from tmol.ligand.detect import LigandInfo, _atom_array_to_smiles

logger = logging.getLogger(__name__)


def _import_pybel():
    """Import openbabel and pybel, suppressing noisy format plugin warnings."""
    import io

    stderr_backup = sys.stderr
    sys.stderr = io.StringIO()
    try:
        from openbabel import openbabel, pybel

        return openbabel, pybel
    finally:
        sys.stderr = stderr_backup


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
    """Obtain a canonical SMILES string for a detected ligand.

    Tier 1: Return the CCD canonical SMILES if available (pre-resolved
    during detection and stored on ligand_info.ccd_smiles).

    Tier 2: Write the ligand's AtomArray (from the CIF, with bonds) to a
    MOL block via Biotite, load into RDKit, and convert to SMILES.  If
    the molecule has no bonds, rdDetermineBonds infers them from 3D
    coordinates.

    Args:
        ligand_info: A LigandInfo with ccd_smiles, atom_array, and
            fallback atom_names / elements / coords.

    Returns:
        A canonical SMILES string for the molecule.

    Raises:
        ValueError: If SMILES cannot be determined by any tier.
    """
    # Tier 1: CCD canonical SMILES
    if ligand_info.ccd_smiles:
        logger.debug(
            "Using CCD SMILES for %s: %s",
            ligand_info.res_name,
            ligand_info.ccd_smiles,
        )
        return ligand_info.ccd_smiles

    # Tier 2: CIF sub-array -> MOL -> RDKit -> SMILES
    smi = _atom_array_to_smiles(ligand_info.atom_array)
    if smi:
        logger.debug(
            "Perceived SMILES from CIF AtomArray for %s: %s",
            ligand_info.res_name,
            smi,
        )
        return smi

    raise ValueError(
        f"Failed to perceive SMILES for residue {ligand_info.res_name}: "
        f"not in CCD and CIF sub-array conversion failed"
    )


def protonate_ligand_smiles(
    smiles: str,
    ph: float = 7.4,
    precision: float = 0.1,
) -> str:
    """Protonate a SMILES string at a target pH using dimorphite_dl.

    Calls the vendored Dimorphite-DL ``Protonate`` iterator in-process
    rather than spawning a subprocess, avoiding temp-file I/O and an
    extra SMILES round-trip.

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
        from tmol.ligand.dimorphite_dl import Protonate

        args = {
            "smiles": smiles,
            "min_ph": ph,
            "max_ph": ph,
            "pka_precision": precision,
            "max_variants": 128,
            "silent": True,
        }
        for protonated_smi in Protonate(args):
            result = protonated_smi.split("\t")[0]
            logger.debug(
                "Protonated SMILES at pH %.1f: %s -> %s",
                ph,
                smiles,
                result,
            )
            return result
    except Exception:
        logger.warning("Dimorphite-DL protonation failed for %s", smiles)
    return smiles
