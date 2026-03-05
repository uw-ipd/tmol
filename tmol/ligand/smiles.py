"""SMILES perception and protonation via dimorphite_dl.

SMILES are obtained via a tiered strategy:
  1. CCD lookup -- the LigandInfo carries a pre-resolved canonical SMILES
     from the wwPDB Chemical Component Dictionary when available.
  2. CIF sub-array -- the ligand AtomArray (with bonds from the CIF) is
     written to MOL via Biotite, loaded into RDKit, and converted to SMILES.
     If the MOL block has no bonds (novel ligand), RDKit infers them from
     coordinates via rdDetermineBonds.

Protonation at a target pH uses the vendored dimorphite_dl script.
"""

import logging
import subprocess
import sys
import tempfile
from pathlib import Path

from tmol.ligand.detect import LigandInfo, _atom_array_to_smiles

logger = logging.getLogger(__name__)
_VENDORED_DIMORPHITE = Path(__file__).resolve().parent / "dimorphite_dl.py"


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

    Args:
        smiles: Input SMILES string (may be un-protonated).
        ph: Target pH for protonation (both min and max are set to this).
        precision: pKa precision in standard deviations from mean.

    Returns:
        The protonated SMILES string. If dimorphite_dl returns multiple
        variants, the first one is used. If protonation fails, the
        original SMILES is returned unchanged.
    """
    # Replicate guangfeng/ligand_prep protonation workflow using vendored script:
    # python dimorphite_dl.py --smiles_file ... --min_ph X --max_ph X
    # --output_file ... --pka_precision Y
    try:
        if not _VENDORED_DIMORPHITE.exists():
            logger.warning(
                "Vendored dimorphite script not found: %s", _VENDORED_DIMORPHITE
            )
            return smiles

        with tempfile.TemporaryDirectory(prefix="tmol_dimorphite_") as tmpdir:
            tmp = Path(tmpdir)
            smiles_file = tmp / "designs.smi"
            output_file = tmp / "designs.prot.smi"
            smiles_file.write_text(f"{smiles}\tmol\n")

            proc = subprocess.run(
                [
                    sys.executable,
                    str(_VENDORED_DIMORPHITE),
                    "--smiles_file",
                    str(smiles_file),
                    "--min_ph",
                    str(ph),
                    "--max_ph",
                    str(ph),
                    "--output_file",
                    str(output_file),
                    "--pka_precision",
                    str(precision),
                ],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                logger.warning(
                    "Vendored dimorphite invocation failed for %s: %s",
                    smiles,
                    proc.stderr.strip(),
                )
                return smiles

            if output_file.exists():
                for line in output_file.read_text().splitlines():
                    parts = line.strip().split()
                    if parts:
                        result = parts[0]
                        logger.debug(
                            "Vendored protonated SMILES at pH %.1f: %s -> %s",
                            ph,
                            smiles,
                            result,
                        )
                        return result
    except Exception:
        logger.warning("Vendored dimorphite protonation failed for %s", smiles)
    return smiles
