"""Where AtomWorks puts a hydrogen, tmol keeps it: this notices when their ideal
hydrogen geometries drift apart."""

import biotite.structure as struc
import biotite.structure.info as info
import numpy as np
import pytest
from atomworks.constants import STANDARD_AA
from atomworks.experimental.protonation import add_hydrogens, assign_hydrogens

from tmol.io import biotite_from_pose_stack, pose_stack_from_biotite


def _tripeptide(middle):
    """GLY-<middle>-GLY, heavy atoms only, with the backbone joined."""
    parts = []
    for i, code in enumerate(("GLY", middle, "GLY")):
        residue = info.residue(code)
        residue = residue[residue.element != "H"]
        if i < 2:
            residue = residue[residue.atom_name != "OXT"]
        residue.res_id[:] = i + 1
        residue.chain_id[:] = "A"
        if parts:
            previous = parts[-1]
            carbon, alpha, oxygen = (
                previous.coord[previous.atom_name == name][0]
                for name in ("C", "CA", "O")
            )
            direction = sum(
                (carbon - neighbor) / np.linalg.norm(carbon - neighbor)
                for neighbor in (alpha, oxygen)
            )
            # Join the actual ports; arbitrary residue offsets can leave a
            # declared peptide bond several Angstroms longer than a real bond.
            nitrogen = residue.coord[residue.atom_name == "N"][0]
            residue.coord += (
                carbon + 1.33 * direction / np.linalg.norm(direction) - nitrogen
            )
        parts.append(residue)

    joined = struc.concatenate(parts)
    for i in range(2):
        c = int(np.flatnonzero((joined.res_id == i + 1) & (joined.atom_name == "C"))[0])
        n = int(np.flatnonzero((joined.res_id == i + 2) & (joined.atom_name == "N"))[0])
        joined.bonds.add_bond(c, n, struc.BondType.SINGLE)
    joined.set_annotation("is_polymer", np.ones(len(joined), dtype=bool))
    joined.set_annotation("pn_unit_iid", np.full(len(joined), "A_1"))
    joined.set_annotation("charge", np.zeros(len(joined), dtype=int))
    return joined


def _hydrogens(atom_array, res_id):
    residue = atom_array[atom_array.res_id == res_id]
    is_h = np.isin(residue.element.astype(str), ("H", "D"))
    finite = np.isfinite(residue.coord).all(axis=-1)
    return residue.coord[is_h & finite]


@pytest.mark.parametrize("res_name", STANDARD_AA)
def test_tmol_keeps_the_hydrogens_atomworks_placed(res_name, torch_device):
    if torch_device.type != "cpu":
        pytest.skip("placement is device independent; one device is enough")

    protonated = add_hydrogens(assign_hydrogens(_tripeptide(res_name), ph=7.4))
    placed = _hydrogens(protonated, 2)
    assert len(placed) > 0, f"AtomWorks placed no hydrogen on {res_name}"

    pose_stack = pose_stack_from_biotite(protonated, torch_device=torch_device)
    kept = _hydrogens(biotite_from_pose_stack(pose_stack), 2)

    # Each hydrogen AtomWorks placed has one of tmol's on top of it.
    for position in placed:
        assert np.min(np.linalg.norm(kept - position, axis=-1)) < 1e-3
