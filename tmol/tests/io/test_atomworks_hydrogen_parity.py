"""Where AtomWorks puts a hydrogen, tmol keeps it.

Both libraries build hydrogens from internal coordinates against the same ideal
geometry, so a structure protonated by one and read by the other should not move
a hydrogen. A hydrogen tmol cannot account for is one it rebuilds from its own
table, and this is what notices when the two tables drift apart.
"""

import numpy as np
import pytest

pytest.importorskip("atomworks")

import biotite.structure as struc  # noqa: E402
import biotite.structure.info as info  # noqa: E402
from atomworks.protonation import ensure_hydrogens  # noqa: E402

from tmol.io import biotite_from_pose_stack, pose_stack_from_biotite  # noqa: E402

STANDARD = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]


def _tripeptide(middle):
    """GLY-<middle>-GLY, heavy atoms only, with the backbone joined."""
    parts, offset = [], 0.0
    for i, code in enumerate(("GLY", middle, "GLY")):
        residue = info.residue(code)
        residue = residue[residue.element != "H"]
        if i < 2:
            residue = residue[residue.atom_name != "OXT"]
        residue.res_id[:] = i + 1
        residue.chain_id[:] = "A"
        residue.coord = residue.coord + np.array([offset, 0.0, 0.0])
        offset += 3.5
        parts.append(residue)

    joined = parts[0]
    for part in parts[1:]:
        joined = joined + part
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


@pytest.mark.parametrize("res_name", STANDARD)
def test_tmol_keeps_the_hydrogens_atomworks_placed(res_name, torch_device):
    if torch_device.type != "cpu":
        pytest.skip("placement is device independent; one device is enough")

    protonated = ensure_hydrogens(
        _tripeptide(res_name), ph=7.4, silence_rdkit_warnings=True
    )
    placed = _hydrogens(protonated, 2)
    assert len(placed) > 0, f"AtomWorks placed no hydrogen on {res_name}"

    pose_stack = pose_stack_from_biotite(protonated, torch_device=torch_device)
    rebuilt = biotite_from_pose_stack(pose_stack)
    rebuilt = rebuilt[0] if isinstance(rebuilt, (list, tuple)) else rebuilt
    kept = _hydrogens(rebuilt, 2)

    # Each hydrogen AtomWorks placed has one of tmol's on top of it.
    for position in placed:
        assert np.min(np.linalg.norm(kept - position, axis=-1)) < 1e-3
