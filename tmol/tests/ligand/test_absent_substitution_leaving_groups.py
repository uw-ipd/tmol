"""A declared bond at a carbonyl or phosphoryl site displaces one terminal branch."""

import biotite.structure as struc
import numpy as np
import pytest

from tmol.ligand._input_repair import get_absent_substitution_leaving_groups


@pytest.mark.parametrize(
    "carbonyl, absent, connection_atoms, expected",
    [
        pytest.param(True, {"O3"}, {"C1"}, {"C1": ("O3",)}, id="single_unobserved"),
        pytest.param(
            True, {"O2", "HO2"}, {"C1"}, {"C1": ("O2", "HO2")}, id="own_hydrogens"
        ),
        pytest.param(True, {"O2", "HO2", "O3"}, {"C1"}, {}, id="two_ambiguous"),
        pytest.param(True, set(), {"C1"}, {}, id="resolved_never_displaced"),
        pytest.param(False, {"O3"}, {"C1"}, {}, id="no_double_bonded_oxygen"),
        pytest.param(True, {"O3"}, set(), {}, id="no_connection_atoms"),
        pytest.param(True, {"O3"}, {"O1"}, {}, id="other_connection_atom"),
    ],
)
def test_absent_substitution_leaving_groups(
    carbonyl, absent, connection_atoms, expected
):
    """C1 bears a (double-bonded) O1 and terminal O2(-HO2) and O3; ``absent`` are NaN."""
    template = struc.AtomArray(5)
    template.atom_name[:] = ["C1", "O1", "O2", "HO2", "O3"]
    template.element[:] = ["C", "O", "O", "H", "O"]
    template.res_name[:] = "LIG"
    template.coord[:] = 0.0
    double = struc.BondType.DOUBLE if carbonyl else struc.BondType.SINGLE
    single = int(struc.BondType.SINGLE)
    template.bonds = struc.BondList(
        5,
        np.array([[0, 1, int(double)], [0, 2, single], [2, 3, single], [0, 4, single]]),
    )
    residue = template.copy()
    residue.coord[np.isin(residue.atom_name, list(absent))] = np.nan
    groups = get_absent_substitution_leaving_groups(template, residue, connection_atoms)
    assert groups == expected
