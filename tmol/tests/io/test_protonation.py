"""The protonation state tmol takes from AtomWorks for the residues it builds."""

import biotite.structure
import numpy
import pytest

from tmol.database import ParameterDatabase
from tmol.io import atom_array_from_cif
from tmol.io import _protonation as protonation
from tmol.tests.data import data_path


@pytest.mark.parametrize(
    "path",
    [
        ("cif", "3N0I.cif"),
        ("metal_fixtures", "zn_tetrahedral_3ks3.cif.gz"),
        ("metal_fixtures", "cua_ba3_2cua.cif.gz"),
    ],
    ids=["3n0i", "3ks3_zinc", "2cua_copper"],
)
def test_states_by_context_match_the_whole_structure(path):
    """A residue's state read from its bonded context is the whole structure's."""
    structure = atom_array_from_cif(data_path(*path))
    heavy = structure[structure.element != "H"]
    forms = protonation.database_forms(ParameterDatabase.get_default().chemical)
    protonation._STATES.clear()
    marked = protonation.with_atomworks_hydrogens(
        heavy, forms=forms, residue_names=list(forms)
    )
    variant = marked.get_annotation(protonation.PROTONATION_VARIANT)

    starts = biotite.structure.get_residue_starts(marked, add_exclusive_stop=True)
    disulfides = protonation.find_disulfides(marked)
    extra = numpy.r_[
        protonation._polymer_gap_links(marked, starts, {}),
        numpy.c_[disulfides, numpy.ones(len(disulfides), dtype=int)],
    ]
    parent, _, (charged, charge), _, free = protonation._placed_hydrogens(
        marked, marked.res_name != "HOH", 7.4, extra
    )
    count = numpy.bincount(parent, minlength=len(marked))
    count[free] = -1
    charges = numpy.zeros(len(marked), dtype=int)
    charges[charged] = charge

    checked = 0
    for start, stop in zip(starts[:-1], starts[1:]):
        name = marked.res_name[start]
        if name in forms and forms[name][0]:
            state = {
                str(marked.atom_name[i]): (int(count[i]), int(charges[i]))
                for i in range(start, stop)
            }
            assert variant[start] == protonation._variant(forms[name], state)
            checked += 1
    assert checked > 10
