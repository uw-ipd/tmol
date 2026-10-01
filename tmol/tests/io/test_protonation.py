"""The protonation state tmol takes from AtomWorks for the residues it builds."""

import biotite.structure
import numpy
import pytest
from atomworks.io.utils.ccd import atom_array_from_ccd_code

from tmol.database import ParameterDatabase
from tmol.database.chemical import DEPROTONATED_VAR_IND
from tmol.io import atom_array_from_cif
from tmol.io import _protonation as protonation
from tmol.tests.data import data_path


@pytest.mark.parametrize(
    "path",
    [
        ("cif", "3N0I.cif"),
        ("metal_fixtures", "zn_tetrahedral_3ks3.cif.zst"),
        ("metal_fixtures", "cua_ba3_2cua.cif.zst"),
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
    disulfides = protonation._disulfides(marked)
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


def _cysteine_on_zinc(distance):
    cys = atom_array_from_ccd_code("CYS")
    cys = cys[(cys.element != "H") & (cys.atom_name != "OXT")]
    zinc = atom_array_from_ccd_code("ZN")
    zinc.res_id[:] = 2
    sg, cb = (cys.coord[cys.atom_name == name][0] for name in ("SG", "CB"))
    zinc.coord[0] = sg + distance * (sg - cb) / numpy.linalg.norm(sg - cb)
    site = biotite.structure.concatenate([cys, zinc])
    site.set_annotation("charge", numpy.r_[numpy.zeros(len(cys), int), 2])
    site.bonds = biotite.structure.connect_via_residue_names(site)
    return site, numpy.array([[len(cys), numpy.flatnonzero(cys.atom_name == "SG")[0]]])


@pytest.mark.parametrize("lengths", [(2.3, 3.6), (3.6, 2.3)], ids=["near", "far"])
def test_a_cached_state_follows_the_metal_bond_length(lengths):
    """A thiolate on a bonded zinc, a thiol past AtomWorks' reach, in either order."""
    forms = protonation.database_forms(ParameterDatabase.get_default().chemical)
    protonation._STATES.clear()
    for length in lengths:
        site, coordination = _cysteine_on_zinc(length)
        marked = protonation.with_atomworks_hydrogens(
            site, coordination=coordination, forms=forms
        )
        variant = marked.get_annotation(protonation.PROTONATION_VARIANT)[0]
        assert variant == (DEPROTONATED_VAR_IND if length < 3 else 0)


def test_a_ligand_numbered_after_its_chain_is_not_bonded_to_it():
    """3T14 FAD 500 follows MET 418 in chain A; its pyrophosphate stays a dianion."""
    structure = atom_array_from_cif(
        data_path("atomworks_regressions", "sulfur_attachments_3t14.cif.zst")
    )
    heavy = structure[(structure.element != "H") & (structure.res_name != "HOH")]
    protonated = protonation.with_atomworks_hydrogens(heavy)
    phosphate = (protonated.res_name == "FAD") & numpy.isin(
        protonated.atom_name, ("O1A", "O2A", "O1P", "O2P")
    )
    assert protonated.charge[phosphate].sum() == -2


def test_free_nucleotides_of_one_chain_are_not_linked():
    """Two AMP ligands of one chain and entity (8gpb) are no dinucleotide: each O3' is a hydroxyl."""
    structure = atom_array_from_cif(
        data_path("sweep_regressions", "free_nucleotides_8gpb.cif.zst")
    )
    protonated = protonation.with_atomworks_hydrogens(
        structure[structure.element != "H"]
    )
    for o3 in numpy.flatnonzero(protonated.atom_name == "O3'"):
        partners = protonated.element[protonated.bonds.get_bonds(o3)[0]]
        assert sorted(partners) == ["C", "H"]


def test_a_heme_split_from_its_iron_keeps_the_porphyrin_dianion():
    """155C HEM's Fe becomes an ion residue; NA and NC stay anionic, not radicals."""
    from tmol.io._pose_stack_from_biotite import _with_input_hydrogens

    structure = atom_array_from_cif(data_path("cif", "155c__1__1.A__1.B.cif"))
    heavy = structure[(structure.element != "H") & (structure.res_name != "HOH")]
    chemdb = ParameterDatabase.get_default().chemical
    protonated = _with_input_hydrogens(heavy, 7.4, None, chemdb, True)
    heme = protonated[protonated.res_name == "HEM"]
    nitrogens = numpy.isin(heme.atom_name, ("NA", "NB", "NC", "ND"))
    assert heme.charge[nitrogens].sum() == -2
