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


def _hydrogens_on(pose, chain, number):
    """Block type, class and {heavy atom: hydrogens} of residue ``chain`` ``number``."""
    info, pbt = pose.pdb_info, pose.packed_block_types
    block = next(
        b
        for b in range(pose.max_n_blocks)
        if (str(info.chain_labels[0, b]), int(info.residue_labels[0, b]))
        == (chain, number)
    )
    index = int(pose.block_type_ind[0, block])
    bt, is_h = pbt.active_block_types[index], pbt.atom_is_hydrogen[index].tolist()
    got = {a.name: 0 for a, h in zip(bt.atoms, is_h) if not h}
    for a, b in numpy.asarray(bt.bond_indices).reshape(-1, 2):
        if is_h[b] and not is_h[a]:
            got[bt.atoms[a].name] += 1
    return bt.name, bt.io_equiv_class, got


@pytest.mark.parametrize(
    "fixture, residues",
    [
        ("metal_lysine_2r1w.cif.zst", [("B", 62)]),  # LYS NZ on Mg: a neutral amine
        ("metal_amine_terminus_3ppd.cif.zst", [("A", 1)]),  # GLY 1 amine on Zn
        ("cysteine_heme_1cch.cif.zst", [("A", 83)]),  # pyrroles of a Cys-bonded heme
        ("metal_phosphate_7bad.cif.zst", [("A", 103)]),  # PO4 O2 on Mg
        # a free PO4 and two on metals, each its own state
        ("per_copy_phosphate_6m8q.cif.zst", [("A", 504), ("A", 505), ("B", 504)]),
        ("polar_hydrogens_10gs.pdb.zst", [("A", 47), ("A", 71), ("A", 101)]),
    ],
    ids=["2r1w_lys", "3ppd_nterm", "1cch_heme", "7bad_po4", "6m8q_po4s", "10gs_polar_h"],
)
def test_residue_types_take_the_atomworks_state(
    fixture, residues, monkeypatch, torch_device
):
    """Each residue's type carries the hydrogens AtomWorks gives each heavy atom.

    AtomWorks is asked about the whole input once, with the bonds and drawn
    hydrogens tmol hands it; a histidine tautomer it leaves free is summed.
    """
    from tmol.io import atom_array_from_file, pose_stack_from_biotite

    placed, reference = protonation._placed_hydrogens, []

    def whole(model, heavy, ph, extra, *declared):
        if not reference:
            every = ~numpy.isin(model.element, ("H", "D")) & (model.res_name != "HOH")
            parent, *_, free = placed(model, every, ph, extra, *declared)
            count = numpy.bincount(parent, minlength=len(model))
            reference.append((model, count, set(free.tolist())))
        return placed(model, heavy, ph, extra, *declared)

    monkeypatch.setattr(protonation, "_placed_hydrogens", whole)
    protonation._STATES.clear()
    structure = atom_array_from_file(data_path("sweep_regressions", fixture))
    pose = pose_stack_from_biotite(
        structure, torch_device, prepare_ligands=True, ligand_seed=0
    )
    model, count, free = reference[0]
    for chain, number in residues:
        name, equiv, got = _hydrogens_on(pose, chain, number)
        mine = numpy.flatnonzero(
            (model.chain_id == chain)
            & (model.res_id == number)
            & (model.res_name == equiv)
            & ~numpy.isin(model.element, ("H", "D"))
        )
        fixed = {str(model.atom_name[i]): int(count[i]) for i in mine if i not in free}
        assert {a: got[a] for a in fixed} == fixed, name
        ring = [i for i in mine if i in free]
        assert sum(got[str(model.atom_name[i])] for i in ring) == count[ring].sum()
