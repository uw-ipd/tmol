"""Explicit terminal chemistry wins over inferred polymer gap connections."""

import numpy as np
import pytest

from tmol.database import ParameterDatabase
from tmol.io import atom_array_from_file, pose_stack_from_biotite
from tmol.io._input_termini import EXPLICIT_TERMINI, with_stated_termini
from tmol.tests.data import data_path
from tmol.tests.io.test_protonation import _hydrogens_on
from tmol.tests.ligand.test_ligand_entry_paths import _score_and_minimize_ligand


def _short_gap():
    return atom_array_from_file(
        data_path("sweep_regressions", "stated_terminus_5lh4.pdb")
    )


@pytest.mark.parametrize("bondless_consecutive", [False, True])
def test_supplied_neutral_terminus_breaks_a_short_gap(
    torch_device, bondless_consecutive
):
    structure = _short_gap()
    terminal_residue = 132
    if bondless_consecutive:
        structure.bonds = None
        terminal_residue = 131
        structure.res_id[structure.res_id == 132] = terminal_residue
    original = structure.copy()
    pose = pose_stack_from_biotite(structure, torch_device, no_optH=True)
    assert _hydrogens_on(pose, "A", terminal_residue)[2]["N"] == 2
    assert pose.inter_residue_connections[0, 0, :, 0].lt(0).all()
    assert pose.inter_residue_connections[0, 1, :, 0].lt(0).all()
    block_type = pose.packed_block_types.active_block_types[
        int(pose.block_type_ind[0, 1])
    ]
    offset = int(pose.block_coord_offset[0, 1])
    supplied = structure[
        (structure.res_id == terminal_residue) & np.isfinite(structure.coord).all(-1)
    ]
    actual = pose.coords[
        0, [offset + block_type.atom_to_idx[n] for n in supplied.atom_name]
    ]
    np.testing.assert_allclose(actual.cpu(), supplied.coord, atol=1e-5)
    np.testing.assert_array_equal(structure.coord, original.coord)
    if original.bonds is None:
        assert structure.bonds is None
    else:
        np.testing.assert_array_equal(
            structure.bonds.as_array(), original.bonds.as_array()
        )
    _score_and_minimize_ligand(pose, ParameterDatabase.get_default())


def test_supplied_terminus_does_not_remove_an_explicit_covalent_bond():
    structure = _short_gap()
    c = np.flatnonzero((structure.res_id == 130) & (structure.atom_name == "C"))[0]
    n = np.flatnonzero((structure.res_id == 132) & (structure.atom_name == "N"))[0]
    structure.bonds.add_bond(int(c), int(n), 1)
    with pytest.raises(ValueError, match="Stated terminus.*ALA/N -- A:130:SER/C"):
        with_stated_termini(structure, ParameterDatabase.get_default().chemical)


def test_nonterminal_hydrogens_do_not_close_polymer_connections():
    from atomworks.io.utils.ccd import atom_array_from_ccd_code

    structure = atom_array_from_ccd_code("SER")
    structure = structure[~np.isin(structure.atom_name, ("H2", "H3", "OXT", "HXT"))]
    marked = with_stated_termini(structure, ParameterDatabase.get_default().chemical)
    if EXPLICIT_TERMINI in marked.get_annotation_categories():
        assert not marked.get_annotation(EXPLICIT_TERMINI).any()


@pytest.mark.parametrize("bonds_given", [False, True])
def test_stated_termini_use_database_residue_aliases(bonds_given):
    from atomworks.io.utils.ccd import atom_array_from_ccd_code

    structure = atom_array_from_ccd_code("MSE")
    if not bonds_given:
        structure.bonds = None
    original = structure.copy()
    marked = with_stated_termini(structure, ParameterDatabase.get_default().chemical)
    assert (marked.get_annotation(EXPLICIT_TERMINI) == 3).all()
    np.testing.assert_array_equal(marked.res_name, original.res_name)
    np.testing.assert_array_equal(marked.coord, original.coord)
    if not bonds_given:
        assert structure.bonds is None
        assert marked.bonds is not None
