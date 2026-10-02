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


@pytest.mark.parametrize("fixture,residue", [("2gyi", 65), ("1hdq", 274)])
def test_pdb_reader_preserves_supplied_short_gap_termini(
    torch_device, fixture, residue
):
    structure = atom_array_from_file(
        data_path("sweep_regressions", f"supplied_terminus_{fixture}.pdb")
    )
    terminal = structure[
        (structure.res_id == residue) & np.isfinite(structure.coord).all(-1)
    ]
    assert {"H1", "H2"} <= set(terminal.atom_name)
    pose = pose_stack_from_biotite(structure, torch_device, no_optH=True)
    assert _hydrogens_on(pose, "A", residue)[2]["N"] == 2
    assert pose.inter_residue_connections[0, :, :, 0].lt(0).all()
    block_type = pose.packed_block_types.active_block_types[
        int(pose.block_type_ind[0, 1])
    ]
    offset = int(pose.block_coord_offset[0, 1])
    actual = pose.coords[
        0, [offset + block_type.atom_to_idx[n] for n in terminal.atom_name]
    ]
    np.testing.assert_allclose(actual.cpu(), terminal.coord, atol=1e-5)
    _score_and_minimize_ligand(pose, ParameterDatabase.get_default())


@pytest.mark.parametrize("renamed", ["hydrogen", "heavy_element", "heavy_parent"])
def test_terminal_name_requires_chemical_identity(renamed):
    from atomworks.io.utils.ccd import atom_array_from_ccd_code

    structure = atom_array_from_ccd_code("SER")
    structure = structure[~np.isin(structure.atom_name, ("H2", "H3", "OXT", "HXT"))]
    if renamed == "hydrogen":
        structure.atom_name[structure.atom_name == "HG"] = "H3"
    elif renamed == "heavy_element":
        structure.atom_name[structure.atom_name == "CB"] = "OXT"
    else:
        structure.atom_name[structure.atom_name == "OG"] = "OXT"
    original = structure.copy()
    marked = with_stated_termini(structure, ParameterDatabase.get_default().chemical)
    if EXPLICIT_TERMINI in marked.get_annotation_categories():
        assert not marked.get_annotation(EXPLICIT_TERMINI).any()
    np.testing.assert_array_equal(marked.coord, original.coord)
    np.testing.assert_array_equal(marked.bonds.as_array(), original.bonds.as_array())


def test_generated_polymer_terminal_names_respect_supplied_parents(torch_device):
    from tmol.io import biotite_from_pose_stack

    structure = atom_array_from_file(
        data_path("sweep_regressions", "terminal_name_collision_1a8i.pdb")
    )
    pose, context = pose_stack_from_biotite(
        structure,
        torch_device,
        prepare_ligands=True,
        no_optH=True,
        ligand_seed=0,
        return_context=True,
    )
    assert _hydrogens_on(pose, "A", 680)[2]["O3"] == 1
    assert _hydrogens_on(pose, "A", 680)[2]["N"] == 1
    assert (pose.inter_residue_connections[0, 1, :2, 0] >= 0).all()
    actual = biotite_from_pose_stack(pose, context.canonical_ordering)
    for mode in ("given", "partial", "absent"):
        original = structure.copy()
        h = int(
            np.flatnonzero((original.res_id == 680) & (original.atom_name == "H3"))[0]
        )
        o = int(
            np.flatnonzero((original.res_id == 680) & (original.atom_name == "O3"))[0]
        )
        if mode == "partial":
            original.bonds.remove_bond(h, o)
        elif mode == "absent":
            original.bonds = None
        marked = with_stated_termini(
            original, context.parameter_database.chemical, context.canonical_ordering
        )
        if EXPLICIT_TERMINI in marked.get_annotation_categories():
            assert not marked.get_annotation(EXPLICIT_TERMINI)[
                marked.res_id == 680
            ].any()
        assert marked.bonds.get_bonds(h)[0].tolist() == [o]
        np.testing.assert_array_equal(marked.coord, original.coord)
    # Known generated names can explicitly retain the supplied O3-H coordinate.
    named = structure.copy()
    o3_h = (named.res_id == 680) & (named.atom_name == "H3")
    named.atom_name[o3_h] = "HO3"
    retained = pose_stack_from_biotite(
        named, torch_device, context=context, no_optH=True, trust_hydrogen_names=True
    )
    retained_atoms = biotite_from_pose_stack(retained, context.canonical_ordering)
    actual_h = retained_atoms[
        (retained_atoms.res_id == 680) & (retained_atoms.atom_name == "HO3")
    ]
    np.testing.assert_allclose(actual_h.coord, named.coord[o3_h], atol=1e-5)
    for residue in (679, 680, 681):
        source = structure[(structure.res_id == residue) & (structure.element != "H")]
        dest = actual[actual.res_id == residue]
        by_name = dict(zip(dest.atom_name, dest.coord))
        np.testing.assert_allclose(
            [by_name[n] for n in source.atom_name], source.coord, atol=1e-5
        )
    _score_and_minimize_ligand(pose, context.parameter_database)


def test_terminal_oxygen_coordinated_to_metal_still_closes_port():
    import biotite.structure as struc
    from atomworks.io.utils.ccd import atom_array_from_ccd_code

    structure = atom_array_from_ccd_code("ALA")
    structure = structure[~np.isin(structure.atom_name, ("H2", "H3", "HXT"))]
    metal = struc.AtomArray(1)
    metal.res_name[:] = "ZN"
    metal.atom_name[:] = "ZN"
    metal.element[:] = "ZN"
    metal.res_id[:] = 2
    metal.coord[:] = [10, 10, 10]
    structure = struc.concatenate([structure, metal])
    oxygen = int(np.flatnonzero(structure.atom_name == "OXT")[0])
    structure.bonds.add_bond(oxygen, len(structure) - 1, struc.BondType.COORDINATION)
    marked = with_stated_termini(structure, ParameterDatabase.get_default().chemical)
    assert (marked.get_annotation(EXPLICIT_TERMINI)[:-1] == 2).all()
    np.testing.assert_array_equal(marked.bonds.as_array(), structure.bonds.as_array())


def test_pdb_explicit_connection_does_not_erase_supplied_terminus(tmp_path):
    from pathlib import Path

    source = Path(
        data_path("sweep_regressions", "supplied_terminus_1hdq.pdb")
    ).read_text()
    serial = {
        (int(line[22:26]), line[12:16].strip()): int(line[6:11])
        for line in source.splitlines()
        if line.startswith(("ATOM  ", "HETATM"))
    }
    path = tmp_path / "contradiction.pdb"
    path.write_text(
        source.replace(
            "END\n", f"CONECT{serial[273, 'C']:5d}{serial[274, 'N']:5d}\nEND\n"
        )
    )
    with pytest.raises(ValueError, match="Stated terminus.*THR/N -- A:273:ASP/C"):
        atom_array_from_file(path)


def test_pocket_terminal_oxygen_survives_unrelated_hydrogen_name(torch_device):
    structure = atom_array_from_file(
        data_path("sweep_regressions", "terminal_pocket_2jdm.pdb")
    )
    pose, context = pose_stack_from_biotite(
        structure,
        torch_device,
        prepare_ligands=True,
        no_optH=True,
        ligand_seed=0,
        return_context=True,
    )
    block_type = pose.packed_block_types.active_block_types[
        int(pose.block_type_ind[0, 0])
    ]
    assert "OXT" in block_type.atom_to_idx
    assert not {"up", "down"} & {c.name for c in block_type.connections}
    source = structure[
        (structure.res_name == "GLY")
        & (structure.res_id == structure.res_id[0])
        & (structure.element != "H")
    ]
    assert len(source) == 5
    offset = int(pose.block_coord_offset[0, 0])
    actual = pose.coords[
        0, [offset + block_type.atom_to_idx[n] for n in source.atom_name]
    ]
    np.testing.assert_allclose(actual.cpu(), source.coord, atol=1e-5)
    # The cropped pocket's default amino-terminus convention is unchanged.
    assert (
        _hydrogens_on(
            pose,
            str(pose.pdb_info.chain_labels[0, 0]),
            int(pose.pdb_info.residue_labels[0, 0]),
        )[2]["N"]
        == 3
    )
    _score_and_minimize_ligand(pose, context.parameter_database)


@pytest.mark.parametrize("trade_heavy", [False, True])
def test_terminal_selection_never_trades_a_supplied_atom(trade_heavy):
    """More hydrogen-name matches cannot compensate for losing a supplied oxygen."""
    from types import SimpleNamespace
    import torch
    from tmol.io.details._select_from_canonical import _termini_the_atoms_state

    # Supplied atoms: backbone O, generic H, H2, H3, terminal OXT.
    # Original amino terminus cannot hold H, H2, H3 or OXT. A compatible
    # carboxyl terminus may recover OXT; an unusual patch that drops O must lose.
    absent = torch.tensor(
        [[False, True, True, True, True], [trade_heavy, True, False, False, False]]
    )
    candidates = torch.zeros((1, 4, 1, 1), dtype=torch.int64)
    candidates[0, 3, 0, 0] = 1
    pbt = SimpleNamespace(
        canonical_ordering_annotation=SimpleNamespace(
            var_combo_candidate_bt_index=candidates,
            var_combo_is_real_candidate=torch.ones_like(candidates, dtype=torch.bool),
            bt_canonical_atom_is_absent=absent,
        )
    )
    result = _termini_the_atoms_state(
        pbt,
        torch.ones((1, 1, 5), dtype=torch.bool),
        torch.zeros((1, 1), dtype=torch.int64),
        torch.zeros((1, 1), dtype=torch.int64),
        torch.zeros((1, 1), dtype=torch.int64),
        torch.tensor([[[False, True]]]),
        torch.ones((1, 1), dtype=torch.bool),
    )
    assert result.item() == (0 if trade_heavy else 3)
