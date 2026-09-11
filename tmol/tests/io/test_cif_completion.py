"""Missing-atom completion preserves residue instances and bond indices."""

import biotite.structure as struc
import numpy as np

from tmol.io import _cif


def component():
    return _cif._component_array(
        "ZZZ", ["C1", "C2", "C3"], ["C"] * 3, [("C1", "C2", 1), ("C2", "C3", 1)]
    )


def test_insertion_codes_keep_missing_atoms_local():
    template = component()
    first, second = template[:2], template.copy()
    first.ins_code[:] = "A"
    second.ins_code[:] = "B"
    first.coord[:] = 1
    second.coord[:] = 2
    original = struc.concatenate([first, second])
    result = _cif.with_unresolved_atoms(original, {"ZZZ": template}, use_ccd=False)
    assert result.array_length() == 6
    np.testing.assert_array_equal(result.ins_code, ["A"] * 3 + ["B"] * 3)
    np.testing.assert_array_equal(result.atom_name, ["C1", "C2", "C3"] * 2)
    assert np.isnan(result.coord[2]).all()
    np.testing.assert_array_equal(result.coord[[0, 1, 3, 4, 5]], original.coord)
    assert {tuple(b) for b in result.bonds.as_array()} == {
        (0, 1, 1),
        (1, 2, 1),
        (3, 4, 1),
        (4, 5, 1),
    }
    assert original.array_length() == 5


def test_connections_distinguish_insertion_codes():
    first, second = component(), component()
    first.ins_code[:] = "A"
    second.ins_code[:] = "B"
    original = struc.concatenate([first, second])
    original.bonds.add_bond(2, 3, struc.BondType.SINGLE)
    assert _cif._connection_atoms_by_name(original) == {"ZZZ": {"C1", "C3"}}


def test_dictionary_is_read_once_per_component_per_call(monkeypatch):
    template = component()
    residues = []
    for i in range(6):
        residue = template[:2].copy()
        residue.res_id[:] = i + 1
        residues.append(residue)
    original = struc.concatenate(residues)
    calls = []
    monkeypatch.setattr(
        _cif,
        "_component_dictionary_template",
        lambda name: calls.append(name) or template,
    )
    result = _cif.with_unresolved_atoms(original, {})
    assert calls == ["ZZZ"]
    assert result.array_length() == 18
    assert result.bonds.get_bond_count() == 12


def test_inserted_atoms_remap_existing_cross_residue_bonds():
    template = component()
    first, second = template[:2].copy(), template[:2].copy()
    second.res_id[:] = 2
    original = struc.concatenate([first, second])
    original.bonds.add_bond(0, 2, struc.BondType.SINGLE)
    original_bonds = original.bonds.as_array().copy()
    result = _cif._inserted(
        original, [0, 2], {0: (["C3"], template), 2: (["C3"], template)}
    )
    assert {tuple(b) for b in result.bonds.as_array()} == {
        (0, 1, 1),
        (0, 3, 1),
        (1, 2, 1),
        (3, 4, 1),
        (4, 5, 1),
    }
    np.testing.assert_array_equal(original.bonds.as_array(), original_bonds)


def test_numbered_hydrogen_names_do_not_block_heavy_atom_completion():
    template = component()
    residue = template.copy()
    residue.atom_name[-1] = "1H"
    residue.element[-1] = "H"
    result = _cif.with_unresolved_atoms(residue, {"ZZZ": template}, use_ccd=False)
    assert list(result.atom_name) == ["C1", "C2", "1H", "C3"]


def test_representative_respects_insertion_codes():
    from tmol.ligand._detect import _representative_instance

    template = component()
    first, second = template[:2].copy(), template.copy()
    first.ins_code[:] = "A"
    second.ins_code[:] = "B"
    second.coord[:] = 1
    original = struc.concatenate([first, second])
    result = _representative_instance(original, struc.get_residue_starts(original), 0)
    assert result.array_length() == 3
    assert (result.ins_code == "B").all()


def test_canonical_residue_selection_does_not_merge_chains():
    from tmol.ligand._preparation import _canonical_residue_array

    first, second = component(), component()
    second.chain_id[:] = "B"
    result = _canonical_residue_array(struc.concatenate([first, second]), "ZZZ")
    assert result.array_length() == 3
    assert (result.chain_id == "A").all()


def test_ligand_detection_sees_bonds_between_insertion_codes():
    from tmol.ligand._detect import _cross_residue_bond_atoms

    first, second = component(), component()
    first.ins_code[:] = "A"
    second.ins_code[:] = "B"
    original = struc.concatenate([first, second])
    original.bonds.add_bond(2, 3, struc.BondType.SINGLE)
    linked, _ = _cross_residue_bond_atoms(original)
    assert linked[("A", 1, "ZZZ")] == {"C1", "C3"}
