"""Geometry may reject supplied topology, but may only revise inferred edges."""

import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile, CIFCategory, set_structure
import numpy as np
import pytest

from tmol.io import atom_array_from_cif, canonical_form_from_biotite
from tmol.io._atomworks_reader import (
    INFERRED_POLYMER_BOND,
    _mark_inferred_polymer_bonds,
)
from tmol.tests.data import data_path


def _gapped_dipeptide():
    atoms = PDBFile.read(data_path("pdb", "1ubq.pdb")).get_structure(model=1)
    atoms = atoms[(atoms.chain_id == "A") & (atoms.res_id <= 2)]
    atoms.coord[atoms.res_id == 2] += [20, 0, 0]
    atoms.bonds = struc.connect_via_residue_names(atoms)
    carbon = int(np.flatnonzero((atoms.res_id == 1) & (atoms.atom_name == "C"))[0])
    nitrogen = int(np.flatnonzero((atoms.res_id == 2) & (atoms.atom_name == "N"))[0])
    return atoms, carbon, nitrogen


@pytest.mark.parametrize(
    "source",
    [
        "array",
        "CONECT",
        "LINK",
        "LINK_decreasing",
        "LINK_nonconsecutive",
        "struct_conn",
    ],
)
def test_supplied_polymer_bond_is_not_silently_cut(tmp_path, torch_device, source):
    atoms, carbon, nitrogen = _gapped_dipeptide()
    if source == "LINK_decreasing":
        atoms.res_id = 3 - atoms.res_id
    elif source == "LINK_nonconsecutive":
        atoms.res_id = 2 * atoms.res_id + 8
    if source == "struct_conn":
        file = CIFFile()
        set_structure(file, atoms, include_bonds=True)
        # Standard peptide links are omitted by the writer; declare this one.
        file.block["struct_conn"] = CIFCategory(
            {
                "id": ["explicit"],
                "conn_type_id": ["covale"],
                "pdbx_value_order": ["sing"],
                "ptnr1_label_asym_id": ["A"],
                "ptnr1_label_seq_id": ["1"],
                "ptnr1_label_comp_id": ["MET"],
                "ptnr1_label_atom_id": ["C"],
                "ptnr2_label_asym_id": ["A"],
                "ptnr2_label_seq_id": ["2"],
                "ptnr2_label_comp_id": ["GLN"],
                "ptnr2_label_atom_id": ["N"],
            }
        )
        path = tmp_path / "explicit.cif"
        file.write(path)
        atoms = atom_array_from_cif(path)
    elif source != "array":
        atoms.bonds = None
        file = PDBFile()
        file.set_structure(atoms)
        if source == "CONECT":
            file.lines.append(f"CONECT{carbon + 1:5d}{nitrogen + 1:5d}")
        else:
            link = list(" " * 80)
            link[:6] = "LINK  "
            for offset, atom in ((12, carbon), (42, nitrogen)):
                link[offset : offset + 4] = f"{atoms.atom_name[atom]:>4}"
                link[offset + 5 : offset + 8] = atoms.res_name[atom]
                link[offset + 9] = atoms.chain_id[atom]
                link[offset + 10 : offset + 14] = f"{atoms.res_id[atom]:4d}"
            file.lines.append("".join(link))
        path = tmp_path / "explicit.pdb"
        file.write(path)
        atoms = atom_array_from_cif(path)

    coordinates, bonds = atoms.coord.copy(), atoms.bonds.as_array().copy()
    with pytest.raises(
        ValueError, match="Declared polymer bond crosses a geometry gap"
    ):
        canonical_form_from_biotite(atoms, torch_device)
    retained = canonical_form_from_biotite(
        atoms, torch_device, missing_density_distance_threshold=0
    )
    assert retained.covalent_bonds.shape[0] == 1
    np.testing.assert_equal(atoms.coord, coordinates)
    np.testing.assert_array_equal(atoms.bonds.as_array(), bonds)


def test_inferred_edge_tags_survive_reordering_but_do_not_match_other_inputs():
    first, carbon, nitrogen = _gapped_dipeptide()
    second = first.copy()
    for atoms in (first, second):
        _mark_inferred_polymer_bonds(atoms, set())
        tags = atoms.get_annotation(INFERRED_POLYMER_BOND)
        assert tags[carbon] and tags[carbon] == tags[nitrogen]
        reordered = atoms[[nitrogen, carbon]]
        assert reordered.get_annotation(INFERRED_POLYMER_BOND).tolist() == [
            tags[carbon],
            tags[carbon],
        ]
    assert (
        first.get_annotation(INFERRED_POLYMER_BOND)[carbon]
        != second.get_annotation(INFERRED_POLYMER_BOND)[nitrogen]
    )


def test_supplied_or_ambiguous_ports_are_not_marked_inferred():
    atoms, carbon, nitrogen = _gapped_dipeptide()
    _mark_inferred_polymer_bonds(atoms, {frozenset((carbon, nitrogen))})
    assert not atoms.get_annotation(INFERRED_POLYMER_BOND).any()
    sidechain = int(np.flatnonzero((atoms.res_id == 2) & (atoms.atom_name == "CB"))[0])
    atoms.bonds.add_bond(carbon, sidechain, struc.BondType.SINGLE)
    _mark_inferred_polymer_bonds(atoms, set())
    assert not atoms.get_annotation(INFERRED_POLYMER_BOND).any()


@pytest.mark.parametrize("inferred", [False, True])
def test_gap_uses_the_connected_ports_not_the_reverse_contact(torch_device, inferred):
    atoms, carbon, nitrogen = _gapped_dipeptide()
    first_n = np.flatnonzero((atoms.res_id == 1) & (atoms.atom_name == "N"))[0]
    second_c = np.flatnonzero((atoms.res_id == 2) & (atoms.atom_name == "C"))[0]
    atoms.coord[atoms.res_id == 2] += (
        atoms.coord[first_n] - atoms.coord[second_c] + [1.3, 0, 0]
    )
    assert np.linalg.norm(atoms.coord[carbon] - atoms.coord[nitrogen]) > 2.4
    if inferred:
        _mark_inferred_polymer_bonds(atoms, set())
        result = canonical_form_from_biotite(atoms, torch_device)
        assert result.covalent_bonds.shape[0] == 0
        assert bool(result.res_not_connected[0, 0, 1])
        assert bool(result.res_not_connected[0, 1, 0])
    else:
        with pytest.raises(ValueError, match="Declared polymer bond"):
            canonical_form_from_biotite(atoms, torch_device)
