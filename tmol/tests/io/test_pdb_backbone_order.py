"""Explicit backbone links, rather than PDB record order, place modified residues."""

import numpy as np
from biotite.structure import get_residue_starts
from biotite.structure.io.pdb import PDBFile

from tmol.io import atom_array_from_file
from tmol.io._atomworks_reader import (
    _polymer_from_backbone_bonds,
    _read_pdb,
    _stated_hetero_bonds,
)
from tmol.tests.data import data_path

SOURCE = data_path() / "sweep_regressions" / "backbone_hetero_order_4fut.pdb"


def test_appended_modified_residues_rejoin_their_explicit_backbone():
    source = PDBFile.read(SOURCE).get_structure(model=1)
    result = atom_array_from_file(SOURCE)
    starts = get_residue_starts(result)
    assert result.res_id[starts].tolist() == [11, 12, 13, 17, 18, 19]
    assert set(result.chain_id) == {"A"}
    assert result.is_polymer.all()
    assert len(result) == len(source)
    coordinates = {(int(a.res_id), a.atom_name): a.coord for a in source}
    np.testing.assert_array_equal(
        result.coord, [coordinates[(int(a.res_id), a.atom_name)] for a in result]
    )
    actual = {
        frozenset(
            (
                (int(result.res_id[i]), result.atom_name[i]),
                (int(result.res_id[j]), result.atom_name[j]),
            )
        )
        for i, j, _ in result.bonds.as_array()
        if result.res_id[i] != result.res_id[j]
    }
    assert actual == {
        frozenset(((left, "C"), (right, "N")))
        for left, right in [(11, 12), (12, 13), (17, 18), (18, 19)]
    }


def test_unlinked_hetero_residues_do_not_join_by_matching_author_chain(tmp_path):
    lines = SOURCE.read_text().splitlines()
    kind = {
        int(line[6:11]): line[:6]
        for line in lines
        if line.startswith(("ATOM  ", "HETATM"))
    }
    unlinked = []
    for line in lines:
        if line.startswith("CONECT"):
            ids = [
                int(line[k : k + 5])
                for k in range(6, len(line), 5)
                if line[k : k + 5].strip()
            ]
            ids = [ids[0]] + [n for n in ids[1:] if kind[n] == kind[ids[0]]]
            line = "CONECT" + "".join(f"{n:5d}" for n in ids)
        unlinked.append(line)
    path = tmp_path / "unlinked.pdb"
    path.write_text("\n".join(unlinked) + "\n")
    result = _read_pdb(path, 1)
    assert np.all(result.chain_id[result.hetero] != result.auth_asym_id[result.hetero])


def test_declared_backbone_preserves_atom_order_when_numbers_decrease(tmp_path):
    numbering = {11: 100, 12: 10, 13: 90, 17: 80, 18: 20, 19: 70}
    lines = []
    for line in SOURCE.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            line = line[:22] + f"{numbering[int(line[22:26])]:4d}" + line[26:]
        lines.append(line)
    path = tmp_path / "nonmonotonic.pdb"
    path.write_text("\n".join(lines) + "\n")
    result = _read_pdb(path, 1)
    assert result.res_id[get_residue_starts(result)].tolist() == list(
        numbering.values()
    )
    source = PDBFile.read(path).get_structure(model=1, extra_fields=["atom_id"])
    coordinates = dict(zip(source.atom_id.tolist(), source.coord))
    np.testing.assert_array_equal(
        result.coord, [coordinates[i] for i in result.atom_id]
    )


def test_linked_residues_preserve_distinct_chains_with_shared_author_id():
    file = PDBFile.read(SOURCE)
    source = file.get_structure(model=1, include_bonds=True, extra_fields=["atom_id"])
    source.set_annotation("auth_asym_id", source.chain_id.copy())
    source.bonds, declared = _stated_hetero_bonds(source, file.lines)
    source.chain_id[source.res_id >= 17] = "B"
    source.chain_id[source.hetero] = "H"
    result = _polymer_from_backbone_bonds(source, declared_pairs=declared)
    starts = get_residue_starts(result)
    assert result.res_id[starts].tolist() == [11, 12, 13, 17, 18, 19]
    assert result.chain_id[starts].tolist() == ["A", "A", "A", "B", "B", "B"]
    assert set(result.auth_asym_id) == {"A"}


def test_in_sequence_modified_residues_without_conect_keep_backbone(tmp_path):
    records = [
        line
        for line in SOURCE.read_text().splitlines()
        if line.startswith(("ATOM  ", "HETATM"))
        and line[76:78].strip() not in ("H", "D")
    ]
    records.sort(key=lambda line: int(line[22:26]))
    records = [line[:6] + f"{i:5d}" + line[11:] for i, line in enumerate(records, 1)]
    path = tmp_path / "in_sequence.pdb"
    path.write_text("\n".join(records) + "\nEND\n")
    expected = atom_array_from_file(SOURCE)
    expected = expected[~np.isin(expected.element, ("H", "D"))]
    result = atom_array_from_file(path)
    np.testing.assert_array_equal(result.res_id, expected.res_id)
    np.testing.assert_array_equal(result.atom_name, expected.atom_name)
    np.testing.assert_array_equal(result.coord, expected.coord)
    assert result.is_polymer.all()
    assert {tuple(row) for row in result.bonds.as_array()} == {
        tuple(row) for row in expected.bonds.as_array()
    }
