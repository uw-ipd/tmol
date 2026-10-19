import io

import numpy
import pytest
import torch
import zstandard

from tmol.io import (
    atom_array_from_file,
    atom_array_from_pose_stack,
    file_from_atom_array,
    file_from_pose_stack,
    pose_stack_from_atom_array,
    pose_stack_from_file,
)
from tmol.io._structure_io import _infer_format
from tmol.tests.data import data_path

MOL2 = data_path("protein_ligand_test") / "ace.lig.mol2"


def _sdf_text():
    fixture = data_path("pdbbind_regressions", "cyclic_acylguanidinium_3ge7.sdf.zst")
    return zstandard.decompress(fixture.read_bytes()).decode()


def test_format_is_inferred_from_the_extension(tmp_path):
    for name, fmt in [
        ("x.pdb", "pdb"),
        ("x.ent", "pdb"),
        ("x.cif", "cif"),
        ("x.cif.zst", "cif"),
        ("x.bcif", "cif"),
        ("x.mol2", "mol2"),
        ("x.sdf", "sdf"),
        ("x.mol", "sdf"),
    ]:
        assert _infer_format(tmp_path / name) == fmt


def test_format_is_inferred_from_the_contents(ubq_pdb):
    cif = file_from_atom_array(atom_array_from_file(ubq_pdb), format="cif")
    assert _infer_format(ubq_pdb) == "pdb"
    assert _infer_format(io.StringIO(ubq_pdb)) == "pdb"
    assert _infer_format(cif) == "cif"
    assert _infer_format(MOL2.read_text()) == "mol2"
    assert _infer_format(_sdf_text()) == "sdf"
    with pytest.raises(ValueError, match="Cannot infer"):
        _infer_format("not\na structure\n")


def test_contents_read_like_the_file(ubq_pdb, tmp_path):
    from_text = atom_array_from_file(ubq_pdb)
    path = tmp_path / "ubq.txt"
    path.write_text(ubq_pdb)
    from_unknown_extension = atom_array_from_file(path)
    from_stated_format = atom_array_from_file(path, "pdb")
    for array in (from_unknown_extension, from_stated_format):
        numpy.testing.assert_array_equal(array.atom_name, from_text.atom_name)
        numpy.testing.assert_allclose(array.coord, from_text.coord)

    ligand = atom_array_from_file(MOL2, res_name="LG1")
    from_mol2_text = atom_array_from_file(MOL2.read_text(), res_name="LG1")
    numpy.testing.assert_array_equal(from_mol2_text.atom_name, ligand.atom_name)
    assert len(atom_array_from_file(_sdf_text())) > 0


def test_only_pdb_and_cif_are_written(ubq_pdb, tmp_path):
    array = atom_array_from_file(ubq_pdb)
    for fmt in ("mol2", "sdf"):
        with pytest.raises(ValueError, match="format must be one of"):
            file_from_atom_array(array, format=fmt)
    with pytest.raises(ValueError, match="format must be one of"):
        file_from_atom_array(array, tmp_path / "x.sdf")
    with pytest.raises(ValueError, match="Pass format="):
        file_from_atom_array(array)


@pytest.mark.parametrize("fmt", ["pdb", "cif"])
def test_written_pose_reads_back(fmt, ubq_pdb, pertuzumab_pdb, tmp_path):
    device = torch.device("cpu")
    for pdb in (ubq_pdb, pertuzumab_pdb):
        ps = pose_stack_from_file(pdb, device, no_optH=True)
        path = file_from_pose_stack(ps, tmp_path / f"out.{fmt}")
        text = file_from_pose_stack(ps, format=fmt)
        # the text route writes the same structure (the header carries a timestamp)
        from_text = atom_array_from_file(text, fmt)
        from_path = atom_array_from_file(path)
        numpy.testing.assert_array_equal(from_text.atom_name, from_path.atom_name)
        numpy.testing.assert_array_equal(from_text.coord, from_path.coord)
        ps2 = pose_stack_from_file(path, device, no_optH=True)

        # PDB keeps three decimals
        torch.testing.assert_close(ps.coords, ps2.coords, atol=1e-3, rtol=0)
        numpy.testing.assert_equal(ps.pdb_info.chain_labels, ps2.pdb_info.chain_labels)
        numpy.testing.assert_equal(
            ps.pdb_info.residue_labels, ps2.pdb_info.residue_labels
        )


def test_ligand_hydrogens_mode_is_validated():
    with pytest.raises(ValueError, match="ligand_hydrogens must be one of"):
        pose_stack_from_atom_array(
            atom_array_from_file(MOL2),
            torch.device("cpu"),
            prepare_ligands=True,
            ligand_hydrogens="trust",
        )


def _ligand_hydrogens(pose_stack, context):
    array = atom_array_from_pose_stack(pose_stack, context.canonical_ordering)
    is_h = array.element == "H"
    return dict(zip(array.atom_name[is_h], array.coord[is_h]))


def test_input_ligand_hydrogens_are_kept_only_on_their_parents():
    device = torch.device("cpu")
    source = atom_array_from_file(MOL2)
    is_h = source.element == "H"
    supplied = dict(zip(source.atom_name[is_h], source.coord[is_h]))

    def build(array, mode):
        return _ligand_hydrogens(
            *pose_stack_from_atom_array(
                array,
                device,
                prepare_ligands=True,
                ligand_hydrogens=mode,
                return_context=True,
            )
        )

    kept = build(source, "input")
    shared = sorted(set(kept) & set(supplied))
    assert shared
    for name in shared:
        numpy.testing.assert_allclose(kept[name], supplied[name], atol=1e-3)

    rebuilt = build(source, "names")
    assert any(
        not numpy.allclose(rebuilt[name], supplied[name], atol=1e-3)
        for name in shared
        if name in rebuilt
    )

    # push one hydrogen out along its bond, far from bond length
    moved_name = shared[0]
    moved = source.copy()
    h = numpy.flatnonzero((moved.atom_name == moved_name) & is_h)[0]
    parent = moved.bonds.get_bonds(h)[0][0]
    bond = moved.coord[h] - moved.coord[parent]
    moved.coord[h] = moved.coord[parent] + 2.5 * bond / numpy.linalg.norm(bond)
    displaced = build(moved, "input")
    assert not numpy.allclose(displaced[moved_name], moved.coord[h], atol=1e-3)
    for name in shared[1:]:
        numpy.testing.assert_allclose(displaced[name], supplied[name], atol=1e-3)
