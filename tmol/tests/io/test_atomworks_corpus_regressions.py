"""Difficult, unmodified AtomWorks fixtures and their explicit input contracts."""

from pathlib import Path

import numpy as np
import biotite.structure.io.pdbx as pdbx
import pytest
import torch

from tmol.io import atom_array_from_cif, pose_stack_from_cif
from tmol.io._pose_stack_from_biotite import (
    _map_atoms_to_canonical,
    canonical_form_from_biotite,
    canonical_ordering_for_biotite,
)
from tmol.score import beta2016_score_function

DATA = Path(__file__).parents[1] / "data" / "atomworks_regressions"


def test_unknown_heavy_atom_is_not_silently_deleted():
    # The fixture deliberately conflicts: label XYZ versus author CG. The
    # native reader legitimately uses CG; the label-based view must reject XYZ.
    cif = pdbx.CIFFile.read(DATA / "unknown_heavy_atom_1a8o.cif")
    array = pdbx.get_structure(cif, model=1, use_author_fields=False)
    assert np.count_nonzero(array.atom_name == "XYZ") == 1
    with pytest.raises(ValueError, match="Heavy atoms.*ASP.*XYZ"):
        canonical_form_from_biotite(array, torch.device("cpu"))


def test_author_named_view_preserves_the_conflicting_label_atom_coordinate():
    path = DATA / "unknown_heavy_atom_1a8o.cif"
    source = pdbx.CIFFile.read(path).block["atom_site"]
    selected = source["label_atom_id"].as_array(str) == "XYZ"
    expected = np.column_stack(
        [
            source[name].as_array(float)[selected]
            for name in ("Cartn_x", "Cartn_y", "Cartn_z")
        ]
    )
    array = atom_array_from_cif(path)
    actual = array[(array.res_id == 152) & (array.atom_name == "CG")].coord
    np.testing.assert_allclose(actual, expected, atol=1e-5)


@pytest.mark.parametrize("element", ["H", "D"])
def test_unrecognized_hydrogen_names_can_be_rebuilt(element):
    mask, atoms, residues = _map_atoms_to_canonical(
        canonical_ordering_for_biotite(),
        np.array([0, 0]),
        ["ALA", "ALA"],
        ["CA", "extra_H"],
        ["C", element],
    )
    np.testing.assert_array_equal(mask, [True, False])
    assert len(atoms) == len(residues) == 1


def test_schiff_base_cannot_lose_its_incomplete_lysine_partner():
    with pytest.raises(ValueError, match="discard.*covalent"):
        pose_stack_from_cif(
            DATA / "schiff_base_double_bond.cif",
            torch.device("cpu"),
            prepare_ligands=True,
            ligand_seed=20260909,
            no_optH=True,
        )


def test_entirely_unresolved_ligand_keeps_its_chemical_identity():
    array = atom_array_from_cif(DATA / "unresolved_unl.cif")
    ligand = array[array.res_name == "UNL"]
    assert len(ligand) == 28
    assert np.isnan(ligand.coord).all()
    assert ligand.bonds.get_bond_count() > 0
    # There is no placement anchor. Do not manufacture a positioned ligand.
    with pytest.raises(RuntimeError, match="UNL|missing"):
        pose_stack_from_cif(
            DATA / "unresolved_unl.cif",
            torch.device("cpu"),
            prepare_ligands=True,
            ligand_seed=20260909,
            no_optH=True,
        )


@pytest.mark.parametrize(
    "filename", ["conditional_generation.cif", "acetylated_peptide_1j8z.cif"]
)
def test_backbone_only_and_crosslinked_modified_peptides_build_and_score(
    filename, torch_device
):
    path = DATA / filename
    pose, context = pose_stack_from_cif(
        path,
        torch_device,
        prepare_ligands=True,
        ligand_seed=20260909,
        no_optH=True,
        return_context=True,
    )
    assert torch.isfinite(pose.coords[pose.real_atoms]).all()
    coords = pose.coords.detach().clone().requires_grad_()
    score = beta2016_score_function(torch_device, param_db=context.parameter_database)
    energy = score.render_whole_pose_scoring_module(pose)(coords)
    energy.sum().backward()
    assert torch.isfinite(energy).all()
    assert torch.isfinite(coords.grad).all()
    if filename == "acetylated_peptide_1j8z.cif":
        types = [
            pose.packed_block_types.active_block_types[int(i)]
            for i in pose.block_type_ind[0]
        ]
        assert types[4].name.split(":")[0] == "BCX"
        assert types[4].properties.polymer.is_polymer
        assert (
            int(pose.inter_residue_connections[0, 3, types[3].up_connection_ind, 0])
            == 4
        )
        assert (
            int(pose.inter_residue_connections[0, 4, types[4].up_connection_ind, 0])
            == 5
        )
