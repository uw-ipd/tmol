"""Metal coordination on real structures, checked against expected.yaml."""

import os

import numpy
import pytest
import torch
from yaml import safe_load

import tmol.io.details._metal_detection as metal_detection
from tmol.io import atom_array_from_cif, pose_stack_from_biotite
from tmol.tests.data import data_path

FIXTURE_DIR = data_path("metal_fixtures")

with open(os.path.join(FIXTURE_DIR, "expected.yaml")) as infile:
    EXPECTED = safe_load(infile)["fixtures"]


def fixture_params():
    for stem, spec in EXPECTED.items():
        marks = []
        if "xfail" in spec:
            marks.append(pytest.mark.xfail(reason=spec["xfail"], strict=True))
        yield pytest.param(stem, marks=marks, id=stem)


@pytest.fixture(scope="module")
def built():
    """Build each fixture once, keeping what metal detection decided."""
    cache = {}

    def build(stem):
        if stem in cache:
            return cache[stem]
        captured = []
        find = metal_detection.find_metal_geometries

        def recording(canonical_ordering, chemical_db, res_types, coords, **kwargs):
            variants, assignments = find(
                canonical_ordering, chemical_db, res_types, coords, **kwargs
            )
            captured.append((canonical_ordering, res_types, assignments))
            return variants, assignments

        metal_detection.find_metal_geometries = recording
        try:
            structure = atom_array_from_cif(os.path.join(FIXTURE_DIR, stem + ".cif.gz"))
            pose_stack = pose_stack_from_biotite(
                structure, torch.device("cpu"), prepare_ligands=True
            )
        finally:
            metal_detection.find_metal_geometries = find
        cache[stem] = (pose_stack, captured[-1])
        return cache[stem]

    return build


def residue_key(pose_stack, pose, res):
    info = pose_stack.pdb_info
    return (str(info.chain_labels[pose, res]), int(info.residue_labels[pose, res]))


@pytest.mark.parametrize("stem", fixture_params())
def test_metal_geometry_selects_the_block_type(built, stem):
    pose_stack, _ = built(stem)
    spec = EXPECTED[stem]
    names = {}
    for res, bt_index in enumerate(pose_stack.block_type_ind[0].tolist()):
        if bt_index >= 0:
            bt = pose_stack.packed_block_types.active_block_types[bt_index]
            names[residue_key(pose_stack, 0, res)] = bt.name
    for metal in spec["metals"]:
        geometry = metal["geometry"] or "irregular"
        assert names[(metal["chain"], metal["res"])] == f"{metal['comp']}_{geometry}"


@pytest.mark.parametrize("stem", fixture_params())
def test_detected_donors_match_expected(built, stem):
    pose_stack, (co, res_types, assignments) = built(stem)
    spec = EXPECTED[stem]
    found = {}
    for (pose, res), got in assignments:
        donors = set()
        for other, atom in got.donor_atoms:
            name3 = co.restype_io_equiv_classes[int(res_types[pose, other])]
            atom_name = co.restypes_ordered_atom_names[name3][atom]
            donors.add((*residue_key(pose_stack, pose, other), atom_name))
        found[residue_key(pose_stack, pose, res)] = (donors, got)

    for metal in spec["metals"]:
        key = (metal["chain"], metal["res"])
        donors, got = found[key]
        expected = {
            (d["chain"], d["res"], d["atom"])
            for d in spec["donors"][f"{metal['chain']}/{metal['comp']}/{metal['res']}"]
        }
        assert donors == expected, key
        if "n_open_sites" in metal:
            assert got.n_open_sites == metal["n_open_sites"], key


@pytest.mark.parametrize("stem", fixture_params())
def test_every_donor_is_built_in_a_form_that_can_donate(built, stem, default_database):
    # a histidine coordinating through NE2 must be HIS_D, a cysteine CYS_D
    pose_stack, (co, res_types, assignments) = built(stem)
    donor_type = {
        at.name for at in default_database.chemical.atom_types if at.is_metal_donor
    }
    for (pose, _), got in assignments:
        for other, atom in got.donor_atoms:
            name3 = co.restype_io_equiv_classes[int(res_types[pose, other])]
            atom_name = co.restypes_ordered_atom_names[name3][atom]
            bt_index = int(pose_stack.block_type_ind[pose, other])
            bt = pose_stack.packed_block_types.active_block_types[bt_index]
            atom_type = {a.name: a.atom_type for a in bt.atoms}[atom_name]
            assert atom_type in donor_type, (bt.name, atom_name)


@pytest.mark.parametrize("stem", fixture_params())
def test_every_atom_is_built(built, stem):
    # site virtuals have no frame of their own; they must still be placed
    pose_stack, _ = built(stem)
    assert numpy.isfinite(pose_stack.coords.numpy()).all()
