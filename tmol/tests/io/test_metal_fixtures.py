"""Metal coordination on real structures, checked against expected.yaml."""

import os

import numpy
import pytest
import torch
from yaml import safe_load

import tmol.io.details._metal_detection as metal_detection
from tmol.io import (
    add_metal_coordination,
    atom_array_from_cif,
    canonical_form_from_pose_stack,
    pose_stack_from_biotite,
    pose_stack_from_canonical_form,
    remove_metal_coordination,
)
from tmol.kinematics import EdgeType, FoldForest
from tmol.pose import PoseStackBuilder
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


@pytest.mark.parametrize("stem", fixture_params())
def test_every_donor_is_connected_to_its_metal(built, stem):
    pose_stack, (co, res_types, assignments) = built(stem)
    pbt = pose_stack.packed_block_types
    irc = pose_stack.inter_residue_connections64
    for (pose, metal), got in assignments:
        metal_bt = pbt.active_block_types[int(pose_stack.block_type_ind[pose, metal])]
        partners = set()
        for name in metal_bt.metal_sites[0].site_connections:
            res, conn = irc[pose, metal, metal_bt.connection_to_cidx[name]].tolist()
            if res >= 0:
                bt = pbt.active_block_types[int(pose_stack.block_type_ind[pose, res])]
                partners.add((res, bt.connections[conn].atom))
        expected = set()
        for res, atom in got.donor_atoms:
            name3 = co.restype_io_equiv_classes[int(res_types[pose, res])]
            expected.add((res, co.restypes_ordered_atom_names[name3][atom]))
        assert partners == expected, metal_bt.name


@pytest.mark.parametrize("stem", fixture_params())
def test_metal_is_reached_by_a_jump(built, stem):
    # metal bonds close on the tree instead of building it, so the ion keeps
    #    its rigid-body freedom
    pose_stack, (_, _, assignments) = built(stem)
    ff = FoldForest.reasonable_fold_forest(pose_stack)
    edges = ff.edges[0, : ff.n_edges[0]].tolist()
    jumps = {int(EdgeType.jump), int(EdgeType.root_jump)}
    for (_, metal), _ in assignments:
        into = [e[0] for e in edges if e[2] == metal]
        assert len(into) == 1 and into[0] in jumps


def residue_index(pose_stack, chain, label):
    info = pose_stack.pdb_info
    for res in range(pose_stack.max_n_blocks):
        if (str(info.chain_labels[0, res]), int(info.residue_labels[0, res])) == (
            chain,
            label,
        ):
            return res
    raise KeyError((chain, label))


def block_type_names(pose_stack):
    bts = pose_stack.packed_block_types.active_block_types
    return [bts[i].name if i >= 0 else None for i in pose_stack.block_type_ind[0]]


def metal_bonds(pose_stack):
    """{(metal, site): (donor, donor atom)} for every filled metal site."""
    pbt = pose_stack.packed_block_types
    irc = pose_stack.inter_residue_connections64[0].tolist()
    out = {}
    for res, bt_ind in enumerate(pose_stack.block_type_ind64[0].tolist()):
        if bt_ind < 0 or not pbt.active_block_types[bt_ind].metal_sites:
            continue
        bt = pbt.active_block_types[bt_ind]
        for k, name in enumerate(bt.metal_sites[0].site_connections):
            partner, conn = irc[res][bt.connection_to_cidx[name]]
            if partner >= 0:
                other = pbt.active_block_types[pose_stack.block_type_ind64[0, partner]]
                out[(res, k)] = (partner, other.connections[conn].atom)
    return out


@pytest.mark.parametrize("stem", fixture_params())
def test_round_trip_keeps_metal_coordination(built, stem):
    pose_stack, (co, _, _) = built(stem)
    cf = canonical_form_from_pose_stack(co, pose_stack)
    rebuilt = pose_stack_from_canonical_form(
        co,
        pose_stack.packed_block_types,
        *cf,
        trust_hydrogen_names=True,
        find_additional_metal_coordination=False,
    )
    assert block_type_names(rebuilt) == block_type_names(pose_stack)
    assert metal_bonds(rebuilt) == metal_bonds(pose_stack)
    torch.testing.assert_close(rebuilt.coords, pose_stack.coords)


def test_remove_then_add_a_donor(built):
    pose_stack, (co, _, _) = built("zn_tetrahedral_3ks3")
    zn = residue_index(pose_stack, "A", 262)
    his = residue_index(pose_stack, "A", 119)
    bonds = metal_bonds(pose_stack)
    (site,) = [k for (m, k), d in bonds.items() if m == zn and d == (his, "ND1")]

    opened = remove_metal_coordination(co, pose_stack, 0, zn, site)
    assert (zn, site) not in metal_bonds(opened)
    assert len(metal_bonds(opened)) == len(bonds) - 1
    his_bt = opened.packed_block_types.active_block_types[
        opened.block_type_ind64[0, his]
    ]
    assert all(c.kinematic for c in his_bt.connections), his_bt.name

    # the donor goes back to the site its virtual still points at
    closed = add_metal_coordination(co, opened, 0, zn, his, "ND1")
    assert metal_bonds(closed) == bonds
    assert block_type_names(closed) == block_type_names(pose_stack)


def test_one_atom_bridges_two_metals(built):
    pose_stack, (co, _, _) = built("cu_zn_sod_3f7l")
    cu = residue_index(pose_stack, "A", 201)
    his118 = residue_index(pose_stack, "A", 118)
    his61 = residue_index(pose_stack, "A", 61)
    (site,) = [
        k for (m, k), d in metal_bonds(pose_stack).items() if d == (his118, "NE2")
    ]
    opened = remove_metal_coordination(co, pose_stack, 0, cu, site)
    bridged = add_metal_coordination(co, opened, 0, cu, his61, "ND1", site=site)

    holders = [m for (m, _), d in metal_bonds(bridged).items() if d == (his61, "ND1")]
    assert sorted(holders) == sorted([cu, residue_index(pose_stack, "A", 203)])
    bt = bridged.packed_block_types.active_block_types[
        bridged.block_type_ind64[0, his61]
    ]
    assert sum(c.atom == "ND1" and not c.kinematic for c in bt.connections) == 2


def test_donor_forms_accumulate_and_stack():
    # built from the default context, the second structure's donors extend
    #    the first's packed set rather than starting a new one
    def build(stem):
        structure = atom_array_from_cif(os.path.join(FIXTURE_DIR, stem + ".cif.gz"))
        return pose_stack_from_biotite(structure, torch.device("cpu"))

    zinc = build("zn_tetrahedral_3ks3")
    iron = build("fe_rubredoxin_30oh")
    first, grown = zinc.packed_block_types, iron.packed_block_types
    n = len(first.chem_db.residues)
    assert n <= len(grown.chem_db.residues)
    assert all(
        a is b for a, b in zip(grown.chem_db.residues[:n], first.chem_db.residues)
    )

    # a structure needing nothing new reuses the newest packed set as is
    assert build("zn_tetrahedral_3ks3").packed_block_types is grown

    stacked = PoseStackBuilder.from_poses([zinc, iron], torch.device("cpu"))
    assert stacked.packed_block_types is grown
    assert block_type_names(stacked)[: zinc.max_n_blocks] == block_type_names(zinc)
