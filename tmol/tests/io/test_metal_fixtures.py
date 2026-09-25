"""Metal coordination on real structures, checked against expected.yaml."""

import os

import biotite.structure as struc
from biotite.structure.io import pdbx
import numpy
import pytest
import torch
from yaml import safe_load

import tmol.io.details._metal_detection as metal_detection
from tmol.io import (
    add_metal_coordination,
    atom_array_from_cif,
    biotite_from_pose_stack,
    canonical_form_from_pose_stack,
    pose_stack_from_biotite,
    pose_stack_from_canonical_form,
    remove_metal_coordination,
    write_pose_stack_pdb,
)
from tmol.database.chemical import site_connections
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
        expected = metal.get("block_type") or (
            f"{metal['comp']}_{metal['geometry'] or 'irregular'}"
        )
        assert names[(metal["chain"], metal["res"])] == expected


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
    # a histidine coordinating through NE2 must be HIS_D, a cysteine CYS_DEP
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
        for name in site_connections(metal_bt):
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


def labeled_metal_bonds(pose_stack):
    """Metal-donor bonds by author labels, independent of residue order."""
    return {
        (residue_key(pose_stack, 0, m), residue_key(pose_stack, 0, d), atom)
        for (m, _), (d, atom) in metal_bonds(pose_stack).items()
    }


@pytest.mark.parametrize("stem", ["sf4_ferredoxin_2fdn", "sf4_ferredoxin_1fdn"])
def test_cluster_sites_face_their_donors_in_either_naming(
    built, stem, default_database
):
    # 2FDN names its cubane atoms in the mirror sense of the SF4 template,
    #    1FDN in the same sense; every free site must face its own cysteine
    from tmol.tests.score.metal.metal_oracle import restraints

    pose_stack, _ = built(stem)
    site_rows, _, fan_rows, fan_params = restraints(default_database, pose_stack)
    xyz = pose_stack.coords[0].double()
    offset = pose_stack.block_coord_offset64[0]

    def at(block, atom):
        return xyz[offset[block] + atom]

    assert len(site_rows) == 8
    for _, metal_block, metal, virt, donor_block, donor in site_rows:
        m = at(metal_block, metal)
        ray = at(metal_block, virt) - m
        ray = ray / ray.norm()
        delta = at(donor_block, donor) - m
        assert float((delta - (delta @ ray) * ray).norm()) < 1.0
    for (_, block, a, b), (l0, _) in zip(fan_rows, fan_params):
        assert abs(float((at(block, a) - at(block, b)).norm()) - l0) < 0.3


def test_written_pdb_has_no_virtual_atoms(built, tmp_path):
    pose_stack, _ = built("zn_tetrahedral_3ks3")
    path = tmp_path / "out.pdb"
    write_pose_stack_pdb(pose_stack, str(path))
    with open(path) as infile:
        zinc = [line[12:16].strip() for line in infile if line[17:20].strip() == "ZN"]
    assert zinc == ["ZN"]


def test_exported_structure_rebuilds_the_same_coordination(built, tmp_path):
    pose_stack, (co, _, _) = built("zn_tetrahedral_3ks3")
    structure = biotite_from_pose_stack(pose_stack, co)
    assert list(structure.atom_name[structure.res_name == "ZN"]) == ["ZN"]
    cif = pdbx.CIFFile()
    pdbx.set_structure(cif, structure)
    cif.write(str(tmp_path / "out.cif"))
    rebuilt = pose_stack_from_biotite(
        atom_array_from_cif(str(tmp_path / "out.cif")),
        torch.device("cpu"),
        prepare_ligands=True,
    )
    assert labeled_metal_bonds(rebuilt) == labeled_metal_bonds(pose_stack)

    with_virtuals = biotite_from_pose_stack(pose_stack, co, include_virtual_atoms=True)
    zinc = with_virtuals.atom_name[with_virtuals.res_name == "ZN"]
    assert list(zinc) == ["ZN", "V1", "V2", "V3", "V4"]


def zinc_with_declared_bonds(donors, bond_type):
    """3KS3, with the zinc declared bonded to (res_id, res_name, atom) donors."""
    structure = atom_array_from_cif(
        os.path.join(FIXTURE_DIR, "zn_tetrahedral_3ks3.cif.gz")
    )
    (zinc,) = numpy.flatnonzero(structure.res_name == "ZN")
    # replace the file's own metalc declarations
    kept = structure.bonds.as_array()
    kept = kept[(kept[:, 0] != zinc) & (kept[:, 1] != zinc)]
    bonds = struc.BondList(structure.array_length(), kept)
    for res_id, res_name, atom_name in donors:
        (atom,) = numpy.flatnonzero(
            (structure.res_id == res_id)
            & (structure.res_name == res_name)
            & (structure.atom_name == atom_name)
            & (structure.chain_id == structure.chain_id[0])
        )
        bonds.add_bond(zinc, atom, bond_type)
    structure.bonds = bonds
    return structure, zinc


EXPECTED_ZINC_DONORS = [
    (d["res"], d["comp"], d["atom"])
    for d in EXPECTED["zn_tetrahedral_3ks3"]["donors"]["A/ZN/262"]
]


@pytest.mark.parametrize(
    "bond_type",
    [struc.BondType.ANY, struc.BondType.COORDINATION],
    ids=["pdb_conect", "cif_metalc"],
)
def test_declared_metal_bonds_are_coordination(built, bond_type):
    # a PDB lists coordination in CONECT; a CIF's metalc reads as COORDINATION
    pose_stack, _ = built("zn_tetrahedral_3ks3")
    structure, _ = zinc_with_declared_bonds(EXPECTED_ZINC_DONORS, bond_type)
    rebuilt, context = pose_stack_from_biotite(
        structure, torch.device("cpu"), prepare_ligands=True, return_context=True
    )
    assert labeled_metal_bonds(rebuilt) == labeled_metal_bonds(pose_stack)
    cf = canonical_form_from_pose_stack(context.canonical_ordering, rebuilt)
    assert cf.covalent_bonds is None


def test_declared_bond_beyond_cutoff_is_kept(built):
    pose_stack, _ = built("zn_tetrahedral_3ks3")
    structure, zinc = zinc_with_declared_bonds([], struc.BondType.ANY)
    # the nearest backbone carbonyl, well past the detection cutoff
    carbonyl = numpy.flatnonzero(
        (structure.atom_name == "O") & (structure.res_name != "HOH")
    )
    dist = numpy.linalg.norm(structure.coord[carbonyl] - structure.coord[zinc], axis=1)
    far = carbonyl[dist > 3.0][numpy.argmin(dist[dist > 3.0])]
    far_donor = (int(structure.res_id[far]), str(structure.res_name[far]), "O")
    structure, _ = zinc_with_declared_bonds([far_donor], struc.BondType.ANY)

    rebuilt = pose_stack_from_biotite(
        structure, torch.device("cpu"), prepare_ligands=True
    )
    bonds = labeled_metal_bonds(rebuilt)
    assert (("A", 262), ("A", far_donor[0]), "O") in bonds
    assert labeled_metal_bonds(pose_stack) <= bonds


def test_only_declared_bonds_without_detection(built):
    structure, _ = zinc_with_declared_bonds(
        EXPECTED_ZINC_DONORS[:2], struc.BondType.ANY
    )
    rebuilt = pose_stack_from_biotite(
        structure,
        torch.device("cpu"),
        prepare_ligands=True,
        find_additional_metal_coordination=False,
    )
    donors = {(donor[1], atom) for _, donor, atom in labeled_metal_bonds(rebuilt)}
    assert donors == {(res, atom) for res, _, atom in EXPECTED_ZINC_DONORS[:2]}
