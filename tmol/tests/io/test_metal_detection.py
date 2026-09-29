import types

import numpy
import pytest
import torch

import tmol.database
from tmol.database.chemical import (
    is_metal_cluster,
    metal_geometry_variant_index,
    metal_table,
    special_case_variant_index,
)
from tmol.io import CanonicalOrdering
from tmol.io.details._metal_detection import (
    MetalSiteAssignment,
    assign_one,
    build_canonical_metal_tables,
    find_metal_geometries,
    gather_candidates,
    ideal_distances,
    place_site_virtuals,
)
from tmol.io.details._protonation_variants import select_protonation_variants


def ion_named(element, ox):
    for ion in metal_table()["ions"]:
        if ion["element"] == element and ion["oxidation_state"] == ox:
            return ion
    raise KeyError(f"{element}{ox}")


def vertices(name):
    for g in metal_table()["geometries"]:
        if g["name"] == name:
            return numpy.asarray(g["vertices"])
    raise KeyError(name)


def place(directions, distance):
    """Donors at a given distance along each direction, metal at the origin."""
    d = numpy.asarray(directions, dtype=numpy.float64)
    return d / numpy.linalg.norm(d, axis=1, keepdims=True) * distance


def test_missing_donor_element_falls_back_rather_than_raising():
    dists = ideal_distances(ion_named("Zn", 2), metal_table()["donor_radii"])
    keep, _ = gather_candidates(
        numpy.zeros(3), place([[1, 0, 0]], 2.0), ["Se"], dists, 0.55
    )
    assert len(keep) == 1


def test_derived_distances_fill_only_what_was_not_measured():
    ion = ion_named("Ca", 2)
    dists = ideal_distances(ion, metal_table()["donor_radii"])
    assert dists["O"] == ion["distances"]["O"], "a measured value must survive"
    assert "S" not in ion["distances"], "calcium-sulfur was not measured"
    assert dists["S"] == pytest.approx(ion["ionic_radius"] + 1.549, abs=1e-3)


def test_candidates_are_gathered_without_reference_to_geometry():
    # the cutoff scales with the donor element, not with any polyhedron
    dists = ideal_distances(ion_named("Zn", 2), metal_table()["donor_radii"])
    # zinc-oxygen is 2.033 measured, so 0.55 A past it admits out to 2.58
    xyz = numpy.array([[2.03, 0, 0], [2.45, 0, 0], [2.70, 0, 0], [4.0, 0, 0]])
    keep, excess = gather_candidates(numpy.zeros(3), xyz, ["O"] * 4, dists, 0.55)
    assert list(keep) == [0, 1], "2.70 and 4.0 A are both past tolerance"
    assert excess[0] < excess[1]


def _seven_zinc_contacts():
    # the case that hard-exits in Rosetta: zinc tops out at six sites, so the
    #    contact furthest past ideal has to go
    donors = place(list(vertices("octahedral")) + [[1.0, 1.0, 1.0]], 2.03)
    donors[6] *= 1.2
    return donors


@pytest.mark.parametrize(
    "ion, donors, element, geometry, expected",
    [
        pytest.param(
            ("Zn", 2),
            place(vertices("tetrahedral"), 2.03),
            "N",
            None,
            dict(geometry="tetrahedral", n_donors=4, n_open_sites=0),
            id="clean_tetrahedral_zinc",
        ),
        # five waters dropped on input; the geometry has to come from the table
        pytest.param(
            ("Mg", 2),
            place([[0, 0, 1]], 2.07),
            "O",
            None,
            dict(geometry="octahedral", n_donors=1, n_open_sites=5),
            id="magnesium_with_one_donor",
        ),
        # every contact coordinates; none is dropped
        pytest.param(
            ("Ca", 2),
            place([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]], 2.34),
            "O",
            None,
            dict(
                geometry=None, how_chosen="untemplated", n_donors=5, n_open_sites=None
            ),
            id="untemplated_calcium",
        ),
        # a fifth donor promotes zinc to a larger geometry instead of being dropped
        pytest.param(
            ("Zn", 2),
            place(vertices("trigonal_bipyramidal"), 2.03),
            "N",
            None,
            dict(geometry="trigonal_bipyramidal", n_donors=5, rejected=()),
            id="fifth_donor_promotes_zinc",
        ),
        pytest.param(
            ("Zn", 2),
            _seven_zinc_contacts(),
            "N",
            None,
            dict(geometry="octahedral", n_donors=6, rejected=(6,)),
            id="seventh_donor_drops_the_weakest",
        ),
        pytest.param(
            ("Zn", 2),
            place(vertices("tetrahedral"), 2.03),
            "N",
            "octahedral",
            dict(geometry="octahedral", how_chosen="declared", n_open_sites=2),
            id="declared_geometry_overrides_inference",
        ),
        pytest.param(
            ("Cu", 2),
            place(vertices("square_planar"), 1.99),
            "N",
            None,
            dict(geometry="square_planar"),
            id="square_planar_copper_is_not_tetrahedral",
        ),
        # nothing to fit means nothing to claim
        pytest.param(
            ("Zn", 2),
            numpy.zeros((0, 3)),
            "N",
            None,
            dict(geometry=None, n_donors=0),
            id="isolated_metal",
        ),
    ],
)
def test_assign_one(ion, donors, element, geometry, expected):
    got = assign_one(
        numpy.zeros(3),
        ion_named(*ion),
        donors,
        [element] * len(donors),
        metal_table(),
        geometry=geometry,
    )
    assert isinstance(got, MetalSiteAssignment)
    assert {name: getattr(got, name) for name in expected} == expected
    if got.geometry is not None:
        sites = got.vertex_for_donor
        assert len(set(sites)) == len(sites) == got.n_donors, "one donor per site"
        assert set(sites) <= set(range(len(vertices(got.geometry))))


def canonical_ordering(db):
    return CanonicalOrdering.from_chemdb(db.chemical)


def metal_tables(db):
    return build_canonical_metal_tables(
        canonical_ordering(db), db.chemical, metal_table()
    )


def find(db, residues, **kwargs):
    """find_metal_geometries on one pose of ``(name3, {atom: xyz})`` residues."""
    co = canonical_ordering(db)
    index_of = {c: i for i, c in enumerate(co.restype_io_equiv_classes)}
    res_types = torch.tensor([[index_of[n] for n, _ in residues]], dtype=torch.int32)
    coords = torch.full((1, len(residues), co.max_n_canonical_atoms, 3), numpy.nan)
    for i, (name, atoms) in enumerate(residues):
        for atom, xyz in atoms.items():
            coords[0, i, co.restypes_atom_index_mapping[name][atom]] = torch.tensor(xyz)
    return find_metal_geometries(co, db.chemical, res_types, coords, **kwargs)


def test_every_supported_ion_is_found_in_the_canonical_ordering(
    default_database: tmol.database.ParameterDatabase,
):
    co = canonical_ordering(default_database)
    tables = metal_tables(default_database)
    found = {co.restype_io_equiv_classes[i] for i in tables.ion_for_class}
    expected = {ion["name3"] for ion in metal_table()["ions"]}
    assert found == expected, "every ion in the table must be selectable on input"


def test_the_metal_atom_index_points_at_the_metal(
    default_database: tmol.database.ParameterDatabase,
):
    co = canonical_ordering(default_database)
    tables = metal_tables(default_database)
    metal_atom = {
        res.io_equiv_class: res.metal_sites[0].metal_atom
        for res in default_database.chemical.residues
        if res.metal_sites
    }
    for cls, index in tables.metal_atom_index.items():
        name3 = co.restype_io_equiv_classes[cls]
        assert co.restypes_ordered_atom_names[name3][index] == metal_atom[name3]


def test_the_metal_atom_is_named_by_element_not_component(
    default_database: tmol.database.ParameterDatabase,
):
    # CU1, FE2 and 3CO name their single atom CU, FE and CO; a cluster numbers
    #    its metals
    element_of = {t.name: t.element for t in default_database.chemical.atom_types}
    for res in default_database.chemical.residues:
        if is_metal_cluster(res):
            continue
        type_of = {a.name: a.atom_type for a in res.atoms}
        for site in res.metal_sites:
            assert site.metal_atom == element_of[type_of[site.metal_atom]].upper()


@pytest.mark.parametrize(
    "res, atom, element",
    [
        ("ASP", "OD1", "O"),
        # either tautomer nitrogen may donate
        ("HIS", "ND1", "N"),
        ("HIS", "NE2", "N"),
        # the thiolate and phenolate forms donate
        ("CYS", "SG", "S"),
        ("TYR", "OH", "O"),
        ("ALA", "O", "O"),
        ("ALA", "CB", ""),
        ("G", "OP1", "O"),
        ("HOH", "O", "Owat"),
    ],
)
def test_donor_elements_are_tabulated(
    default_database: tmol.database.ParameterDatabase, res, atom, element
):
    co = canonical_ordering(default_database)
    tables = metal_tables(default_database)
    row = co.restype_io_equiv_classes.index(res)
    column = co.restypes_atom_index_mapping[res][atom]
    assert tables.donor_element[row, column] == element


@pytest.mark.parametrize(
    "res, atom, hydrogens",
    [("CYS", "SG", {"HG"}), ("TYR", "OH", {"HH"}), ("HIS", "NE2", {"HE2"})],
)
def test_a_donor_holding_its_hydrogen_is_known_by_that_hydrogen(
    default_database: tmol.database.ParameterDatabase, res, atom, hydrogens
):
    # the thiol and phenol forms cannot donate; their hydrogen still marks the atom
    co = canonical_ordering(default_database)
    tables = metal_tables(default_database)
    index = co.restypes_atom_index_mapping[res]
    names = co.restypes_ordered_atom_names[res]
    row = tables.donor_hydrogens[co.restype_io_equiv_classes.index(res), index[atom]]
    assert {names[h] for h in row.tolist() if h >= 0} == hydrogens


def test_geometry_is_carried_as_a_res_type_variant(
    default_database: tmol.database.ParameterDatabase,
):
    """The index find_metal_geometries emits must select the matching block type."""
    # both carboxylate oxygens of one aspartate at a coordinating distance
    asp = {"OD1": [2.03, 0.0, 0.0], "OD2": [0.0, 2.03, 0.0]}
    variants, assignments = find(
        default_database, [("ZN", {"ZN": [0.0, 0.0, 0.0]}), ("ASP", asp)]
    )
    assert variants[0, 1] == 0, "a non-metal keeps the default variant"
    ((_, got),) = assignments
    assert got.n_donors == 2, "both carboxylate oxygens are in range"
    assert int(variants[0, 0]) == metal_geometry_variant_index(got.geometry)
    assert got.geometry in ion_named("Zn", 2)["geometries"]


@pytest.mark.parametrize(
    "metal, geometries, expected",
    [
        ("ZN", {(0, 0): "octahedral"}, "octahedral"),
        # it must still resolve to a block type, or pose construction has
        #    nothing to build
        ("MG", None, "octahedral"),
    ],
    ids=["declared_geometry", "nothing_nearby"],
)
def test_a_lone_metal_takes_its_declared_or_default_geometry(
    default_database: tmol.database.ParameterDatabase, metal, geometries, expected
):
    variants, _ = find(
        default_database, [(metal, {metal: [0.0, 0.0, 0.0]})], geometries=geometries
    )
    assert int(variants[0, 0]) == metal_geometry_variant_index(expected)


def _assignment_with_donors(donor_atoms):
    return MetalSiteAssignment(
        metal=0,
        element="Zn",
        oxidation_state=2,
        geometry="tetrahedral",
        how_chosen="declared",
        donors=tuple(range(len(donor_atoms))),
        vertex_for_donor=tuple(range(len(donor_atoms))),
        n_open_sites=4 - len(donor_atoms),
        donor_atoms=tuple(donor_atoms),
    )


def _present(co, database, residues):
    """[1, n, A] presence of each residue's heavy atoms and the named hydrogens."""
    elements = {at.name: at.element for at in database.chemical.atom_types}
    present = torch.zeros(
        (1, len(residues), co.max_n_canonical_atoms), dtype=torch.bool
    )
    for i, (name, hydrogens) in enumerate(residues):
        index = co.restypes_atom_index_mapping[name]
        for res in database.chemical.residues:
            if res.io_equiv_class != name:
                continue
            for a in res.atoms:
                if a.name in index and elements.get(a.atom_type) != "H":
                    present[0, i, index[a.name]] = True
        for h in hydrogens:
            present[0, i, index[h]] = True
    return present


@pytest.mark.parametrize(
    "res, hydrogens, coordinating, preset, expected_base",
    [
        # heavy atoms only, as from coordinates without hydrogens
        ("HIS", (), "NE2", False, "HIS_D"),
        ("HIS", (), "ND1", False, "HIS"),
        ("CYS", (), "SG", False, "CYS_DEP"),
        ("TYR", (), "OH", False, "TYR_DEP"),
        ("SER", (), "OG", False, "SER"),
        ("ASP", (), "OD1", False, "ASP"),
        # the hydrogens presented decide; for HIS the kernel has already chosen
        #    the tautomer the ring hydrogens show
        ("CYS", ("H", "HA", "HB2", "HB3"), None, False, "CYS_DEP"),
        ("CYS", ("H", "HA", "HB2", "HB3", "HG"), "SG", False, "CYS"),
        (
            "TYR",
            ("H", "HA", "HB2", "HB3", "HD1", "HD2", "HE1", "HE2"),
            None,
            False,
            "TYR_DEP",
        ),
        ("HIS", ("H", "HA", "HB2", "HB3", "HD1", "HD2", "HE1"), "NE2", True, "HIS_D"),
        ("HIS", ("H", "HA", "HB2", "HB3", "HD2", "HE1", "HE2"), None, True, "HIS"),
    ],
)
def test_protonation_variant_follows_coordination_and_hydrogens(
    default_database: tmol.database.ParameterDatabase,
    res,
    hydrogens,
    coordinating,
    preset,
    expected_base,
):
    co = canonical_ordering(default_database)
    index_of = {c: i for i, c in enumerate(co.restype_io_equiv_classes)}
    res_types = torch.tensor([[index_of["ZN"], index_of[res]]], dtype=torch.int32)
    expected = next(
        r for r in default_database.chemical.residues if r.name == expected_base
    )
    variants = torch.zeros_like(res_types)
    if preset:
        variants[0, 1] = special_case_variant_index(expected)
    given = variants.clone()
    assignments = []
    if coordinating is not None:
        j = co.restypes_atom_index_mapping[res][coordinating]
        assignments = [((0, 0), _assignment_with_donors([(1, j)]))]

    got = select_protonation_variants(
        co,
        default_database.chemical,
        res_types,
        variants,
        _present(co, default_database, [("ZN", ()), (res, hydrogens)]),
        assignments,
    )
    assert int(got[0, 1]) == special_case_variant_index(expected)
    assert torch.equal(variants, given), "the input variants are not modified"


def test_deprotonated_forms_are_never_the_default_variant(
    default_database: tmol.database.ParameterDatabase,
):
    # structure input picks among a class by variant index first; sharing index
    # 0 with CYS or TYR would let a hydrogen-free input fall into the thiolate
    for res in default_database.chemical.residues:
        if res.base_name in ("CYS_DEP", "TYR_DEP", "DCYS_DEP", "DTYR_DEP"):
            assert special_case_variant_index(res) != 0, res.name


def test_a_disulfide_cysteine_is_not_a_donor_candidate(
    default_database: tmol.database.ParameterDatabase,
):
    co = canonical_ordering(default_database)
    residues = [("ZN", {"ZN": [0.0, 0.0, 0.0]}), ("CYS", {"SG": [2.3, 0.0, 0.0]})]

    _, free = find(default_database, residues)
    ((_, got),) = free
    assert got.donor_atoms == ((1, co.restypes_atom_index_mapping["CYS"]["SG"]),)

    _, bonded = find(
        default_database,
        residues,
        excluded_donor_residues=torch.tensor([[False, True]]),
    )
    ((_, got),) = bonded
    assert got.donor_atoms == ()


def test_site_virtuals_point_at_the_donors_that_took_their_vertices(
    default_restype_set,
):
    (zinc,) = [
        rt for rt in default_restype_set.residue_types if rt.name == "ZN_tetrahedral"
    ]
    site = zinc.metal_sites[0]
    pbt = types.SimpleNamespace(active_block_types=[zinc])

    # a zinc away from the origin with three donors on tetrahedral directions,
    # rotated so the fan cannot be right by accident
    metal = numpy.array([3.0, -1.0, 2.0])
    angle = numpy.radians(37.0)
    turn = numpy.array(
        [
            [numpy.cos(angle), -numpy.sin(angle), 0.0],
            [numpy.sin(angle), numpy.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    directions = vertices("tetrahedral")[:3] @ turn.T
    donors = metal + place(directions, 2.03)
    got = assign_one(
        metal,
        ion_named("Zn", 2),
        donors,
        ["N"] * 3,
        metal_table(),
        geometry="tetrahedral",
    )

    n_atoms = len(zinc.atoms)
    block_coords = torch.zeros((1, 1, n_atoms, 3), dtype=torch.float32)
    block_coords[0, 0, zinc.atom_to_idx[site.metal_atom]] = torch.tensor(metal)
    missing = torch.ones((1, 1, n_atoms), dtype=torch.bool)
    missing[0, 0, zinc.atom_to_idx[site.metal_atom]] = False

    block_types = torch.zeros((1, 1), dtype=torch.int64)
    coords, missing = place_site_virtuals(
        pbt, block_types, block_coords, missing, [((0, 0), got)]
    )
    assert not missing.any(), "every virtual is placed"

    virts = coords[0, 0, [zinc.atom_to_idx[v] for v in site.site_virts]].numpy()
    fan = virts - metal

    def cosine(a, b):
        return a @ b / numpy.linalg.norm(a) / numpy.linalg.norm(b)

    for donor, vertex in zip(directions, got.vertex_for_donor):
        assert cosine(fan[vertex], donor) == pytest.approx(1.0, abs=1e-5)
    # the open site takes the fourth tetrahedral direction
    (open_vertex,) = set(range(4)) - set(got.vertex_for_donor)
    expected = vertices("tetrahedral")[3] @ turn.T
    assert cosine(fan[open_vertex], expected) == pytest.approx(1.0, abs=1e-5)
