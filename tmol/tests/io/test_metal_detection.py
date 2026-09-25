import numpy
import pytest
from yaml import safe_load

import tmol.database
from tmol.database.chemical import is_metal_cluster
from tmol.io.details._metal_detection import (
    MetalSiteAssignment,
    assign_one,
    gather_candidates,
    ideal_distances,
)


def table():
    import os

    path = os.path.join(
        os.path.dirname(tmol.database.__file__), "default", "chemical", "metals.yaml"
    )
    with open(path) as infile:
        return safe_load(infile)


def ion_named(element, ox):
    for ion in table()["ions"]:
        if ion["element"] == element and ion["oxidation_state"] == ox:
            return ion
    raise KeyError(f"{element}{ox}")


def vertices(name):
    for g in table()["geometries"]:
        if g["name"] == name:
            return numpy.asarray(g["vertices"])
    raise KeyError(name)


def place(directions, distance):
    """Donors at a given distance along each direction, metal at the origin."""
    d = numpy.asarray(directions, dtype=numpy.float64)
    return d / numpy.linalg.norm(d, axis=1, keepdims=True) * distance


def test_missing_donor_element_falls_back_rather_than_raising():
    dists = ideal_distances(ion_named("Zn", 2), table()["donor_radii"])
    keep, _ = gather_candidates(
        numpy.zeros(3), place([[1, 0, 0]], 2.0), ["Se"], dists, 0.55
    )
    assert len(keep) == 1


def test_derived_distances_fill_only_what_was_not_measured():
    ion = ion_named("Ca", 2)
    dists = ideal_distances(ion, table()["donor_radii"])
    assert dists["O"] == ion["distances"]["O"], "a measured value must survive"
    assert "S" not in ion["distances"], "calcium-sulfur was not measured"
    assert dists["S"] == pytest.approx(ion["ionic_radius"] + 1.549, abs=1e-3)


def test_candidates_are_gathered_without_reference_to_geometry():
    # the cutoff scales with the donor element, not with any polyhedron
    dists = ideal_distances(ion_named("Zn", 2), table()["donor_radii"])
    # zinc-oxygen is 2.033 measured, so 0.55 A past it admits out to 2.58
    xyz = numpy.array([[2.03, 0, 0], [2.45, 0, 0], [2.70, 0, 0], [4.0, 0, 0]])
    keep, excess = gather_candidates(numpy.zeros(3), xyz, ["O"] * 4, dists, 0.55)
    assert list(keep) == [0, 1], "2.70 and 4.0 A are both past tolerance"
    assert excess[0] < excess[1]


def test_a_clean_tetrahedral_zinc_is_recovered():
    ion = ion_named("Zn", 2)
    donors = place(vertices("tetrahedral"), 2.03)
    got = assign_one(numpy.zeros(3), ion, donors, ["N"] * 4, table())
    assert got.geometry == "tetrahedral"
    assert got.n_donors == 4
    assert got.n_open_sites == 0
    assert sorted(got.vertex_for_donor) == [0, 1, 2, 3], "each site takes one donor"


def test_a_magnesium_with_one_donor_still_gets_its_octahedron():
    # five waters dropped on input; the geometry has to come from the table
    ion = ion_named("Mg", 2)
    donors = place([[0, 0, 1]], 2.07)
    got = assign_one(numpy.zeros(3), ion, donors, ["O"], table())
    assert got.geometry == "octahedral"
    assert got.n_donors == 1
    assert got.n_open_sites == 5, "the empty sites are real and must be counted"


def test_untemplated_ions_report_donors_but_no_geometry():
    ion = ion_named("Ca", 2)
    donors = place([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]], 2.34)
    got = assign_one(numpy.zeros(3), ion, donors, ["O"] * 5, table())
    assert got.geometry is None
    assert got.how_chosen == "untemplated"
    assert got.n_donors == 5, "every contact coordinates; none is dropped"
    assert got.n_open_sites is None


def test_a_fifth_donor_promotes_zinc_rather_than_being_discarded():
    # zinc allows trigonal bipyramidal, so five donors are accommodated by
    # choosing a larger geometry instead of throwing one away
    ion = ion_named("Zn", 2)
    donors = place(vertices("trigonal_bipyramidal"), 2.03)
    got = assign_one(numpy.zeros(3), ion, donors, ["N"] * 5, table())
    assert got.geometry == "trigonal_bipyramidal"
    assert got.n_donors == 5
    assert got.rejected == ()


def test_more_donors_than_any_geometry_holds_drops_the_weakest():
    # the case that hard-exits in Rosetta. Zinc tops out at six sites, so a
    # seventh contact has to go, and it should be the one furthest past ideal
    ion = ion_named("Zn", 2)
    directions = list(vertices("octahedral")) + [[1.0, 1.0, 1.0]]
    donors = place(directions, 2.03)
    donors[6] *= 1.2
    got = assign_one(numpy.zeros(3), ion, donors, ["N"] * 7, table())
    assert got.geometry == "octahedral"
    assert got.n_donors == 6
    assert got.rejected == (6,), "the weakest contact is the one dropped"
    assert len(set(got.vertex_for_donor)) == 6, "no site takes two donors"


def test_a_declared_geometry_overrides_inference():
    ion = ion_named("Zn", 2)
    donors = place(vertices("tetrahedral"), 2.03)
    got = assign_one(
        numpy.zeros(3), ion, donors, ["N"] * 4, table(), geometry="octahedral"
    )
    assert got.geometry == "octahedral"
    assert got.how_chosen == "declared"
    assert got.n_open_sites == 2


def test_square_planar_copper_is_not_called_tetrahedral():
    ion = ion_named("Cu", 2)
    donors = place(vertices("square_planar"), 1.99)
    got = assign_one(numpy.zeros(3), ion, donors, ["N"] * 4, table())
    assert got.geometry == "square_planar"


def canonical_ordering(db):
    from tmol.io import CanonicalOrdering

    return CanonicalOrdering.from_chemdb(db.chemical)


def metal_tables(db):
    from tmol.io.details._metal_detection import build_canonical_metal_tables

    return build_canonical_metal_tables(canonical_ordering(db), db.chemical, table())


def test_every_supported_ion_is_found_in_the_canonical_ordering(
    default_database: tmol.database.ParameterDatabase,
):
    co = canonical_ordering(default_database)
    tables = metal_tables(default_database)
    found = {co.restype_io_equiv_classes[i] for i in tables.ion_for_class}
    expected = {ion["name3"] for ion in table()["ions"]}
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


def test_donors_are_tabulated_for_protein_and_nucleic_acid(
    default_database: tmol.database.ParameterDatabase,
):
    co = canonical_ordering(default_database)
    tables = metal_tables(default_database)
    index_of = {c: i for i, c in enumerate(co.restype_io_equiv_classes)}

    def donor(res, atom):
        return tables.donor_element[
            index_of[res], co.restypes_atom_index_mapping[res][atom]
        ]

    assert donor("ASP", "OD1") == "O"
    assert donor("HIS", "ND1") == "N", "either tautomer nitrogen may donate"
    assert donor("HIS", "NE2") == "N", "either tautomer nitrogen may donate"
    assert donor("CYS", "SG") == "S", "the thiolate form donates"
    assert donor("TYR", "OH") == "O", "the phenolate form donates"
    assert donor("ALA", "O") == "O", "backbone carbonyls coordinate"
    assert donor("ALA", "CB") == "", "carbon never donates"
    if "RG" in index_of:
        assert donor("RG", "OP1") == "O", "nucleic acid phosphate donates"


def test_water_is_tabulated_apart_from_other_oxygens(
    default_database: tmol.database.ParameterDatabase,
):
    co = canonical_ordering(default_database)
    tables = metal_tables(default_database)
    index_of = {c: i for i, c in enumerate(co.restype_io_equiv_classes)}
    if "HOH" not in index_of:
        pytest.skip("water is not in this chemical database")
    j = co.restypes_atom_index_mapping["HOH"]["O"]
    assert tables.donor_element[index_of["HOH"], j] == "Owat"


def test_geometry_is_carried_as_a_res_type_variant(
    default_database: tmol.database.ParameterDatabase,
):
    """The index find_metal_geometries emits must select the matching block type."""
    import torch

    from tmol.database.chemical import geometry_for_metal_variant_index
    from tmol.io.details._metal_detection import find_metal_geometries

    co = canonical_ordering(default_database)
    index_of = {c: i for i, c in enumerate(co.restype_io_equiv_classes)}
    n_atoms = co.max_n_canonical_atoms

    # one zinc, and one aspartate placed so both carboxylate oxygens sit at a
    # coordinating distance
    res_types = torch.tensor([[index_of["ZN"], index_of["ASP"]]], dtype=torch.int32)
    coords = torch.full((1, 2, n_atoms, 3), float("nan"), dtype=torch.float32)
    coords[0, 0, co.restypes_atom_index_mapping["ZN"]["ZN"]] = torch.tensor(
        [0.0, 0.0, 0.0]
    )
    for atom, xyz in (("OD1", [2.03, 0.0, 0.0]), ("OD2", [0.0, 2.03, 0.0])):
        coords[0, 1, co.restypes_atom_index_mapping["ASP"][atom]] = torch.tensor(xyz)

    variants, assignments = find_metal_geometries(
        co, default_database.chemical, res_types, coords, table()
    )
    assert variants[0, 1] == 0, "a non-metal keeps the default variant"
    geometry = geometry_for_metal_variant_index(int(variants[0, 0]))
    assert geometry in table()["ions"][0]["geometries"] or geometry in (
        "tetrahedral",
        "trigonal_bipyramidal",
        "octahedral",
    )
    assert len(assignments) == 1
    ((_, got),) = assignments
    assert got.n_donors == 2, "both carboxylate oxygens are in range"


def test_a_declared_geometry_reaches_the_variant_index(
    default_database: tmol.database.ParameterDatabase,
):
    import torch

    from tmol.database.chemical import metal_geometry_variant_index
    from tmol.io.details._metal_detection import find_metal_geometries

    co = canonical_ordering(default_database)
    index_of = {c: i for i, c in enumerate(co.restype_io_equiv_classes)}
    res_types = torch.tensor([[index_of["ZN"]]], dtype=torch.int32)
    coords = torch.full((1, 1, co.max_n_canonical_atoms, 3), float("nan"))
    coords[0, 0, co.restypes_atom_index_mapping["ZN"]["ZN"]] = torch.zeros(3)

    variants, _ = find_metal_geometries(
        co,
        default_database.chemical,
        res_types,
        coords.to(torch.float32),
        table(),
        geometries={(0, 0): "octahedral"},
    )
    assert int(variants[0, 0]) == metal_geometry_variant_index("octahedral")


def test_a_metal_with_nothing_nearby_still_gets_its_default_geometry(
    default_database: tmol.database.ParameterDatabase,
):
    # it must still resolve to a block type, or pose construction has nothing
    # to build
    import torch

    from tmol.database.chemical import geometry_for_metal_variant_index
    from tmol.io.details._metal_detection import find_metal_geometries

    co = canonical_ordering(default_database)
    index_of = {c: i for i, c in enumerate(co.restype_io_equiv_classes)}
    res_types = torch.tensor([[index_of["MG"]]], dtype=torch.int32)
    coords = torch.full((1, 1, co.max_n_canonical_atoms, 3), float("nan"))
    coords[0, 0, co.restypes_atom_index_mapping["MG"]["MG"]] = torch.zeros(3)

    variants, _ = find_metal_geometries(
        co, default_database.chemical, res_types, coords.to(torch.float32), table()
    )
    assert geometry_for_metal_variant_index(int(variants[0, 0])) == "octahedral"


def test_an_isolated_metal_coordinates_nothing():
    ion = ion_named("Zn", 2)
    got = assign_one(numpy.zeros(3), ion, numpy.zeros((0, 3)), [], table())
    assert isinstance(got, MetalSiteAssignment)
    assert got.n_donors == 0
    assert got.geometry is None, "nothing to fit means nothing to claim"


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


@pytest.mark.parametrize(
    "res, atom, expected_base",
    [
        ("HIS", "NE2", "HIS_D"),
        ("HIS", "ND1", "HIS"),
        ("CYS", "SG", "CYS_DEP"),
        ("TYR", "OH", "TYR_DEP"),
        ("SER", "OG", "SER"),
        ("ASP", "OD1", "ASP"),
    ],
)
def test_coordination_selects_the_form_that_can_donate(
    default_database: tmol.database.ParameterDatabase, res, atom, expected_base
):
    import torch

    from tmol.database.chemical import special_case_variant_index
    from tmol.io.details._metal_detection import select_donor_variants

    co = canonical_ordering(default_database)
    index_of = {c: i for i, c in enumerate(co.restype_io_equiv_classes)}
    res_types = torch.tensor([[index_of["ZN"], index_of[res]]], dtype=torch.int32)
    variants = torch.zeros_like(res_types)
    j = co.restypes_atom_index_mapping[res][atom]

    got = select_donor_variants(
        co,
        default_database.chemical,
        res_types,
        variants,
        [((0, 0), _assignment_with_donors([(1, j)]))],
    )
    expected = next(
        r for r in default_database.chemical.residues if r.name == expected_base
    )
    assert int(got[0, 1]) == special_case_variant_index(expected)
    assert int(variants[0, 1]) == 0, "the input variants are not modified"


def test_deprotonated_forms_are_never_the_default_variant(
    default_database: tmol.database.ParameterDatabase,
):
    # structure input picks among a class by variant index first; sharing index
    # 0 with CYS or TYR would let a hydrogen-free input fall into the thiolate
    from tmol.database.chemical import special_case_variant_index

    for res in default_database.chemical.residues:
        if res.base_name in ("CYS_DEP", "TYR_DEP", "DCYS_DEP", "DTYR_DEP"):
            assert special_case_variant_index(res) != 0, res.name


def test_a_disulfide_cysteine_is_not_a_donor_candidate(
    default_database: tmol.database.ParameterDatabase,
):
    import torch

    from tmol.io.details._metal_detection import find_metal_geometries

    co = canonical_ordering(default_database)
    index_of = {c: i for i, c in enumerate(co.restype_io_equiv_classes)}
    res_types = torch.tensor([[index_of["ZN"], index_of["CYS"]]], dtype=torch.int32)
    coords = torch.full((1, 2, co.max_n_canonical_atoms, 3), float("nan"))
    coords[0, 0, co.restypes_atom_index_mapping["ZN"]["ZN"]] = torch.zeros(3)
    coords[0, 1, co.restypes_atom_index_mapping["CYS"]["SG"]] = torch.tensor(
        [2.3, 0.0, 0.0]
    )
    coords = coords.to(torch.float32)

    _, free = find_metal_geometries(
        co, default_database.chemical, res_types, coords, table()
    )
    ((_, got),) = free
    assert got.donor_atoms == ((1, co.restypes_atom_index_mapping["CYS"]["SG"]),)

    _, bonded = find_metal_geometries(
        co,
        default_database.chemical,
        res_types,
        coords,
        table(),
        excluded_donor_residues=torch.tensor([[False, True]]),
    )
    ((_, got),) = bonded
    assert got.donor_atoms == ()


def test_site_virtuals_point_at_the_donors_that_took_their_vertices(
    default_restype_set,
):
    import types

    import torch

    from tmol.io.details._metal_detection import place_site_virtuals

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
        metal, ion_named("Zn", 2), donors, ["N"] * 3, table(), geometry="tetrahedral"
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
