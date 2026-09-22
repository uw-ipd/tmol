from types import SimpleNamespace

import pytest

import tmol.database
from tmol.database.chemical import GEOMETRY_SITE_COUNT, MetalSite
from tmol.database._patched_chemdb import _validate_raw_residue_metal_sites

# The supported set, chosen against PDB entry counts. Asserted exactly so that
# adding or dropping an ion is a deliberate edit and not a drift.
EXPECTED_METALS = {
    ("Mg", 2),
    ("Ca", 2),
    ("Na", 1),
    ("K", 1),
    ("Mn", 2),
    ("Fe", 2),
    ("Fe", 3),
    ("Co", 2),
    ("Co", 3),
    ("Ni", 2),
    ("Cu", 1),
    ("Cu", 2),
    ("Zn", 2),
    ("Cd", 2),
    ("Hg", 2),
}


def metal_types(db):
    return [at for at in db.chemical.atom_types if at.is_metal]


def test_metal_atom_types_present(default_database: tmol.database.ParameterDatabase):
    found = {(at.element, at.oxidation_state) for at in metal_types(default_database)}
    assert found == EXPECTED_METALS


def test_metal_atom_types_are_fully_specified(
    default_database: tmol.database.ParameterDatabase,
):
    elements = {el.name for el in default_database.chemical.element_types}
    for at in metal_types(default_database):
        assert at.oxidation_state is not None, f"{at.name} has no oxidation state"
        assert at.oxidation_state > 0, f"{at.name} has a non-positive oxidation state"
        assert at.element in elements, f"{at.name} has an unknown element"


def test_only_metals_carry_an_oxidation_state(
    default_database: tmol.database.ParameterDatabase,
):
    for at in default_database.chemical.atom_types:
        if not at.is_metal:
            assert (
                at.oxidation_state is None
            ), f"{at.name} is not a metal but declares an oxidation state"


def test_metal_elements_have_correct_atomic_numbers(
    default_database: tmol.database.ParameterDatabase,
):
    expected = {
        "Na": 11,
        "Mg": 12,
        "K": 19,
        "Ca": 20,
        "Mn": 25,
        "Fe": 26,
        "Co": 27,
        "Ni": 28,
        "Cu": 29,
        "Zn": 30,
        "Cd": 48,
        "Hg": 80,
    }
    elements = {
        el.name: el.atomic_number for el in default_database.chemical.element_types
    }
    for name, number in expected.items():
        assert elements[name] == number


def donor_names(db):
    return {at.name for at in db.chemical.atom_types if at.is_metal_donor}


def test_sulfur_donates_to_metals_without_accepting_hydrogen_bonds(
    default_database: tmol.database.ParameterDatabase,
):
    # the reason is_metal_donor is not just is_acceptor: cysteine thiolate and
    # methionine thioether are the defining soft-metal ligands and neither is
    # an hbond acceptor in Rosetta's typing
    by_name = {at.name: at for at in default_database.chemical.atom_types}
    for name in ("Sthio", "S"):
        assert by_name[name].is_metal_donor, f"{name} should donate to metals"
        assert not by_name[name].is_acceptor, (
            f"{name} is not an hbond acceptor; if that changed, revisit whether "
            "is_metal_donor still needs to be a separate flag"
        )


def test_only_deprotonated_histidine_nitrogens_donate(
    default_database: tmol.database.ParameterDatabase,
):
    # a protonated ring nitrogen has no lone pair to give; tautomer resolution
    # chooses which of the two faces the metal sees
    donors = donor_names(default_database)
    assert "NhisDDepro" in donors
    assert "NhisEDepro" in donors
    assert "NhisD" not in donors
    assert "NhisE" not in donors


def test_only_deprotonated_thiol_and_phenol_donate(
    default_database: tmol.database.ParameterDatabase,
):
    # the thiolate and phenolate coordinate; the protonated forms do not, so a
    # coordinating cysteine or tyrosine is built as CYS_D or TYR_D
    donors = donor_names(default_database)
    assert "Sthio" in donors and "OOC" in donors
    assert "SH1" not in donors
    assert "OHphenol" not in donors


def test_nucleic_acid_donors_are_declared(
    default_database: tmol.database.ParameterDatabase,
):
    # phosphate, guanine O6 and base ring nitrogens are what the alkali and
    # alkaline-earth sites in nucleic acids actually coordinate
    donors = donor_names(default_database)
    for name in ("OOP", "Oet2", "ObaccG", "Obacc", "Nbacc"):
        assert name in donors, f"{name} should donate to metals"


def test_backbone_carbonyl_donates(default_database: tmol.database.ParameterDatabase):
    assert "OCbb" in donor_names(default_database), (
        "backbone carbonyl coordination is common and is why donors are declared "
        "per atom type rather than per residue"
    )


def test_water_donates_when_it_is_present(
    default_database: tmol.database.ParameterDatabase,
):
    # structure input filters HOH, so a site left open by a dropped water is
    # carried by the implicit model rather than by this type. The flag is here
    # for the paths that do retain waters, not for the common one.
    assert "Owat" in donor_names(default_database)


def test_metal_donors_are_oxygen_nitrogen_or_sulfur(
    default_database: tmol.database.ParameterDatabase,
):
    for at in default_database.chemical.atom_types:
        if at.is_metal_donor:
            assert at.element in ("O", "N", "S"), f"{at.name} is an implausible donor"


def test_metals_are_not_donors(default_database: tmol.database.ParameterDatabase):
    for at in default_database.chemical.atom_types:
        if at.is_metal:
            assert not at.is_metal_donor, f"{at.name} cannot coordinate itself"


def test_free_site_count_subtracts_internal_satisfiers():
    # a heme iron: octahedral, four porphyrin nitrogens, two axial sites free
    heme = MetalSite(
        metal_atom="FE",
        geometry="octahedral",
        internal_satisfiers=("NA", "NB", "NC", "ND"),
        site_virts=("V1", "V2"),
    )
    assert heme.n_free_sites == 2

    # a free ion satisfies nothing internally
    zinc = MetalSite(metal_atom="ZN", geometry="tetrahedral")
    assert zinc.n_free_sites == 4


def test_untemplated_geometry_has_no_free_site_count():
    # Ca/Na/K restrain metal-ligand distances only, so occupancy comes from the
    # structure and there is no vertex budget to report
    calcium = MetalSite(metal_atom="CA", geometry="irregular")
    assert calcium.n_free_sites is None
    assert GEOMETRY_SITE_COUNT["irregular"] is None


def test_unknown_geometry_is_rejected():
    import cattr

    from tmol.database.chemical import CoordinationGeometry

    with pytest.raises(ValueError):
        cattr.structure("octahedral-ish", CoordinationGeometry)


def fake_residue(metal_sites, atoms=("FE", "NA", "V1"), virtual=("V1",)):
    return SimpleNamespace(
        name="TST",
        atoms=tuple(SimpleNamespace(name=name) for name in atoms),
        properties=SimpleNamespace(virtual=tuple(virtual)),
        metal_sites=tuple(metal_sites),
    )


def validate(res):
    _validate_raw_residue_metal_sites(res, {a.name for a in res.atoms})


def test_metal_site_naming_a_missing_atom_is_rejected():
    res = fake_residue(
        [MetalSite(metal_atom="FE", geometry="octahedral", internal_satisfiers=("NZ",))]
    )
    with pytest.raises(RuntimeError, match="does not have"):
        validate(res)


def test_free_site_virt_must_be_virtual():
    res = fake_residue(
        [MetalSite(metal_atom="FE", geometry="octahedral", site_virts=("NA",))]
    )
    with pytest.raises(RuntimeError, match="not virtual"):
        validate(res)


def test_oversubscribed_geometry_is_rejected():
    res = fake_residue(
        [
            MetalSite(
                metal_atom="FE",
                geometry="linear",
                internal_satisfiers=("NA", "FE"),
                site_virts=("V1",),
            )
        ],
    )
    with pytest.raises(RuntimeError, match="but names"):
        validate(res)


def test_untemplated_geometry_is_not_budget_checked():
    # many donors on an irregular ion is the normal case, not an error
    res = fake_residue(
        [
            MetalSite(
                metal_atom="FE",
                geometry="irregular",
                internal_satisfiers=("NA", "V1"),
            )
        ]
    )
    validate(res)


def geometry_table():
    import os

    from yaml import safe_load

    path = os.path.join(
        os.path.dirname(tmol.database.__file__), "default", "chemical", "metals.yaml"
    )
    with open(path) as infile:
        return safe_load(infile)


def pairwise_angles(vectors):
    import math

    import numpy

    out = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            cos = numpy.dot(vectors[i], vectors[j]) / (
                numpy.linalg.norm(vectors[i]) * numpy.linalg.norm(vectors[j])
            )
            out.append(math.degrees(math.acos(max(-1.0, min(1.0, cos)))))
    return sorted(out)


def test_generated_icoors_rebuild_the_intended_polyhedra():
    """The internal coordinates the generator emits must rebuild its own vertices.

    Guards the inversion of build_coords_from_icoors, where theta is the
    supplement of the bond angle and a wrong convention still yields plausible
    numbers. Compared as a set of pairwise angles, because the rebuilt fan is
    free to be rotated as a whole.
    """
    import math

    import numpy

    from tmol.chemical._ideal_coords import build_coords_from_icoors
    from tmol.support.chemical._add_metal_ions import icoors_for_sites

    dist = 2.1
    for geometry in geometry_table()["geometries"]:
        vertices = geometry["vertices"]
        if len(vertices) < 2:
            continue
        rows = icoors_for_sites(vertices, dist)
        n = len(rows)
        ancestors = numpy.tile(numpy.array([0, 1, 2], dtype=numpy.int32), (n, 1))
        geom = numpy.array(
            [[math.radians(p), math.radians(t), d] for p, t, d in rows],
            dtype=numpy.float64,
        )
        xyz = build_coords_from_icoors(ancestors, geom)
        built = [xyz[i] - xyz[0] for i in range(1, n)]
        target = [numpy.array(v) * dist for v in vertices]

        for vector in built:
            assert numpy.linalg.norm(vector) == pytest.approx(
                dist, abs=1e-3
            ), f"{geometry['name']}: a site sits at the wrong distance"
        for got, want in zip(pairwise_angles(built), pairwise_angles(target)):
            assert got == pytest.approx(
                want, abs=1e-3
            ), f"{geometry['name']}: rebuilt geometry does not match its vertices"


def test_collinear_leading_vertices_are_rejected():
    # V1 and V2 define the dihedral frame, so a trans pair listed first leaves
    # it undefined; that must raise rather than emit silent NaNs
    from tmol.support.chemical._add_metal_ions import icoors_for_sites

    with pytest.raises(ValueError, match="collinear"):
        icoors_for_sites([[0, 0, 1], [0, 0, -1], [1, 0, 0]], 2.1)


def test_metal_ion_block_types_are_loaded(
    default_database: tmol.database.ParameterDatabase,
):
    metal_types = {at.name for at in metal_types_list(default_database)}
    by_name = {r.name: r for r in default_database.chemical.residues}
    generated = [r for r in default_database.chemical.residues if r.metal_sites]
    assert generated, "no generated metal ion residue types were loaded"

    for res in generated:
        assert len(res.metal_sites) == 1
        site = res.metal_sites[0]
        assert res.atoms[0].name == site.metal_atom, "the metal must be atom 0"
        assert res.atoms[0].atom_type in metal_types
        assert site.internal_satisfiers == (), "a free ion satisfies nothing itself"
        # every templated site is marked, so detection can fill all of them
        if site.n_free_sites is not None:
            assert len(site.site_virts) == site.n_free_sites
        else:
            assert site.site_virts == (), "untemplated ions get no site waters"
    assert "ZN_tetrahedral" in by_name
    assert "CA_irregular" in by_name


def metal_types_list(db):
    return [at for at in db.chemical.atom_types if at.is_metal]


def test_every_metal_atom_has_a_zero_elec_charge(
    default_database: tmol.database.ParameterDatabase,
):
    # a missing charge raises at block-type setup, and a nonzero one would
    # double-count the attraction metal_coordination is meant to carry
    charges = default_database.scoring.elec.atom_charge_parameters
    generated = {r.name for r in default_database.chemical.residues if r.metal_sites}
    seen = {c.res: [] for c in charges if c.res in generated}
    for c in charges:
        if c.res in generated:
            seen[c.res].append(c.charge)
    assert set(seen) == generated, "some metal ion types have no charge entries"
    for res, values in seen.items():
        assert all(v == 0.0 for v in values), f"{res} has a nonzero charge"


def test_existing_residues_declare_no_metal_sites(
    default_database: tmol.database.ParameterDatabase,
):
    # regression guard for the schema addition: the field is optional, and
    # nothing in the hand-maintained database has gained one by accident. The
    # generated ion types are the only residues that may declare sites.
    generated = {
        f"{i['name3']}_{g}" for i in geometry_table()["ions"] for g in i["geometries"]
    }
    for res in default_database.chemical.residues:
        if res.name in generated:
            continue
        assert res.metal_sites == (), f"{res.name} unexpectedly declares metal sites"
