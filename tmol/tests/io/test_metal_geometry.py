import math

import numpy
import pytest
from yaml import safe_load

import tmol.database
from tmol.io.details._metal_geometry import (
    choose_geometry,
    fit_geometry,
    unit,
)


def geometry_table():
    import os

    path = os.path.join(
        os.path.dirname(tmol.database.__file__), "default", "chemical", "metals.yaml"
    )
    with open(path) as infile:
        return safe_load(infile)


def vertices_for():
    return {g["name"]: g["vertices"] for g in geometry_table()["geometries"]}


def rotate(vectors, axis, degrees):
    """Rodrigues rotation, so a fit has to be rotation-invariant to pass."""
    axis = unit(numpy.asarray(axis, dtype=numpy.float64))
    t = math.radians(degrees)
    k = numpy.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    rot = numpy.eye(3) + math.sin(t) * k + (1 - math.cos(t)) * (k @ k)
    return numpy.asarray(vectors, dtype=numpy.float64) @ rot.T


def test_a_perfect_polyhedron_fits_itself_exactly():
    for name, verts in vertices_for().items():
        if not verts:
            continue
        fit = fit_geometry(numpy.asarray(verts), numpy.asarray(verts))
        assert fit.rms_angle == pytest.approx(0.0, abs=1e-6), name
        assert fit.n_open_sites == 0, name


def test_the_fit_is_rotation_invariant():
    # the polyhedron is free to point anywhere, so an arbitrarily rotated copy
    # of it must still fit perfectly
    verts = numpy.asarray(vertices_for()["octahedral"])
    turned = rotate(verts, [0.3, -0.7, 0.5], 37.0)
    fit = fit_geometry(turned, verts)
    assert fit.rms_angle == pytest.approx(0.0, abs=1e-6)


def test_partial_occupancy_still_fits():
    # waters are dropped on input, so most sites arrive empty; three of an
    # octahedron's six donors must still identify it
    verts = numpy.asarray(vertices_for()["octahedral"])
    fit = fit_geometry(rotate(verts[:3], [1.0, 1.0, 0.0], 21.0), verts)
    assert fit.rms_angle == pytest.approx(0.0, abs=1e-6)
    assert fit.n_open_sites == 3


def test_more_donors_than_sites_does_not_fit():
    verts = numpy.asarray(vertices_for()["tetrahedral"])
    assert fit_geometry(numpy.asarray(vertices_for()["octahedral"]), verts) is None


def test_tetrahedral_and_square_planar_are_told_apart():
    # the case coordination number alone cannot resolve: four donors either way
    v = vertices_for()
    tet = numpy.asarray(v["tetrahedral"])
    sqp = numpy.asarray(v["square_planar"])
    assert fit_geometry(tet, tet).rms_angle < 1e-6
    assert fit_geometry(tet, sqp).rms_angle > 20.0
    assert fit_geometry(sqp, sqp).rms_angle < 1e-6
    assert fit_geometry(sqp, tet).rms_angle > 20.0


def test_choose_prefers_the_geometry_the_directions_support():
    v = vertices_for()
    allowed = ["tetrahedral", "square_planar", "octahedral"]
    chosen, why = choose_geometry(numpy.asarray(v["tetrahedral"]), allowed, v)
    assert chosen.geometry == "tetrahedral"
    assert why == "directions"


def test_a_nested_tie_falls_to_preference_order_not_to_the_smaller_polyhedron():
    # two donors at 90 degrees fit square planar and octahedral equally well,
    # since the smaller polyhedron's vertices are a subset of the larger's.
    # Whichever the ion lists first wins -- empty sites are not evidence,
    # because input drops the waters that would have filled them
    v = vertices_for()
    donors = numpy.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    for allowed in (["square_planar", "octahedral"], ["octahedral", "square_planar"]):
        chosen, why = choose_geometry(donors, allowed, v)
        assert chosen.geometry == allowed[0]
        assert why == "preference order"


def test_untemplated_geometry_declines_to_choose():
    v = vertices_for()
    donors = numpy.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    chosen, why = choose_geometry(donors, ["irregular"], v)
    assert chosen is None
    assert why == "untemplated"


def test_every_ion_can_be_fitted_from_a_single_donor():
    # the worst real case: a magnesium whose five waters were dropped. No fit
    # can discriminate, so the ion's first listed geometry must carry it
    table = geometry_table()
    v = vertices_for()
    donors = numpy.asarray([[0.0, 0.0, 1.0]])
    for ion in table["ions"]:
        chosen, why = choose_geometry(donors, ion["geometries"], v)
        if ion["geometries"] == ["irregular"]:
            assert chosen is None
            continue
        assert chosen is not None, f"{ion['element']}{ion['oxidation_state']}"
        assert chosen.geometry == ion["geometries"][0], (
            f"{ion['element']}{ion['oxidation_state']}: a lone donor should fall "
            "back on the most common geometry"
        )
