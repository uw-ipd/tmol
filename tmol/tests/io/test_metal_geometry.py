import math

import numpy
import pytest

from tmol.database.chemical import metal_table
from tmol.io.details._metal_geometry import (
    choose_geometry,
    fit_geometry,
    unit,
)


def vertices_for():
    return {g["name"]: g["vertices"] for g in metal_table()["geometries"]}


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


TWO_AT_90 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


@pytest.mark.parametrize(
    "donors, allowed, expected, why",
    [
        (
            vertices_for()["tetrahedral"],
            ["tetrahedral", "square_planar", "octahedral"],
            "tetrahedral",
            "directions",
        ),
        # a nested tie (square planar's vertices are a subset of octahedral's) goes
        #    to the ion's first listed geometry; empty sites are not evidence
        (
            TWO_AT_90,
            ["square_planar", "octahedral"],
            "square_planar",
            "preference order",
        ),
        (TWO_AT_90, ["octahedral", "square_planar"], "octahedral", "preference order"),
        (TWO_AT_90, ["irregular"], None, "untemplated"),
    ],
    ids=[
        "directions",
        "nested_tie_square_planar_first",
        "nested_tie_octahedral_first",
        "untemplated",
    ],
)
def test_choose_geometry(donors, allowed, expected, why):
    chosen, reason = choose_geometry(numpy.asarray(donors), allowed, vertices_for())
    assert (None if chosen is None else chosen.geometry) == expected
    assert reason == why


def test_every_ion_can_be_fitted_from_a_single_donor():
    # the worst real case: a magnesium whose five waters were dropped. No fit
    # can discriminate, so the ion's first listed geometry must carry it
    v = vertices_for()
    donors = numpy.asarray([[0.0, 0.0, 1.0]])
    for ion in metal_table()["ions"]:
        chosen, why = choose_geometry(donors, ion["geometries"], v)
        if ion["geometries"] == ["irregular"]:
            assert chosen is None
            continue
        assert chosen is not None, f"{ion['element']}{ion['oxidation_state']}"
        assert chosen.geometry == ion["geometries"][0], (
            f"{ion['element']}{ion['oxidation_state']}: a lone donor should fall "
            "back on the most common geometry"
        )
