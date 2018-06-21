import attr

import pytest

import torch

from math import nan

from tmol.score.factory import Factory

from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.score_components import ScoreComponent, ScoreComponentClasses
from tmol.score.interatomic_distance import (
    InteratomicDistanceGraphBase,
    NaiveInteratomicDistanceGraph,
    BlockedInteratomicDistanceGraph,
)

from tmol.utility.reactive import reactive_attrs, reactive_property


@pytest.fixture(scope="function")
def seterr_ignore():
    """Silent numpy nan-comparison warnings within a test class."""

    import numpy

    old = numpy.seterr(all="ignore")

    yield

    numpy.seterr(**old)


@reactive_attrs(auto_attribs=True)
class ThresholdDistanceCountIntraScore:
    target: "ThresholdDistanceCount" = attr.ib()

    @reactive_property
    def component_total(target):
        return target.total_threshold_count


@reactive_attrs(auto_attribs=True)
class ThresholdDistanceCountInterScore:
    target_i: "ThresholdDistanceCount" = attr.ib()
    target_j: "ThresholdDistanceCount" = attr.ib()

    @reactive_property
    def component_total(target_i, target_j):
        raise ValueError()


@reactive_attrs(auto_attribs=True)
class ThresholdDistanceCount(
    CartesianAtomicCoordinateProvider,
    InteratomicDistanceGraphBase,
    ScoreComponent,
    Factory,
):
    total_score_components = ScoreComponentClasses(
        name="threshold_count",
        intra_container=ThresholdDistanceCountIntraScore,
        inter_container=ThresholdDistanceCountInterScore,
    )

    threshold_distance: float

    def factory_for(obj, **_):
        return dict(threshold_distance=6.0)

    @property
    def component_atom_pair_dist_threshold(self):
        return self.threshold_distance

    @reactive_property
    def total_threshold_count(
        stack_depth, atom_pair_inds, atom_pair_dist, threshold_distance
    ):
        "number of bonds under threshold distance"
        result = atom_pair_dist.new_zeros((stack_depth,))

        result.put_(
            atom_pair_inds[:, 0],
            (atom_pair_dist < threshold_distance).to(dtype=torch.float),
            accumulate=True,
        )

        return result


@pytest.fixture(
    params=[NaiveInteratomicDistanceGraph, BlockedInteratomicDistanceGraph],
    ids=["naive", "blocked"],
)
def threshold_distance_score_class(request):
    interatomic_distance_component = request.param

    @reactive_attrs
    class TestGraph(ThresholdDistanceCount, interatomic_distance_component):
        @reactive_property
        def system_size(coords) -> int:
            return len(coords)

    return TestGraph


@pytest.fixture
def multilayer_test_coords():
    """A stacked test system with random coordinate clusters.

    clusters:
        8 coordinate block, 6 populated w/ unit vectors x,y,z,-x,-y,-z

    layers:
        0-------1------2
        1-------2------0
        ----0---1---2---
        -------012------
    """

    cluster_coords = torch.Tensor(
        [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
            (nan, nan, nan),
            (nan, nan, nan),
        ]
    )

    layout = torch.Tensor([[-8, 0, 8], [8, 0, -8], [-4, 0, 4], [-1, 0, 1]])

    # Convert to coordinates offsets along the x axis.
    offsets = layout[..., None] * torch.Tensor([1, 0, 0])
    assert offsets.shape == layout.shape + (3,)
    assert not (offsets[..., 0] == 0).all()
    assert (offsets[..., 1] == 0).all()
    assert (offsets[..., 2] == 0).all()

    coords = offsets[:, :, None, :] + cluster_coords
    assert coords.shape == layout.shape + cluster_coords.shape
    assert ((coords[0, 0, :6] == cluster_coords[:6] + torch.Tensor([-8, 0, 0]))).all()

    return coords.reshape((4, 8 * 3, 3))
