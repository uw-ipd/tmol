import attr

import pytest

import torch

from tmol.score.factory import Factory

from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.score_components import ScoreComponent, ScoreComponentClasses
from tmol.score.interatomic_distance import (
    InteratomicDistanceGraphBase,
    # NaiveInteratomicDistanceGraph,
    BlockedInteratomicDistanceGraph,
    InterLayerAtomPairs,
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
        "number of bonds under threshold distance"

        return (
            torch.sparse_coo_tensor(
                target.atom_pair_inds[:, 0][None, :],
                (target.atom_pair_dist < target.threshold_distance).to(
                    dtype=torch.float
                ),
                (target.stack_depth,),
                device=target.atom_pair_inds.device,
            )
            .coalesce()
            .to_dense()
        )


@reactive_attrs(auto_attribs=True)
class ThresholdDistanceCountInterScore:
    target_i: "ThresholdDistanceCount" = attr.ib()
    target_j: "ThresholdDistanceCount" = attr.ib()

    @reactive_property
    def component_total(target_i, target_j):
        assert target_i.threshold_distance == target_j.threshold_distance
        assert target_i.atom_pair_block_size == target_j.atom_pair_block_size

        pind = InterLayerAtomPairs.for_coord_blocks(
            target_i.atom_pair_block_size,
            target_i.coord_blocks,
            target_j.coord_blocks,
            target_i.threshold_distance,
        ).inds

        ci = target_i.coords.detach()
        cj = target_j.coords.detach()

        pdist = (ci[pind[:, 0], pind[:, 1]] - cj[pind[:, 2], pind[:, 3]]).norm(dim=-1)

        return (
            torch.sparse_coo_tensor(
                pind[:, [0, 2]].t(),
                (pdist < target_i.threshold_distance).to(dtype=torch.float),
                (target_i.stack_depth, target_j.stack_depth),
                device=pind.device,
            )
            .coalesce()
            .to_dense()
        )


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


@pytest.fixture(
    # params=[NaiveInteratomicDistanceGraph, BlockedInteratomicDistanceGraph],
    # ids=["naive", "blocked"],
    params=[BlockedInteratomicDistanceGraph],
    ids=["blocked"],
)
def threshold_distance_score_class(request):
    interatomic_distance_component = request.param

    @reactive_attrs
    class TestGraph(ThresholdDistanceCount, interatomic_distance_component):
        @reactive_property
        def system_size(coords) -> int:
            return len(coords)

    return TestGraph
