from math import nan

import pytest

import torch

from tmol.score.factory import Factory

from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.score_components import (
    ScoreComponent,
    ScoreComponentClasses,
    InterScoreGraph,
    IntraScoreGraph,
)
from tmol.score.interatomic_distance import (
    InteratomicDistanceGraphBase,
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


@pytest.fixture
def multilayer_test_offsets():
    layout = torch.Tensor([[-8, 0, 8], [8, 0, -8], [-4, 0, 4], [-1, 0, 1]])

    # Convert to coordinates offsets along the x axis.
    offsets = layout[..., None] * torch.Tensor([1, 0, 0])
    assert offsets.shape == layout.shape + (3,)
    assert not (offsets[..., 0] == 0).all()
    assert (offsets[..., 1] == 0).all()
    assert (offsets[..., 2] == 0).all()

    return offsets


@pytest.fixture
def multilayer_test_coords(multilayer_test_offsets):
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

    offsets = multilayer_test_offsets

    coords = offsets[:, :, None, :] + cluster_coords
    assert coords.shape == offsets.shape[:2] + cluster_coords.shape
    assert ((coords[0, 0, :6] == cluster_coords[:6] + torch.Tensor([-8, 0, 0]))).all()

    return coords.reshape((4, 8 * 3, 3))


@reactive_attrs(auto_attribs=True)
class ThresholdDistanceCountIntraScore(IntraScoreGraph):
    @reactive_property
    def total_threshold_count(target):
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
class ThresholdDistanceCountInterScore(InterScoreGraph):
    @reactive_property
    def total_threshold_count(target_i, target_j):
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


@pytest.fixture(params=[BlockedInteratomicDistanceGraph], ids=["blocked"])
def threshold_distance_score_class(request):
    interatomic_distance_component = request.param

    @reactive_attrs
    class TestGraph(ThresholdDistanceCount, interatomic_distance_component):
        pass

    return TestGraph
