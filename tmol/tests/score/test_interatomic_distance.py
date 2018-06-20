import pytest
import torch
import numpy

from scipy.spatial.distance import pdist, squareform

from tmol.score.factory import Factory

from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.total_score import ScoreComponentAttributes, TotalScoreComponentsGraph
from tmol.score.interatomic_distance import (
    InteratomicDistanceGraphBase,
    NaiveInteratomicDistanceGraph,
    BlockedInteratomicDistanceGraph,
    BlockedDistanceAnalysis,
)

from tmol.utility.reactive import reactive_attrs, reactive_property

from math import nan

from argparse import Namespace


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


# def test_blocked_interatomic_distance_nulls(multilayer_test_coords):
#     """Test that interatomic distance properly handles null blocks."""
#     null_padded = multilayer_test_coords.new_full((4, 24 + 8, 3), nan)
#     null_padded[:, :24, :] = multilayer_test_coords

#     bda = BlockedDistanceAnalysis.setup(
#         block_size=8, coords=multilayer_test_coords
#     )
#     np_bda = BlockedDistanceAnalysis.setup(block_size=8, coords=null_padded)
#     assert (bda.block_triu_inds == np_bda.block_triu_inds).all()


def test_blocked_interatomic_distance_layered(multilayer_test_coords):
    threshold_distance = 6.0

    # dense_expected_block_interactions = torch.Tensor([
    #     [  #0-------1------2
    #         [1, 0, 0],
    #         [0, 1, 0],
    #         [0, 0, 1],
    #     ],
    #     [  #2-------1------0
    #         [1, 0, 0],
    #         [0, 1, 0],
    #         [0, 0, 1],
    #     ],
    #     [  #----0---1---2---
    #         [1, 1, 0],
    #         [1, 1, 1],
    #         [0, 1, 1],
    #     ],
    #     [  #-------012------
    #         [1, 1, 1],
    #         [1, 1, 1],
    #         [1, 1, 1],
    #     ],
    # ]).to(dtype=torch.uint8)

    triu_expected_block_interactions = torch.Tensor(
        [[2, 0], [2, 2], [3, 0], [3, 1], [3, 2]]
    ).to(dtype=torch.long)

    bda = BlockedDistanceAnalysis.setup(coords=multilayer_test_coords, block_size=8)
    assert (
        bda.block_triu_inds
        == torch.Tensor([[0, 1], [0, 2], [1, 2]]).to(dtype=torch.long)
    ).all()  # yapf: disable
    assert (
        (bda.block_triu_min_dist < threshold_distance).nonzero()
        == triu_expected_block_interactions
    ).all()  # yapf: disable


@reactive_attrs(auto_attribs=True)
class ThresholdDistanceCount(
    CartesianAtomicCoordinateProvider,
    InteratomicDistanceGraphBase,
    TotalScoreComponentsGraph,
    Factory,
):
    threshold_distance: float

    def factory_for(obj, **_):
        return dict(threshold_distance=6.0)

    @property
    def component_total_score_terms(self):
        return ScoreComponentAttributes(
            name="threshold_count", total="total_threshold_count"
        )

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

    @staticmethod
    def test_class_for(interatomic_distance_component):
        @reactive_attrs
        class TestGraph(ThresholdDistanceCount, interatomic_distance_component):
            @reactive_property
            def system_size(coords) -> int:
                return len(coords)

        return TestGraph


@pytest.mark.benchmark(group="interatomic_distance_calculation")
@pytest.mark.parametrize(
    "interatomic_distance_component",
    [NaiveInteratomicDistanceGraph, BlockedInteratomicDistanceGraph],
    ids=["naive", "blocked"],
)
def test_interatomic_distance_stacked(
    multilayer_test_coords, interatomic_distance_component, torch_device
):
    expected_counts = []
    for l in range(len(multilayer_test_coords)):
        scipy_distance = pdist(multilayer_test_coords[l])
        scipy_count = numpy.count_nonzero(
            scipy_distance[~numpy.isnan(pdist(multilayer_test_coords[l]))] < 6.0
        )
        expected_counts.append(scipy_count)

    total_score = (
        ThresholdDistanceCount.test_class_for(interatomic_distance_component)
        .build_for(
            Namespace(
                stack_depth=multilayer_test_coords.shape[0],
                system_size=multilayer_test_coords.shape[1],
                coords=multilayer_test_coords,
                threshold_distance=6.0,
                atom_pair_block_size=8,
                device=torch_device,
            )
        )
        .total_score
    )

    assert total_score.shape == (4,)
    assert (total_score.new_tensor(expected_counts) == total_score).all()


@pytest.mark.benchmark(group="interatomic_distance_calculation")
@pytest.mark.parametrize(
    "interatomic_distance_component",
    [NaiveInteratomicDistanceGraph, BlockedInteratomicDistanceGraph],
    ids=["naive", "blocked"],
)
def test_interatomic_distance_ubq_smoke(
    benchmark, ubq_system, interatomic_distance_component, torch_device
):
    dgraph = ThresholdDistanceCount.test_class_for(
        interatomic_distance_component
    ).build_for(ubq_system, drop_missing_atoms=True, device=torch_device)

    scipy_distance = pdist(ubq_system.coords)
    scipy_count = numpy.count_nonzero(
        scipy_distance[~numpy.isnan(scipy_distance)] < 6.0
    )

    layer = dgraph.atom_pair_inds[:, 0]
    fa = dgraph.atom_pair_inds[:, 1]
    ta = dgraph.atom_pair_inds[:, 2]

    assert (layer == 0).all()

    numpy.testing.assert_allclose(
        numpy.nan_to_num(squareform(scipy_distance)[fa, ta]),
        numpy.nan_to_num(dgraph.atom_pair_dist.detach()),
        rtol=1e-4,
    )

    @benchmark
    def total_score():
        # Reset graph by setting coord values,
        # triggering full recalc.
        dgraph.coords = dgraph.coords

        # Calculate total score, rather than atom pair distances
        # As naive implemenation returns a more precise set of distances
        # to the resulting score function.
        return dgraph.total_score

    assert total_score.shape == (1,)
    assert (scipy_count == total_score).all()
