import pytest
import numpy

from scipy.spatial.distance import pdist, cdist, squareform

from argparse import Namespace


@pytest.mark.benchmark(group="interatomic_distance_calculation")
def test_interatomic_distance_stacked(
    multilayer_test_coords, threshold_distance_score_class, torch_device, seterr_ignore
):
    threshold_distance = 6.0
    tc = multilayer_test_coords

    intra_layer_counts = [
        numpy.nansum(pdist(tc[l]) < threshold_distance) for l in range(len(tc))
    ]

    inter_layer_counts = [
        [numpy.nansum(cdist(tc[i], tc[j]) < threshold_distance) for j in range(len(tc))]
        for i in range(len(tc))
    ]

    score_state = threshold_distance_score_class.build_for(
        Namespace(
            stack_depth=multilayer_test_coords.shape[0],
            system_size=multilayer_test_coords.shape[1],
            coords=multilayer_test_coords,
            threshold_distance=6.0,
            atom_pair_block_size=8,
            device=torch_device,
        )
    )

    intra_total = threshold_distance_score_class.intra_score(score_state).total

    assert intra_total.shape == (4,)
    assert (intra_total.new_tensor(intra_layer_counts) == intra_total).all()

    inter_total = threshold_distance_score_class.inter_score(
        score_state, score_state
    ).total

    assert inter_total.shape == (4, 4)
    assert (inter_total.new_tensor(inter_layer_counts) == inter_total).all()


@pytest.mark.benchmark(group="interatomic_distance_calculation")
def test_interatomic_distance_ubq_smoke(
    benchmark, ubq_system, threshold_distance_score_class, torch_device, seterr_ignore
):
    dgraph = threshold_distance_score_class.build_for(
        ubq_system, drop_missing_atoms=True, device=torch_device
    )

    scipy_distance = pdist(ubq_system.coords)
    scipy_count = numpy.nansum(scipy_distance < 6.0)

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
        return dgraph.intra_score(dgraph).total

    assert total_score.shape == (1,)
    assert (scipy_count == total_score).all()
