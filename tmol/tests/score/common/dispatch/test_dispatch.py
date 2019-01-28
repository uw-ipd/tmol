import pytest

import numpy
import torch

from scipy.spatial.distance import cdist

from tmol.utility.cpp_extension import load, relpaths, modulename, cuda_if_available
from tmol.tests.benchmark import subfixture, make_subfixture

import sparse


@pytest.mark.benchmark(group="dispatch")
@pytest.mark.parametrize("dispatch_type", [
    "exhaustive",
    "naive",
    "exhaustive_omp",
    "aabb",
])
def test_dispatch(benchmark, dispatch_type, torch_device, ubq_system):
    compiled = load(
        modulename(__name__),
        cuda_if_available(
            relpaths(__file__, ["test.cpp", "test.pybind.cpp", "test.cu"])
        ),
    )

    @subfixture(benchmark)
    def scipy_dist():
        return cdist(ubq_system.coords, ubq_system.coords)

    scipy_dist[numpy.isnan(scipy_dist)] = 10.0

    coords = torch.from_numpy(ubq_system.coords).to(device=torch_device)

    dispatch_name = f"{dispatch_type}_dispatch"
    dispatch_func = getattr(compiled, dispatch_name)

    @make_subfixture(benchmark, f".{dispatch_name}")
    def dispatched():
        return dispatch_func(coords)

    dind, dscore = dispatched
    numpy.testing.assert_array_equal(
        sparse.COO(
            dind.cpu().numpy().T, dscore.cpu().numpy(), scipy_dist.shape
        ).todense(),
        scipy_dist < 6.0,
    )


@pytest.mark.benchmark(group="dispatch")
@pytest.mark.parametrize("dispatch_type", [
    "exhaustive",
    "naive",
])
def test_triu_dispatch(benchmark, dispatch_type, torch_device, ubq_system):
    compiled = load(
        modulename(__name__),
        cuda_if_available(
            relpaths(__file__, ["test.cpp", "test.pybind.cpp", "test.cu"])
        ),
    )

    @subfixture(benchmark)
    def scipy_dist():
        return cdist(ubq_system.coords, ubq_system.coords)

    scipy_dist[numpy.isnan(scipy_dist)] = 10.0

    coords = torch.from_numpy(ubq_system.coords).to(device=torch_device)

    dispatch_name = f"{dispatch_type}_triu_dispatch"
    dispatch_func = getattr(compiled, dispatch_name)

    @make_subfixture(benchmark, f".{dispatch_name}")
    def dispatched():
        return dispatch_func(coords)

    dind, dscore = dispatched
    numpy.testing.assert_array_equal(
        sparse.COO(
            dind.cpu().numpy().T, dscore.cpu().numpy(), scipy_dist.shape
        ).todense(),
        numpy.triu(scipy_dist < 6.0),
    )
