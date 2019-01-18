import numpy
import torch

from scipy.spatial.distance import cdist

from tmol.utility.cpp_extension import load, relpaths, modulename
from tmol.tests.benchmark import subfixture, make_subfixture

import sparse


def test_cpu_dispatch(benchmark, ubq_system):
    compiled = load(modulename(__name__), relpaths(__file__, "test_dispatch.cpp"))

    @subfixture(benchmark)
    def scipy_dist():
        idist = cdist(ubq_system.coords, ubq_system.coords)
        idist[numpy.isnan(idist)] = 10.0
        return idist

    dispatch_types = [n for n in dir(compiled) if "dispatch" in n]

    assert len(dispatch_types) == 2

    for dispatch_name in dispatch_types:
        dispatch = getattr(compiled, dispatch_name)

        @make_subfixture(benchmark, f".{dispatch_name}")
        def dispatched():
            dind, dscore = dispatch(torch.from_numpy(ubq_system.coords))
            return sparse.COO(dind.numpy().T, dscore.numpy(), scipy_dist.shape)

        numpy.testing.assert_array_equal(dispatched.todense(), scipy_dist < 6.0)
