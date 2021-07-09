import threading

import pytest

import torch
import torch.cuda

cuda_available = torch.cuda.is_available()

requires_cuda = pytest.mark.skipif(not cuda_available, reason="Requires cuda.")


@pytest.fixture(params=[requires_cuda("cuda")])
# @pytest.fixture(params=["cpu", requires_cuda("cuda")])
# @pytest.fixture(params=["cpu"])
def torch_device(request):
    """Paramterized test fixure covering cpu & cuda torch devices."""

    if request.param == "cpu":
        device = torch.device("cpu")
    elif request.param == "cuda":
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        raise NotImplementedError

    # Perform a "warmup" computation on the device, ensuring that it is
    # initialized and available.
    torch.arange(100, device=device).sum()

    return device


def cuda_not_implemented(f):
    """Parametrize 'torch_device' as an xfail via NotImplementedError."""
    return pytest.mark.parametrize(
        "torch_device",
        [
            (torch.device("cpu")),
            pytest.param(
                torch.device("cuda"),
                marks=[
                    requires_cuda,
                    pytest.mark.xfail(strict=True, raises=NotImplementedError),
                ],
            ),
        ],
    )(f)


@pytest.fixture
def torch_backward_coverage(cov):
    """Torch hook to enable coverage in backward pass.

    Returns a hook function used to enable coverage tracing during
    pytorch backward passes. Torch runs all backward passes in a
    non-main thread, not spawned by the standard 'threading'
    interface, so coverage does not trace the thread.

    Example:

    result = custom_func(input)

    # enable the hook
    result.register_hook(torch_backward_coverage)

    # call backward via sum so hook fires before custom_op backward
    result.sum().backward()
    """

    if cov:
        cov.collector.added_tracers = {threading.get_ident()}

        def add_tracer(_):
            tid = threading.get_ident()
            if tid not in cov.collector.added_tracers:
                print(f"pytorch backward trace: {tid}")
                cov.collector.added_tracers.add(tid)
                cov.collector._start_tracer()

    else:

        def add_tracer(_):
            pass

    return add_tracer
